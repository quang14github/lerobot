# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""How the federated VLA study builds and fine-tunes the pretrained SmolVLA policy.

Two modes, both starting from `lerobot/smolvla_base` (the whole pretrained policy, not the
VLM-only initialization):

- ``full``: backbone and action head (`train_expert_only=false`, `freeze_vision_encoder=false`):
  0.393B trainable parameters. This is not SmolVLA's default recipe, which trains only the action
  expert (99.9M), but matches "backbone and action head" and the 0.40B the study's reference
  (DECENT-VLA) reports. SmolVLA's own code still freezes the parts of its VLM that the action expert
  never reads (`lm_head`, the final norm and the last layer).
- ``lora``: rank-64, alpha-128 LoRA on every attention and MLP projection of the vision encoder,
  the language backbone and the action expert, plus the small state/action/time projections trained
  at full rank (a rank-64 adapter on a 32-wide projection would be larger than the layer itself).
  In the backbone's last layer only the key and value projections are adapted: the action expert
  reads nothing else of that layer, so other adapters there would never get a gradient.

The CLI flags returned here are used verbatim by both the federated and the centralized arm, and the
trainable sets they produce are pinned by `tests/federated/test_vla_modes.py`.
"""

POLICY = "smolvla"
PRETRAINED = "lerobot/smolvla_base"
MODES = ("lora", "full")

LORA_RANK = 64
LORA_ALPHA = 128

# smolvla_base keeps the first 16 layers of SmolVLM2 (num_vlm_layers=16).
LAST_BACKBONE_LAYER = 15

_ATTN_MLP = r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
_VISION = r"(self_attn\.(q|k|v|out)_proj|mlp\.fc[12])"
_BACKBONE = r"vlm\.model\.text_model\.layers"

# PEFT matches this against module names of the policy (before PEFT adds its own prefix).
LORA_TARGETS = (
    r"model\.vlm_with_expert\.("
    rf"{_BACKBONE}\.(?!{LAST_BACKBONE_LAYER}\.)\d+\.{_ATTN_MLP}"
    rf"|{_BACKBONE}\.{LAST_BACKBONE_LAYER}\.self_attn\.(k|v)_proj"
    rf"|lm_expert\.layers\.\d+\.{_ATTN_MLP}"
    rf"|vlm\.model\.vision_model\.encoder\.layers\.\d+\.{_VISION})"
)

# Trained (and federated) at full rank alongside the adapters.
LORA_FULL_RANK_MODULES = [
    "state_proj",
    "action_in_proj",
    "action_out_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
]

# Actions executed between replans at evaluation. Training is unaffected (the chunk stays 50).
N_ACTION_STEPS = 10


def policy_args(mode: str, device: str = "cuda", dtype: str = "float32") -> list[str]:
    """`--policy.*` flags that build SmolVLA from its pretrained checkpoint for `mode`.

    `dtype="null"` keeps SmolVLA's own bfloat16 weights, as in its published code.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    return [
        f"--policy.path={PRETRAINED}",
        # Features come from the dataset, not from smolvla_base's SO-100 camera layout.
        "--policy.input_features=null",
        "--policy.output_features=null",
        # Both modes: LoRA adapters inside the VLM need it in train mode and on the graph.
        "--policy.train_expert_only=false",
        "--policy.freeze_vision_encoder=false",
        # Train float32 weights (compute runs in bf16 autocast): smolvla_base is stored in bfloat16,
        # where an AdamW step of 1e-4 rounds to no change for about half of the weights.
        f"--policy.dtype={dtype}",
        "--policy.push_to_hub=false",
        f"--policy.device={device}",
        f"--policy.n_action_steps={N_ACTION_STEPS}",
    ]


def peft_args(mode: str) -> list[str]:
    """`--peft.*` flags for `mode` (none for full fine-tuning)."""
    if mode == "full":
        return []
    return [
        "--peft.method_type=LORA",
        f"--peft.r={LORA_RANK}",
        f"--peft.lora_alpha={LORA_ALPHA}",
        f"--peft.target_modules={LORA_TARGETS}",
        f"--peft.full_training_modules=[{','.join(LORA_FULL_RANK_MODULES)}]",
    ]


def scheduler_args(rounds: int, local_steps: int, num_clients: int, arm: str) -> list[str]:
    """Put both arms on the same LR-schedule shape relative to their budgets.

    SmolVLA's preset warms up for 1,000 steps and anneals over 30,000 (1/30 of the horizon). The
    federated LR advances once per round, so its horizon is `rounds`; the centralized arm takes
    `rounds * local_steps * num_clients` steps. Stretching both to their own budget keeps peak and
    floor LR unchanged and puts every round-equivalent checkpoint of both arms at the same point of
    the schedule. Left alone, the centralized arm would anneal within its first 30,000 steps and
    spend the remaining 88% of a 250,000-step budget at the floor LR.
    """
    warmup_rounds = int(1_000 * rounds / 30_000)
    if arm == "federated":
        return [
            f"--policy.scheduler_warmup_steps={warmup_rounds}",
            f"--policy.scheduler_decay_steps={rounds}",
        ]
    per_round = local_steps * num_clients
    return [
        f"--policy.scheduler_warmup_steps={warmup_rounds * per_round}",
        f"--policy.scheduler_decay_steps={rounds * per_round}",
    ]
