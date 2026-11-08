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
"""How the federated VLA study builds and fine-tunes each pretrained policy.

Two policies, two modes:

- ``full``: the parameters the policy's fine-tuning recipe trains, backbone and action head included.
  SmolVLA unfreezes its VLM (`train_expert_only=false`, `freeze_vision_encoder=false`), which is not
  SmolVLA's default (action head only) but matches the 0.40B the study's reference reports. pi0.5
  trains everything except its two `lm_head`s, which the action model never uses and which would
  otherwise sit in the optimizer and in every federated update with no gradient.
- ``lora``: rank-64, alpha-128 LoRA on every attention and MLP projection of the vision encoder,
  the language backbone and the action expert, plus the small action/time projections trained at
  full rank (a rank-64 adapter on a 32-wide projection would be larger than the layer itself).

The CLI flags returned here are used verbatim by both the federated and the centralized arm, and the
trainable sets they produce are pinned by `tests/federated/test_vla_modes.py`.
"""

import json

from .tasks import BenchmarkSpec

POLICIES = ("smolvla", "pi05")
MODES = ("lora", "full")

LORA_RANK = 64
LORA_ALPHA = 128

PRETRAINED = {
    "smolvla": "lerobot/smolvla_base",
    "pi05": "lerobot/pi05_base",
}

_ATTN_MLP = r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
_VISION = r"(self_attn\.(q|k|v|out)_proj|mlp\.fc[12])"

# PEFT matches these against module names of the policy (before PEFT adds its own prefix).
LORA_TARGETS = {
    "pi05": (
        r"model\.paligemma_with_expert\.("
        rf"paligemma\.model\.language_model\.layers\.\d+\.{_ATTN_MLP}"
        rf"|gemma_expert\.model\.layers\.\d+\.{_ATTN_MLP}"
        rf"|paligemma\.model\.vision_tower\.vision_model\.encoder\.layers\.\d+\.{_VISION})"
    ),
    "smolvla": (
        r"model\.vlm_with_expert\.("
        rf"vlm\.model\.text_model\.layers\.\d+\.{_ATTN_MLP}"
        rf"|lm_expert\.layers\.\d+\.{_ATTN_MLP}"
        rf"|vlm\.model\.vision_model\.encoder\.layers\.\d+\.{_VISION})"
    ),
}

# Trained (and federated) at full rank alongside the adapters.
LORA_FULL_RANK_MODULES = {
    "pi05": ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"],
    "smolvla": [
        "state_proj",
        "action_in_proj",
        "action_out_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
    ],
}

# Actions executed between replans at evaluation. Training is unaffected (the chunk stays 50).
N_ACTION_STEPS = 10

# pi0.5 always fills three image slots (base + two wrists); missing cameras are added empty.
PI05_IMAGE_SLOTS = 3

PI05_NORMALIZATION = {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}


def num_cameras(bench: BenchmarkSpec) -> int:
    """Cameras in the benchmark's dataset (and eval env)."""
    return 2 if bench.env_type == "libero" else 1


def policy_args(policy: str, mode: str, bench: BenchmarkSpec, device: str = "cuda") -> list[str]:
    """`--policy.*` flags that build `policy` from its pretrained checkpoint for `mode`."""
    if policy not in POLICIES or mode not in MODES:
        raise ValueError(f"unknown policy/mode {policy}/{mode}")
    common = [
        "--policy.push_to_hub=false",
        f"--policy.device={device}",
        f"--policy.n_action_steps={N_ACTION_STEPS}",
    ]
    if policy == "smolvla":
        return [
            f"--policy.path={PRETRAINED['smolvla']}",
            # Features come from the dataset, not from smolvla_base's SO-100 camera layout.
            "--policy.input_features=null",
            "--policy.output_features=null",
            # Both modes: LoRA adapters inside the VLM need it in train mode and on the graph.
            "--policy.train_expert_only=false",
            "--policy.freeze_vision_encoder=false",
            *common,
        ]
    return [
        "--policy.type=pi05",
        f"--policy.pretrained_path={PRETRAINED['pi05']}",
        f"--policy.normalization_mapping={json.dumps(PI05_NORMALIZATION)}",
        f"--policy.empty_cameras={PI05_IMAGE_SLOTS - num_cameras(bench)}",
        "--policy.dtype=float32",
        "--policy.gradient_checkpointing=true",
        "--policy.freeze_vision_encoder=false",
        "--policy.train_expert_only=false",
        "--policy.freeze_lm_heads=true",
        *common,
    ]


def peft_args(policy: str, mode: str) -> list[str]:
    """`--peft.*` flags for `mode` (none for full fine-tuning)."""
    if mode == "full":
        return []
    return [
        "--peft.method_type=LORA",
        f"--peft.r={LORA_RANK}",
        f"--peft.lora_alpha={LORA_ALPHA}",
        f"--peft.target_modules={LORA_TARGETS[policy]}",
        f"--peft.full_training_modules=[{','.join(LORA_FULL_RANK_MODULES[policy])}]",
    ]


def scheduler_args(rounds: int, local_steps: int, num_clients: int, arm: str) -> list[str]:
    """Put both arms on the same LR-schedule shape relative to their budgets.

    Both policies' presets warm up for 1,000 steps and anneal over 30,000 (1/30 of the horizon).
    The federated LR advances once per round, so its horizon is `rounds`; the centralized arm takes
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
