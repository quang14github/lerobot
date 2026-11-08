#!/usr/bin/env python

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
"""Each fine-tuning mode of the VLA study trains exactly the intended parameters.

The policies are built on the meta device (no weights, no GPU) from the very CLI flags the study
passes to the trainers, so these tests also cover how those flags parse.
"""

import dataclasses
import re
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from lerobot.common.train_utils import verify_pretrained_weights
from lerobot.configs import FeatureType, PolicyFeature, parser
from lerobot.federated.config import FederatedTrainPipelineConfig
from lerobot.federated.tasks import BENCHMARKS
from lerobot.federated.vla_study import LORA_FULL_RANK_MODULES, peft_args, policy_args

pytest.importorskip("peft")
pytest.importorskip("transformers")

LIBERO = BENCHMARKS["libero_object"]

# Exact trainable counts, kept here so any change to the definitions is a visible diff.
EXPECTED = {
    ("pi05", "full"): 3_353_433_872,
    ("pi05", "lora"): 143_115_296,
    ("smolvla", "full"): 392_904_096,
    ("smolvla", "lora"): 43_283_280,
}
# Adapted Linear layers per component: (vision encoder, language backbone, action expert).
EXPECTED_ADAPTERS = {"pi05": (162, 126, 126), "smolvla": (72, 112, 112)}


@parser.wrap()
def _parse(cfg: FederatedTrainPipelineConfig) -> FederatedTrainPipelineConfig:
    return cfg


def build(policy: str, mode: str, tmp_path: Path):
    """The policy the trainers would build for (policy, mode), on the meta device."""
    from lerobot.policies.factory import get_policy_class

    args = [
        f"--dataset.repo_id={LIBERO.dataset}",
        f"--output_dir={tmp_path / 'run'}",
        *policy_args(policy, mode, LIBERO, device="cpu"),
        *peft_args(policy, mode),
    ]
    try:
        with patch.object(sys, "argv", ["lerobot-fl-train", *args]):
            cfg = _parse()
            cfg.validate()
    except OSError as err:  # smolvla_base's config.json is not in the cache and there is no network
        pytest.skip(f"cannot resolve the pretrained config: {err}")
    cfg.policy.input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    cfg.policy.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    if policy == "smolvla":
        cfg.policy.load_vlm_weights = False  # structure only; the weights come from smolvla_base
    cfg.policy.device = "meta"
    with torch.device("meta"):
        try:
            model = get_policy_class(cfg.policy.type)(cfg.policy)
        except OSError as err:
            pytest.skip(f"cannot resolve the VLM config: {err}")
        if cfg.peft is not None:
            model = model.wrap_with_peft(peft_cli_overrides=dataclasses.asdict(cfg.peft))
    return cfg, model


def trainable(model) -> dict[str, int]:
    return {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}


@pytest.mark.parametrize(("policy", "mode"), list(EXPECTED))
def test_trainable_parameter_count(policy, mode, tmp_path):
    _, model = build(policy, mode, tmp_path)
    assert sum(trainable(model).values()) == EXPECTED[(policy, mode)]


def test_pi05_full_trains_everything_but_the_unused_lm_heads(tmp_path):
    _, model = build("pi05", "full", tmp_path)
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert sorted(frozen) == [
        "model.paligemma_with_expert.gemma_expert.lm_head.weight",
        "model.paligemma_with_expert.paligemma.lm_head.weight",
    ]
    names = trainable(model)
    assert any("vision_tower" in n for n in names) and any("embed_tokens" in n for n in names)


def test_smolvla_full_unfreezes_the_backbone(tmp_path):
    _, model = build("smolvla", "full", tmp_path)
    names = trainable(model)
    for part in ("vision_model", "text_model", "lm_expert", "connector", "action_out_proj"):
        assert any(part in n for n in names), part


@pytest.mark.parametrize("policy", ["pi05", "smolvla"])
def test_lora_adapts_vision_backbone_and_action_head(policy, tmp_path):
    _, model = build(policy, "lora", tmp_path)
    names = trainable(model)
    adapted = {n.rsplit(".lora_A", 1)[0] for n in names if ".lora_A." in n}
    components = Counter()
    for n in adapted:
        if re.search(r"vision_(tower|model)", n):
            components["vision"] += 1
        elif "gemma_expert" in n or "lm_expert" in n:
            components["expert"] += 1
        else:
            components["backbone"] += 1
    assert (components["vision"], components["backbone"], components["expert"]) == EXPECTED_ADAPTERS[policy]
    # Everything trainable is an adapter or one of the listed full-rank modules.
    others = {n for n in names if ".lora_A." not in n and ".lora_B." not in n}
    full_rank = LORA_FULL_RANK_MODULES[policy]
    assert others and all(any(f".{m}." in n for m in full_rank) for n in others), sorted(others)[:5]
    assert {m for m in full_rank if any(f".{m}." in n for n in others)} == set(full_rank)


def test_guard_accepts_loaded_weights_and_rejects_random_ones(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 2))
    save_file(
        {k: v.clone().to(torch.bfloat16) for k, v in model.state_dict().items()},
        tmp_path / "model.safetensors",
    )
    model.load_state_dict({k: v.to(torch.bfloat16).float() for k, v in model.state_dict().items()})
    assert verify_pretrained_weights(model, tmp_path) == 4
    with torch.no_grad():
        model[1].weight.add_(1.0)
    with pytest.raises(RuntimeError, match="does not hold its pretrained weights"):
        verify_pretrained_weights(model, tmp_path)
