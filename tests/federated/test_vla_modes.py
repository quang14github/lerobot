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
"""Each fine-tuning mode of the VLA study trains exactly the intended SmolVLA parameters.

The policy is built on the meta device (no weights, no GPU) from the very CLI flags the study
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
from lerobot.federated.vla_study import LAST_BACKBONE_LAYER, LORA_FULL_RANK_MODULES, peft_args, policy_args

pytest.importorskip("peft")
pytest.importorskip("transformers")

# Exact trainable counts, kept here so any change to the definitions is a visible diff.
EXPECTED = {"full": 392_904_096, "lora": 42_361_680}
# Adapted Linear layers: (vision encoder, language backbone, action expert). The backbone's last
# layer only gets k/v adapters (2 of its 7 projections): 15 x 7 + 2 = 107.
EXPECTED_ADAPTERS = (72, 107, 112)


@parser.wrap()
def _parse(cfg: FederatedTrainPipelineConfig) -> FederatedTrainPipelineConfig:
    return cfg


def build(mode: str, tmp_path: Path):
    """The policy the trainers would build for `mode`, on the meta device."""
    from lerobot.policies.factory import get_policy_class

    args = [
        "--dataset.repo_id=lerobot/libero",
        f"--output_dir={tmp_path / 'run'}",
        *policy_args(mode, device="cpu"),
        *peft_args(mode),
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


@pytest.mark.parametrize("mode", list(EXPECTED))
def test_trainable_parameter_count(mode, tmp_path):
    _, model = build(mode, tmp_path)
    assert sum(trainable(model).values()) == EXPECTED[mode]


def test_full_unfreezes_the_backbone(tmp_path):
    _, model = build("full", tmp_path)
    names = trainable(model)
    for part in ("vision_model", "text_model", "lm_expert", "connector", "action_out_proj"):
        assert any(part in n for n in names), part


def test_lora_adapts_vision_backbone_and_action_head(tmp_path):
    _, model = build("lora", tmp_path)
    names = trainable(model)
    adapted = {n.rsplit(".lora_A", 1)[0] for n in names if ".lora_A." in n}
    components = Counter()
    for n in adapted:
        if re.search(r"vision_model", n):
            components["vision"] += 1
        elif "lm_expert" in n:
            components["expert"] += 1
        else:
            components["backbone"] += 1
    assert (components["vision"], components["backbone"], components["expert"]) == EXPECTED_ADAPTERS
    last = f"text_model.layers.{LAST_BACKBONE_LAYER}."
    assert sorted(n.split(last)[1] for n in adapted if last in n) == ["self_attn.k_proj", "self_attn.v_proj"]
    # Everything trainable is an adapter or one of the listed full-rank modules.
    others = {n for n in names if ".lora_A." not in n and ".lora_B." not in n}
    assert others and all(any(f".{m}." in n for m in LORA_FULL_RANK_MODULES) for n in others), sorted(others)[
        :5
    ]
    assert {m for m in LORA_FULL_RANK_MODULES if any(f".{m}." in n for n in others)} == set(
        LORA_FULL_RANK_MODULES
    )


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
