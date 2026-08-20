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
"""Every federated fail-fast should fire at config time, not hours into a run."""

import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.federated.config import FederatedConfig, FederatedTrainPipelineConfig, PartitionConfig
from lerobot.policies.act.configuration_act import ACTConfig


def make_config(tmp_path, **overrides) -> FederatedTrainPipelineConfig:
    """A minimal valid federated config; ACT keeps construction cheap."""
    kwargs = {
        "dataset": DatasetConfig(repo_id="lerobot/dummy"),
        # push_to_hub defaults to True with no repo_id, which the inherited validation rejects.
        "policy": ACTConfig(push_to_hub=False),
        "output_dir": tmp_path / "run",
        "fl": FederatedConfig(rounds=10, local_steps=5),
        "partition": PartitionConfig(num_clients=4),
    }
    kwargs.update(overrides)
    return FederatedTrainPipelineConfig(**kwargs)


def test_scheduler_horizon_is_the_round_count(tmp_path):
    """The LR schedule advances once per round, so `steps` must mean rounds here.

    Leaving `steps` at its centralized default would stretch the decay over 100k rounds and
    pin the LR near its peak for the entire run.
    """
    cfg = make_config(tmp_path, fl=FederatedConfig(rounds=37, local_steps=5))
    cfg.validate()

    assert cfg.steps == 37


def test_accepts_a_reasonable_config(tmp_path):
    make_config(tmp_path).validate()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"fl": FederatedConfig(rounds=0)}, "fl.rounds must be >= 1"),
        ({"fl": FederatedConfig(local_steps=0)}, "fl.local_steps must be >= 1"),
        ({"fl": FederatedConfig(clients_per_round=99)}, "fl.clients_per_round must be in"),
        ({"fl": FederatedConfig(client_weighting="frame")}, "fl.client_weighting must be"),
        ({"fl": FederatedConfig(prox_mu=-1.0)}, "fl.prox_mu must be >= 0"),
    ],
)
def test_rejects_invalid_federated_settings(tmp_path, overrides, match):
    with pytest.raises(ValueError, match=match):
        make_config(tmp_path, **overrides).validate()


def test_rejects_unknown_server_optimizer(tmp_path):
    from lerobot.federated.config import ServerConfig

    with pytest.raises(ValueError, match="Unknown server.type"):
        make_config(tmp_path, server=ServerConfig(type="fedsgd")).validate()


def test_rejects_env_eval_without_an_env(tmp_path):
    with pytest.raises(ValueError, match="requires an environment"):
        make_config(tmp_path, fl=FederatedConfig(env_eval_freq_rounds=5)).validate()


def test_rejects_streaming_datasets(tmp_path):
    """Client shards are episode allowlists, which a streaming dataset cannot honor."""
    with pytest.raises(NotImplementedError, match="streaming"):
        make_config(tmp_path, dataset=DatasetConfig(repo_id="lerobot/dummy", streaming=True)).validate()


def test_rejects_ema(tmp_path):
    """An EMA shadow would track whichever client trained last, not the global model."""
    cfg = make_config(tmp_path)
    cfg.ema.enable = True

    with pytest.raises(NotImplementedError, match="ema.enable"):
        cfg.validate()


def test_clients_per_round_zero_means_all(tmp_path):
    make_config(tmp_path, fl=FederatedConfig(clients_per_round=0)).validate()
