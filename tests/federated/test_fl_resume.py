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
"""A federated run stopped mid-way and resumed ends bit-identical to an uninterrupted one (CPU)."""

import csv
import json

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")
pytest.importorskip("diffusers", reason="diffusers is required (install lerobot[diffusion])")

from safetensors.torch import load_file

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.federated.config import FederatedConfig, FederatedTrainPipelineConfig, PartitionConfig
from lerobot.federated.run_state import TrainingInterrupted

REPO_ID = "dummy/fl_resume"
N_TASKS, EPISODES_PER_TASK, EPISODE_LENGTH = 3, 2, 10


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("fl_resume") / "dataset"
    features = {
        "action": {"dtype": "float32", "shape": (4,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (4,), "names": None},
        "observation.images.top": {
            "dtype": "image",
            "shape": (32, 32, 3),
            "names": ["height", "width", "channel"],
        },
    }
    dataset = LeRobotDataset.create(repo_id=REPO_ID, fps=30, features=features, root=root)
    rng = np.random.default_rng(0)
    for task in range(N_TASKS):
        for _ in range(EPISODES_PER_TASK):
            for _ in range(EPISODE_LENGTH):
                dataset.add_frame(
                    {
                        "action": rng.standard_normal(4).astype(np.float32),
                        "observation.state": rng.standard_normal(4).astype(np.float32),
                        "observation.images.top": rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8),
                        "task": f"task_{task}",
                    }
                )
            dataset.save_episode()
    dataset.finalize()
    return root


def make_cfg(root, output_dir, *, stop_after_rounds=0, reset_optimizer=True, local_steps=3):
    from lerobot.policies.factory import make_policy_config

    policy = make_policy_config(
        "diffusion",
        device="cpu",
        push_to_hub=False,
        n_obs_steps=2,
        horizon=8,
        n_action_steps=4,
        drop_n_last_frames=0,
        down_dims=(32, 64),
        diffusion_step_embed_dim=32,
        spatial_softmax_num_keypoints=8,
        num_inference_steps=2,
        pretrained_backbone_weights=None,
        use_group_norm=True,
    )
    return FederatedTrainPipelineConfig(
        dataset=DatasetConfig(repo_id=REPO_ID, root=str(root)),
        policy=policy,
        output_dir=output_dir,
        batch_size=2,
        num_workers=0,
        seed=7,
        log_freq=0,
        env_eval_freq=0,
        fl=FederatedConfig(
            rounds=4,
            local_steps=local_steps,
            save_freq_rounds=2,
            auto_resume=True,
            stop_after_rounds=stop_after_rounds,
            reset_optimizer_each_round=reset_optimizer,
            client_weighting="uniform",
        ),
        partition=PartitionConfig(strategy="iid", num_clients=2, seed=1),
    )


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def final_weights(output_dir):
    return load_file(output_dir / "checkpoints" / "000004" / "pretrained_model" / "model.safetensors")


@pytest.mark.parametrize("reset_optimizer", [True, False])
def test_stop_and_resume_matches_an_uninterrupted_run(dataset_root, tmp_path, reset_optimizer):
    from lerobot.scripts.lerobot_fl_train import fl_train

    straight = tmp_path / "straight"
    fl_train(make_cfg(dataset_root, straight, reset_optimizer=reset_optimizer))

    chained = tmp_path / "chained"
    with pytest.raises(TrainingInterrupted):
        fl_train(make_cfg(dataset_root, chained, stop_after_rounds=2, reset_optimizer=reset_optimizer))
    state = json.loads((chained / "fl_state" / "state.json").read_text())
    assert state["round"] == 2
    assert [int(r["round"]) for r in rows(chained / "rounds.csv")] == [1, 2]
    assert not (chained / "TRAINING_COMPLETE").exists()

    fl_train(make_cfg(dataset_root, chained, reset_optimizer=reset_optimizer))
    assert (chained / "TRAINING_COMPLETE").exists()
    assert [int(r["round"]) for r in rows(chained / "rounds.csv")] == [1, 2, 3, 4]

    a, b = final_weights(straight), final_weights(chained)
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a), "resumed run diverged from the uninterrupted one"
    losses = [(r["round"], r["client_id"], r["loss"]) for r in rows(straight / "client_losses.csv")]
    assert losses == [(r["round"], r["client_id"], r["loss"]) for r in rows(chained / "client_losses.csv")]
    assert [r["lr"] for r in rows(straight / "checkpoints.csv")] == [
        r["lr"] for r in rows(chained / "checkpoints.csv")
    ]

    manifest = json.loads((chained / "manifest.json").read_text())
    assert manifest["num_episodes"] == N_TASKS * EPISODES_PER_TASK
    assert all(row["clients_holding"] == 2 for row in manifest["tasks"].values())


def test_resume_refuses_changed_settings(dataset_root, tmp_path):
    from lerobot.scripts.lerobot_fl_train import fl_train

    out = tmp_path / "run"
    with pytest.raises(TrainingInterrupted):
        fl_train(make_cfg(dataset_root, out, stop_after_rounds=1))
    with pytest.raises(ValueError, match="fl.local_steps"):
        fl_train(make_cfg(dataset_root, out, local_steps=5))
