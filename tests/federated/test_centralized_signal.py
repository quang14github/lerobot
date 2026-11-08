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
"""The centralized arm of the VLA study: checkpoint on a signal, prune, resume (CPU)."""

import csv
import os
import signal

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")
pytest.importorskip("diffusers", reason="diffusers is required (install lerobot[diffusion])")

from safetensors.torch import load_file

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.federated.run_state import TrainingInterrupted
from lerobot.utils.constants import PRETRAINED_MODEL_DIR, TRAINING_STATE_DIR

REPO_ID = "dummy/central_signal"


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("central") / "dataset"
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
    for episode in range(3):
        for _ in range(10):
            dataset.add_frame(
                {
                    "action": rng.standard_normal(4).astype(np.float32),
                    "observation.state": rng.standard_normal(4).astype(np.float32),
                    "observation.images.top": rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8),
                    "task": f"task_{episode}",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    return root


def make_cfg(root, output_dir):
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
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=REPO_ID, root=str(root)),
        policy=policy,
        output_dir=output_dir,
        steps=8,
        save_freq=4,
        batch_size=2,
        num_workers=0,
        seed=3,
        log_freq=0,
        env_eval_freq=0,
        checkpoint_on_signal=True,
        prune_checkpoints=True,
    )
    cfg.optimizer = policy.get_optimizer_preset()
    cfg.scheduler = policy.get_scheduler_preset()
    cfg.validate = lambda: None  # built in-process; skip the CLI-oriented validation
    return cfg


def test_signal_checkpoint_prune_and_resume(dataset_root, tmp_path, monkeypatch):
    import lerobot.scripts.lerobot_train as lerobot_train

    straight = tmp_path / "straight"
    lerobot_train.train(make_cfg(dataset_root, straight))

    real_update = lerobot_train.update_policy
    calls = {"n": 0}

    def update_then_signal(*args, **kwargs):
        out = real_update(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 6:
            os.kill(os.getpid(), signal.SIGUSR1)
        return out

    chained = tmp_path / "chained"
    monkeypatch.setattr(lerobot_train, "update_policy", update_then_signal)
    with pytest.raises(TrainingInterrupted):
        lerobot_train.train(make_cfg(dataset_root, chained))
    monkeypatch.setattr(lerobot_train, "update_policy", real_update)

    ckpts = chained / "checkpoints"
    assert (ckpts / "last").resolve().name == "000006"
    assert (ckpts / "000006" / TRAINING_STATE_DIR).is_dir()
    assert not (ckpts / "000004" / TRAINING_STATE_DIR).exists(), "older checkpoint kept its optimizer state"
    assert (ckpts / "000004" / PRETRAINED_MODEL_DIR / "model.safetensors").is_file()

    resume = make_cfg(dataset_root, chained)
    resume.resume = True
    resume.checkpoint_path = (ckpts / "last").resolve()
    # What `--resume=true --config_path=...` sets during validation, which this in-process config skips.
    resume.policy.pretrained_path = resume.checkpoint_path / PRETRAINED_MODEL_DIR
    lerobot_train.train(resume)

    assert (chained / "TRAINING_COMPLETE").is_file()
    assert sorted(p.name for p in ckpts.iterdir() if not p.is_symlink()) == ["000004", "000008"], (
        "the off-grid signal checkpoint should be gone"
    )
    with open(chained / "checkpoints.csv") as f:
        assert [int(r["step"]) for r in csv.DictReader(f)] == [4, 6, 8]

    a = load_file(straight / "checkpoints" / "000008" / PRETRAINED_MODEL_DIR / "model.safetensors")
    b = load_file(ckpts / "000008" / PRETRAINED_MODEL_DIR / "model.safetensors")
    assert all(torch.equal(a[k], b[k]) for k in a), "resumed centralized run diverged"
