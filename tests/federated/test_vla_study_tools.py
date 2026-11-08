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
"""The study tooling in examples/federated/vla_study: grid order, trainer commands, aggregation."""

import csv
import json
import sys
from pathlib import Path

import pytest

STUDY_DIR = Path(__file__).resolve().parents[2] / "examples" / "federated" / "vla_study"
sys.path.insert(0, str(STUDY_DIR))

import collect_results  # noqa: E402
import grid  # noqa: E402


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for key in (
        "ROUNDS",
        "LOCAL_STEPS",
        "SEEDS",
        "PARTITION_SEEDS",
        "BENCHMARKS",
        "RUN_FEDERATED",
        "RUN_CENTRALIZED",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("STUDY_ROOT", str(tmp_path / "study"))
    monkeypatch.setenv("BENCHMARKS", "libero_object metaworld_mt10")
    s = grid.Settings.from_env()
    for bench in s.benchmarks:
        path = grid.task_map_path(s, bench)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"episode_tasks": {"3": "a", "1": "b"}}))
    return s


def test_runs_finish_complete_comparisons_first(settings):
    labels = [r.label() for r in grid.runs_for(1, settings)]
    assert labels[:4] == [
        "lora/libero_object/fed-lora-taskdisjoint-s1000-p1",
        "lora/libero_object/cen-lora-s1000",
        "lora/metaworld_mt10/fed-lora-taskdisjoint-s1000-p1",
        "lora/metaworld_mt10/cen-lora-s1000",
    ]
    assert labels[4].endswith("s2000-p2") and len(labels) == 12
    phase2 = [r.run_id() for r in grid.runs_for(2, settings)]
    assert phase2[:2] == ["fed-lora-iid-s1000-p1", "fed-lora-sharedcore-s1000-p1"]
    assert not any(r.startswith("cen") for r in phase2), "phase 2 reuses the centralized runs"


def test_seeds_and_partition_seeds_must_pair(monkeypatch, tmp_path):
    monkeypatch.setenv("STUDY_ROOT", str(tmp_path))
    monkeypatch.setenv("SEEDS", "1000 2000")
    monkeypatch.setenv("PARTITION_SEEDS", "1")
    with pytest.raises(ValueError, match="pair"):
        grid.Settings.from_env()


def _flags(cmd):
    return dict(a.split("=", 1) for a in cmd if a.startswith("--"))


def test_federated_and_centralized_commands_match_the_plan(settings):
    fed = grid.RunSpec("metaworld_mt10", "federated", "sharedcore", 2000, 2)
    flags = _flags(grid.federated_command(fed, settings, Path("/out")))
    assert flags["--dataset.episodes"] == "[1,3]"
    assert flags["--partition.strategy"] == "shared_core" and flags["--partition.seed"] == "2"
    assert flags["--partition.num_shared_tasks"] == "5"
    assert flags["--fl.rounds"] == "200" and flags["--fl.local_steps"] == "250"
    assert flags["--fl.client_weighting"] == "uniform" and flags["--fl.auto_resume"] == "true"
    assert flags["--peft.r"] == "64" and flags["--peft.lora_alpha"] == "128"
    assert (
        flags["--policy.scheduler_warmup_steps"] == "6" and flags["--policy.scheduler_decay_steps"] == "200"
    )

    cen = grid.RunSpec("metaworld_mt10", "centralized", None, 2000, None)
    flags = _flags(grid.centralized_command(cen, settings, Path("/out"), resume=False))
    assert flags["--steps"] == "250000" and flags["--save_freq"] == "25000"
    assert flags["--policy.scheduler_warmup_steps"] == "7500"
    assert flags["--policy.scheduler_decay_steps"] == "250000"
    assert flags["--peft.r"] == "64" and flags["--checkpoint_on_signal"] == "true"
    resume = grid.centralized_command(cen, settings, Path("/out"), resume=True)
    assert _flags(resume)["--config_path"].endswith("checkpoints/last/pretrained_model/train_config.json")


def test_overrides_replace_flags_instead_of_repeating_them(settings):
    cmd = grid.baseline_command("libero_object", settings, Path("/b"), "cpu")
    assert sum(a.startswith("--fl.rounds=") for a in cmd) == 1 and "--fl.rounds=0" in cmd
    assert "--policy.device=cpu" in cmd and "--policy.device=cuda" not in cmd


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))


def _result(rates: dict) -> dict:
    return {
        "tasks": {k: {"success_rate": v, "n_episodes": 10} for k, v in rates.items()},
        "protocol": {"n_episodes_per_task": 10, "seed": 1000, "batch_size": 1, "n_action_steps": 10},
    }


def test_aggregation_tolerates_missing_and_broken_runs(tmp_path):
    root = tmp_path
    fed = root / "studies" / "s1" / "fed-lora-taskdisjoint-s1000-p1"
    _write(fed / "run.json", {"bench": "libero_object", "policy": "smolvla", "mode": "lora", "arm": "federated",
                              "partition": "taskdisjoint", "seed": 1000, "partition_seed": 1, "rounds": 40,
                              "save_every": 20, "round_steps": 1})  # fmt: skip
    _write(
        fed / "train" / "manifest.json",
        {"tasks": {"t0": {"clients_holding": 1, "episode_share": {"0": 1.0, "1": 0.0}}}},
    )
    _write(fed / "eval" / "round_0020" / "result.json", _result({"t0": 40.0}))
    _write(fed / "train" / "client_losses.csv", "round,client_id,loss\n1,0,0.5\n1,1,0.7\n")
    cen = root / "studies" / "s2" / "cen-lora-s1000"
    _write(cen / "run.json", {"bench": "libero_object", "policy": "smolvla", "mode": "lora", "arm": "centralized",
                              "partition": None, "seed": 1000, "partition_seed": None, "rounds": 40,
                              "save_every": 20, "round_steps": 1250})  # fmt: skip
    _write(cen / "train" / "TRAINING_COMPLETE", "{}")
    _write(cen / "eval" / "round_0020" / "result.json", _result({"t0": 60.0}))
    _write(cen / "eval" / "round_0040" / "result.json", _result({"t0": 70.0}))
    broken = root / "studies" / "s3" / "fed-full-iid-s1000-p1"
    _write(broken / "run.json", "{not json")
    base = root / "baselines" / "libobject_smolvla"
    _write(base / "run.json", {"bench": "libero_object", "policy": "smolvla", "mode": "pretrained", "arm": "pretrained",
                               "partition": None, "seed": None, "partition_seed": None, "rounds": 0})  # fmt: skip
    _write(base / "eval" / "round_0000" / "result.json", _result({"t0": 5.0}))

    assert collect_results.collect(root) == 0
    rows = list(csv.DictReader((root / "results.csv").open()))
    assert [r["arm"] for r in rows] == ["federated", "centralized", "centralized", "pretrained"]
    assert rows[0]["n_clients_holding"] == "1" and rows[0]["task_share_by_client"] == "1.000;0.000"
    assert rows[1]["partition_type"] == "pooled" and rows[2]["round"] == "40"
    assert list(rows[0])[:13] == collect_results.RESULT_COLUMNS[:13]
    missing = (root / "missing.txt").read_text()
    assert "fed-lora-taskdisjoint" in missing and "missing [40]" in missing
    assert "unreadable" in missing and "cen-lora-s1000" not in missing
    losses = list(csv.DictReader((root / "client_losses.csv").open()))
    assert len(losses) == 2 and losses[0]["mode"] == "lora"
