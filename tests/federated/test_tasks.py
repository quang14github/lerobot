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
"""Task-key -> episode resolution, including Meta-World's push-v3 / push-back-v3 clash."""

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.federated.tasks import BENCHMARKS, MT10, benchmark_episodes, resolve_task_episodes

PUSH = "Push the puck to a goal"


class FakeMetadata:
    """Episode table plus data files carrying a per-frame `task_id`, like lerobot/metaworld_mt50."""

    data_path = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

    def __init__(self, root: Path, episodes: list[tuple[str, int]], frames: int = 4):
        self.repo_id = "fake/metaworld"
        self.root = root
        self.total_episodes = len(episodes)
        self.episodes = {
            "tasks": [[text] for text, _ in episodes],
            "data/chunk_index": [0] * len(episodes),
            # Two files, so resolution has to read across file boundaries.
            "data/file_index": [ep % 2 for ep in range(len(episodes))],
            "dataset_from_index": [ep * frames for ep in range(len(episodes))],
            "dataset_to_index": [(ep + 1) * frames for ep in range(len(episodes))],
        }
        for file_index in (0, 1):
            eps = [ep for ep in range(len(episodes)) if ep % 2 == file_index]
            table = pa.table(
                {
                    "episode_index": [ep for ep in eps for _ in range(frames)],
                    # Stored as length-1 lists, the layout of int16 [1] features.
                    "task_id": [[episodes[ep][1]] for ep in eps for _ in range(frames)],
                }
            )
            path = root / self.data_path.format(chunk_index=0, file_index=file_index)
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, path)

    def _pull_from_repo(self, allow_patterns=None):  # pragma: no cover - must never be needed here
        raise AssertionError(f"unexpected download of {allow_patterns}")


@pytest.fixture
def push_meta(tmp_path):
    # Episodes 0-3 are push-v3 (id 38), 4-7 push-back-v3 (id 37) under the SAME instruction,
    # 8-9 reach-v3 (id 40).
    episodes = [(PUSH, 38)] * 4 + [(PUSH, 37)] * 4 + [("Reach a goal position", 40)] * 2
    return FakeMetadata(tmp_path, episodes)


def test_push_v3_excludes_push_back_episodes(push_meta):
    resolved = resolve_task_episodes(
        push_meta,
        {"push-v3": PUSH, "reach-v3": "Reach a goal position"},
        frame_task_ids={"push-v3": 38, "reach-v3": 40},
    )
    assert resolved == {"push-v3": [0, 1, 2, 3], "reach-v3": [8, 9]}


def test_instruction_lookup_alone_refuses_a_shared_instruction(push_meta):
    with pytest.raises(ValueError, match="share an instruction"):
        resolve_task_episodes(push_meta, {"push-v3": PUSH, "push-back-v3": PUSH})


def test_a_task_with_no_matching_frames_is_an_error(push_meta):
    with pytest.raises(ValueError, match="task_id=99"):
        resolve_task_episodes(push_meta, {"push-v3": PUSH}, frame_task_ids={"push-v3": 99})


def test_benchmark_specs_list_ten_tasks_each():
    for bench in BENCHMARKS.values():
        assert len(bench.task_keys) == 10
        assert len(set(bench.instructions().values())) == 10
    assert BENCHMARKS["metaworld_mt10"].frame_task_ids()["push-v3"] == 38
    assert BENCHMARKS["libero_goal"].env_flags()[-1] == "--env.task_ids=[0,1,2,3,4,5,6,7,8,9]"
    assert BENCHMARKS["metaworld_mt10"].env_flags()[-1] == "--env.task=" + ",".join(MT10)


def _cached(repo_id: str) -> bool:
    root = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot")) / "hub"
    return (root / f"datasets--{repo_id.replace('/', '--')}").is_dir()


@pytest.mark.skipif(
    not _cached("lerobot/metaworld_mt50"), reason="lerobot/metaworld_mt50 not in the local cache"
)
def test_real_mt10_resolves_push_v3_to_its_own_fifty_episodes():
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    resolved = benchmark_episodes(
        LeRobotDatasetMetadata("lerobot/metaworld_mt50"), BENCHMARKS["metaworld_mt10"]
    )
    assert resolved["push-v3"] == list(range(2000, 2050))
    assert all(len(eps) == 50 for eps in resolved.values())
    assert not set(range(2100, 2150)) & {ep for eps in resolved.values() for ep in eps}, (
        "push-back-v3 leaked in"
    )
