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
"""Partitioning is pure metadata arithmetic, so these tests need no dataset on disk."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.federated.config import PartitionConfig
from lerobot.federated.partition import episode_tasks, format_partition, partition_episodes


class FakeMetadata:
    """The slice of `LeRobotDatasetMetadata` that the partitioner reads."""

    def __init__(self, tasks_per_episode: list[str], frames_per_episode: int = 100):
        self.repo_id = "fake/dataset"
        boundaries = [i * frames_per_episode for i in range(len(tasks_per_episode) + 1)]
        self.episodes = {
            "tasks": [[task] for task in tasks_per_episode],
            "dataset_from_index": boundaries[:-1],
            "dataset_to_index": boundaries[1:],
        }
        self.total_episodes = len(tasks_per_episode)


def make_meta(num_tasks: int = 10, episodes_per_task: int = 5) -> FakeMetadata:
    return FakeMetadata([f"task-{t}-v3" for t in range(num_tasks) for _ in range(episodes_per_task)])


def all_episodes(shards) -> list[int]:
    return sorted(ep for shard in shards for ep in shard.episodes)


@pytest.mark.parametrize("strategy", ["uniform", "dirichlet", "task"])
def test_partition_is_an_exact_cover(strategy):
    """Every episode goes to exactly one client — no loss, no duplication across clients."""
    meta = make_meta()
    shards = partition_episodes(meta, PartitionConfig(strategy=strategy, num_clients=5))

    assert all_episodes(shards) == list(range(meta.total_episodes))
    assert len(shards) == 5
    assert all(shard.num_episodes > 0 for shard in shards)


def test_task_strategy_gives_clients_disjoint_tasks():
    """The pathological non-IID case: no task is shared between two clients."""
    shards = partition_episodes(make_meta(num_tasks=10), PartitionConfig(strategy="task", num_clients=5))

    seen: set[str] = set()
    for shard in shards:
        assert not (seen & set(shard.tasks)), "a task leaked across clients"
        seen |= set(shard.tasks)
        assert len(shard.tasks) == 2
    assert len(seen) == 10


def test_uniform_strategy_spreads_every_task_across_clients():
    """The IID baseline: clients look statistically alike."""
    shards = partition_episodes(
        make_meta(num_tasks=10, episodes_per_task=20), PartitionConfig(strategy="uniform", num_clients=4)
    )

    for shard in shards:
        assert len(shard.tasks) == 10


def test_dirichlet_alpha_controls_skew():
    """Small alpha concentrates each task on few clients; large alpha approaches uniform."""
    meta = make_meta(num_tasks=20, episodes_per_task=20)
    skewed = partition_episodes(
        meta, PartitionConfig(strategy="dirichlet", num_clients=8, dirichlet_alpha=0.05)
    )
    even = partition_episodes(
        meta, PartitionConfig(strategy="dirichlet", num_clients=8, dirichlet_alpha=100.0)
    )

    def mean_tasks(shards):
        return sum(len(shard.tasks) for shard in shards) / len(shards)

    assert mean_tasks(skewed) < mean_tasks(even)


def test_frame_counts_are_tracked_for_client_weighting():
    """Client weights come from these counts, so they must reflect the shard, not the dataset."""
    shards = partition_episodes(
        FakeMetadata(["a", "a", "b", "b"], frames_per_episode=50),
        PartitionConfig(strategy="task", num_clients=2),
    )

    assert sum(shard.num_frames for shard in shards) == 4 * 50
    assert all(shard.num_frames == 100 for shard in shards)


def test_partition_respects_an_episode_allowlist():
    """`--dataset.episodes` narrows the pool before the split."""
    shards = partition_episodes(
        make_meta(num_tasks=4, episodes_per_task=5),
        PartitionConfig(strategy="uniform", num_clients=2),
        episodes=[0, 1, 2, 3, 10, 11, 12, 13],
    )

    assert all_episodes(shards) == [0, 1, 2, 3, 10, 11, 12, 13]


def test_same_seed_reproduces_the_split():
    meta = make_meta()
    first = partition_episodes(meta, PartitionConfig(strategy="dirichlet", num_clients=4, seed=7))
    second = partition_episodes(meta, PartitionConfig(strategy="dirichlet", num_clients=4, seed=7))
    other = partition_episodes(meta, PartitionConfig(strategy="dirichlet", num_clients=4, seed=8))

    assert [s.episodes for s in first] == [s.episodes for s in second]
    assert [s.episodes for s in first] != [s.episodes for s in other]


def test_task_strategy_rejects_more_clients_than_tasks():
    with pytest.raises(ValueError, match="exceeds the 3 task"):
        partition_episodes(make_meta(num_tasks=3), PartitionConfig(strategy="task", num_clients=5))


def test_rejects_more_clients_than_episodes():
    with pytest.raises(ValueError, match="every client needs at least one episode"):
        partition_episodes(FakeMetadata(["a", "b"]), PartitionConfig(strategy="uniform", num_clients=100))


def test_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown partition.strategy"):
        partition_episodes(make_meta(), PartitionConfig(strategy="gossip", num_clients=2))


def test_format_partition_lists_every_client():
    shards = partition_episodes(make_meta(), PartitionConfig(strategy="task", num_clients=5))
    table = format_partition(shards)

    assert len(table.splitlines()) == 6  # header + one row per client
    assert "task-0-v3" in table


def test_explicit_shares_one_task_between_two_clients():
    """Two clients on the same task, a third on its own - the asymmetric federation.

    "Sharing a task" still means disjoint episodes: data never moves in federated learning, so
    the shared task is split between its holders rather than duplicated.
    """
    meta = make_meta(num_tasks=2, episodes_per_task=50)
    shards = partition_episodes(
        meta,
        PartitionConfig(
            strategy="explicit",
            num_clients=3,
            task_assignment={"0": ["task-0-v3"], "1": ["task-0-v3"], "2": ["task-1-v3"]},
        ),
    )

    assert shards[0].tasks == ["task-0-v3"]
    assert shards[1].tasks == ["task-0-v3"]
    assert shards[2].tasks == ["task-1-v3"]
    # The shared task is halved; the solo client keeps all of its own.
    assert shards[0].num_episodes == 25
    assert shards[1].num_episodes == 25
    assert shards[2].num_episodes == 50
    assert not set(shards[0].episodes) & set(shards[1].episodes)
    assert all_episodes(shards) == list(range(100))


def test_explicit_rejects_an_unknown_task():
    with pytest.raises(ValueError, match="absent from the selected episodes"):
        partition_episodes(
            make_meta(num_tasks=2),
            PartitionConfig(
                strategy="explicit", num_clients=2, task_assignment={"0": ["nope-v3"], "1": ["task-0-v3"]}
            ),
        )


def test_explicit_rejects_a_client_outside_the_range():
    with pytest.raises(ValueError, match="outside"):
        partition_episodes(
            make_meta(num_tasks=2),
            PartitionConfig(
                strategy="explicit", num_clients=2, task_assignment={"5": ["task-0-v3"], "1": ["task-1-v3"]}
            ),
        )


def test_explicit_requires_an_assignment():
    with pytest.raises(ValueError, match="requires partition.task_assignment"):
        partition_episodes(make_meta(), PartitionConfig(strategy="explicit", num_clients=2))


class FramesOnlyMetadata:
    """Metadata whose episode table has no `tasks` column, like `lerobot/libero`.

    The task survives only as each frame's `task_index`, in data files laid out on disk the way
    a v3 dataset stores them - including episodes interleaved across tasks and split over files.
    """

    data_path = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

    def __init__(self, root, task_per_episode: list[int], task_names: list[str], frames: int = 4):
        self.repo_id = "fake/frames-only"
        self.root = root
        self.total_episodes = len(task_per_episode)
        self.tasks = pd.DataFrame({"task_index": range(len(task_names))}, index=task_names)
        # Two episodes per file, so the reader has to visit several files.
        file_of = [ep // 2 for ep in range(self.total_episodes)]
        boundaries = [i * frames for i in range(self.total_episodes + 1)]
        self.episodes = {
            "dataset_from_index": boundaries[:-1],
            "dataset_to_index": boundaries[1:],
            "data/chunk_index": [0] * self.total_episodes,
            "data/file_index": file_of,
        }
        for file_index in sorted(set(file_of)):
            eps = [ep for ep, f in enumerate(file_of) if f == file_index]
            path = root / self.data_path.format(chunk_index=0, file_index=file_index)
            path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.table(
                {
                    "episode_index": [ep for ep in eps for _ in range(frames)],
                    "task_index": [task_per_episode[ep] for ep in eps for _ in range(frames)],
                }
            )
            pq.write_table(table, path)

    def _pull_from_repo(self, **_):
        raise AssertionError("every data file is already on disk; nothing should be downloaded")


def test_episode_tasks_fall_back_to_frame_task_index(tmp_path):
    names = ["open the drawer", "put the bowl on the stove", "turn on the stove"]
    meta = FramesOnlyMetadata(tmp_path, task_per_episode=[2, 0, 1, 0, 2, 1], task_names=names)

    assert episode_tasks(meta, [0, 1, 2, 3, 4, 5]) == {
        0: names[2],
        1: names[0],
        2: names[1],
        3: names[0],
        4: names[2],
        5: names[1],
    }


def test_task_partition_works_without_a_tasks_column(tmp_path):
    """The partitioner must not depend on the optional episode-table `tasks` column."""
    names = [f"libero task {t}" for t in range(3)]
    meta = FramesOnlyMetadata(
        tmp_path, task_per_episode=[t for _ in range(4) for t in range(3)], task_names=names
    )
    shards = partition_episodes(meta, PartitionConfig(strategy="task", num_clients=3))

    assert sorted(task for shard in shards for task in shard.tasks) == names
    assert all(shard.num_episodes == 4 and shard.num_frames == 16 for shard in shards)
