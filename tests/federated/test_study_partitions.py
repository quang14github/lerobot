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
"""The three partitions of the federated VLA study: task-disjoint, IID and shared-core.

Ten tasks with uneven episode counts (like LIBERO's 33-50 demos per task), five clients.
"""

from collections import Counter

import pytest

from lerobot.federated.config import PartitionConfig
from lerobot.federated.partition import (
    check_exact_cover,
    episode_table,
    partition_episodes,
    partition_manifest,
)

NUM_CLIENTS = 5
EPISODES_PER_TASK = [44, 45, 47, 46, 33, 42, 50, 36, 49, 40]


class FakeMetadata:
    """The slice of `LeRobotDatasetMetadata` that the partitioner reads."""

    def __init__(self):
        self.repo_id = "fake/dataset"
        tasks = [f"task-{t}" for t, n in enumerate(EPISODES_PER_TASK) for _ in range(n)]
        # Uneven episode lengths, so frame shares and episode shares differ.
        lengths = [50 + (i * 37) % 90 for i in range(len(tasks))]
        bounds = [0]
        for length in lengths:
            bounds.append(bounds[-1] + length)
        self.episodes = {
            "tasks": [[t] for t in tasks],
            "dataset_from_index": bounds[:-1],
            "dataset_to_index": bounds[1:],
        }
        self.total_episodes = len(tasks)


@pytest.fixture
def meta():
    return FakeMetadata()


def split(meta, strategy, seed, **kwargs):
    return partition_episodes(
        meta, PartitionConfig(strategy=strategy, num_clients=NUM_CLIENTS, seed=seed, **kwargs)
    )


def tasks_at(shards, meta):
    """client id -> Counter(task -> episodes held)."""
    _, task_of, _ = episode_table(meta, None)
    return {s.client_id: Counter(task_of[ep] for ep in s.episodes) for s in shards}


@pytest.mark.parametrize("strategy", ["task", "iid", "shared_core"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_episode_is_assigned_exactly_once(meta, strategy, seed):
    shards = split(meta, strategy, seed)
    held = sorted(ep for s in shards for ep in s.episodes)
    assert held == list(range(meta.total_episodes))
    check_exact_cover(shards, list(range(meta.total_episodes)))


@pytest.mark.parametrize("strategy", ["task", "iid", "shared_core"])
def test_partition_is_reproducible_from_its_seed(meta, strategy):
    assert [s.episodes for s in split(meta, strategy, 1)] == [s.episodes for s in split(meta, strategy, 1)]
    assert [s.episodes for s in split(meta, strategy, 1)] != [s.episodes for s in split(meta, strategy, 2)]


def test_task_disjoint_gives_each_client_two_whole_tasks(meta):
    counts = tasks_at(split(meta, "task", 1), meta)
    owners = Counter()
    for client_tasks in counts.values():
        assert len(client_tasks) == 2
        for task, n in client_tasks.items():
            assert n == EPISODES_PER_TASK[int(task.split("-")[1])], "a task was split"
            owners[task] += 1
    assert set(owners.values()) == {1} and len(owners) == 10


def test_task_disjoint_assignment_is_shuffled_by_seed(meta):
    """Seeds must change which client holds which tasks, not only the client order."""
    pairings = {
        seed: frozenset(frozenset(c) for c in tasks_at(split(meta, "task", seed), meta).values())
        for seed in (1, 2, 3)
    }
    assert len(set(pairings.values())) > 1


def test_iid_gives_every_client_an_equal_share_of_every_task(meta):
    shards = split(meta, "iid", 1)
    counts = tasks_at(shards, meta)
    for t, total in enumerate(EPISODES_PER_TASK):
        shares = [counts[c][f"task-{t}"] for c in range(NUM_CLIENTS)]
        assert sum(shares) == total
        assert max(shares) - min(shares) <= 1
        assert min(shares) >= total // NUM_CLIENTS
    sizes = [s.num_episodes for s in shards]
    assert max(sizes) - min(sizes) <= 1, "leftover episodes must rotate between clients"


def test_iid_shares_are_random_not_contiguous(meta):
    shards = split(meta, "iid", 1)
    first_task = [ep for ep in shards[0].episodes if ep < EPISODES_PER_TASK[0]]
    assert first_task != list(range(min(first_task), min(first_task) + len(first_task)))


def test_shared_core_definition(meta):
    for seed in (1, 2, 3):
        counts = tasks_at(split(meta, "shared_core", seed), meta)
        holders = Counter(task for client_tasks in counts.values() for task in client_tasks)
        shared = sorted(t for t, n in holders.items() if n == NUM_CLIENTS)
        unique = sorted(t for t, n in holders.items() if n == 1)
        assert len(shared) == 5 and len(unique) == 5 and len(holders) == 10
        for task in shared:
            shares = [counts[c][task] for c in range(NUM_CLIENTS)]
            assert max(shares) - min(shares) <= 1
        unique_per_client = [sum(1 for t in counts[c] if t in unique) for c in range(NUM_CLIENTS)]
        assert unique_per_client == [1] * NUM_CLIENTS
        for task in unique:
            assert (
                sum(counts[c][task] for c in range(NUM_CLIENTS)) == EPISODES_PER_TASK[int(task.split("-")[1])]
            )


def test_shared_core_draws_the_shared_tasks_from_the_seed(meta):
    def shared(seed):
        counts = tasks_at(split(meta, "shared_core", seed), meta)
        holders = Counter(task for client_tasks in counts.values() for task in client_tasks)
        return frozenset(t for t, n in holders.items() if n == NUM_CLIENTS)

    assert len({shared(s) for s in (1, 2, 3)}) > 1


def test_exact_cover_check_rejects_drops_and_duplicates(meta):
    shards = split(meta, "iid", 1)
    episodes = list(range(meta.total_episodes))
    dropped = shards[0].episodes.pop()
    with pytest.raises(ValueError, match="dropped"):
        check_exact_cover(shards, episodes)
    shards[1].episodes.append(shards[2].episodes[0])
    shards[0].episodes.append(dropped)
    with pytest.raises(ValueError, match="duplicated"):
        check_exact_cover(shards, episodes)


def test_task_map_replaces_instructions(meta):
    """Two tasks recorded under one instruction stay apart when a task map names them."""
    task_map = {ep: f"key-{ep % 2}" for ep in range(meta.total_episodes)}
    shards = partition_episodes(
        meta, PartitionConfig(strategy="task", num_clients=2, seed=1), task_map=task_map
    )
    assert sorted(len(s.tasks) for s in shards) == [1, 1]
    assert {t for s in shards for t in s.tasks} == {"key-0", "key-1"}


def test_manifest_reports_shares_and_holders(meta):
    cfg = PartitionConfig(strategy="shared_core", num_clients=NUM_CLIENTS, seed=1)
    shards = partition_episodes(meta, cfg)
    _, task_of, frames_of = episode_table(meta, None)
    manifest = partition_manifest(shards, task_of, frames_of, cfg)
    assert manifest["num_episodes"] == meta.total_episodes
    for task, row in manifest["tasks"].items():
        assert sum(row["episode_share"].values()) == pytest.approx(1.0)
        assert sum(row["frame_share"].values()) == pytest.approx(1.0)
        assert row["clients_holding"] in (1, NUM_CLIENTS)
        assert row["episodes"] == EPISODES_PER_TASK[int(task.split("-")[1])]
    assert sum(c["num_episodes"] for c in manifest["clients"]) == meta.total_episodes
