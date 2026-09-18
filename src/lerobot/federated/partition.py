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
"""Split a `LeRobotDataset`'s episodes across federated clients.

The partition is computed once, on the server, from dataset *metadata* only — no frames are
read. Each client is then handed an episode allowlist that it passes to `LeRobotDataset(...,
episodes=...)`, which is the same mechanism as `--dataset.episodes` on the CLI.

Three strategies, in increasing order of realism:

- ``uniform``: episodes shuffled and split evenly. IID; the sanity-check baseline where
  federated averaging should closely track centralized training.
- ``dirichlet``: every task's episodes are split across clients with proportions drawn from
  ``Dir(alpha)``. ``alpha -> inf`` approaches uniform, ``alpha -> 0`` approaches disjoint
  tasks; ``alpha ~= 0.5`` is the usual "moderately non-IID" setting in the FL literature.
- ``task``: each client owns a disjoint set of tasks. The pathological label-skew case, and
  the one that matters for Meta-World: a client that has never seen `door-open-v3` must still
  end up with a global policy that can do it.
"""

import logging
from dataclasses import dataclass

import numpy as np

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

from .config import PartitionConfig

PARTITION_STRATEGIES = ("uniform", "dirichlet", "task", "explicit")


@dataclass
class ClientShard:
    """One client's private slice of the dataset."""

    client_id: int
    episodes: list[int]
    tasks: list[str]
    num_frames: int

    @property
    def num_episodes(self) -> int:
        """Number of episodes this client holds."""
        return len(self.episodes)


def _read_episode_table(
    meta: LeRobotDatasetMetadata, episodes: list[int] | None
) -> tuple[list[int], dict[int, str], dict[int, int]]:
    """Return the episode indices in play, plus their task and frame count.

    Rows of `meta.episodes` are indexed by episode index, matching how
    `lerobot.datasets.factory.make_train_eval_datasets` reads the same table.
    """
    if meta.episodes is None:
        raise ValueError(f"Dataset {meta.repo_id} has no episode metadata to partition.")

    ep_tasks = meta.episodes["tasks"]
    ep_from = meta.episodes["dataset_from_index"]
    ep_to = meta.episodes["dataset_to_index"]

    indices = list(range(meta.total_episodes)) if episodes is None else list(episodes)
    # An episode may carry several task strings; the first is the canonical one, as used by
    # the eval_split logic in the dataset factory.
    task_of = {ep: (ep_tasks[ep][0] if ep_tasks[ep] else "") for ep in indices}
    frames_of = {ep: int(ep_to[ep]) - int(ep_from[ep]) for ep in indices}
    return indices, task_of, frames_of


def _split_by_task(
    indices: list[int], task_of: dict[int, str], num_clients: int, rng: np.random.Generator
) -> list[list[int]]:
    tasks = sorted({task_of[ep] for ep in indices})
    if num_clients > len(tasks):
        raise ValueError(
            f"partition.strategy='task' gives each client a disjoint set of tasks, but "
            f"num_clients={num_clients} exceeds the {len(tasks)} task(s) in the dataset. "
            f"Lower num_clients, or use strategy='dirichlet' to split within tasks."
        )
    # Shuffle tasks before chunking so client 0 doesn't always get the alphabetically-first
    # tasks; the chunks themselves stay contiguous and near-equal in size.
    shuffled = list(rng.permutation(tasks))
    groups = np.array_split(np.arange(len(shuffled)), num_clients)
    per_client: list[list[int]] = []
    for group in groups:
        owned = {shuffled[i] for i in group}
        per_client.append([ep for ep in indices if task_of[ep] in owned])
    return per_client


def _split_dirichlet(
    indices: list[int],
    task_of: dict[int, str],
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    if alpha <= 0:
        raise ValueError(f"partition.dirichlet_alpha must be > 0, got {alpha}")

    by_task: dict[str, list[int]] = {}
    for ep in indices:
        by_task.setdefault(task_of[ep], []).append(ep)

    per_client: list[list[int]] = [[] for _ in range(num_clients)]
    for task in sorted(by_task):
        eps = np.array(by_task[task])
        rng.shuffle(eps)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        # cumulative cut points, so every episode lands with exactly one client
        cuts = (np.cumsum(proportions) * len(eps)).astype(int)[:-1]
        for client_id, chunk in enumerate(np.split(eps, cuts)):
            per_client[client_id].extend(int(ep) for ep in chunk)
    return per_client


def _split_explicit(
    indices: list[int],
    task_of: dict[int, str],
    num_clients: int,
    assignment: dict[str, list[str]],
    rng: np.random.Generator,
) -> list[list[int]]:
    """Give each client the tasks it is named for, splitting shared tasks between their holders.

    A task named by one client is exclusive to it; a task named by several is divided evenly
    among exactly those clients. Episodes stay disjoint either way - clients that "share a task"
    hold different episodes of it, as they must when the point is that data never moves.
    """
    holders: dict[str, list[int]] = {}
    for client_key, task_names in assignment.items():
        client_id = int(client_key)
        if not 0 <= client_id < num_clients:
            raise ValueError(
                f"partition.task_assignment names client {client_id}, outside "
                f"[0, partition.num_clients={num_clients})."
            )
        for task in task_names:
            holders.setdefault(task, []).append(client_id)

    available = {task_of[ep] for ep in indices}
    unknown = sorted(set(holders) - available)
    if unknown:
        raise ValueError(
            f"partition.task_assignment names task(s) absent from the selected episodes: {unknown}. "
            f"Available: {sorted(available)[:10]}{'...' if len(available) > 10 else ''}"
        )

    by_task: dict[str, list[int]] = {}
    for ep in indices:
        by_task.setdefault(task_of[ep], []).append(ep)

    per_client: list[list[int]] = [[] for _ in range(num_clients)]
    for task, client_ids in sorted(holders.items()):
        eps = np.array(sorted(by_task[task]))
        rng.shuffle(eps)
        for client_id, chunk in zip(sorted(client_ids), np.array_split(eps, len(client_ids)), strict=True):
            per_client[client_id].extend(int(ep) for ep in chunk)
    return per_client


def _split_uniform(indices: list[int], num_clients: int, rng: np.random.Generator) -> list[list[int]]:
    shuffled = rng.permutation(indices)
    return [[int(ep) for ep in chunk] for chunk in np.array_split(shuffled, num_clients)]


def partition_episodes(
    meta: LeRobotDatasetMetadata,
    cfg: PartitionConfig,
    episodes: list[int] | None = None,
) -> list[ClientShard]:
    """Split `episodes` (default: all of them) into `cfg.num_clients` disjoint shards.

    Args:
        meta: Metadata of the dataset to partition. Only the episode table is read.
        cfg: Strategy, client count, Dirichlet concentration and seed.
        episodes: Optional episode allowlist to partition within (e.g. from
            `--dataset.episodes`). `None` means every episode in the dataset.

    Returns:
        One `ClientShard` per client, in client-id order.

    Raises:
        ValueError: On an unknown strategy, a client count the strategy cannot satisfy, or a
            draw that leaves some client with no data.
    """
    if cfg.num_clients < 2:
        raise ValueError(f"partition.num_clients must be >= 2 for federated training, got {cfg.num_clients}")
    if cfg.strategy not in PARTITION_STRATEGIES:
        raise ValueError(
            f"Unknown partition.strategy={cfg.strategy!r}; expected one of {PARTITION_STRATEGIES}"
        )

    indices, task_of, frames_of = _read_episode_table(meta, episodes)
    if len(indices) < cfg.num_clients:
        raise ValueError(
            f"Cannot split {len(indices)} episode(s) across {cfg.num_clients} clients — "
            f"every client needs at least one episode."
        )

    rng = np.random.default_rng(cfg.seed)
    if cfg.strategy == "explicit":
        if not cfg.task_assignment:
            raise ValueError("partition.strategy='explicit' requires partition.task_assignment.")
        per_client = _split_explicit(indices, task_of, cfg.num_clients, cfg.task_assignment, rng)
    elif cfg.strategy == "task":
        per_client = _split_by_task(indices, task_of, cfg.num_clients, rng)
    elif cfg.strategy == "uniform":
        per_client = _split_uniform(indices, cfg.num_clients, rng)
    else:
        # A Dirichlet draw can starve a client outright; redraw rather than crash the run
        # several hours in, and only give up if the concentration is hopeless for this
        # client count.
        for attempt in range(cfg.max_draw_attempts):
            per_client = _split_dirichlet(indices, task_of, cfg.num_clients, cfg.dirichlet_alpha, rng)
            if all(len(eps) > 0 for eps in per_client):
                break
            logging.warning(
                "Dirichlet draw %d/%d left a client with no episodes; redrawing.",
                attempt + 1,
                cfg.max_draw_attempts,
            )
        else:
            raise ValueError(
                f"Could not draw a Dirichlet partition giving all {cfg.num_clients} clients data "
                f"after {cfg.max_draw_attempts} attempts (alpha={cfg.dirichlet_alpha}). Raise "
                f"partition.dirichlet_alpha or lower partition.num_clients."
            )

    shards = []
    for client_id, eps in enumerate(per_client):
        eps = sorted(eps)
        shards.append(
            ClientShard(
                client_id=client_id,
                episodes=eps,
                tasks=sorted({task_of[ep] for ep in eps}),
                num_frames=sum(frames_of[ep] for ep in eps),
            )
        )

    empty = [shard.client_id for shard in shards if shard.num_episodes == 0]
    if empty:
        raise ValueError(f"Partition left client(s) {empty} with no episodes; adjust the partition config.")
    return shards


def format_partition(shards: list[ClientShard], max_tasks_shown: int = 4) -> str:
    """Render the partition as a log-friendly table."""
    lines = [f"{'client':>6}  {'episodes':>8}  {'frames':>10}  {'tasks':>5}  sample tasks"]
    for shard in shards:
        sample = ", ".join(shard.tasks[:max_tasks_shown])
        if len(shard.tasks) > max_tasks_shown:
            sample += f", ... (+{len(shard.tasks) - max_tasks_shown})"
        lines.append(
            f"{shard.client_id:>6}  {shard.num_episodes:>8}  {shard.num_frames:>10}  "
            f"{len(shard.tasks):>5}  {sample}"
        )
    return "\n".join(lines)
