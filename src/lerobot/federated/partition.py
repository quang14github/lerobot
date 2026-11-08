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

Strategies:

- ``uniform``: episodes shuffled and split evenly. IID in expectation; the sanity-check baseline
  where federated averaging should closely track centralized training.
- ``iid``: every client gets an equal-sized random share of *every* task's episodes (sizes differ
  by at most one episode per task, and leftovers rotate between clients so totals stay balanced).
- ``dirichlet``: every task's episodes are split across clients with proportions drawn from
  ``Dir(alpha)``. ``alpha -> inf`` approaches uniform, ``alpha -> 0`` approaches disjoint
  tasks; ``alpha ~= 0.5`` is the usual "moderately non-IID" setting in the FL literature.
- ``task``: each client owns a disjoint set of tasks. The pathological label-skew case, and
  the one that matters for Meta-World: a client that has never seen `door-open-v3` must still
  end up with a global policy that can do it. Tasks are shuffled by the seed before dealing.
- ``shared_core``: ``num_shared_tasks`` tasks, drawn by the seed, are split equally across all
  clients (as in ``iid``); every other task is given whole to one client.
- ``explicit``: the tasks each client holds are named in ``task_assignment``.

Tasks are the dataset's instruction strings, unless ``task_map`` names each episode's task key -
needed when two tasks share an instruction (see `lerobot.federated.tasks`).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

from .config import PartitionConfig

PARTITION_STRATEGIES = ("uniform", "iid", "dirichlet", "task", "shared_core", "explicit")


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


def load_task_map(path: str | Path) -> dict[int, str]:
    """Read a `{"episode_tasks": {"<episode>": "<task key>"}}` file into episode -> task key."""
    payload = json.loads(Path(path).read_text())
    return {int(ep): str(task) for ep, task in payload["episode_tasks"].items()}


def episode_table(
    meta: LeRobotDatasetMetadata, episodes: list[int] | None, task_map: dict[int, str] | None = None
) -> tuple[list[int], dict[int, str], dict[int, int]]:
    """Return the episode indices in play, plus their task and frame count.

    Rows of `meta.episodes` are indexed by episode index, matching how
    `lerobot.datasets.factory.make_train_eval_datasets` reads the same table. With `task_map`,
    tasks are its task keys instead of the dataset's instruction strings, and every episode in
    play must appear in it.
    """
    if meta.episodes is None:
        raise ValueError(f"Dataset {meta.repo_id} has no episode metadata to partition.")

    ep_from = meta.episodes["dataset_from_index"]
    ep_to = meta.episodes["dataset_to_index"]

    if episodes is None:
        indices = sorted(task_map) if task_map is not None else list(range(meta.total_episodes))
    else:
        indices = list(episodes)
    if task_map is not None:
        unmapped = [ep for ep in indices if ep not in task_map]
        if unmapped:
            raise ValueError(f"partition.task_map names no task for episode(s) {unmapped[:10]}.")
        task_of = {ep: task_map[ep] for ep in indices}
    else:
        task_of = episode_tasks(meta, indices)
    frames_of = {ep: int(ep_to[ep]) - int(ep_from[ep]) for ep in indices}
    return indices, task_of, frames_of


def episode_tasks(meta: LeRobotDatasetMetadata, indices: list[int]) -> dict[int, str]:
    """Map each episode in `indices` to its canonical task string.

    Read from the episode table's `tasks` column when it has one. Some converted datasets
    (e.g. `lerobot/libero`) omit that column, leaving the task only on the frames themselves;
    for those, the task of an episode's first frame is read from the data parquet files.
    """
    columns = getattr(meta.episodes, "column_names", meta.episodes)
    if "tasks" in columns:
        ep_tasks = meta.episodes["tasks"]
        # An episode may carry several task strings; the first is the canonical one, as used by
        # the eval_split logic in the dataset factory.
        return {ep: (ep_tasks[ep][0] if ep_tasks[ep] else "") for ep in indices}
    return _episode_tasks_from_frames(meta, indices)


def _episode_tasks_from_frames(meta: LeRobotDatasetMetadata, indices: list[int]) -> dict[int, str]:
    """Recover episode tasks from the per-frame `task_index`, fetching only the data files needed.

    Only the two small index columns are read, and only the parquet files holding the requested
    episodes are downloaded - no videos - so this stays cheap even on large datasets.
    """
    chunk_col = meta.episodes["data/chunk_index"]
    file_col = meta.episodes["data/file_index"]
    files = sorted(
        {meta.data_path.format(chunk_index=chunk_col[ep], file_index=file_col[ep]) for ep in indices}
    )
    missing = [f for f in files if not (meta.root / f).is_file()]
    if missing:
        logging.info(
            "Episode table of %s has no 'tasks' column; fetching %d data file(s) to read tasks from frames",
            meta.repo_id,
            len(missing),
        )
        meta._pull_from_repo(allow_patterns=missing)

    task_names = dict(zip(meta.tasks["task_index"].tolist(), meta.tasks.index.tolist(), strict=True))
    wanted = set(indices)
    task_of: dict[int, str] = {}
    for f in files:
        table = pq.read_table(meta.root / f, columns=["episode_index", "task_index"])
        ep_col = table.column("episode_index").to_numpy()
        task_col = table.column("task_index").to_numpy()
        # Frames are stored in episode order, so the first occurrence is the episode's first frame.
        for ep, first in zip(*np.unique(ep_col, return_index=True), strict=True):
            if int(ep) in wanted:
                task_of[int(ep)] = str(task_names[int(task_col[first])])

    unresolved = sorted(wanted - set(task_of))
    if unresolved:
        raise ValueError(
            f"Could not find frames for episode(s) {unresolved[:10]} of {meta.repo_id} in its data files."
        )
    return task_of


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


def _group_by_task(indices: list[int], task_of: dict[int, str]) -> dict[str, list[int]]:
    by_task: dict[str, list[int]] = {}
    for ep in indices:
        by_task.setdefault(task_of[ep], []).append(ep)
    return {task: sorted(eps) for task, eps in sorted(by_task.items())}


class _EqualSplitter:
    """Splits a task's episodes into `num_clients` random shares whose sizes differ by at most one.

    When a task's episode count does not divide evenly, the clients that receive the extra
    episode are taken in turn from a rotating pointer (starting at a seeded offset), so across
    many tasks no client is systematically favoured and client totals also differ by at most one.
    """

    def __init__(self, num_clients: int, rng: np.random.Generator):
        self.num_clients = num_clients
        self.rng = rng
        self.offset = int(rng.integers(num_clients))

    def split(self, task: str, eps: list[int]) -> list[list[int]]:
        if len(eps) < self.num_clients:
            raise ValueError(
                f"Task {task!r} has {len(eps)} episode(s), fewer than the {self.num_clients} clients "
                f"that must each receive a share of it."
            )
        shuffled = [int(ep) for ep in self.rng.permutation(eps)]
        base, extra = divmod(len(shuffled), self.num_clients)
        lucky = {(self.offset + j) % self.num_clients for j in range(extra)}
        self.offset = (self.offset + extra) % self.num_clients
        shares, start = [], 0
        for client_id in range(self.num_clients):
            size = base + (1 if client_id in lucky else 0)
            shares.append(shuffled[start : start + size])
            start += size
        return shares


def _split_iid(
    indices: list[int], task_of: dict[int, str], num_clients: int, rng: np.random.Generator
) -> list[list[int]]:
    splitter = _EqualSplitter(num_clients, rng)
    per_client: list[list[int]] = [[] for _ in range(num_clients)]
    for task, eps in _group_by_task(indices, task_of).items():
        for client_id, share in enumerate(splitter.split(task, eps)):
            per_client[client_id].extend(share)
    return per_client


def _split_shared_core(
    indices: list[int],
    task_of: dict[int, str],
    num_clients: int,
    num_shared: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Split `num_shared` seeded tasks equally across all clients; give each other task to one.

    Unique tasks are dealt over a seeded client order, one each when there are `num_clients` of them.
    """
    by_task = _group_by_task(indices, task_of)
    tasks = list(by_task)
    if not 0 <= num_shared <= len(tasks):
        raise ValueError(f"partition.num_shared_tasks={num_shared} must be in [0, {len(tasks)}].")
    order = [str(t) for t in rng.permutation(tasks)]
    shared, unique = sorted(order[:num_shared]), order[num_shared:]
    if len(unique) < num_clients and num_shared == 0:
        raise ValueError("shared_core with no shared tasks needs at least one unique task per client.")

    per_client: list[list[int]] = [[] for _ in range(num_clients)]
    splitter = _EqualSplitter(num_clients, rng)
    for task in shared:
        for client_id, share in enumerate(splitter.split(task, by_task[task])):
            per_client[client_id].extend(share)
    client_order = [int(c) for c in rng.permutation(num_clients)]
    for i, task in enumerate(unique):
        per_client[client_order[i % num_clients]].extend(by_task[task])
    return per_client


def _split_uniform(indices: list[int], num_clients: int, rng: np.random.Generator) -> list[list[int]]:
    shuffled = rng.permutation(indices)
    return [[int(ep) for ep in chunk] for chunk in np.array_split(shuffled, num_clients)]


def partition_episodes(
    meta: LeRobotDatasetMetadata,
    cfg: PartitionConfig,
    episodes: list[int] | None = None,
    task_map: dict[int, str] | None = None,
) -> list[ClientShard]:
    """Split `episodes` (default: all of them) into `cfg.num_clients` disjoint shards.

    Args:
        meta: Metadata of the dataset to partition. Only the episode table is read.
        cfg: Strategy, client count, Dirichlet concentration and seed.
        episodes: Optional episode allowlist to partition within (e.g. from
            `--dataset.episodes`). `None` means every episode in the dataset.
        task_map: Optional episode -> task key mapping that replaces the instruction strings as
            the unit of partitioning. Defaults to the file named by `cfg.task_map`, if any.

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

    if task_map is None and cfg.task_map:
        task_map = load_task_map(cfg.task_map)
    indices, task_of, frames_of = episode_table(meta, episodes, task_map)
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
    elif cfg.strategy == "iid":
        per_client = _split_iid(indices, task_of, cfg.num_clients, rng)
    elif cfg.strategy == "shared_core":
        per_client = _split_shared_core(indices, task_of, cfg.num_clients, cfg.num_shared_tasks, rng)
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
    check_exact_cover(shards, indices)
    return shards


def check_exact_cover(shards: list[ClientShard], episodes: list[int]) -> None:
    """Raise unless every episode in `episodes` is held by exactly one client, and no other is."""
    held: dict[int, list[int]] = {}
    for shard in shards:
        for ep in shard.episodes:
            held.setdefault(ep, []).append(shard.client_id)
    expected = set(episodes)
    dropped = sorted(expected - set(held))
    duplicated = {ep: owners for ep, owners in held.items() if len(owners) > 1}
    foreign = sorted(set(held) - expected)
    if dropped or duplicated or foreign:
        raise ValueError(
            f"Partition is not an exact cover of its {len(expected)} episodes: "
            f"dropped={dropped[:10]} (total {len(dropped)}), "
            f"duplicated={dict(list(duplicated.items())[:10])} (total {len(duplicated)}), "
            f"unknown={foreign[:10]} (total {len(foreign)})."
        )


def partition_manifest(
    shards: list[ClientShard],
    task_of: dict[int, str],
    frames_of: dict[int, int],
    cfg: PartitionConfig,
) -> dict:
    """Describe a partition for auditing: each client's episodes, and each task's share at each client.

    Shares are given both by episodes and by frames. JSON-serializable; written next to every run.
    """
    tasks = sorted({task_of[ep] for shard in shards for ep in shard.episodes})
    task_eps = {t: sum(1 for s in shards for ep in s.episodes if task_of[ep] == t) for t in tasks}
    task_frames = {
        t: sum(frames_of[ep] for s in shards for ep in s.episodes if task_of[ep] == t) for t in tasks
    }

    clients = []
    for shard in shards:
        per_task: dict[str, dict[str, int]] = {}
        for ep in shard.episodes:
            entry = per_task.setdefault(task_of[ep], {"episodes": 0, "frames": 0})
            entry["episodes"] += 1
            entry["frames"] += frames_of[ep]
        clients.append(
            {
                "client_id": shard.client_id,
                "num_episodes": shard.num_episodes,
                "num_frames": shard.num_frames,
                "tasks": dict(sorted(per_task.items())),
                "episodes": list(shard.episodes),
            }
        )

    task_table = {}
    for task in tasks:
        episode_share, frame_share = {}, {}
        for client in clients:
            counts = client["tasks"].get(task, {"episodes": 0, "frames": 0})
            episode_share[str(client["client_id"])] = counts["episodes"] / task_eps[task]
            frame_share[str(client["client_id"])] = counts["frames"] / task_frames[task]
        task_table[task] = {
            "episodes": task_eps[task],
            "frames": task_frames[task],
            "clients_holding": sum(1 for share in episode_share.values() if share > 0),
            "episode_share": episode_share,
            "frame_share": frame_share,
        }

    return {
        "strategy": cfg.strategy,
        "seed": cfg.seed,
        "num_clients": len(shards),
        "num_shared_tasks": cfg.num_shared_tasks if cfg.strategy == "shared_core" else None,
        "num_episodes": sum(c["num_episodes"] for c in clients),
        "num_frames": sum(c["num_frames"] for c in clients),
        "tasks": task_table,
        "clients": clients,
    }


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
