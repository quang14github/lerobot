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
"""Benchmark task keys, and the dataset episodes behind each one.

A *task key* is what the study and the eval speak: a Meta-World slug (``push-v3``) or
``<suite>_<id>`` for LIBERO (``libero_object_3``). The datasets only know natural-language
instructions, and an instruction does not always identify a task:

- In ``lerobot/metaworld_mt50``, ``push-v3`` and ``push-back-v3`` were recorded under the same
  instruction ("Push the puck to a goal") and therefore share one ``task_index``. Upstream later
  gave ``push-back-v3`` its own description in ``metaworld_config.json``, so an instruction lookup
  no longer notices the clash and silently returns both tasks' 100 episodes. The per-frame
  ``task_id`` column still tells them apart (38 is ``push-v3``, 37 is ``push-back-v3``, matching
  ``TASK_NAME_TO_ID``), so Meta-World tasks are resolved by that column.
- In ``lerobot/libero`` every instruction is unique (40 tasks, 4 suites), so LIBERO tasks are
  resolved by instruction, and any instruction shared by two requested keys is refused.

The LIBERO instructions are listed here rather than read from the ``libero`` package so that
resolving a task never imports LIBERO, whose first import prompts on stdin.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

from .partition import episode_tasks

MT10: tuple[str, ...] = (
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    "door-open-v3",
    "drawer-open-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "peg-insert-side-v3",
    "window-open-v3",
    "window-close-v3",
)

# Task id -> instruction, in the order of LIBERO's own benchmark (`get_task(i).language`).
LIBERO_INSTRUCTIONS: dict[str, tuple[str, ...]] = {
    "libero_object": (
        "pick up the alphabet soup and place it in the basket",
        "pick up the cream cheese and place it in the basket",
        "pick up the salad dressing and place it in the basket",
        "pick up the bbq sauce and place it in the basket",
        "pick up the ketchup and place it in the basket",
        "pick up the tomato sauce and place it in the basket",
        "pick up the butter and place it in the basket",
        "pick up the milk and place it in the basket",
        "pick up the chocolate pudding and place it in the basket",
        "pick up the orange juice and place it in the basket",
    ),
    "libero_goal": (
        "open the middle drawer of the cabinet",
        "put the bowl on the stove",
        "put the wine bottle on top of the cabinet",
        "open the top drawer and put the bowl inside",
        "put the bowl on top of the cabinet",
        "push the plate to the front of the stove",
        "put the cream cheese in the bowl",
        "turn on the stove",
        "put the bowl on the plate",
        "put the wine bottle on the rack",
    ),
    "libero_spatial": (
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "pick up the black bowl next to the ramekin and place it on the plate",
        "pick up the black bowl from table center and place it on the plate",
        "pick up the black bowl on the cookie box and place it on the plate",
        "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        "pick up the black bowl on the ramekin and place it on the plate",
        "pick up the black bowl next to the cookie box and place it on the plate",
        "pick up the black bowl on the stove and place it on the plate",
        "pick up the black bowl next to the plate and place it on the plate",
        "pick up the black bowl on the wooden cabinet and place it on the plate",
    ),
}


@dataclass(frozen=True)
class BenchmarkSpec:
    """One benchmark of the study: its dataset, its eval env and its ten task keys."""

    name: str
    short: str
    dataset: str
    env_type: str
    suite: str | None
    task_keys: tuple[str, ...]

    def instructions(self) -> dict[str, str]:
        """Task key -> the instruction the dataset (and the eval env) uses for it."""
        if self.env_type == "libero":
            texts = LIBERO_INSTRUCTIONS[self.suite]
            return {key: texts[int(key.rsplit("_", 1)[1])] for key in self.task_keys}
        descriptions = _metaworld_config()["TASK_DESCRIPTIONS"]
        return {key: descriptions[key] for key in self.task_keys}

    def frame_task_ids(self) -> dict[str, int] | None:
        """Task key -> per-frame `task_id`, for datasets where instructions are ambiguous."""
        if self.env_type != "metaworld":
            return None
        ids = _metaworld_config()["TASK_NAME_TO_ID"]
        return {key: int(ids[key]) for key in self.task_keys}

    def env_flags(self, task_keys: list[str] | None = None) -> list[str]:
        """`lerobot-eval` / `--env.*` flags rolling out exactly `task_keys` (default: all ten)."""
        keys = list(task_keys or self.task_keys)
        if self.env_type == "libero":
            ids = ",".join(str(int(key.rsplit("_", 1)[1])) for key in keys)
            return ["--env.type=libero", f"--env.task={self.suite}", f"--env.task_ids=[{ids}]"]
        return ["--env.type=metaworld", f"--env.task={','.join(keys)}"]

    def eval_task_key(self, task_group: str, task_id: int) -> str:
        """The task key of one `per_task` entry of an eval_info.json."""
        if self.env_type == "libero":
            return f"{task_group}_{int(task_id)}"
        return str(task_group)


def _libero(suite: str, short: str) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=suite,
        short=short,
        dataset="lerobot/libero",
        env_type="libero",
        suite=suite,
        task_keys=tuple(f"{suite}_{i}" for i in range(10)),
    )


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "libero_object": _libero("libero_object", "libobject"),
    "libero_goal": _libero("libero_goal", "libgoal"),
    "libero_spatial": _libero("libero_spatial", "libspatial"),
    "metaworld_mt10": BenchmarkSpec(
        name="metaworld_mt10",
        short="mt10",
        dataset="lerobot/metaworld_mt50",
        env_type="metaworld",
        suite=None,
        task_keys=MT10,
    ),
}


def _metaworld_config() -> dict:
    import lerobot.sim.integrations

    return json.loads((Path(lerobot.sim.integrations.__file__).parent / "metaworld_config.json").read_text())


def episode_column_values(
    meta: LeRobotDatasetMetadata, episodes: list[int], column: str
) -> dict[int, set[int]]:
    """The distinct values of a per-frame integer column, for each episode in `episodes`.

    Only that column (plus `episode_index`) is read, and only from the data files holding the
    requested episodes, which are fetched first when missing - no videos are downloaded.
    """
    chunk_col = meta.episodes["data/chunk_index"]
    file_col = meta.episodes["data/file_index"]
    files = sorted(
        {meta.data_path.format(chunk_index=chunk_col[ep], file_index=file_col[ep]) for ep in episodes}
    )
    missing = [f for f in files if not (meta.root / f).is_file()]
    if missing:
        logging.info("Fetching %d data file(s) of %s to read '%s'", len(missing), meta.repo_id, column)
        meta._pull_from_repo(allow_patterns=missing)

    wanted = set(episodes)
    values: dict[int, set[int]] = {}
    for f in files:
        table = pq.read_table(meta.root / f, columns=["episode_index", column])
        ep_col = table.column("episode_index").to_numpy()
        # Scalar features may be stored as length-1 lists; flatten either layout to one value per frame.
        arr = table.column(column).combine_chunks()
        if (
            pa.types.is_list(arr.type)
            or pa.types.is_large_list(arr.type)
            or pa.types.is_fixed_size_list(arr.type)
        ):
            arr = arr.flatten()
        val_col = np.asarray(arr.to_numpy(zero_copy_only=False)).reshape(len(ep_col), -1)[:, 0]
        for ep in np.unique(ep_col):
            if int(ep) in wanted:
                values[int(ep)] = {int(v) for v in np.unique(val_col[ep_col == ep])}
    unresolved = sorted(wanted - set(values))
    if unresolved:
        raise ValueError(f"No frames found for episode(s) {unresolved[:10]} of {meta.repo_id}.")
    return values


def resolve_task_episodes(
    meta: LeRobotDatasetMetadata,
    instructions: dict[str, str],
    frame_task_ids: dict[str, int] | None = None,
) -> dict[str, list[int]]:
    """Map each task key to the episodes recorded for that task, and nothing else.

    Args:
        meta: Metadata of the dataset.
        instructions: Task key -> instruction. Episodes are first matched on it.
        frame_task_ids: Task key -> per-frame `task_id`. When given, the instruction matches are
            narrowed to episodes whose frames all carry that id, which is what separates
            `push-v3` from `push-back-v3` in Meta-World.

    Returns:
        Task key -> sorted episode indices. Keys never share an episode.

    Raises:
        ValueError: When a key matches no episode, when two keys cannot be told apart, or when an
            episode mixes several `task_id`s.
    """
    task_of = episode_tasks(meta, list(range(meta.total_episodes)))
    by_instruction: dict[str, list[int]] = {}
    for ep, text in task_of.items():
        by_instruction.setdefault(text, []).append(ep)

    if frame_task_ids is None:
        clashes = {}
        for key, text in instructions.items():
            twins = [other for other, t in instructions.items() if t == text and other != key]
            if twins:
                clashes[key] = twins
        if clashes:
            raise ValueError(
                f"Task keys share an instruction and cannot be separated without per-frame ids: {clashes}"
            )

    resolved: dict[str, list[int]] = {}
    for key, text in instructions.items():
        candidates = sorted(by_instruction.get(text, []))
        if not candidates:
            raise ValueError(f"No episodes of {meta.repo_id} carry the instruction of {key!r}: {text!r}")
        if frame_task_ids is not None:
            ids = episode_column_values(meta, candidates, "task_id")
            mixed = {ep: sorted(v) for ep, v in ids.items() if len(v) != 1}
            if mixed:
                raise ValueError(f"Episodes mix several task_id values: {dict(list(mixed.items())[:5])}")
            candidates = [ep for ep in candidates if ids[ep] == {frame_task_ids[key]}]
            if not candidates:
                raise ValueError(
                    f"No episode with instruction {text!r} has task_id={frame_task_ids[key]} ({key!r})."
                )
        resolved[key] = candidates

    seen: dict[int, str] = {}
    for key, eps in resolved.items():
        for ep in eps:
            if ep in seen:
                raise ValueError(f"Episode {ep} resolved to both {seen[ep]!r} and {key!r}.")
            seen[ep] = key
    return resolved


def benchmark_episodes(meta: LeRobotDatasetMetadata, bench: BenchmarkSpec) -> dict[str, list[int]]:
    """Task key -> episodes, for every task of `bench`."""
    return resolve_task_episodes(meta, bench.instructions(), bench.frame_task_ids())
