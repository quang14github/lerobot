#!/usr/bin/env python
"""Collect every evaluation of the federated VLA study into one CSV.

  python examples/federated/vla_study/collect_results.py [STUDY_ROOT]

Writes, under STUDY_ROOT (default outputs/fedvla):

  results.csv          one row per (run, evaluated round, task)
  client_losses.csv    every federated run's per-client, per-round training loss
  checkpoint_lrs.csv   the learning rate at every checkpoint, both arms
  missing.txt          runs without results yet, failed runs, unreadable files

Missing, running and failed runs are reported, never fatal. Centralized runs are the pooled
reference for every partition; they appear once per seed with partition_type "pooled". The
untrained models appear with arm "pretrained" at round 0.
"""

import csv
import json
import sys
from pathlib import Path

RESULT_COLUMNS = [
    "benchmark",
    "policy",
    "mode",
    "partition_type",
    "seed",
    "partition_seed",
    "arm",
    "round",
    "task",
    "n_clients_holding",
    "task_share_by_client",
    "success_rate",
    "n_episodes",
    "run",
]


def _read_json(path: Path, problems: list[str]):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as err:
        problems.append(f"unreadable {path}: {err}")
        return None


def _task_columns(manifest: dict | None, task: str, arm: str) -> tuple[str, str]:
    """(clients holding the task, its episode share at each client in client-id order)."""
    if arm in ("centralized", "pretrained"):
        return ("1", "1.000") if arm == "centralized" else ("", "")
    if not manifest or task not in manifest.get("tasks", {}):
        return "", ""
    row = manifest["tasks"][task]
    shares = row["episode_share"]
    return str(row["clients_holding"]), ";".join(f"{shares[c]:.3f}" for c in sorted(shares, key=int))


def _run_rows(run_dir: Path, root: Path, problems: list[str], protocols: set) -> list[dict]:
    run = _read_json(run_dir / "run.json", problems)
    if run is None:
        return []
    arm = run.get("arm", "")
    manifest = None
    if arm == "federated":
        manifest_path = run_dir / "train" / "manifest.json"
        manifest = _read_json(manifest_path, problems) if manifest_path.exists() else None
    rows = []
    for result_path in (
        sorted((run_dir / "eval").glob("round_*/result.json")) if (run_dir / "eval").exists() else []
    ):
        result = _read_json(result_path, problems)
        if result is None:
            continue
        protocols.add(json.dumps(result.get("protocol", {}), sort_keys=True))
        rnd = int(result_path.parent.name.removeprefix("round_"))
        for task, metrics in sorted(result["tasks"].items()):
            holding, shares = _task_columns(manifest, task, arm)
            rows.append(
                {
                    "benchmark": run.get("bench", ""),
                    "policy": run.get("policy", ""),
                    "mode": run.get("mode", ""),
                    "partition_type": "pooled" if arm == "centralized" else (run.get("partition") or ""),
                    "seed": run.get("seed") if run.get("seed") is not None else "",
                    "partition_seed": run.get("partition_seed")
                    if run.get("partition_seed") is not None
                    else "",
                    "arm": arm,
                    "round": rnd,
                    "task": task,
                    "n_clients_holding": holding,
                    "task_share_by_client": shares,
                    "success_rate": f"{metrics['success_rate']:.2f}",
                    "n_episodes": metrics["n_episodes"],
                    "run": str(run_dir.relative_to(root)),
                }
            )
    rounds = run.get("rounds", 0)
    expected = (
        {0}
        if arm == "pretrained"
        else set(range(run.get("save_every", 20), rounds + 1, run.get("save_every", 20)))
    )
    have = (
        {int(p.parent.name.removeprefix("round_")) for p in (run_dir / "eval").glob("round_*/result.json")}
        if (run_dir / "eval").exists()
        else set()
    )
    status = (
        "FAILED"
        if (run_dir / "FAILED").exists()
        else "complete"
        if (run_dir / "train" / "TRAINING_COMPLETE").exists()
        else "training"
    )
    if expected - have or (status != "complete" and arm != "pretrained"):
        problems.append(
            f"{run_dir.relative_to(root)}: {status}; evaluated rounds {sorted(have)}; missing {sorted(expected - have)}"
        )
    return rows


def _copy_csv(src: Path, extra: dict, writer_holder: dict, out: Path) -> None:
    with src.open() as f:
        for row in csv.DictReader(f):
            row = {**extra, **row}
            if "writer" not in writer_holder:
                writer_holder["file"] = out.open("w", newline="")
                writer_holder["writer"] = csv.DictWriter(writer_holder["file"], fieldnames=list(row))
                writer_holder["writer"].writeheader()
            writer_holder["writer"].writerow(row)


def collect(root: Path) -> int:
    root = Path(root)
    problems: list[str] = []
    protocols: set = set()
    run_dirs = sorted(p.parent for p in (root / "studies").glob("*/*/run.json")) + sorted(
        p.parent for p in (root / "baselines").glob("*/run.json")
    )
    rows = []
    for run_dir in run_dirs:
        try:
            rows += _run_rows(run_dir, root, problems, protocols)
        except Exception as err:  # noqa: BLE001 - one broken run must not hide the others
            problems.append(f"{run_dir}: {type(err).__name__}: {err}")
    if len(protocols) > 1:
        problems.append(f"WARNING: evaluations used different protocols: {sorted(protocols)}")

    tmp = root / "results.csv.tmp"
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(root / "results.csv")

    losses, lrs = {}, {}
    for run_dir in run_dirs:
        run_path = run_dir / "run.json"
        try:
            run = json.loads(run_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue  # already reported above
        identity = {
            k: run.get(k, "")
            for k in ("bench", "policy", "mode", "arm", "partition", "seed", "partition_seed")
        }
        identity["run"] = str(run_dir.relative_to(root))
        try:
            if (run_dir / "train" / "client_losses.csv").exists():
                _copy_csv(
                    run_dir / "train" / "client_losses.csv", identity, losses, root / "client_losses.csv"
                )
            if (run_dir / "train" / "checkpoints.csv").exists():
                _copy_csv(run_dir / "train" / "checkpoints.csv", identity, lrs, root / "checkpoint_lrs.csv")
        except (OSError, csv.Error) as err:
            problems.append(f"{run_dir}: could not read its CSV logs: {err}")
    for holder in (losses, lrs):
        if "file" in holder:
            holder["file"].close()

    (root / "missing.txt").write_text("\n".join(problems) + ("\n" if problems else ""))
    print(f"{len(rows)} result rows from {len(run_dirs)} runs -> {root / 'results.csv'}")
    print(f"{len(problems)} runs incomplete or with problems -> {root / 'missing.txt'}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import grid

    default = grid.Settings.from_env().study_root
    sys.exit(collect(Path(sys.argv[1]) if len(sys.argv) > 1 else default))
