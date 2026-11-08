#!/usr/bin/env python
"""Run the federated SmolVLA (LoRA) fine-tuning study from a terminal.

See README.md in this folder. Every command can be stopped with Ctrl-C (the current run saves its
state and stops; a second Ctrl-C aborts) and rerun to continue where it stopped.

  prepare        resolve every benchmark's tasks to episodes and check every partition
  prefetch       download the checkpoint and the datasets' episodes up front (optional)
  plan           list the runs of a phase in order, with their progress
  run            train the runs in order (resuming each), evaluating each one when it finishes
  eval           evaluate every saved, not yet evaluated checkpoint (can run in another terminal)
  baseline       build and evaluate the untrained model on each benchmark
  status         progress of every run
  aggregate      collect all evaluations into results.csv
  smoke          tiny end-to-end tests; bench-workers / bench-eval: throughput measurements

Settings are environment variables (see grid.py), e.g. `RUN_CENTRALIZED=0 python study.py run`.
"""

import argparse
import contextlib
import csv
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import grid  # noqa: E402
from grid import REPO_ROOT, RunSpec, Settings  # noqa: E402

EXIT_INTERRUPTED = 75

# --------------------------------------------------------------------------------------------
# Process plumbing
# --------------------------------------------------------------------------------------------
_child: subprocess.Popen | None = None
_stop = {"requested": False}


def _on_signal(signum, frame):  # noqa: ARG001
    """Ctrl-C reaches the child directly (same process group): just remember to stop after it.
    A second Ctrl-C aborts. SIGTERM is forwarded, since only this process receives it."""
    if signum == signal.SIGINT and _stop["requested"]:
        raise KeyboardInterrupt
    if not _stop["requested"]:
        print(
            f"\n*** {signal.Signals(signum).name}: stopping after the current run saves its state", flush=True
        )
    _stop["requested"] = True
    if signum != signal.SIGINT and _child is not None and _child.poll() is None:
        _child.send_signal(signum)


signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def stamp() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"


def run_logged(cmd: list, log_path: Path, env: dict | None = None) -> int:
    """Run `cmd`, streaming its output to the terminal and appending it to `log_path`."""
    global _child
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        header = f"\n# {now()} on {socket.gethostname()}\n# {shlex.join(map(str, cmd))}\n"
        log.write(header)
        print(header, flush=True)
        _child = subprocess.Popen(
            list(map(str, cmd)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=REPO_ROOT,
            env={**os.environ, **(env or {})},
        )
        for line in _child.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = _child.wait()
        _child = None
        log.write(f"# exit {rc} at {now()}\n")
    return rc


@contextlib.contextmanager
def gpu_logger(path: Path):
    """nvidia-smi utilization and memory every 30 s to CSV, while the block runs (when available)."""
    proc = None
    if shutil.which("nvidia-smi") and os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "":
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = "timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
        proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv", "-l", "30"],
            stdout=path.open("a"),
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # keep Ctrl-C away from it; it is stopped below
        )
    try:
        yield
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait()


def git_commit() -> str:
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    tmp.replace(path)


def append_tsv(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row), delimiter="\t")
        if new:
            writer.writeheader()
        writer.writerow(row)


def acquire(lock: Path) -> bool:
    """A lock file shared by every process of the study (e.g. `run` and an `eval` in another
    terminal). A lock left by a process that no longer runs on this machine is stale."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                json.dump({"host": socket.gethostname(), "pid": os.getpid(), "time": time.time()}, f)
            return True
        except FileExistsError:
            try:
                info = json.loads(lock.read_text())
            except (OSError, json.JSONDecodeError):
                info = {}
            stale = time.time() - float(info.get("time", 0)) > 7 * 86400
            if info.get("host") == socket.gethostname():
                try:
                    os.kill(int(info.get("pid", -1)), 0)
                except (ProcessLookupError, ValueError):
                    stale = True
                except PermissionError:
                    pass
            if stale:
                lock.unlink(missing_ok=True)
                continue
            return False
    return False


# --------------------------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------------------------
def prepare_benchmark(s: Settings, bench_name: str) -> dict:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.federated.config import PartitionConfig
    from lerobot.federated.partition import check_exact_cover, episode_table, partition_episodes
    from lerobot.federated.tasks import BENCHMARKS, benchmark_episodes

    bench = BENCHMARKS[bench_name]
    meta = LeRobotDatasetMetadata(bench.dataset)
    resolved = benchmark_episodes(meta, bench)
    task_map = {ep: key for key, eps in resolved.items() for ep in eps}
    episodes = sorted(task_map)
    _, _, frames_of = episode_table(meta, episodes, task_map)

    # Every partition the study can draw must hand out exactly the episodes the centralized arm
    # pools - all of them, once each. Check them all now rather than inside a long run.
    checks = []
    for partition, (strategy, _) in grid.PARTITIONS.items():
        for pseed in s.partition_seeds:
            cfg = PartitionConfig(strategy=strategy, num_clients=s.num_clients, seed=pseed)
            shards = partition_episodes(meta, cfg, episodes=episodes, task_map=task_map)
            check_exact_cover(shards, episodes)
            checks.append(
                {
                    "partition": partition,
                    "partition_seed": pseed,
                    "client_episodes": [sh.num_episodes for sh in shards],
                    "client_tasks": [sorted(sh.tasks) for sh in shards],
                }
            )
    out = grid.prep_dir(s, bench_name)
    write_json(
        out / "task_map.json",
        {
            "dataset": bench.dataset,
            "benchmark": bench_name,
            "episode_tasks": {str(ep): key for ep, key in task_map.items()},
            "tasks": resolved,
        },
    )
    write_json(out / "partition_checks.json", checks)
    instructions = bench.instructions()
    with (out / "tasks.tsv").open("w") as f:
        f.write("task\tepisodes\tframes\tinstruction\n")
        for key, eps in resolved.items():
            f.write(f"{key}\t{len(eps)}\t{sum(frames_of[e] for e in eps)}\t{instructions[key]}\n")
    print(
        f"{bench_name}: {len(episodes)} episodes over {len(resolved)} tasks; {len(checks)} partitions checked"
    )
    return resolved


def ensure_prepared(s: Settings, bench_name: str) -> None:
    if not grid.task_map_path(s, bench_name).is_file():
        prepare_benchmark(s, bench_name)


def cmd_prepare(args, s: Settings) -> int:
    for bench in args.benchmarks or s.benchmarks:
        prepare_benchmark(s, bench)
    return 0


def cmd_prefetch(args, s: Settings) -> int:
    from huggingface_hub import snapshot_download

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.federated import vla_study
    from lerobot.federated.tasks import BENCHMARKS

    revisions = {}
    for repo in (vla_study.PRETRAINED, "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"):
        path = snapshot_download(repo)
        revisions[repo] = Path(path).name
        print(f"model {repo}: {path}")
    benches = args.benchmarks or s.benchmarks
    for bench_name in benches:
        prepare_benchmark(s, bench_name)
        ds = LeRobotDataset(BENCHMARKS[bench_name].dataset, episodes=grid.benchmark_episodes(s, bench_name))
        print(f"dataset {BENCHMARKS[bench_name].dataset} ({bench_name}): {ds.num_episodes} episodes cached")
    if any(BENCHMARKS[b].env_type == "libero" for b in benches):
        # LIBERO prompts on its first import; declining writes its default config.
        subprocess.run([sys.executable, "-c", "import libero.libero"], input="N\n", text=True, check=False)
        out = subprocess.run(
            [sys.executable, "-c", "from libero.libero import get_assets_path; print(get_assets_path())"],
            capture_output=True,
            text=True,
        )
        print(f"LIBERO assets: {out.stdout.strip() or out.stderr.strip()[-300:]}")
    write_json(s.study_root / "prep" / "revisions.json", revisions)
    return 0


# --------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------
def train_dir(run_dir: Path) -> Path:
    return run_dir / "train"


def is_complete(run_dir: Path) -> bool:
    return (train_dir(run_dir) / "TRAINING_COMPLETE").is_file()


def progress(spec: RunSpec, s: Settings) -> float:
    """Rounds (or round-equivalents) completed so far."""
    out = train_dir(spec.run_dir(s))
    if spec.arm == "federated":
        state = out / "fl_state" / "state.json"
        return float(json.loads(state.read_text())["round"]) if state.is_file() else 0.0
    last = out / "checkpoints" / "last"
    return int(last.resolve().name) / spec.round_steps(s) if last.exists() else 0.0


def write_study_env(spec: RunSpec, s: Settings) -> None:
    """study.env: the study's settings and code version, written once (sessions.tsv logs later ones)."""
    path = spec.study_dir(s) / "study.env"
    if path.exists():
        return
    values = {
        "STUDY": spec.study_id(s),
        "STARTED": now(),
        "GIT_COMMIT": git_commit(),
        "POLICY": grid.POLICY,
        "MODE": spec.mode,
        "BENCHMARK": spec.bench,
        "ARM": spec.arm,
        "PARTITION": spec.partition or "pooled",
        "HOST": socket.gethostname(),
        **s.as_env(),
    }
    revisions = s.study_root / "prep" / "revisions.json"
    if revisions.is_file():
        values["CHECKPOINT_REVISIONS"] = revisions.read_text().replace("\n", " ")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"# Settings of study {spec.study_id(s)}, written by examples/federated/vla_study/study.py\n")
        for key, value in values.items():
            f.write(f"{key}='{str(value).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n")


def write_run_json(spec: RunSpec, s: Settings) -> None:
    path = spec.run_dir(s) / "run.json"
    if path.exists():
        return
    write_json(
        path,
        {
            **spec.to_json(),
            "policy": grid.POLICY,
            "study": spec.study_id(s),
            "run": spec.run_id(),
            "rounds": s.rounds,
            "local_steps": s.local_steps,
            "num_clients": s.num_clients,
            "round_steps": spec.round_steps(s),
            "save_every": s.save_every,
            "batch_size": s.batch_size,
            "client_weighting": s.client_weighting,
            "client_opt_state": s.client_opt_state,
        },
    )


def train_run(spec: RunSpec, s: Settings, overrides: list[str] | None = None) -> int:
    """Train one run to completion, continuing it if it was started before. Returns the trainer's
    exit code: 0 complete, 75 stopped cleanly (rerun to continue), anything else failed."""
    run_dir = spec.run_dir(s)
    print(f"\n=== train {spec.label()}  ({run_dir})", flush=True)
    if is_complete(run_dir):
        print("    already complete")
        return 0
    ensure_prepared(s, spec.bench)
    write_study_env(spec, s)
    write_run_json(spec, s)
    out = train_dir(run_dir)
    if spec.arm == "federated":
        cmd = grid.federated_command(spec, s, out, overrides=overrides)
    else:
        resume = (out / "checkpoints" / "last" / "training_state").is_dir()
        if not resume and out.exists():
            # An earlier session stopped before the first checkpoint: nothing to resume, and the
            # trainer refuses to start in an existing directory. Keep it for inspection.
            out.rename(out.with_name(f"train.aborted-{stamp()}"))
        cmd = grid.centralized_command(spec, s, out, resume=resume, overrides=overrides)

    before, start = progress(spec, s), time.time()
    session = stamp()
    with gpu_logger(run_dir / "logs" / f"gpu_{session}.csv"):
        rc = run_logged(cmd, run_dir / "logs" / f"train_{session}.log")
    after, hours = progress(spec, s), (time.time() - start) / 3600
    append_tsv(
        run_dir / "sessions.tsv",
        {
            "session": session,
            "host": socket.gethostname(),
            "commit": git_commit(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start)),
            "hours": round(hours, 3),
            "exit": rc,
            "rounds_before": before,
            "rounds_after": after,
            "rounds_per_hour": round((after - before) / hours, 3) if hours > 0 else "",
        },
    )
    if rc == 0 and is_complete(run_dir):
        print(f"    complete ({after:g} rounds)")
    elif rc == EXIT_INTERRUPTED:
        print(f"    stopped at round {after:g}; rerun the same command to continue")
    else:
        print(f"!!  training exited with {rc} at round {after:g}; see {run_dir / 'logs'}")
    return rc


# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------
def eval_targets(run_dir: Path, run: dict) -> list[tuple[int, Path]]:
    """Unevaluated checkpoints on the eval grid, never past the last fully written checkpoint."""
    ckpts = train_dir(run_dir) / "checkpoints"
    if not ckpts.is_dir():
        return []
    last = ckpts / "last"
    newest = int(last.resolve().name) if last.exists() else -1
    out = []
    for d in sorted(ckpts.iterdir()):
        if d.is_symlink() or not d.name.isdigit() or int(d.name) > newest:
            continue
        step = int(d.name)
        if step % run["round_steps"]:
            continue
        rnd = step // run["round_steps"]
        on_grid = rnd == 0 or rnd == run["rounds"] or rnd % run["save_every"] == 0
        if not on_grid or (d / "PRUNED").exists() or not (d / "pretrained_model" / "config.json").exists():
            continue
        if (run_dir / "eval" / f"round_{rnd:04d}" / "result.json").exists():
            continue
        out.append((rnd, d))
    return out


def evaluate_run(
    run_dir: Path, s: Settings, device: str, only_round: int | None = None, tasks: str = ""
) -> int:
    run = json.loads((run_dir / "run.json").read_text())
    targets = [t for t in eval_targets(run_dir, run) if only_round is None or t[0] == only_round]
    failures = 0
    for rnd, ckpt in targets:
        if _stop["requested"]:
            break
        lock = run_dir / "eval" / f"round_{rnd:04d}.lock"
        if not acquire(lock):
            print(f"    round {rnd}: being evaluated by another process")
            continue
        try:
            out = run_dir / "eval" / f"round_{rnd:04d}"
            cmd = [
                sys.executable,
                HERE / "eval_run.py",
                "--checkpoint",
                ckpt / "pretrained_model",
                "--benchmark",
                run["bench"],
                "--out",
                out,
                "--device",
                device,
                "--n-episodes",
                s.eval_episodes,
                "--seed",
                s.eval_seed,
                "--batch-size",
                s.eval_batch_size,
            ]
            if tasks:
                cmd += ["--tasks", tasks]
            rc = run_logged(cmd, run_dir / "logs" / f"eval_round_{rnd:04d}_{stamp()}.log")
            if rc != 0 and not _stop["requested"]:
                failures += 1
                print(f"!!  evaluation of round {rnd} failed (exit {rc})")
        finally:
            lock.unlink(missing_ok=True)
    return 1 if failures else 0


def build_baseline(bench: str, s: Settings, device: str) -> Path:
    """The untrained model of a benchmark as a round-0 checkpoint, built once."""
    bdir = grid.baseline_dir(s, bench)
    ready = bdir / "train" / "checkpoints" / "000000" / "pretrained_model" / "config.json"
    if ready.exists():
        return bdir
    lock = bdir / "build.lock"
    while not acquire(lock):  # another process is building it
        time.sleep(10)
        if ready.exists():
            return bdir
    try:
        ensure_prepared(s, bench)
        write_json(
            bdir / "run.json",
            {
                "policy": grid.POLICY,
                "mode": "pretrained",
                "bench": bench,
                "arm": "pretrained",
                "partition": None,
                "seed": None,
                "partition_seed": None,
                "rounds": 0,
                "round_steps": 1,
                "save_every": s.save_every,
            },
        )
        shutil.rmtree(bdir / "train", ignore_errors=True)
        rc = run_logged(
            grid.baseline_command(bench, s, bdir / "train", device), bdir / "logs" / f"build_{stamp()}.log"
        )
        if rc != 0:
            raise RuntimeError(f"building the untrained baseline for {bench} failed (exit {rc})")
    finally:
        lock.unlink(missing_ok=True)
    return bdir


def baseline(bench: str, s: Settings, device: str) -> int:
    bdir = grid.baseline_dir(s, bench)
    print(f"\n=== baseline: untrained {grid.POLICY} on {bench}", flush=True)
    if (bdir / "eval" / "round_0000" / "result.json").exists():
        print("    already evaluated")
        return 0
    build_baseline(bench, s, device)
    return evaluate_run(bdir, s, device)


# --------------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------------
def selected_runs(args, s: Settings) -> list[RunSpec]:
    if getattr(args, "phase", None):
        phases = [args.phase]
    else:
        # `run` and `plan` default to Phase 1; `eval` and `status` cover whatever exists.
        phases = [1, 2] if args.cmd in ("eval", "status") else [1]
    runs = [spec for phase in phases for spec in grid.runs_for(phase, s)]
    if getattr(args, "only", None):
        runs = [r for r in runs if args.only in r.label()]
    return runs


def cmd_plan(args, s: Settings) -> int:
    runs = selected_runs(args, s)
    for i, spec in enumerate(runs):
        state = "done" if is_complete(spec.run_dir(s)) else f"{progress(spec, s):g}/{s.rounds}"
        print(f"{i:3d}  {spec.label():<55} {state}")
    print(f"\n{len(runs)} runs of {s.rounds} rounds x {s.local_steps} local steps x {s.num_clients} clients")
    return 0


def cmd_run(args, s: Settings) -> int:
    evaluate = not args.no_eval
    if evaluate and (args.phase or 1) == 1:
        for bench in s.benchmarks:
            if _stop["requested"]:
                return EXIT_INTERRUPTED
            baseline(bench, s, s.eval_device)
    for spec in selected_runs(args, s):
        if _stop["requested"]:
            return EXIT_INTERRUPTED
        rc = train_run(spec, s)
        if rc == EXIT_INTERRUPTED or _stop["requested"]:
            return EXIT_INTERRUPTED
        if rc != 0:
            if args.keep_going:
                continue
            return rc
        if evaluate:
            evaluate_run(spec.run_dir(s), s, s.eval_device)
    print("\nall selected runs are complete")
    return 0


def cmd_eval(args, s: Settings) -> int:
    device = args.device or s.eval_device
    failures = 0
    if not args.no_baseline:
        for bench in s.benchmarks:
            if not _stop["requested"]:
                failures += baseline(bench, s, device)
    for spec in selected_runs(args, s):
        run_dir = spec.run_dir(s)
        if _stop["requested"]:
            break
        if (run_dir / "run.json").exists():
            print(f"\n=== eval {spec.label()}")
            failures += evaluate_run(run_dir, s, device)
    return 1 if failures else 0


def cmd_baseline(args, s: Settings) -> int:
    return max((baseline(b, s, args.device or s.eval_device) for b in s.benchmarks), default=0)


def cmd_status(args, s: Settings) -> int:
    print(f"{'run':<55} {'rounds':>9} {'state':<9} {'evaluated':>9} {'sessions':>8} {'hours':>7}")
    for spec in selected_runs(args, s):
        run_dir = spec.run_dir(s)
        if not run_dir.exists():
            continue
        state = "done" if is_complete(run_dir) else "partial"
        evals = (
            len(list((run_dir / "eval").glob("round_*/result.json"))) if (run_dir / "eval").exists() else 0
        )
        sessions = (
            list(csv.DictReader((run_dir / "sessions.tsv").open(), delimiter="\t"))
            if (run_dir / "sessions.tsv").exists()
            else []
        )
        hours = sum(float(r["hours"]) for r in sessions)
        print(
            f"{spec.label():<55} {progress(spec, s):>4.0f}/{s.rounds:<4} {state:<9} "
            f"{evals:>4}/{s.rounds // s.save_every:<4} {len(sessions):>8} {hours:>7.1f}"
        )
    for bench in s.benchmarks:
        done = (grid.baseline_dir(s, bench) / "eval" / "round_0000" / "result.json").exists()
        print(f"baseline {bench}: {'evaluated' if done else 'pending'}")
    return 0


def cmd_aggregate(args, s: Settings) -> int:
    import collect_results

    return collect_results.collect(s.study_root)


# --------------------------------------------------------------------------------------------
# Smoke tests
# --------------------------------------------------------------------------------------------
SMOKE_SETTINGS = {
    "ROUNDS": "2",
    "LOCAL_STEPS": "3",
    "SAVE_EVERY": "1",
    "BATCH_SIZE": os.environ.get("SMOKE_BATCH_SIZE", "8"),
    "NUM_WORKERS": "1",
    "CEN_NUM_WORKERS": "2",
    "EVAL_EPISODES": "1",
    "SEEDS": "1000",
    "PARTITION_SEEDS": "1",
}
# Benchmark key -> (benchmark, federated partition, single task to keep or "" for all ten).
SMOKE_BENCHES = {
    "libero_object": ("libero_object", "taskdisjoint", ""),
    "metaworld": ("metaworld_mt10", "iid", "reach-v3"),
}


def smoke_settings(root: Path, **extra: str) -> Settings:
    os.environ.update({**SMOKE_SETTINGS, **extra, "STUDY_ROOT": str(root)})
    return Settings.from_env()


def _restrict_task_map(s: Settings, bench: str, task: str) -> None:
    """A task map holding only one task's episodes (the single-task Meta-World smoke test)."""
    path = grid.task_map_path(s, bench)
    payload = json.loads(path.read_text())
    if task and set(payload["episode_tasks"].values()) != {task}:
        payload["episode_tasks"] = {ep: key for ep, key in payload["episode_tasks"].items() if key == task}
        payload["tasks"] = {task: payload["tasks"][task]}
        write_json(path, payload)


def _size_gb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**3, 3)


def cmd_smoke_case(args, s: Settings) -> int:
    bench_key, arm = args.name.split("-")
    bench, partition, only_task = SMOKE_BENCHES[bench_key]
    s = smoke_settings(s.study_root / "smoke" / bench_key)
    ensure_prepared(s, bench)
    _restrict_task_map(s, bench, only_task)
    federated = arm == "fed"
    spec = RunSpec(
        bench,
        "federated" if federated else "centralized",
        partition if federated else None,
        1000,
        1 if federated else None,
    )
    run_dir = spec.run_dir(s)
    shutil.rmtree(run_dir, ignore_errors=True)
    start = time.time()
    rc = train_run(spec, s, overrides=["--fl.audit_gradients=true"] if federated else None)
    train_s = time.time() - start
    ok_train = rc == 0 and is_complete(run_dir)
    eval_start = time.time()
    rc_eval = (
        evaluate_run(run_dir, s, s.eval_device, only_round=s.rounds, tasks=only_task or "libero_object_0")
        if ok_train
        else 1
    )
    out = train_dir(run_dir)
    summary = {
        "case": args.name,
        "train_ok": ok_train,
        "eval_ok": rc_eval == 0 and (run_dir / "eval" / f"round_{s.rounds:04d}" / "result.json").exists(),
        "train_s": round(train_s, 1),
        "eval_s": round(time.time() - eval_start, 1),
        "total_s": round(time.time() - start, 1),
        "checkpoint_gb": {d.name: _size_gb(d) for d in sorted((out / "checkpoints").glob("[0-9]*"))}
        if (out / "checkpoints").exists()
        else {},
        "fl_state_gb": _size_gb(out / "fl_state") if (out / "fl_state").exists() else None,
    }
    summary["under_15_min"] = summary["total_s"] < 900
    for name in ("trainable_params.json", "grad_audit.json"):
        if (out / name).exists():
            payload = json.loads((out / name).read_text())
            payload.pop("by_module", None)
            if "no_grad_names" in payload:
                payload["no_grad_names"] = payload["no_grad_names"][:50]
            summary[name.removesuffix(".json")] = payload
    if (out / "rounds.csv").exists():
        summary["rounds"] = list(csv.DictReader((out / "rounds.csv").open()))
    result = run_dir / "eval" / f"round_{s.rounds:04d}" / "result.json"
    if result.exists():
        summary["eval"] = json.loads(result.read_text())
    write_json(s.study_root / "summaries" / f"{args.name}.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rounds", "eval")}, indent=1))
    return 0 if summary["train_ok"] and summary["eval_ok"] else 1


def cmd_smoke_resume(args, s: Settings) -> int:
    """Interrupt a real run partway (SIGINT, as Ctrl-C), restart it, check it continues where it stopped."""
    # Long enough rounds that the interrupt lands mid-run rather than after the last round.
    s = smoke_settings(s.study_root / "smoke" / "resume", ROUNDS="4", SAVE_EVERY="2", LOCAL_STEPS="20")
    ensure_prepared(s, "libero_object")
    report = {"checks": {}}
    for arm in ("federated", "centralized"):
        federated = arm == "federated"
        spec = RunSpec(
            "libero_object",
            arm,
            "taskdisjoint" if federated else None,
            1000,
            1 if federated else None,
        )
        run_dir = spec.run_dir(s)
        shutil.rmtree(run_dir, ignore_errors=True)
        write_run_json(spec, s)
        out = train_dir(run_dir)

        def command(resume: bool, spec=spec, out=out) -> list:
            if spec.arm == "federated":
                return grid.federated_command(spec, s, out)
            return grid.centralized_command(spec, s, out, resume=resume)

        log = run_dir / "logs" / "first.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as f:
            proc = subprocess.Popen(
                list(map(str, command(False))),
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
                start_new_session=True,
            )
        # Interrupt once the run is past round 2 (federated) or its first checkpoint (centralized).
        while proc.poll() is None:
            rounds_csv = out / "rounds.csv"
            if federated and rounds_csv.exists() and len(rounds_csv.read_text().strip().splitlines()) >= 3:
                break
            if not federated and any((out / "checkpoints").glob("[0-9]*")):
                time.sleep(2)
                break
            time.sleep(1)
        proc.send_signal(signal.SIGINT)
        rc_first = proc.wait()
        stopped_at = progress(spec, s)
        rc_second = run_logged(command(True), run_dir / "logs" / "second.log")
        text = (run_dir / "logs" / "second.log").read_text()
        check = {
            "first_exit": rc_first,
            "stopped_at_round": stopped_at,
            "second_exit": rc_second,
            "complete": is_complete(run_dir),
        }
        if federated:
            rounds = [int(r["round"]) for r in csv.DictReader((out / "rounds.csv").open())]
            check["rounds_logged"] = rounds
            check["resumed_from_saved_round"] = f"Resuming after round {int(stopped_at)}" in text
            check["pass"] = (
                rc_first == EXIT_INTERRUPTED
                and rounds == list(range(1, s.rounds + 1))
                and check["resumed_from_saved_round"]
                and check["complete"]
            )
        else:
            steps = [int(r["step"]) for r in csv.DictReader((out / "checkpoints.csv").open())]
            check["checkpoint_steps"] = steps
            check["pass"] = (
                rc_first == EXIT_INTERRUPTED and check["complete"] and steps[-1] == spec.total_steps(s)
            )
        report["checks"][arm] = check
    report["pass"] = all(c["pass"] for c in report["checks"].values())
    write_json(s.study_root / "summaries" / "resume.json", report)
    print(json.dumps(report, indent=1))
    return 0 if report["pass"] else 1


def cmd_smoke(args, s: Settings) -> int:
    cases = [["smoke-case", f"{b}-{a}"] for b in SMOKE_BENCHES for a in ("fed", "cen")]
    cases += [["smoke-resume"]]
    if args.only:
        cases = [c for c in cases if args.only in " ".join(c)]
    failed = []
    for case in cases:
        if _stop["requested"]:
            break
        rc = subprocess.run([sys.executable, HERE / "study.py", *case], cwd=REPO_ROOT).returncode
        if rc:
            failed.append(" ".join(case))
    print(
        f"\nsmoke tests: {len(cases) - len(failed)}/{len(cases)} passed"
        + (f"; failed: {failed}" if failed else "")
    )
    print(f"summaries: {s.study_root / 'smoke'}/*/summaries/")
    return 1 if failed else 0


# --------------------------------------------------------------------------------------------
# Throughput measurements
# --------------------------------------------------------------------------------------------
def cmd_bench_case(args, s: Settings) -> int:
    bench, workers = args.name.split("-")
    root = s.study_root / "bench"
    s = smoke_settings(
        root,
        ROUNDS="2",
        SAVE_EVERY="2",
        LOCAL_STEPS=os.environ.get("BENCH_LOCAL_STEPS", "30"),
        BATCH_SIZE=os.environ.get("BATCH_SIZE", "64"),
        NUM_WORKERS=workers.removeprefix("w"),
    )
    ensure_prepared(s, bench)
    spec = RunSpec(bench, "federated", "taskdisjoint", 1000, 1)
    run_dir = root / "runs" / args.name
    shutil.rmtree(run_dir, ignore_errors=True)
    out = train_dir(run_dir)
    # Training only: no checkpoint is written.
    cmd = grid.federated_command(spec, s, out, overrides=["--save_checkpoint=false"])
    with gpu_logger(run_dir / "gpu.csv"):
        rc = run_logged(cmd, run_dir / "train.log")
    rows = list(csv.DictReader((out / "rounds.csv").open())) if (out / "rounds.csv").exists() else []
    steady = rows[-1] if len(rows) >= 2 else {}  # round 1 includes starting the data-loader workers
    result = {
        "case": args.name,
        "mode": grid.MODE,
        "benchmark": bench,
        "workers_per_client": int(workers.removeprefix("w")),
        "batch_size": s.batch_size,
        "exit": rc,
        "steps_per_s": float(steady["steps_per_s"]) if steady else None,
        "gpu_mem_peak_gb": float(steady["gpu_mem_peak_gb"]) if steady else None,
        "rss_with_workers_gb": float(steady["rss_with_workers_gb"]) if steady else None,
        "trainable": json.loads((out / "trainable_params.json").read_text())["trainable"]
        if (out / "trainable_params.json").exists()
        else None,
        "fl_state_gb": _size_gb(out / "fl_state") if (out / "fl_state").exists() else None,
        "host": socket.gethostname(),
    }
    write_json(root / "results" / f"{args.name}.json", result)
    print(json.dumps(result, indent=1))
    return 0 if rc == 0 else 1


def cmd_bench_workers(args, s: Settings) -> int:
    cases = [f"libero_object-w{w}" for w in args.workers] + [f"metaworld_mt10-w{args.workers[-1]}"]
    failed = []
    for case in cases:
        if _stop["requested"]:
            break
        if subprocess.run([sys.executable, HERE / "study.py", "bench-case", case], cwd=REPO_ROOT).returncode:
            failed.append(case)
    print(
        f"\n{len(cases) - len(failed)}/{len(cases)} benchmark cases ran; results in {s.study_root / 'bench' / 'results'}"
    )
    return 1 if failed else 0


def cmd_bench_eval(args, s: Settings) -> int:
    """SmolVLA evaluation speed (episodes/hour) per device: the untrained model, one episode per task."""
    failed = 0
    for device in args.devices:
        for bench in ("libero_object", "metaworld_mt10"):
            if _stop["requested"]:
                return EXIT_INTERRUPTED
            bdir = build_baseline(bench, s, device)
            out = s.study_root / "bench" / "eval" / f"{device}-{bench}"
            env = {"CUDA_VISIBLE_DEVICES": ""} if device == "cpu" else None
            rc = run_logged(
                [
                    sys.executable,
                    HERE / "eval_run.py",
                    "--checkpoint",
                    bdir / "train" / "checkpoints" / "000000" / "pretrained_model",
                    "--benchmark",
                    bench,
                    "--out",
                    out,
                    "--device",
                    device,
                    "--n-episodes",
                    str(args.episodes),
                ],
                out.parent / f"{device}-{bench}.log",
                env=env,
            )
            failed += rc != 0
    return 1 if failed else 0


# --------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("prepare", "prefetch"):
        sub.add_parser(name).add_argument("--benchmarks", nargs="*")
    for name in ("plan", "run", "eval", "status"):
        p = sub.add_parser(name)
        p.add_argument(
            "--phase", type=int, choices=(1, 2), help="1: task-disjoint (+ centralized); 2: IID + shared-core"
        )
        p.add_argument("--only", help="only runs whose label contains this, e.g. 'libero_goal/fed'")
        if name == "run":
            p.add_argument("--no-eval", action="store_true", help="train only (evaluate later with `eval`)")
            p.add_argument(
                "--keep-going", action="store_true", help="continue with the next run after a failure"
            )
        if name == "eval":
            p.add_argument("--device", help="cuda or cpu (default EVAL_DEVICE)")
            p.add_argument("--no-baseline", action="store_true")
    sub.add_parser("baseline").add_argument("--device")
    sub.add_parser("aggregate")
    p = sub.add_parser("smoke")
    p.add_argument("--only", help="only cases containing this, e.g. 'metaworld'")
    sub.add_parser("smoke-case").add_argument("name")
    sub.add_parser("smoke-resume")
    p = sub.add_parser("bench-workers")
    p.add_argument("--workers", nargs="*", type=int, default=[1, 2, 3, 4])
    sub.add_parser("bench-case").add_argument("name")
    p = sub.add_parser("bench-eval")
    p.add_argument("--devices", nargs="*", default=["cuda", "cpu"])
    p.add_argument("--episodes", type=int, default=1, help="episodes per task")

    args = ap.parse_args()
    s = Settings.from_env()
    handlers = {
        "prepare": cmd_prepare,
        "prefetch": cmd_prefetch,
        "plan": cmd_plan,
        "run": cmd_run,
        "eval": cmd_eval,
        "baseline": cmd_baseline,
        "status": cmd_status,
        "aggregate": cmd_aggregate,
        "smoke": cmd_smoke,
        "smoke-case": cmd_smoke_case,
        "smoke-resume": cmd_smoke_resume,
        "bench-workers": cmd_bench_workers,
        "bench-case": cmd_bench_case,
        "bench-eval": cmd_bench_eval,
    }
    try:
        return handlers[args.cmd](args, s)
    except KeyboardInterrupt:
        print("\naborted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
