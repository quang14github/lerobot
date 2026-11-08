# Federated LoRA fine-tuning of SmolVLA

This study fine-tunes the pretrained SmolVLA (`lerobot/smolvla_base`) with LoRA (rank 64, alpha 128).
- **Arms:** federated (5 clients, FedAvg, equal client weights) against a centralized reference
  with the same total compute.
- **Benchmarks:** LIBERO-Object, LIBERO-Goal and LIBERO-Spatial (task ids 0-9), and Meta-World MT10.
- **Machine:** everything runs from a terminal on any machine with an NVIDIA GPU, and evaluation
  can also run on a CPU.

REPORT.md explains the settings.

| File | What it is |
|---|---|
| `grid.py` | the study grid: settings (environment variables), run names and order, trainer commands |
| `study.py` | the commands below |
| `eval_run.py` | evaluates one checkpoint |
| `collect_results.py` | builds `results.csv` from everything evaluated so far |
| `throughput.py` | turns benchmark and smoke-test outputs into a throughput / GPU-hours report |

Run every command from the repository root, in the environment where LeRobot is installed. It
needs SmolVLA, PEFT, LIBERO and Meta-World:
`pip install -e '.[smolvla,peft,libero,metaworld,training]'`.
`S=examples/federated/vla_study` shortens the commands below.

## Settings

All settings are environment variables; `grid.py` lists them with their defaults. The defaults
are the planned study:

| Variable | Default | |
|---|---|---|
| `STUDY_ROOT` | `outputs/fedvla` | where everything is written |
| `ROUNDS`, `LOCAL_STEPS`, `NUM_CLIENTS`, `BATCH_SIZE` | 200, 250, 5, 64 | the centralized arm takes ROUNDS x LOCAL_STEPS x NUM_CLIENTS steps |
| `SAVE_EVERY` | 20 | checkpoint (and evaluation) interval in rounds |
| `SEEDS`, `PARTITION_SEEDS` | `1000 2000 3000`, `1 2 3` | paired one to one |
| `BENCHMARKS` | all four | `libero_object libero_goal libero_spatial metaworld_mt10` |
| `RUN_FEDERATED`, `RUN_CENTRALIZED` | 1, 1 | which arms |
| `NUM_WORKERS` | 3 | data-loader workers per client (5 clients: 15 processes) |
| `CEN_NUM_WORKERS` | 12 | data-loader workers of the centralized arm |
| `CLIENT_WEIGHTING`, `CLIENT_OPT_STATE` | `uniform`, `reset` | |
| `DEVICE`, `EVAL_DEVICE` | `cuda`, `cuda` | `EVAL_DEVICE=cpu` evaluates on the CPU |
| `EVAL_EPISODES`, `EVAL_SEED` | 10, 1000 | per task, for every evaluation |
| `TAG`, `WANDB_PROJECT` | empty | a note in study names; stream to Weights & Biases |

## 1. Prepare

```bash
python $S/study.py prepare
python $S/study.py prefetch     # optional: download the checkpoint and the episodes up front
```

`prepare` maps every task key to its episodes. Meta-World's `push-v3` is separated from
`push-back-v3` by the per-frame `task_id`. It also checks that every partition (task-disjoint,
IID and shared-core, for every partition seed and benchmark) assigns every episode exactly once.
It writes `$STUDY_ROOT/prep/<benchmark>/tasks.tsv` and `partition_checks.json`.

## 2. Smoke tests (minutes each)

```bash
python $S/study.py smoke                       # all cases
python $S/study.py smoke --only metaworld      # a subset
```

The suite runs these cases:

- **Main cases:** federated and centralized, on LIBERO-Object and on one Meta-World task
  (`reach-v3`). Each case trains 2 rounds of 3 local steps (or 30 centralized steps), then
  evaluates the final checkpoint on one episode.
- **`smoke-resume`:** interrupts a federated and a centralized run with Ctrl-C (SIGINT) partway,
  restarts them, and checks that they continue from the saved round and finish. A run refuses to
  resume if a setting that defines it changed.

Summaries are in `$STUDY_ROOT/smoke/*/summaries/`. They hold train and eval status, timings,
checkpoint and state sizes, the trainable-parameter count, and a gradient audit (trainable
parameters that receive no gradient; there should be none).

## 3. Measure throughput (optional, before the long runs)

```bash
python $S/study.py bench-workers               # steps/s for 1-4 workers per client, batch 64
python $S/study.py bench-eval                  # evaluation episodes/hour on cuda and on cpu
python $S/throughput.py > throughput.md
```

`throughput.md` reports:
- steps/s, GPU peak memory and RAM for each worker count;
- disk per checkpoint;
- evaluation episodes/hour on GPU and CPU;
- the planned grid in GPU-hours, with options to reduce `ROUNDS` or `LOCAL_STEPS`.

Use it to set `NUM_WORKERS`, and to decide whether evaluation should run on the CPU.

## 4. Phase 1: task-disjoint partition

```bash
python $S/study.py run
```

- **Order:** `run` first evaluates the untrained model on each benchmark. It then trains the runs
  in this order: for each seed (1000 with partition seed 1, then 2000/2, then 3000/3), and for each
  benchmark, the federated run, then its centralized reference. That way complete comparisons
  finish early.
- **Evaluation:** each run is evaluated once its training completes: rounds 20, 40, ..., 200 on
  all 10 tasks, 10 episodes each, with the same initial states for every arm.
- **Previewing the order:** `python $S/study.py plan` lists the runs without running them.

**Stopping and continuing.**
- **Stopping:** press Ctrl-C. The current run finishes its training step, saves, and the command
  exits. A second Ctrl-C aborts without saving.
- **Continuing:** rerun the same command. Finished runs are skipped, and the interrupted one
  continues exactly where it stopped: weights, round, LR schedule, random state, each client's
  position in its data.
- **One part only:** `--only 'libero_goal/fed'` runs only the matching runs.

**Evaluating in parallel.**
- To keep the GPU on training, run `python $S/study.py run --no-eval` in one terminal and
  `EVAL_DEVICE=cpu python $S/study.py eval` in another.
- `eval` evaluates every saved checkpoint that is not evaluated yet, and can be rerun at any time.
- Locks stop two processes from evaluating the same checkpoint.

**Progress.**

```bash
python $S/study.py status
```

Each run folder `$STUDY_ROOT/studies/<study>/<run>/` contains:
- `run.json`: the run's identity.
- `sessions.tsv`: one row per `run` session, with rounds before and after and rounds/hour.
- `logs/`: training and eval logs, and `gpu_*.csv`, nvidia-smi utilization and memory every 30 s.
- `train/`:
  - `rounds.csv`: steps/s, GPU peak and RAM per round;
  - `client_losses.csv`: each client's training loss per round;
  - `checkpoints.csv`: the LR at every checkpoint;
  - `manifest.json`: each client's episodes, and each task's share at each client;
  - `trainable_params.json`, plus the checkpoints.
- `eval/round_XXXX/result.json`: per-task success rates.

Each study folder also holds `study.env` (every setting and the git commit).

**Results.** Aggregation tolerates missing and failed runs, and can be rerun at any time:

```bash
python $S/study.py aggregate
```

It writes `$STUDY_ROOT/results.csv` with these columns: benchmark, policy, mode, partition type,
seed, partition seed, arm, round, task, clients holding the task, task share at each client,
success rate, episodes. It also writes `client_losses.csv`, `checkpoint_lrs.csv` and
`missing.txt` (runs still training or not yet evaluated). Round 200 is the headline result; the
whole curve is kept.

## 5. Phase 2: IID and shared-core partitions

```bash
python $S/study.py run --phase 2
```

Phase 2 is federated only: the centralized runs of Phase 1 are the reference for every partition.
- **IID:** each client gets a random, equal share (within one episode) of every task's episodes.
- **Shared-core:** the partition seed picks 5 tasks that are split equally across all clients;
  each of the other 5 tasks goes whole to one client.

## Disk

Each checkpoint holds the adapters and the full-rank layers, about 0.16 GB. A centralized
checkpoint also carries its optimizer state, about 0.48 GB, but only the newest one keeps it. All
10 checkpoints of a run are kept, plus the federated run state (~0.16 GB). A run therefore needs
about 2 GB, and the 48 runs of both phases need about 100 GB.
