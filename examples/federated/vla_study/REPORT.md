# Federated LoRA fine-tuning of SmolVLA: decisions, settings and open issues

Branch `fed-vla-finetune`. The study fine-tunes `lerobot/smolvla_base` with LoRA and runs from a
terminal; the README gives the procedure. pi0.5, full fine-tuning and the cluster job tooling were
removed on request.

## What was built

- **Tasks** (`src/lerobot/federated/tasks.py`): resolved by key, with Meta-World `push-v3`
  separated by the per-frame `task_id` (see "Meta-World push-v3" below).
- **Partitions** (`src/lerobot/federated/partition.py`):
  - task-disjoint: already shuffled by the partition seed, so unchanged;
  - IID: an equal share of every task at every client, within one episode, with leftovers
    rotating between clients;
  - shared-core: 5 seeded shared tasks split across all clients, plus 5 unique tasks, one per
    client.
  - An exact-cover check runs for every partition and seed at prepare time, and again inside every
    run. A manifest per run records each client's episodes and each task's share at each client.
- **LoRA definition** (`src/lerobot/federated/vla_study.py`): used verbatim by both arms, and its
  trainable set is pinned by tests.
- **Federated trainer** (`lerobot_fl_train.py`):
  - resume from `fl_state`: server weights and optimizer, round, LR schedule, every RNG, each
    client's exact data position, and kept optimizer states;
  - clean stop on Ctrl-C/SIGTERM;
  - per-client and per-round CSV logs, the LR at every checkpoint, the manifest, a
    trainable-parameter breakdown, a gradient audit, and a round-0 checkpoint for baselines;
  - a check that the pretrained weights really loaded.
- **Centralized trainer** (`lerobot_train.py`): checkpoint-and-exit on Ctrl-C/SIGTERM,
  optimizer state kept only in the newest checkpoint, the LR log and the weights check. Its resume
  was not bit-exact before and now is.
- **Study tooling** (this folder): prepare, run (train + evaluate, resumable), eval (GPU or CPU, in
  parallel with training), status, aggregate, smoke tests, throughput measurements.

## Decisions

1. **LoRA** uses rank 64 and alpha 128. Adapters go on every attention and MLP projection of the
   vision encoder, the language backbone and the action expert: q/k/v/o and gate/up/down, and in
   the vision encoder q/k/v/out and fc1/fc2.
   - In the backbone's last layer only k and v are adapted. A gradient audit showed that the q, o
     and MLP adapters there never get a gradient, because the action expert reads only that
     layer's keys and values.
   - The total is 42,361,680 trainable parameters, against the 0.04B reference.
2. **Trained at full rank alongside LoRA** (and federated): `state_proj`, `action_in_proj`,
   `action_out_proj`, `action_time_mlp_in`, `action_time_mlp_out` (1.63M). A rank-64 adapter on a
   32-wide projection would be larger than the layer.
3. **Learning rate** is SmolVLA's fine-tuning preset (1e-4), with no LoRA-specific scaling.
4. **LR schedule**: both arms keep the preset's shape (warmup = 1/30 of the horizon, cosine down to
   the floor), stretched to their own budget.
   - Federated: warmup 6 rounds, decay over 200 rounds.
   - Centralized: warmup 7,500 steps, decay over 250,000.
   - Every round-equivalent checkpoint of both arms therefore sits at the same point of the
     schedule.
   - Unstretched, the centralized arm would reach its floor LR at step 30,000 and spend 88% of its
     budget there. Peak and floor LR are the published values.
5. **Evaluation replans every 10 actions** (`n_action_steps=10`, as in DECENT-VLA); the action
   chunk is still 50. Earlier SmolVLA runs used 50.
6. **Client optimizer state** is reset every round (see below).

Other choices:
- **Checkpoint:** `lerobot/smolvla_base` at revision `5e8d12a6e2`, loaded whole (not the VLM-only
  initialization).
- **Precision:** the frozen base keeps SmolVLA's own bfloat16 weights. The LoRA adapters and the
  full-rank layers are float32 (PEFT upcasts the adapters), and compute runs in bf16 autocast.
- **Clients:** 5, FedAvg with `CLIENT_WEIGHTING=uniform`. Each client's episode and frame counts
  are logged and written to the manifest.
- **Evaluation protocol:** 10 episodes per task, seed 1000, one environment at a time, float32
  inference.
  - LIBERO replays its fixed initial states 0-9 of each task.
  - Meta-World seeds object positions from the episode seed.
  - So every arm meets the same starting states. Each `result.json` records the protocol, and
    aggregation warns if two protocols were mixed.
- **Untrained baselines:** the pretrained weights with each benchmark's own normalization
  statistics. `smolvla_base` ships its pretraining robot's statistics, which would make an
  untrained evaluation meaningless.
- **Checkpoint selection:** never by evaluation success. Round 200 is the headline result, and the
  curve is kept.

## Published hyperparameters used

| | SmolVLA |
|---|---|
| Optimizer | AdamW, betas (0.9, 0.95), eps 1e-8, weight decay 1e-10 |
| Gradient clip (norm) | 10 |
| Peak / floor LR | 1e-4 / 2.5e-6 |
| Schedule (preset) | 1,000 warmup, cosine over 30,000 |
| Batch size | 64 |
| Source | `src/lerobot/policies/smolvla/configuration_smolvla.py`; `docs/source/smolvla.mdx` |

## Trainable parameters

42,361,680 (0.042B) against DECENT-VLA's 0.04B, which matches. The paper does not name its target
modules; this count includes the vision encoder's adapters (10.6M), and without them it would be
0.03B. The breakdown:
- Language backbone: 16.4M, on 107 linear layers.
- Action expert: 13.7M (112 layers).
- Vision encoder: 10.6M (72 layers).
- Full-rank modules: 1.6M.

## Client optimizer state

Each client's AdamW state is **reset at the start of every round**. This is standard FedAvg and the
trainer's default, and it adds no memory, disk or resume state. With betas (0.9, 0.95) the restart
transient is a few tens of the 250 local steps. The centralized arm keeps its optimizer state
throughout, which is a small advantage to keep in mind when reading the federated-centralized gap.

Keeping the state is available with `CLIENT_OPT_STATE=keep`, and its two bugs are fixed: the copies
now live in CPU RAM rather than on the GPU, and loading one no longer rolls the LR back one round.
It costs 0.34 GB of CPU RAM per client, 1.7 GB for 5 clients.

## Meta-World push-v3

In `lerobot/metaworld_mt50`, `push-v3` and `push-back-v3` were recorded under the same instruction
("Push the puck to a goal") and share one `task_index`. Upstream later gave `push-back-v3` its own
description in `metaworld_config.json`. After that, the earlier study script (`noniid.sh`, now
removed from the branch) could no longer detect the clash, and silently selected 100 episodes for
`push-v3`.

This study resolves Meta-World tasks by the per-frame `task_id`:
- `push-v3` is id 38, episodes 2000-2049, 3,521 frames. Its puck starts at y 0.60-0.70 and is
  pushed out to y 0.80-0.90.
- `push-back-v3` (id 37, episodes 2100-2149) is excluded.

Any earlier result that used `push-v3` through `noniid.sh` was trained on both tasks.

## Throughput

**Earlier federated studies.** These ran on an A100 80GB at batch 64 (W&B project `fedvla`), with
SmolVLA training only its action expert. They are context for the LoRA numbers, not LoRA numbers.

| arm | LIBERO | Meta-World |
|---|---|---|
| centralized | 1.80-1.90 steps/s | 2.31-2.40 steps/s |
| federated (3 clients x 5 local steps) | 1.33-1.53 steps/s | 1.68-1.69 steps/s |

- A step took 0.51-0.53 s on LIBERO and 0.37-0.40 s on Meta-World.
- Data loading took about 0 s with 8-12 workers per client.
- GPU peak memory was 10.4 / 6.6 GB, at about 82% utilization.
- Federated runs lost about 20% to round switching at 5 local steps; at 250 local steps that
  overhead should all but vanish.

**This study's smoke tests**, on this laptop: RTX 5070 Ti 12 GB, WSL, batch 8, one worker per
client, final code. Sizes are real; speeds are not representative of an A100 at batch 64.

| | federated | centralized |
|---|---|---|
| checkpoint | 0.16 GB | 0.48 GB, the newest only (it carries the optimizer state) |
| resumable run state | 0.16 GB | (the newest checkpoint) |
| GPU peak at batch 8 | 9.4 GB (LIBERO), 5.7 GB (Meta-World) | |
| RAM, including data-loader workers | 8.8-9.2 GB | |
| smoke case, 2 rounds + one evaluation episode | 107-120 s | 81-83 s |
| trainable parameters / tensors without a gradient | 42,361,680 / 0 | |

- **Evaluation speed on this laptop:**
  - LIBERO-Object: 137-141 episodes/hour with the policy on the GPU, 67 with it on the CPU
    (20 threads).
  - Meta-World: 72-93 episodes/hour on the GPU.
  - Under WSL, rendering goes through the GPU either way, so a CPU-only machine renders more
    slowly. A run's evaluation is 1,000 episodes, about 7-15 h on a GPU at these rates.
- **Not measured here:**
  - steps/s at batch 64, and its dependence on data-loader workers. Batch 64 needs more than this
    laptop's 12 GB, so run `study.py bench-workers` on the training GPU.
  - evaluation speed on a headless Linux machine (`study.py bench-eval`).

**The planned grid at an estimated 1.2 steps/s** (`throughput.py` replaces it with the measured
rate):

| | per run | Phase 1 (24 runs) | Phase 2 (24 runs) |
|---|---|---|---|
| training | 58 h | 1,389 GPU-hours, 58 days on one GPU | 1,389 GPU-hours |

Time scales with rounds x local steps:

| rounds x local steps | 200 x 250 | 200 x 100 | 100 x 250 | 100 x 100 | 50 x 250 |
|---|---|---|---|---|---|
| Phase 1 days on one GPU | 58 | 23 | 29 | 12 | 14 |

## Open issues

1. **Compute is the main constraint.** One run is 250,000 optimizer steps. At my estimate of 1.2
   steps/s for LoRA SmolVLA on an A100, Phase 1 (24 runs) is about 1,400 GPU-hours, roughly 58 days
   on one GPU, and Phase 2 adds as much again. This is an estimate: the measured rate for SmolVLA
   with only its action expert trained was 1.9-2.4 steps/s. Run `study.py bench-workers` on the
   training GPU, then `throughput.py`, which gives the real figure and the options to reduce
   `ROUNDS` or `LOCAL_STEPS`.
2. **Deviations from published recipes**, approved and reported: the LR schedule is stretched to
   each arm's budget, and evaluation replans every 10 actions.
3. **LoRA learning rate.** LeRobot's PEFT guide suggests about 10x the full fine-tuning LR for
   LoRA (its SmolVLA example uses 1e-3 decaying to 1e-4). The preset's 1e-4 is kept, as no tuning
   was the instruction, but LoRA may be under-trained at it.
4. **Resetting the client optimizer state** gives the federated arm a small handicap against the
   centralized arm, which keeps its AdamW state. `CLIENT_OPT_STATE=keep` is available.
5. **Resume exactness.** Stopped-and-resumed runs are bit-identical to uninterrupted ones on CPU
   (tested, both arms). On a GPU, cuDNN's non-deterministic kernels make them equal up to numerical
   noise.
6. **Earlier `push-v3` results.** Any earlier result that used `push-v3` through `noniid.sh` trained
   on `push-back-v3` too.
7. **Weights & Biases.** If enabled (`WANDB_PROJECT`), each `run` session of a resumed run is a
   separate W&B run. The CSV logs in each run folder are the complete record.
8. **If full fine-tuning comes back**, train float32 weights. SmolVLA builds its VLM and action
   expert in bfloat16, and in bf16 an AdamW step of 1e-4 rounds to no change for 49% of
   `smolvla_base`'s weights (84% for a quarter-LR step, which is typical for Adam). Commit
   `38cee17b` on this branch had an opt-in `SmolVLAConfig.dtype="float32"` for this. LoRA is
   unaffected, because its adapters are float32 and the base is frozen.
9. **pi0.5 was removed** from this branch on request. The generic parts would carry over if it is
   added back: resume, partitions, the weights check and the evaluation tooling. My survey also found
   three pi0.5-specific traps:
   - its loader falls back silently to random weights if the checkpoint can't be read;
   - its published `dtype=bfloat16` loses most small AdamW updates at its learning rate;
   - MT50 has no quantile statistics, so its default quantile normalization fails on Meta-World.
