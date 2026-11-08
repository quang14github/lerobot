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
"""Train a policy with federated averaging.

Requires: pip install 'lerobot[training]'

This is a *simulated* federation: one process holds the global model and plays every client in
turn, so the algorithm can be developed and ablated without a real deployment. What makes it a
federation rather than sharded SGD is that clients never share data — each one only ever sees
its own episode shard — and the server only ever sees weight deltas.

One round:

  1. the server broadcasts the global trainable weights to the selected clients;
  2. each client runs `--fl.local_steps` optimizer steps on its private shard;
  3. each client uploads `local_weights - global_weights`;
  4. the server applies the sample-weighted mean of those deltas via `--server.type`.

```bash
lerobot-fl-train \
    --dataset.repo_id=lerobot/metaworld_mt50 \
    --policy.type=smolvla --policy.load_vlm_weights=true \
    --policy.push_to_hub=false \
    --env.type=metaworld --env.task=medium \
    --partition.strategy=task --partition.num_clients=10 \
    --fl.rounds=200 --fl.local_steps=50 --fl.clients_per_round=3 \
    --server.type=fedadam
```
"""

import copy
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm

from lerobot.common.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    publish_trained_model,
    push_checkpoint_to_hub,
    save_checkpoint,
    update_last_checkpoint,
    verify_pretrained_weights,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import compute_sampler_state
from lerobot.datasets.utils import resolve_episode_indices
from lerobot.distributed import ParallelDims, make_accelerator
from lerobot.envs import make_env_pre_post_processors
from lerobot.federated.aggregate import (
    DeltaAccumulator,
    ProximalTerm,
    compute_delta,
    delta_norm,
    get_trainable_state,
    make_server_optimizer,
    reset_optimizer_state,
    set_trainable_state,
)
from lerobot.federated.config import FederatedConfig, FederatedTrainPipelineConfig
from lerobot.federated.partition import (
    ClientShard,
    episode_table,
    format_partition,
    load_task_map,
    partition_episodes,
    partition_manifest,
)
from lerobot.federated.run_state import (
    EXIT_INTERRUPTED,
    CsvLog,
    RoundSnapshot,
    StopRequest,
    TrainingInterrupted,
    capture_rng_state,
    compare_settings,
    config_fingerprint,
    load_run_state,
    optimizer_state_to_cpu,
    process_memory_gb,
    restore_rng_state,
    save_run_state,
    state_exists,
)
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.factory import ProcessorConfigKwargs
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.import_utils import require_package
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import cycle, format_big_number, init_logging, inside_slurm

from .lerobot_eval import eval_policy_all
from .lerobot_train import _make_eval_envs, _preprocess_dataset_batch, make_dataloaders, update_policy


class ClientRuntime:
    """A client's private dataset and its position in it.

    The dataloader and its iterator are created once and reused across every round the client
    participates in. Rebuilding them per round would restart the shuffled epoch from the top,
    so the client would train on the same first `local_steps * batch_size` samples forever.
    """

    def __init__(self, shard: ClientShard, dataset, dataloader, seed: int, local_steps_done: int = 0):
        self.shard = shard
        self.dataset = dataset
        self.dataloader = dataloader
        # Worker seeds are drawn when an iterator is created. A private generator keeps that draw
        # off the global RNG, so creating iterators at a different moment (as a resumed run does)
        # cannot shift the policy's noise sampling.
        self.dataloader.generator = torch.Generator().manual_seed(seed)
        self.rounds_participated = 0
        self.local_steps_done = local_steps_done
        if local_steps_done > 0:
            # The data order is a pure function of (seed, epoch), so the position is recomputed from
            # the number of batches consumed - prefetched batches never count.
            sampler = dataloader.sampler
            sampler.load_state_dict(
                compute_sampler_state(local_steps_done, len(sampler), dataloader.batch_size, 1)
            )
        self.iterator = cycle(dataloader)

    def next_batch(self):
        return next(self.iterator)

    def position(self) -> dict[str, int]:
        return {"rounds_participated": self.rounds_participated, "local_steps_done": self.local_steps_done}


def _client_config(cfg: FederatedTrainPipelineConfig, shard: ClientShard) -> FederatedTrainPipelineConfig:
    """A shallow view of `cfg` scoped to one client's shard.

    Only `dataset` and `seed` differ, so every other decision (batch size, workers, transforms)
    stays identical across clients. Per-client seeds decorrelate the shuffling: with one shared
    seed, clients holding equal-length shards would march through their data in lockstep.
    """
    client_cfg = copy.copy(cfg)
    client_cfg.dataset = copy.copy(cfg.dataset)
    client_cfg.dataset.episodes = shard.episodes
    # Exclusions were already applied when the partition was computed.
    client_cfg.dataset.exclude_episodes = None
    client_cfg.seed = (cfg.seed or 0) + 1 + shard.client_id
    return client_cfg


def _client_weight(shard: ClientShard, fl: FederatedConfig) -> float:
    if fl.client_weighting == "custom":
        # Validated to hold one positive weight per client, indexed by client id.
        return float(fl.client_weights[shard.client_id])
    if fl.client_weighting == "frames":
        return float(shard.num_frames)
    if fl.client_weighting == "episodes":
        return float(shard.num_episodes)
    return 1.0


def format_client_weights(shards: list[ClientShard], fl: FederatedConfig) -> str:
    """Each client's tasks and its share of the aggregate when every client takes part.

    Printed before training so a custom weighting can be checked against the partition: client ids
    are assigned by the partition (shuffled by its seed), so this is where "client 0" gets a task.
    """
    weights = [_client_weight(shard, fl) for shard in shards]
    total = sum(weights)
    lines = [f"{'client':>6}  {'weight':>10}  {'share':>6}  tasks"]
    for shard, weight in zip(shards, weights, strict=True):
        lines.append(f"{shard.client_id:>6}  {weight:>10g}  {weight / total:>6.1%}  {', '.join(shard.tasks)}")
    return "\n".join(lines)


def _client_label(shard: ClientShard, max_len: int = 22) -> str:
    """Short, stable name for a client's data: its task when it holds exactly one, else a count.

    Task strings in a LeRobotDataset are free-text descriptions, so they are truncated to keep
    the per-round log line readable.
    """
    if len(shard.tasks) == 1:
        task = shard.tasks[0]
        return task if len(task) <= max_len else task[: max_len - 1] + "\u2026"
    return f"{len(shard.tasks)} tasks"


def _select_clients(shards: list[ClientShard], clients_per_round: int, rng: np.random.Generator) -> list[int]:
    """Sample the round's participants without replacement."""
    if clients_per_round in (0, len(shards)):
        return list(range(len(shards)))
    return sorted(int(i) for i in rng.choice(len(shards), size=clients_per_round, replace=False))


def _resolve_global_stats(meta: LeRobotDatasetMetadata, cfg: FederatedTrainPipelineConfig) -> dict:
    """The single set of normalization statistics every client must share.

    This is the quiet correctness requirement of federated VLA training. Normalization stats
    are baked into each client's `NormalizerProcessorStep`; if client A normalizes state with a
    different mean than client B, their weights are fitted to different input spaces and
    averaging them is not meaningful. Deriving them once, on the server, from dataset-level
    metadata makes that impossible to get wrong.

    In a real deployment where the server cannot see the data, replace this with a round-zero
    statistics exchange: have each client report `meta.stats` for its own shard and merge them
    with `lerobot.datasets.compute_stats.aggregate_stats`, which combines counts, means and
    variances without any client revealing a sample.
    """
    return rename_stats(meta.stats, cfg.rename_map)


CLIENT_LOSS_FIELDS = [
    "round",
    "client_id",
    "loss",
    "lr",
    "local_steps",
    "episodes",
    "frames",
    "seconds",
    "steps_per_s",
]
ROUND_FIELDS = [
    "round", "mean_loss", "loss_std", "update_norm", "lr", "round_s", "steps_per_s", "samples_per_s",
    "gpu_mem_peak_gb", "rss_gb", "rss_with_workers_gb",
]  # fmt: skip
CHECKPOINT_FIELDS = ["round", "step_equivalent", "lr", "next_lr", "checkpoint"]


def _write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    tmp.replace(path)


def _trainable_summary(model) -> dict:
    """Trainable/total counts, broken down by the first components of each parameter name."""
    groups: dict[str, int] = {}
    trainable = total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if not param.requires_grad:
            continue
        trainable += param.numel()
        stem = name.removeprefix("base_model.model.")
        key = ".".join(part for part in stem.split(".")[:6] if not part.isdigit())
        groups[key] = groups.get(key, 0) + param.numel()
    return {"trainable": trainable, "total": total, "by_module": dict(sorted(groups.items()))}


def _audit_gradients(policy, unwrapped, runtime, cfg, accelerator, preprocessor, optimizer) -> dict:
    """One forward/backward on a few samples; list trainable parameters left without a gradient.

    LoRA's B matrices start at zero, so its A matrices legitimately get zero (not missing) gradients
    on this first step; only a missing gradient means a parameter is not on the loss's graph.
    """
    n = min(4, len(runtime.dataset))
    batch = runtime.dataloader.collate_fn([runtime.dataset[i] for i in range(n)])
    batch = _preprocess_dataset_batch(batch, runtime.dataset.meta.camera_keys, cfg.rename_map, preprocessor)
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    with accelerator.autocast():
        loss, _ = policy(batch)
    accelerator.backward(loss)
    named = [(name, p) for name, p in unwrapped.named_parameters() if p.requires_grad]
    missing = [name for name, p in named if p.grad is None]
    result = {
        "trainable_tensors": len(named),
        "trainable_params": sum(p.numel() for _, p in named),
        "no_grad_tensors": len(missing),
        "no_grad_params": sum(p.numel() for name, p in named if p.grad is None),
        "no_grad_names": missing,
    }
    optimizer.zero_grad(set_to_none=True)
    return result


@parser.wrap()
def fl_train(cfg: FederatedTrainPipelineConfig):
    """Run federated training end to end, or continue it from `output_dir/fl_state`.

    Args:
        cfg: The federated training config, parsed from the CLI by `parser.wrap()`.

    Raises:
        TrainingInterrupted: After saving the run state on Ctrl-C/SIGTERM (or
            `fl.stop_after_rounds`); `main` turns it into exit code 75.
    """
    require_package("accelerate", extra="training")
    cfg.validate()

    accelerator = make_accelerator(cfg)
    parallel_dims = ParallelDims.from_config(
        cfg.parallelism, accelerator.num_processes, accelerator.device.type
    )
    init_logging(accelerator=accelerator)

    if accelerator.num_processes > 1 or parallel_dims.is_sharded:
        raise NotImplementedError(
            "Federated training runs the client loop in a single process; launch without torchrun "
            "and with the default parallelism. Multi-GPU support would parallelize *clients*, not "
            "shard one model, and is not implemented yet."
        )
    if cfg.is_reward_model_training:
        raise NotImplementedError("Federated training currently supports policies only.")

    output_dir = Path(cfg.output_dir)
    fingerprint, settings = config_fingerprint(cfg.to_dict())
    loaded = None
    if cfg.fl.auto_resume and state_exists(output_dir):
        loaded = load_run_state(output_dir)
        changed = compare_settings(loaded.settings, settings)
        if changed and not cfg.fl.allow_config_change:
            raise ValueError(
                "Refusing to resume: settings that define the run changed since its state was saved:\n  "
                + "\n  ".join(changed)
                + "\nPass --fl.allow_config_change=true to resume anyway."
            )
        for line in changed:
            logging.warning("Resuming with a changed setting: %s", line)
        logging.info(
            colored(f"Resuming after round {loaded.round} of {cfg.fl.rounds}", "yellow", attrs=["bold"])
            + f" from {output_dir / 'fl_state'} (settings fingerprint {fingerprint})"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(pformat(cfg.to_dict()))
    wandb_logger = WandBLogger(cfg) if (cfg.wandb.enable and cfg.wandb.project) else None
    if wandb_logger is None:
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)
    device = accelerator.device
    torch.backends.cudnn.benchmark = not cfg.cudnn_deterministic
    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
    torch.backends.cuda.matmul.allow_tf32 = True

    # --- partition (server side, from metadata only — no frames are read here) ------------------
    logging.info("Reading dataset metadata")
    meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
        repo_type=cfg.dataset.repo_type,
    )
    task_map = load_task_map(cfg.partition.task_map) if cfg.partition.task_map else None
    base_episodes = resolve_episode_indices(
        cfg.dataset.episodes, meta.total_episodes, cfg.dataset.exclude_episodes
    )
    shards = partition_episodes(meta, cfg.partition, episodes=base_episodes, task_map=task_map)
    _, task_of, frames_of = episode_table(meta, base_episodes, task_map)
    logging.info(
        "Partitioned %d episodes across %d clients (strategy=%s):\n%s",
        sum(shard.num_episodes for shard in shards),
        len(shards),
        cfg.partition.strategy,
        format_partition(shards),
    )
    logging.info(
        "Aggregation weights (client_weighting=%s); a round with fewer clients renormalizes over them:\n%s",
        cfg.fl.client_weighting,
        format_client_weights(shards, cfg.fl),
    )
    manifest = partition_manifest(shards, task_of, frames_of, cfg.partition)
    weights = [_client_weight(shard, cfg.fl) for shard in shards]
    manifest["client_weighting"] = cfg.fl.client_weighting
    manifest["client_weights"] = {
        str(s.client_id): w / sum(weights) for s, w in zip(shards, weights, strict=True)
    }
    manifest_path = output_dir / "manifest.json"
    if loaded is not None and manifest_path.exists():
        saved_clients = [c["episodes"] for c in json.loads(manifest_path.read_text())["clients"]]
        if saved_clients != [c["episodes"] for c in manifest["clients"]]:
            raise ValueError(f"The partition differs from the one in {manifest_path}; refusing to resume.")
    _write_json(manifest_path, manifest)

    # Downloads are episode-selective, so each client would otherwise trigger its own
    # `snapshot_download` the first time it is sampled — several stalls scattered through
    # training, and on a metered or rate-limited link, several chances to fail mid-run. Fetch
    # the union of every client's episodes once, up front, while nothing else is waiting.
    all_episodes = sorted({ep for shard in shards for ep in shard.episodes})
    logging.info("Warming the dataset cache for %d episodes across all clients", len(all_episodes))
    warm_cfg = copy.copy(cfg)
    warm_cfg.dataset = copy.copy(cfg.dataset)
    warm_cfg.dataset.episodes = all_episodes
    warm_cfg.dataset.exclude_episodes = None
    make_dataset(warm_cfg)

    # --- global model, built once from the full metadata ---------------------------------------
    logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=meta, rename_map=cfg.rename_map)
    if cfg.policy.pretrained_path is not None:
        verify_pretrained_weights(
            policy, cfg.policy.pretrained_path, revision=getattr(cfg.policy, "pretrained_revision", None)
        )
    if cfg.peft is not None:
        require_package("peft", extra="peft")
        logging.info("Wrapping policy with PEFT adapters")
        policy = policy.wrap_with_peft(peft_cli_overrides=dataclasses.asdict(cfg.peft))

    dataset_stats = _resolve_global_stats(meta, cfg)
    processor_kwargs = ProcessorConfigKwargs(dataset_stats=dataset_stats)
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
                "stats": dataset_stats,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        processor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
                "stats": dataset_stats,
            },
        }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        pretrained_revision=getattr(cfg.policy, "pretrained_revision", None),
        **processor_kwargs,
    )

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    policy, optimizer, lr_scheduler = accelerator.prepare(policy, optimizer, lr_scheduler)
    # Dataloaders are deliberately not `prepare`d: this is a single process, and batches reach
    # the device through the preprocessor's device step.
    unwrapped = accelerator.unwrap_model(policy)

    trainable_names = [name for name, param in unwrapped.named_parameters() if param.requires_grad]
    if loaded is not None:
        if sorted(loaded.global_state) != sorted(trainable_names):
            raise ValueError("The saved server state does not hold this policy's trainable parameters.")
        global_state = loaded.global_state
        set_trainable_state(unwrapped, global_state)
    else:
        global_state = get_trainable_state(unwrapped)
    server_optimizer = make_server_optimizer(cfg.server)
    proximal = None
    if cfg.fl.prox_mu > 0:
        proximal = ProximalTerm(unwrapped, cfg.fl.prox_mu)
        proximal.attach(optimizer)

    # Per-client optimizer state, kept only when the user opts out of the FedAvg default of
    # starting every round fresh. Kept states live in CPU memory; only the active client's is on
    # the GPU.
    client_optimizer_states: dict[int, dict] = {}

    summary = _trainable_summary(unwrapped)
    _write_json(output_dir / "trainable_params.json", summary)
    num_trainable = summary["trainable"]
    payload_mb = sum(value.numel() * 4 for value in global_state.values()) / 1024**2
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(
        f"Exchanged (trainable) params: {format_big_number(num_trainable)} of {format_big_number(summary['total'])}"
    )
    logging.info(
        f"Per-client upload: {payload_mb:.1f} MB/round -> "
        f"{payload_mb * cfg.fl.rounds * (cfg.fl.clients_per_round or len(shards)) / 1024:.2f} GB total"
    )
    if payload_mb > 512:
        logging.warning(
            "Each client uploads %.0f MB per round. Freeze more of the backbone "
            "(--policy.train_expert_only=true) or train adapters instead "
            "(--peft.method_type=lora) to make this practical.",
            payload_mb,
        )

    env_preprocessor = env_postprocessor = None
    if cfg.env is not None:
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=cfg.env, policy_cfg=cfg.policy
        )

    # --- metrics ------------------------------------------------------------------------------
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f", reduction="mean"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "dataloading_s": AverageMeter("data_s", ":.3f", reduction="max"),
        "preprocessing_s": AverageMeter("prep_s", ":.3f", reduction="max"),
        "update_s": AverageMeter("updt_s", ":.3f", reduction="max"),
        "step_s": AverageMeter("step_s", ":.3f", reduction="max"),
        "samples_per_s": AverageMeter("smp/s", ":.0f"),
    }
    if torch.cuda.is_available():
        train_metrics["gpu_mem_gb"] = AverageMeter("mem_gb", ":.2f", reduction="max")
    tracker = MetricsTracker(
        cfg.batch_size,
        meta.total_frames,
        meta.total_episodes,
        train_metrics,
        initial_step=0,
        dp_world_size=parallel_dims.dp_world_size,
    )

    clients: dict[int, ClientRuntime] = {}

    def client_runtime(client_id: int, local_steps_done: int = 0) -> ClientRuntime:
        """Materialize a client's private dataset the first time it is needed."""
        if client_id not in clients:
            shard = shards[client_id]
            logging.info(
                "Creating client %d dataset: %d episodes, %d tasks, %s frames",
                client_id,
                shard.num_episodes,
                len(shard.tasks),
                format_big_number(shard.num_frames),
            )
            client_cfg = _client_config(cfg, shard)
            client_dataset = make_dataset(client_cfg)
            client_dataloader, _ = make_dataloaders(client_cfg, client_dataset, None, 0, parallel_dims)
            clients[client_id] = ClientRuntime(
                shard,
                client_dataset,
                client_dataloader,
                seed=client_cfg.seed,
                local_steps_done=local_steps_done,
            )
        return clients[client_id]

    selection_rng = np.random.default_rng(cfg.seed if cfg.seed is not None else 0)
    client_log = CsvLog(output_dir / "client_losses.csv", CLIENT_LOSS_FIELDS)
    round_log = CsvLog(output_dir / "rounds.csv", ROUND_FIELDS)
    checkpoint_log = CsvLog(output_dir / "checkpoints.csv", CHECKPOINT_FIELDS)
    steps_per_round = cfg.fl.local_steps * (cfg.fl.clients_per_round or len(shards))

    if loaded is not None:
        server_optimizer.load_state_dict(loaded.server_optimizer_state)
        if lr_scheduler is not None and loaded.scheduler_state is not None:
            lr_scheduler.load_state_dict(loaded.scheduler_state)
            # A scheduler's state does not include the LR it last wrote into the optimizer.
            for group, lr in zip(optimizer.param_groups, lr_scheduler.get_last_lr(), strict=True):
                group["lr"] = lr
        selection_rng.bit_generator.state = loaded.selection_rng
        for client_id, position in sorted(loaded.clients.items()):
            runtime = client_runtime(client_id, local_steps_done=position["local_steps_done"])
            runtime.rounds_participated = position["rounds_participated"]
        client_optimizer_states = dict(loaded.client_optimizers)
        for log in (client_log, round_log, checkpoint_log):
            log.truncate_after(loaded.round)
        # Last, after everything that draws from the generators during setup.
        restore_rng_state(loaded.rng)
        start_round = loaded.round + 1
    else:
        start_round = 1
        if cfg.fl.audit_gradients:
            audit = _audit_gradients(
                policy, unwrapped, client_runtime(0), cfg, accelerator, preprocessor, optimizer
            )
            _write_json(output_dir / "grad_audit.json", audit)
            logging.info(
                "Gradient audit: %d of %d trainable tensors (%s params) got no gradient",
                audit["no_grad_tensors"],
                audit["trainable_tensors"],
                format_big_number(audit["no_grad_params"]),
            )
        if cfg.fl.save_initial_checkpoint and cfg.save_checkpoint:
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.fl.rounds, 0)
            logging.info(f"Saving the untrained model (round 0) to {checkpoint_dir}")
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=0,
                cfg=cfg,
                policy=policy,
                optimizer=None,
                scheduler=lr_scheduler,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                accelerator=accelerator,
            )
            update_last_checkpoint(checkpoint_dir)
            lr_now = optimizer.param_groups[0]["lr"]
            checkpoint_log.append(
                [
                    {
                        "round": 0,
                        "step_equivalent": 0,
                        "lr": lr_now,
                        "next_lr": lr_now,
                        "checkpoint": checkpoint_dir.name,
                    }
                ]
            )

    def snapshot(round_: int) -> RoundSnapshot:
        return RoundSnapshot(
            round=round_,
            rng=capture_rng_state(),
            selection_rng=copy.deepcopy(selection_rng.bit_generator.state),
            clients={cid: runtime.position() for cid, runtime in clients.items()},
            client_optimizers=dict(client_optimizer_states),
        )

    def save_state(snap: RoundSnapshot) -> None:
        save_run_state(
            output_dir,
            global_state=global_state,
            server_optimizer_state=server_optimizer.state_dict(),
            scheduler_state=lr_scheduler.state_dict() if lr_scheduler is not None else None,
            snapshot=snap,
            fingerprint=fingerprint,
            settings=settings,
            extra={"rounds_total": cfg.fl.rounds},
        )

    # Everything the next round depends on, as of the last completed round. A round interrupted
    # mid-way is dropped and replayed on resume.
    last = snapshot(start_round - 1)
    last_saved_round = loaded.round if loaded is not None else -1
    stop = StopRequest()
    rounds_this_run = 0
    progbar = tqdm(
        total=cfg.fl.rounds,
        initial=start_round - 1,
        desc="Federated rounds",
        unit="round",
        disable=inside_slurm(),
    )
    logging.info(
        f"Start federated training: rounds {start_round}..{cfg.fl.rounds} x {cfg.fl.local_steps} local steps "
        f"x {cfg.fl.clients_per_round or len(shards)} clients"
    )

    try:
        for rnd in range(start_round, cfg.fl.rounds + 1):
            round_start = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            selected = _select_clients(shards, cfg.fl.clients_per_round, selection_rng)
            round_lrs = [group["lr"] for group in optimizer.param_groups]

            accumulator = DeltaAccumulator()
            client_losses = {}
            client_rows = []
            for client_id in selected:
                if stop.requested:
                    raise TrainingInterrupted
                shard = shards[client_id]
                client_start = time.perf_counter()

                # 1. Broadcast: every client starts the round from the same global weights.
                set_trainable_state(unwrapped, global_state)
                if cfg.fl.reset_optimizer_each_round or client_id not in client_optimizer_states:
                    reset_optimizer_state(optimizer)
                else:
                    optimizer.load_state_dict(client_optimizer_states[client_id])
                    # A saved state carries the LR of the round it was saved in; this round's
                    # schedule position wins.
                    for group, lr in zip(optimizer.param_groups, round_lrs, strict=True):
                        group["lr"] = lr
                if proximal is not None:
                    proximal.set_anchor(global_state)

                runtime = client_runtime(client_id)

                # 2. Local training on private data.
                policy.train()
                local_loss = 0.0
                for _ in range(cfg.fl.local_steps):
                    if stop.requested:
                        raise TrainingInterrupted
                    step_start = time.perf_counter()
                    batch = runtime.next_batch()
                    preprocessing_start = time.perf_counter()
                    tracker.dataloading_s = preprocessing_start - step_start
                    batch = _preprocess_dataset_batch(
                        batch, runtime.dataset.meta.camera_keys, cfg.rename_map, preprocessor
                    )
                    tracker.preprocessing_s = time.perf_counter() - preprocessing_start
                    # The LR schedule is driven per *round* by the server below, so no scheduler here.
                    tracker, _ = update_policy(
                        tracker,
                        policy,
                        batch,
                        optimizer,
                        cfg.optimizer.grad_clip_norm,
                        accelerator=accelerator,
                        lr_scheduler=None,
                    )
                    tracker.step_s = time.perf_counter() - step_start
                    tracker.step()
                    local_loss += tracker.loss.val
                    runtime.local_steps_done += 1
                runtime.rounds_participated += 1

                # 3. Upload: only the weight change leaves the client, and it is folded into the
                # running aggregate immediately rather than kept until the end of the round.
                accumulator.add(compute_delta(unwrapped, global_state), _client_weight(shard, cfg.fl))
                client_losses[client_id] = local_loss / cfg.fl.local_steps
                if not cfg.fl.reset_optimizer_each_round:
                    client_optimizer_states[client_id] = optimizer_state_to_cpu(optimizer.state_dict())
                client_s = time.perf_counter() - client_start
                client_rows.append(
                    {
                        "round": rnd,
                        "client_id": client_id,
                        "loss": client_losses[client_id],
                        "lr": round_lrs[0],
                        "local_steps": cfg.fl.local_steps,
                        "episodes": shard.num_episodes,
                        "frames": shard.num_frames,
                        "seconds": round(client_s, 3),
                        "steps_per_s": round(cfg.fl.local_steps / client_s, 4),
                    }
                )

            # 4. Aggregate and take the server step.
            aggregate = accumulator.result()
            update_norm = delta_norm(aggregate)
            server_optimizer.step(global_state, aggregate)
            set_trainable_state(unwrapped, global_state)
            del accumulator, aggregate
            if lr_scheduler is not None:
                lr_scheduler.step()

            round_s = time.perf_counter() - round_start
            mean_loss = float(np.mean(list(client_losses.values())))
            spread = float(np.std(list(client_losses.values())))
            rss, rss_total = process_memory_gb()
            gpu_peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
            steps_per_s = cfg.fl.local_steps * len(selected) / round_s
            client_log.append(client_rows)
            round_log.append(
                [
                    {
                        "round": rnd,
                        "mean_loss": mean_loss,
                        "loss_std": spread,
                        "update_norm": update_norm,
                        "lr": round_lrs[0],
                        "round_s": round(round_s, 3),
                        "steps_per_s": round(steps_per_s, 4),
                        "samples_per_s": round(steps_per_s * cfg.batch_size, 2),
                        "gpu_mem_peak_gb": round(gpu_peak, 3),
                        "rss_gb": round(rss, 3),
                        "rss_with_workers_gb": round(rss_total, 3),
                    }
                ]
            )
            if cfg.fl.log_freq_rounds > 0 and rnd % cfg.fl.log_freq_rounds == 0:
                if tracker.step_s.avg > 0:
                    tracker.samples_per_s = cfg.batch_size / tracker.step_s.avg
                # Per-client losses, not just their summary: under a task partition the mean can
                # look healthy while one client is diverging on its own task, and that is exactly
                # the failure federated training needs to surface. The task each client holds is
                # named too, so a struggling client is identifiable without cross-referencing the
                # partition table printed at startup.
                per_client = "  ".join(
                    f"c{cid}={client_losses[cid]:.4f}[{_client_label(shards[cid])}]" for cid in selected
                )
                logging.info(
                    "round %d/%d  loss=%.4f (+/-%.4f)  |update|=%.4e  lr=%.2e  %.1fs  %.2f steps/s  "
                    "gpu_peak=%.1fGB  rss=%.1fGB\n    clients: %s",
                    rnd,
                    cfg.fl.rounds,
                    mean_loss,
                    spread,
                    update_norm,
                    round_lrs[0],
                    round_s,
                    steps_per_s,
                    gpu_peak,
                    rss_total,
                    per_client,
                )
                if wandb_logger:
                    log_dict = tracker.to_dict()
                    log_dict.update(
                        {
                            "fl/round": rnd,
                            "fl/clients_sampled": len(selected),
                            "fl/client_loss_mean": mean_loss,
                            # The spread across clients is the drift signal: a value that grows
                            # round over round means local models are diverging faster than the
                            # server can average them back together.
                            "fl/client_loss_std": spread,
                            "fl/update_norm": update_norm,
                            "fl/round_s": round_s,
                            "fl/steps_per_s": steps_per_s,
                        }
                    )
                    for client_id, loss in client_losses.items():
                        log_dict[f"fl/client_{client_id}_loss"] = loss
                    wandb_logger.log_dict(log_dict, rnd)
                tracker.reset_averages()
            progbar.update(1)

            is_last_round = rnd == cfg.fl.rounds
            if cfg.save_checkpoint and (
                is_last_round or (cfg.fl.save_freq_rounds > 0 and rnd % cfg.fl.save_freq_rounds == 0)
            ):
                logging.info(f"Checkpointing the global model after round {rnd}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.fl.rounds, rnd)
                # No optimizer state: it would be the last client's local state, which nothing
                # resumes from (the run state in fl_state is what resumes the federation).
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=rnd,
                    cfg=cfg,
                    policy=policy,
                    optimizer=None,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    accelerator=accelerator,
                )
                update_last_checkpoint(checkpoint_dir)
                checkpoint_log.append(
                    [
                        {
                            "round": rnd,
                            "step_equivalent": rnd * steps_per_round,
                            "lr": round_lrs[0],
                            "next_lr": optimizer.param_groups[0]["lr"],
                            "checkpoint": checkpoint_dir.name,
                        }
                    ]
                )
                if cfg.save_checkpoint_to_hub:
                    push_checkpoint_to_hub(checkpoint_dir, cfg.policy.repo_id, private=cfg.policy.private)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            if cfg.env is not None and (
                cfg.fl.env_eval_freq_rounds > 0 and rnd % cfg.fl.env_eval_freq_rounds == 0
            ):
                step_id = get_step_identifier(rnd, cfg.fl.rounds)
                logging.info(f"Evaluating the global policy after round {rnd}")
                with _make_eval_envs(cfg) as eval_envs, torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_envs,
                        policy=unwrapped,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_round_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                aggregated = eval_info["overall"]
                logging.info("round %d eval: %s", rnd, aggregated)
                if wandb_logger:
                    wandb_logger.log_dict(
                        {f"eval/{k}": v for k, v in aggregated.items() if isinstance(v, int | float)},
                        rnd,
                        mode="eval",
                    )
                policy.train()

            last = snapshot(rnd)
            rounds_this_run += 1
            if cfg.fl.stop_after_rounds and rounds_this_run >= cfg.fl.stop_after_rounds and not is_last_round:
                stop.requested, stop.signal_name = True, "fl.stop_after_rounds"
            if stop.requested or is_last_round or rnd % cfg.fl.resume_save_freq_rounds == 0:
                save_state(last)
                last_saved_round = rnd
            if stop.requested and not is_last_round:
                raise TrainingInterrupted
    except TrainingInterrupted:
        if last.round > last_saved_round:
            save_state(last)
        logging.warning(
            "Stopped (%s) with round %d of %d complete; rerun the same command to continue.",
            stop.signal_name or "requested",
            last.round,
            cfg.fl.rounds,
        )
        raise
    finally:
        stop.restore()
        progbar.close()
        if proximal is not None:
            proximal.detach()

    _write_json(
        output_dir / "TRAINING_COMPLETE",
        {
            "rounds": cfg.fl.rounds,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fingerprint": fingerprint,
        },
    )
    logging.info("Federated training finished")

    # Publishing is opt-in: `publish_trained_model` always pushes and requires a repo id, so
    # gate it the same way the centralized trainer does. The global model is already on disk
    # in `output_dir/checkpoints/` either way.
    if getattr(cfg.policy, "push_to_hub", False):
        model_to_publish = unwrapped.get_base_model() if cfg.peft is not None else unwrapped
        publish_trained_model(
            cfg=cfg,
            model=model_to_publish,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_meta=meta,
            peft_model=unwrapped if cfg.peft is not None else None,
        )


def main():
    init_logging()
    try:
        fl_train()
    except TrainingInterrupted:
        sys.exit(EXIT_INTERRUPTED)


if __name__ == "__main__":
    main()
