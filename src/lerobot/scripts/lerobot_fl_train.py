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
import logging
import time
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
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import make_dataset
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
from lerobot.federated.config import FederatedTrainPipelineConfig
from lerobot.federated.partition import ClientShard, format_partition, partition_episodes
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

    def __init__(self, shard: ClientShard, dataset, dataloader):
        self.shard = shard
        self.dataset = dataset
        self.dataloader = dataloader
        self.iterator = cycle(dataloader)
        self.rounds_participated = 0
        self.local_steps_done = 0

    def next_batch(self):
        return next(self.iterator)


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


def _client_weight(shard: ClientShard, weighting: str) -> float:
    if weighting == "frames":
        return float(shard.num_frames)
    if weighting == "episodes":
        return float(shard.num_episodes)
    return 1.0


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


@parser.wrap()
def fl_train(cfg: FederatedTrainPipelineConfig):
    """Run federated training end to end.

    Args:
        cfg: The federated training config, parsed from the CLI by `parser.wrap()`.
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
    base_episodes = resolve_episode_indices(
        cfg.dataset.episodes, meta.total_episodes, cfg.dataset.exclude_episodes
    )
    shards = partition_episodes(meta, cfg.partition, episodes=base_episodes)
    logging.info(
        "Partitioned %d episodes across %d clients (strategy=%s):\n%s",
        sum(shard.num_episodes for shard in shards),
        len(shards),
        cfg.partition.strategy,
        format_partition(shards),
    )

    # --- global model, built once from the full metadata ---------------------------------------
    logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=meta, rename_map=cfg.rename_map)
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

    global_state = get_trainable_state(unwrapped)
    server_optimizer = make_server_optimizer(cfg.server)
    proximal = None
    if cfg.fl.prox_mu > 0:
        proximal = ProximalTerm(unwrapped, cfg.fl.prox_mu)
        proximal.attach(optimizer)

    # Per-client optimizer state, kept only when the user opts out of the FedAvg default of
    # starting every round fresh.
    client_optimizer_states: dict[int, dict] = {}

    num_trainable = sum(value.numel() for value in global_state.values())
    num_total = sum(p.numel() for p in unwrapped.parameters())
    payload_mb = sum(value.numel() * 4 for value in global_state.values()) / 1024**2
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(
        f"Exchanged (trainable) params: {format_big_number(num_trainable)} of {format_big_number(num_total)}"
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
    selection_rng = np.random.default_rng(cfg.seed if cfg.seed is not None else 0)
    progbar = tqdm(total=cfg.fl.rounds, desc="Federated rounds", unit="round", disable=inside_slurm())
    logging.info(f"Start federated training: {cfg.fl.rounds} rounds x {cfg.fl.local_steps} local steps")

    for rnd in range(1, cfg.fl.rounds + 1):
        round_start = time.perf_counter()
        selected = _select_clients(shards, cfg.fl.clients_per_round, selection_rng)

        accumulator = DeltaAccumulator()
        client_losses = {}
        for client_id in selected:
            shard = shards[client_id]

            # 1. Broadcast: every client starts the round from the same global weights.
            set_trainable_state(unwrapped, global_state)
            if cfg.fl.reset_optimizer_each_round:
                reset_optimizer_state(optimizer)
            elif client_id in client_optimizer_states:
                optimizer.load_state_dict(client_optimizer_states[client_id])
            else:
                reset_optimizer_state(optimizer)
            if proximal is not None:
                proximal.set_anchor(global_state)

            # Materialize the client's private dataset the first time it is picked.
            if client_id not in clients:
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
                clients[client_id] = ClientRuntime(shard, client_dataset, client_dataloader)
            runtime = clients[client_id]

            # 2. Local training on private data.
            policy.train()
            local_loss = 0.0
            for _ in range(cfg.fl.local_steps):
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
            runtime.rounds_participated += 1
            runtime.local_steps_done += cfg.fl.local_steps

            # 3. Upload: only the weight change leaves the client, and it is folded into the
            # running aggregate immediately rather than kept until the end of the round.
            accumulator.add(
                compute_delta(unwrapped, global_state), _client_weight(shard, cfg.fl.client_weighting)
            )
            client_losses[client_id] = local_loss / cfg.fl.local_steps
            if not cfg.fl.reset_optimizer_each_round:
                client_optimizer_states[client_id] = copy.deepcopy(optimizer.state_dict())

        # 4. Aggregate and take the server step.
        aggregate = accumulator.result()
        update_norm = delta_norm(aggregate)
        server_optimizer.step(global_state, aggregate)
        set_trainable_state(unwrapped, global_state)
        del accumulator, aggregate
        if lr_scheduler is not None:
            lr_scheduler.step()

        round_s = time.perf_counter() - round_start
        if cfg.fl.log_freq_rounds > 0 and rnd % cfg.fl.log_freq_rounds == 0:
            if tracker.step_s.avg > 0:
                tracker.samples_per_s = cfg.batch_size / tracker.step_s.avg
            mean_loss = float(np.mean(list(client_losses.values())))
            spread = float(np.std(list(client_losses.values())))
            logging.info(
                "round %d/%d  clients=%s  loss=%.4f (+/-%.4f)  |update|=%.4e  lr=%.2e  %.1fs",
                rnd,
                cfg.fl.rounds,
                selected,
                mean_loss,
                spread,
                update_norm,
                optimizer.param_groups[0]["lr"],
                round_s,
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
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=rnd,
                cfg=cfg,
                policy=policy,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                accelerator=accelerator,
            )
            update_last_checkpoint(checkpoint_dir)
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

    progbar.close()
    if proximal is not None:
        proximal.detach()

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
    fl_train()


if __name__ == "__main__":
    main()
