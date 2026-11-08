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
"""Configuration for federated training.

`FederatedTrainPipelineConfig` extends `TrainPipelineConfig`, so every existing knob
(`--dataset.*`, `--policy.*`, `--peft.*`, `--wandb.*`, ...) keeps working and a federated run is
reproducible from its `train_config.json` the same way a centralized one is. The federated
additions live under `--fl.*`, `--partition.*` and `--server.*`.
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs.train import TrainPipelineConfig

SERVER_OPTIMIZERS = ("fedavg", "fedavgm", "fedadam")
CLIENT_WEIGHTINGS = ("frames", "episodes", "uniform", "custom")


@dataclass
class PartitionConfig:
    """How the dataset is split into per-client shards. See `federated.partition`."""

    # "uniform" (episodes shuffled, IID only in expectation), "iid" (an equal share of every task
    # at every client), "dirichlet" (tunable skew), "task" (disjoint tasks per client),
    # "shared_core" (some tasks split across all clients, the rest each held by one client) or
    # "explicit" (see task_assignment).
    strategy: str = "task"
    num_clients: int = 10
    # Concentration for strategy="dirichlet": smaller is more skewed.
    dirichlet_alpha: float = 0.5
    # Seeded independently of `cfg.seed` so the same split can be reused across runs that
    # differ only in aggregation or model.
    seed: int = 42
    # Dirichlet draws that starve a client are retried this many times before giving up.
    max_draw_attempts: int = 20
    # For strategy="explicit": client index (as a string key) -> the task names it holds. A task
    # named by one client is exclusive to it; a task named by several is split evenly between
    # exactly those clients. That one rule expresses asymmetric federations the other strategies
    # cannot - e.g. two clients sharing one task while a third holds a different one.
    task_assignment: dict[str, list[str]] | None = None
    # For strategy="shared_core": how many tasks are split across every client. The others are
    # each given whole to one client. Which tasks are shared is drawn from `seed`.
    num_shared_tasks: int = 5
    # Optional JSON file {"episode_tasks": {"<episode>": "<task key>"}} naming each episode's task.
    # Without it tasks are the dataset's instruction strings, which cannot tell apart tasks that
    # were recorded under one instruction (Meta-World's push-v3 and push-back-v3). See
    # `lerobot.federated.tasks`.
    task_map: str | None = None


@dataclass
class ServerConfig:
    """Server-side update applied to the aggregated client deltas.

    With `type="fedavg"` and `lr=1.0` this is textbook FedAvg: the new global weights are the
    sample-weighted mean of the clients' local weights. The adaptive variants follow Reddi et
    al., "Adaptive Federated Optimization" (ICLR 2021), which treats the negated aggregate
    delta as a gradient and runs a real optimizer on the server — the standard fix for the
    client drift that plain FedAvg suffers on non-IID shards.
    """

    type: str = "fedavg"
    # Server step size. 1.0 for fedavg reproduces the plain weight average.
    lr: float = 1.0
    beta1: float = 0.9  # fedavgm, fedadam
    beta2: float = 0.99  # fedadam
    eps: float = 1e-3  # fedadam (Reddi et al. use a much larger tau than local Adam)


@dataclass
class FederatedConfig:
    """The federated loop itself: rounds, local work per round, and client sampling."""

    rounds: int = 200
    # Optimizer steps each selected client runs locally per round.
    local_steps: int = 50
    # Clients sampled per round (0 = all of them). Partial participation is the realistic
    # setting and also cuts wall-clock per round.
    clients_per_round: int = 0
    # FedProx proximal weight (Li et al. 2020). 0 disables it. Try 0.01-0.1 when a non-IID
    # partition makes plain FedAvg unstable.
    prox_mu: float = 0.0
    # Standard FedAvg starts each client from fresh optimizer state every round. Set False to
    # carry per-client Adam state across rounds instead — better local progress, but the
    # cached state costs ~2 floats per trainable parameter per client, so only do it with
    # LoRA or another small trainable surface.
    reset_optimizer_each_round: bool = True
    # Weight each client's delta by its frame count ("frames"), its episode count
    # ("episodes"), equally ("uniform"), or by `client_weights` ("custom").
    client_weighting: str = "frames"
    # For client_weighting="custom": one positive weight per client, in client-id order (the
    # weights table logged at startup says which tasks each client holds). Only the ratios
    # matter: each round the participating clients' weights are normalized to sum to one, so
    # [2, 1, 1] gives client 0 half of the update whenever all three take part.
    client_weights: list[float] | None = None
    log_freq_rounds: int = 1
    save_freq_rounds: int = 20
    # Roll out the global policy in the env every N rounds (0 = never). Requires --env.type.
    env_eval_freq_rounds: int = 0
    # Save `checkpoints/000000` - the pretrained model with this run's processors - before round 1.
    # With rounds=0 the run stops right after it, which is how untrained baselines are produced.
    save_initial_checkpoint: bool = False
    # Continue from `output_dir/fl_state` when it exists (an existing output_dir is then expected),
    # otherwise start fresh. This is what lets one command run as a chain of time-limited jobs.
    auto_resume: bool = False
    # Write the resumable run state every N completed rounds (and always on Ctrl-C/SIGTERM/SIGUSR1).
    resume_save_freq_rounds: int = 1
    # Stop as if signalled after this many rounds of the current invocation (0 = never). For tests.
    stop_after_rounds: int = 0
    # Resume even when settings that define the run differ from the saved state.
    allow_config_change: bool = False
    # After setup, run one forward/backward on a few samples and list trainable parameters that
    # get no gradient (written to grad_audit.json). Costs one extra step; for smoke tests.
    audit_gradients: bool = False


@dataclass
class FederatedTrainPipelineConfig(TrainPipelineConfig):
    """A centralized training config plus the federated loop, partition and server settings."""

    fl: FederatedConfig = field(default_factory=FederatedConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    def validate(self) -> None:
        """Fail fast on any federated setting that would only break hours into a run."""
        # `steps` is inherited and drives the LR scheduler's horizon. In a federated run the
        # scheduler advances once per *round* (see the orchestrator), so the horizon is the
        # round count, not the total number of local updates.
        self.steps = max(self.fl.rounds, 1)
        # With auto_resume the output directory of an earlier link is expected to exist; hide it
        # from the parent's "would overwrite" check, which only knows checkpoint-based resume.
        existing_output = None
        if self.fl.auto_resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            existing_output, self.output_dir = self.output_dir, None
        super().validate()
        if existing_output is not None:
            self.output_dir = existing_output

        if self.fl.rounds < 1 and not (self.fl.rounds == 0 and self.fl.save_initial_checkpoint):
            raise ValueError(
                f"fl.rounds must be >= 1 (or 0 with fl.save_initial_checkpoint=true), got {self.fl.rounds}"
            )
        if self.fl.local_steps < 1:
            raise ValueError(f"fl.local_steps must be >= 1, got {self.fl.local_steps}")
        if not 0 <= self.fl.clients_per_round <= self.partition.num_clients:
            raise ValueError(
                f"fl.clients_per_round must be in [0, partition.num_clients="
                f"{self.partition.num_clients}], got {self.fl.clients_per_round}"
            )
        if self.fl.client_weighting not in CLIENT_WEIGHTINGS:
            raise ValueError(
                f"fl.client_weighting must be one of {CLIENT_WEIGHTINGS}, got {self.fl.client_weighting!r}"
            )
        weights = self.fl.client_weights
        if self.fl.client_weighting == "custom":
            if not weights:
                raise ValueError(
                    "fl.client_weighting='custom' needs fl.client_weights, one weight per client, "
                    "e.g. --fl.client_weights=[2,1,1]"
                )
            if len(weights) != self.partition.num_clients:
                raise ValueError(
                    f"fl.client_weights has {len(weights)} entries but partition.num_clients="
                    f"{self.partition.num_clients}; give exactly one weight per client"
                )
            if not all(math.isfinite(w) and w > 0 for w in weights):
                raise ValueError(
                    f"fl.client_weights must all be positive, got {weights}. To leave a client out, "
                    f"change the partition instead."
                )
        elif weights is not None:
            raise ValueError(
                f"fl.client_weights is only used with fl.client_weighting='custom', "
                f"but client_weighting={self.fl.client_weighting!r}"
            )
        if self.fl.prox_mu < 0:
            raise ValueError(f"fl.prox_mu must be >= 0, got {self.fl.prox_mu}")
        if self.server.type not in SERVER_OPTIMIZERS:
            raise ValueError(f"Unknown server.type={self.server.type!r}; expected one of {SERVER_OPTIMIZERS}")
        if self.server.lr <= 0:
            raise ValueError(f"server.lr must be > 0, got {self.server.lr}")
        if self.fl.env_eval_freq_rounds > 0 and self.env is None:
            raise ValueError(
                "fl.env_eval_freq_rounds > 0 requires an environment (e.g. --env.type=metaworld or libero)."
            )

        if self.ema.enable:
            raise NotImplementedError(
                "--ema.enable is not supported under federated training: the shadow would track "
                "one client's local trajectory rather than the global model."
            )
        if self.dataset.streaming:
            raise NotImplementedError(
                "--dataset.streaming is not supported under federated training: client shards are "
                "defined by episode allowlists, which need a map-style dataset."
            )
        if self.resume:
            raise NotImplementedError(
                "--resume is not used by federated runs: pass --fl.auto_resume=true with the original "
                "arguments and --output_dir, and the run continues from output_dir/fl_state."
            )
        if self.fl.resume_save_freq_rounds < 1:
            raise ValueError(
                f"fl.resume_save_freq_rounds must be >= 1, got {self.fl.resume_save_freq_rounds}"
            )

        # `eval_split` holds out the tail episodes of every task *before* partitioning, so the
        # held-out set stays global — but each client would then compute eval loss on a split
        # it never sees. Point users at env eval instead.
        if self.dataset.eval_split > 0:
            logging.warning(
                "dataset.eval_split is held out globally and is not evaluated per client in "
                "federated training; use fl.env_eval_freq_rounds to track global progress."
            )

        total_updates = self.fl.rounds * self.fl.local_steps
        logging.info(
            "Federated plan: %d rounds x %d local steps = %d local updates per participating client; "
            "server=%s, partition=%s over %d clients.",
            self.fl.rounds,
            self.fl.local_steps,
            total_updates,
            self.server.type,
            self.partition.strategy,
            self.partition.num_clients,
        )
