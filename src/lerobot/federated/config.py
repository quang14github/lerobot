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
from dataclasses import dataclass, field

from lerobot.configs.train import TrainPipelineConfig

SERVER_OPTIMIZERS = ("fedavg", "fedavgm", "fedadam")


@dataclass
class PartitionConfig:
    """How the dataset is split into per-client shards. See `federated.partition`."""

    # "uniform" (IID), "dirichlet" (tunable skew), or "task" (disjoint tasks per client).
    strategy: str = "task"
    num_clients: int = 10
    # Concentration for strategy="dirichlet": smaller is more skewed.
    dirichlet_alpha: float = 0.5
    # Seeded independently of `cfg.seed` so the same split can be reused across runs that
    # differ only in aggregation or model.
    seed: int = 42
    # Dirichlet draws that starve a client are retried this many times before giving up.
    max_draw_attempts: int = 20


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
    # ("episodes"), or equally ("uniform").
    client_weighting: str = "frames"
    log_freq_rounds: int = 1
    save_freq_rounds: int = 20
    # Roll out the global policy in the env every N rounds (0 = never). Requires --env.type.
    env_eval_freq_rounds: int = 0


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
        self.steps = self.fl.rounds
        super().validate()

        if self.fl.rounds < 1:
            raise ValueError(f"fl.rounds must be >= 1, got {self.fl.rounds}")
        if self.fl.local_steps < 1:
            raise ValueError(f"fl.local_steps must be >= 1, got {self.fl.local_steps}")
        if not 0 <= self.fl.clients_per_round <= self.partition.num_clients:
            raise ValueError(
                f"fl.clients_per_round must be in [0, partition.num_clients="
                f"{self.partition.num_clients}], got {self.fl.clients_per_round}"
            )
        if self.fl.client_weighting not in ("frames", "episodes", "uniform"):
            raise ValueError(
                f"fl.client_weighting must be 'frames', 'episodes' or 'uniform', "
                f"got {self.fl.client_weighting!r}"
            )
        if self.fl.prox_mu < 0:
            raise ValueError(f"fl.prox_mu must be >= 0, got {self.fl.prox_mu}")
        if self.server.type not in SERVER_OPTIMIZERS:
            raise ValueError(f"Unknown server.type={self.server.type!r}; expected one of {SERVER_OPTIMIZERS}")
        if self.server.lr <= 0:
            raise ValueError(f"server.lr must be > 0, got {self.server.lr}")
        if self.fl.env_eval_freq_rounds > 0 and self.env is None:
            raise ValueError("fl.env_eval_freq_rounds > 0 requires an environment (--env.type=metaworld).")

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
                "--resume is not supported yet for federated runs; restart from the last saved "
                "global checkpoint with --policy.path=<checkpoint>/pretrained_model."
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
