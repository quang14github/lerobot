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
"""Federated training: dataset partitioning, server-side aggregation, and their config.

The training loop itself lives in `lerobot.scripts.lerobot_fl_train` (`lerobot-fl-train`).
"""

from .aggregate import (
    DeltaAccumulator,
    FedAdam,
    FedAvg,
    FedAvgM,
    ProximalTerm,
    ServerOptimizer,
    compute_delta,
    delta_norm,
    get_trainable_state,
    make_server_optimizer,
    reset_optimizer_state,
    set_trainable_state,
    unwrap_optimizer,
    weighted_mean,
)
from .config import FederatedConfig, FederatedTrainPipelineConfig, PartitionConfig, ServerConfig
from .partition import ClientShard, format_partition, partition_episodes

__all__ = [
    "ClientShard",
    "DeltaAccumulator",
    "FedAdam",
    "FedAvg",
    "FedAvgM",
    "FederatedConfig",
    "FederatedTrainPipelineConfig",
    "PartitionConfig",
    "ProximalTerm",
    "ServerConfig",
    "ServerOptimizer",
    "compute_delta",
    "delta_norm",
    "format_partition",
    "get_trainable_state",
    "make_server_optimizer",
    "partition_episodes",
    "reset_optimizer_state",
    "set_trainable_state",
    "unwrap_optimizer",
    "weighted_mean",
]
