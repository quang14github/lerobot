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
"""How much each client's update counts in the server's average."""

import pytest
import torch

from lerobot.federated.aggregate import DeltaAccumulator
from lerobot.federated.config import FederatedConfig
from lerobot.federated.partition import ClientShard
from lerobot.scripts.lerobot_fl_train import _client_weight, format_client_weights

SHARDS = [
    ClientShard(client_id=0, episodes=[0, 1], tasks=["open the drawer"], num_frames=400),
    ClientShard(client_id=1, episodes=[2, 3, 4], tasks=["close the door"], num_frames=300),
    ClientShard(client_id=2, episodes=[5], tasks=["pull the mug"], num_frames=300),
]


@pytest.mark.parametrize(
    ("fl", "expected"),
    [
        (FederatedConfig(client_weighting="frames"), [400.0, 300.0, 300.0]),
        (FederatedConfig(client_weighting="episodes"), [2.0, 3.0, 1.0]),
        (FederatedConfig(client_weighting="uniform"), [1.0, 1.0, 1.0]),
        (FederatedConfig(client_weighting="custom", client_weights=[2, 1, 0.5]), [2.0, 1.0, 0.5]),
    ],
)
def test_client_weight_per_mode(fl, expected):
    assert [_client_weight(shard, fl) for shard in SHARDS] == expected


def test_custom_weights_follow_the_client_id_not_the_list_position():
    """A round that samples only some clients must still give each its own weight."""
    fl = FederatedConfig(client_weighting="custom", client_weights=[5, 1, 2])

    assert _client_weight(SHARDS[2], fl) == 2.0


def test_custom_weights_set_each_clients_share_of_the_aggregate():
    """With weights [2, 1, 1] client 0 contributes half of the averaged update."""
    fl = FederatedConfig(client_weighting="custom", client_weights=[2, 1, 1])
    acc = DeltaAccumulator()
    for shard, value in zip(SHARDS, [4.0, 0.0, 0.0], strict=True):
        acc.add({"w": torch.tensor([value])}, _client_weight(shard, fl))

    assert acc.result()["w"].item() == pytest.approx(2.0)


def test_weights_table_shows_shares_and_tasks():
    table = format_client_weights(
        SHARDS, FederatedConfig(client_weighting="custom", client_weights=[2, 1, 1])
    )

    assert "50.0%" in table and "25.0%" in table
    assert "open the drawer" in table and "pull the mug" in table
    assert len(table.splitlines()) == 4  # header + one row per client
