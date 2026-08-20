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
"""Aggregation semantics, checked on a small nn.Module instead of a real VLA."""

import pytest
import torch
from torch import nn

from lerobot.federated.aggregate import (
    DeltaAccumulator,
    FedAdam,
    FedAvg,
    FedAvgM,
    ProximalTerm,
    compute_delta,
    delta_norm,
    get_trainable_state,
    make_server_optimizer,
    reset_optimizer_state,
    set_trainable_state,
    weighted_mean,
)
from lerobot.federated.config import ServerConfig


class TinyPolicy(nn.Module):
    """Stands in for a policy with a frozen backbone and a small trainable head."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)
        self.register_buffer("token_ids", torch.tensor([7, 8, 9]))
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        return self.head(self.backbone(x))


def test_only_trainable_parameters_are_exchanged():
    """The frozen backbone must never enter the payload — that is the whole cost argument."""
    state = get_trainable_state(TinyPolicy())

    assert set(state) == {"head.weight", "head.bias"}
    assert all(tensor.device.type == "cpu" and tensor.dtype == torch.float32 for tensor in state.values())


def test_snapshot_is_a_copy_not_a_view():
    policy = TinyPolicy()
    state = get_trainable_state(policy)
    with torch.no_grad():
        policy.head.bias.fill_(42.0)

    assert not torch.allclose(state["head.bias"], policy.head.bias.cpu())


def test_set_trainable_state_round_trips():
    policy = TinyPolicy()
    saved = get_trainable_state(policy)
    with torch.no_grad():
        policy.head.weight.fill_(3.0)
    set_trainable_state(policy, saved)

    assert torch.allclose(policy.head.weight, saved["head.weight"])


def test_set_trainable_state_leaves_buffers_alone():
    """Buffers are not parameters; averaging them would corrupt token ids and caches."""
    policy = TinyPolicy()
    state = get_trainable_state(policy)
    set_trainable_state(policy, state)

    assert torch.equal(policy.token_ids, torch.tensor([7, 8, 9]))


def test_set_trainable_state_rejects_foreign_keys():
    with pytest.raises(KeyError, match="absent from the policy"):
        set_trainable_state(TinyPolicy(), {"does.not.exist": torch.zeros(2)})


def test_compute_delta_measures_local_progress():
    policy = TinyPolicy()
    reference = get_trainable_state(policy)
    with torch.no_grad():
        policy.head.bias.add_(0.5)
    delta = compute_delta(policy, reference)

    assert torch.allclose(delta["head.bias"], torch.full_like(delta["head.bias"], 0.5))
    assert torch.allclose(delta["head.weight"], torch.zeros_like(delta["head.weight"]))


def test_weighted_mean_is_sample_weighted():
    """A client with twice the data pulls the aggregate twice as hard."""
    deltas = [{"w": torch.tensor([0.0])}, {"w": torch.tensor([3.0])}]
    aggregate = weighted_mean(deltas, [1.0, 2.0])

    assert aggregate["w"].item() == pytest.approx(2.0)


def test_weighted_mean_normalizes_arbitrary_weights():
    """Frame counts are large and unnormalized; only their ratio may matter."""
    deltas = [{"w": torch.tensor([1.0])}, {"w": torch.tensor([2.0])}]
    small = weighted_mean(deltas, [1.0, 1.0])
    large = weighted_mean(deltas, [50_000.0, 50_000.0])

    assert small["w"].item() == pytest.approx(large["w"].item())


def test_weighted_mean_rejects_degenerate_input():
    with pytest.raises(ValueError, match="No client deltas"):
        weighted_mean([], [])
    with pytest.raises(ValueError, match="positive value"):
        weighted_mean([{"w": torch.tensor([1.0])}], [0.0])


def test_fedavg_at_lr_one_is_the_weight_average():
    """FedAvg's defining property: the global model lands on the mean of the local models."""
    global_state = {"w": torch.tensor([10.0])}
    locals_ = [torch.tensor([12.0]), torch.tensor([16.0])]
    deltas = [{"w": local - global_state["w"]} for local in locals_]

    FedAvg(lr=1.0).step(global_state, weighted_mean(deltas, [1.0, 1.0]))

    assert global_state["w"].item() == pytest.approx(14.0)


def test_fedavg_lr_scales_the_server_step():
    global_state = {"w": torch.tensor([0.0])}
    FedAvg(lr=0.5).step(global_state, {"w": torch.tensor([4.0])})

    assert global_state["w"].item() == pytest.approx(2.0)


def test_fedavgm_accumulates_momentum_across_rounds():
    """Repeated deltas in the same direction should accelerate, unlike plain FedAvg."""
    momentum_state = {"w": torch.tensor([0.0])}
    plain_state = {"w": torch.tensor([0.0])}
    momentum, plain = FedAvgM(lr=1.0, beta1=0.9), FedAvg(lr=1.0)
    for _ in range(3):
        momentum.step(momentum_state, {"w": torch.tensor([1.0])})
        plain.step(plain_state, {"w": torch.tensor([1.0])})

    assert momentum_state["w"].item() > plain_state["w"].item()


def test_fedadam_normalizes_the_step_size():
    """Adaptivity is the point: a huge aggregate must not produce a huge server step."""
    state = {"w": torch.tensor([0.0])}
    FedAdam(lr=0.01, eps=1e-3).step(state, {"w": torch.tensor([1000.0])})

    assert abs(state["w"].item()) < 0.02


def test_fedadam_direction_follows_the_aggregate():
    up, down = {"w": torch.tensor([0.0])}, {"w": torch.tensor([0.0])}
    FedAdam(lr=0.01).step(up, {"w": torch.tensor([1.0])})
    FedAdam(lr=0.01).step(down, {"w": torch.tensor([-1.0])})

    assert up["w"].item() > 0 > down["w"].item()


def test_a_zero_aggregate_leaves_the_global_model_untouched():
    """Identical clients (or none making progress) must be a no-op, whatever the server."""
    for server in (FedAvg(), FedAvgM(), FedAdam()):
        state = {"w": torch.tensor([5.0])}
        server.step(state, {"w": torch.tensor([0.0])})
        assert state["w"].item() == pytest.approx(5.0)


def test_delta_norm_reports_the_global_l2_norm():
    assert delta_norm({"a": torch.tensor([3.0]), "b": torch.tensor([4.0])}) == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("kind", "expected"), [("fedavg", FedAvg), ("fedavgm", FedAvgM), ("fedadam", FedAdam)]
)
def test_server_optimizer_factory(kind, expected):
    assert isinstance(make_server_optimizer(ServerConfig(type=kind)), expected)


def test_server_optimizer_factory_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown server optimizer"):
        make_server_optimizer(ServerConfig(type="fedsgd"))


def test_proximal_term_pulls_gradients_back_toward_the_anchor():
    """FedProx: a parameter that has drifted gains a gradient pushing it home."""
    policy = TinyPolicy()
    anchor = get_trainable_state(policy)
    optimizer = torch.optim.SGD([p for p in policy.parameters() if p.requires_grad], lr=0.0)

    prox = ProximalTerm(policy, mu=0.5)
    prox.attach(optimizer)
    prox.set_anchor(anchor)

    with torch.no_grad():
        policy.head.bias.add_(2.0)  # drift
    policy.head.bias.grad = torch.zeros_like(policy.head.bias)
    optimizer.step()  # lr=0, so only the hook's contribution is observable

    assert torch.allclose(policy.head.bias.grad, torch.full_like(policy.head.bias, 1.0))  # mu * drift
    prox.detach()


def test_proximal_term_is_inert_at_the_anchor():
    policy = TinyPolicy()
    optimizer = torch.optim.SGD([p for p in policy.parameters() if p.requires_grad], lr=0.0)
    prox = ProximalTerm(policy, mu=0.5)
    prox.attach(optimizer)
    prox.set_anchor(get_trainable_state(policy))

    policy.head.bias.grad = torch.zeros_like(policy.head.bias)
    optimizer.step()

    assert torch.allclose(policy.head.bias.grad, torch.zeros_like(policy.head.bias))
    prox.detach()


def test_proximal_term_requires_positive_mu():
    with pytest.raises(ValueError, match="requires mu > 0"):
        ProximalTerm(TinyPolicy(), mu=0.0)


def test_full_round_trip_matches_centralized_averaging():
    """End-to-end: broadcast, two clients train, aggregate — equals averaging their weights."""
    server_policy = TinyPolicy()
    global_state = get_trainable_state(server_policy)

    deltas, client_weights = [], []
    for target in (1.0, 3.0):
        client = TinyPolicy()
        set_trainable_state(client, global_state)
        with torch.no_grad():
            client.head.weight.fill_(target)
        deltas.append(compute_delta(client, global_state))
        client_weights.append(1.0)

    FedAvg(lr=1.0).step(global_state, weighted_mean(deltas, client_weights))
    set_trainable_state(server_policy, global_state)

    assert torch.allclose(server_policy.head.weight, torch.full_like(server_policy.head.weight, 2.0))


def test_accumulator_matches_weighted_mean():
    """The memory-efficient path must be numerically the same as the reference one."""
    deltas = [
        {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.5])},
        {"a": torch.tensor([3.0, -1.0]), "b": torch.tensor([1.5])},
        {"a": torch.tensor([0.0, 4.0]), "b": torch.tensor([-2.0])},
    ]
    weights = [1000.0, 250.0, 3000.0]

    accumulator = DeltaAccumulator()
    for delta, weight in zip(deltas, weights, strict=True):
        accumulator.add(delta, weight)
    streamed = accumulator.result()
    reference = weighted_mean(deltas, weights)

    for name in reference:
        assert torch.allclose(streamed[name], reference[name])


def test_accumulator_rejects_an_empty_round():
    with pytest.raises(ValueError, match="No client deltas"):
        DeltaAccumulator().result()


def test_accumulator_rejects_non_positive_weights():
    with pytest.raises(ValueError, match="must be positive"):
        DeltaAccumulator().add({"w": torch.tensor([1.0])}, 0.0)


def test_reset_optimizer_state_clears_adam_moments():
    """Each round starts from a clean preconditioner, the FedAvg default."""
    policy = TinyPolicy()
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=0.1)
    policy.head.bias.grad = torch.ones_like(policy.head.bias)
    optimizer.step()
    assert optimizer.state, "expected Adam to have accumulated state"

    reset_optimizer_state(optimizer)

    assert not optimizer.state


def test_reset_optimizer_state_sees_through_an_accelerate_wrapper():
    """`accelerator.prepare` returns a wrapper; the state lives on the optimizer inside it."""

    class FakeAcceleratedOptimizer:
        def __init__(self, optimizer):
            self.optimizer = optimizer

    policy = TinyPolicy()
    inner = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=0.1)
    policy.head.bias.grad = torch.ones_like(policy.head.bias)
    inner.step()

    reset_optimizer_state(FakeAcceleratedOptimizer(inner))

    assert not inner.state
