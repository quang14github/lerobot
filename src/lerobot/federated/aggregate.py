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
"""Server-side aggregation and the client-side proximal term.

Only parameters with ``requires_grad`` are exchanged. That is what makes a VLA the size of
SmolVLA tractable to federate: with the default `train_expert_only=True` the frozen VLM
backbone never moves, and under LoRA (`--peft.method_type=lora`) the exchanged tensors are a
few MB rather than a few GB. It also handles buffers correctly by construction — token ids,
positional caches and other non-parameter state stay at the server's copy instead of being
averaged into nonsense.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy

# The exchanged payload: parameter name -> CPU float32 tensor.
StateDelta = dict[str, Tensor]


def unwrap_optimizer(optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """Return the plain `torch.optim.Optimizer` behind accelerate's `AcceleratedOptimizer`."""
    return getattr(optimizer, "optimizer", optimizer)


def reset_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Drop accumulated moments so a client starts the round from a clean slate.

    This is what standard FedAvg assumes: local optimizer state is not part of what the server
    aggregates, so carrying Adam moments from one round's client into the next one's would
    apply a stale preconditioner to freshly averaged weights.
    """
    unwrap_optimizer(optimizer).state.clear()


@torch.no_grad()
def get_trainable_state(policy: PreTrainedPolicy) -> StateDelta:
    """Snapshot the trainable parameters as CPU float32 tensors.

    Args:
        policy: An *unwrapped* policy (call `accelerator.unwrap_model` first, so parameter
            names are free of DDP's `module.` prefix).

    Returns:
        A detached copy of every parameter with `requires_grad=True`.
    """
    return {
        name: param.detach().to(device="cpu", dtype=torch.float32).clone()
        for name, param in policy.named_parameters()
        if param.requires_grad
    }


@torch.no_grad()
def set_trainable_state(policy: PreTrainedPolicy, state: StateDelta) -> None:
    """Write `state` back into `policy` in place, preserving each parameter's dtype/device."""
    params = dict(policy.named_parameters())
    missing = [name for name in state if name not in params]
    if missing:
        raise KeyError(f"State contains parameters absent from the policy: {missing[:5]}")
    for name, value in state.items():
        params[name].copy_(value.to(device=params[name].device, dtype=params[name].dtype))


@torch.no_grad()
def compute_delta(policy: PreTrainedPolicy, reference: StateDelta) -> StateDelta:
    """Return `policy_weights - reference` for the trainable parameters, on CPU in float32."""
    params = dict(policy.named_parameters())
    return {
        name: params[name].detach().to(device="cpu", dtype=torch.float32) - value
        for name, value in reference.items()
    }


@torch.no_grad()
def weighted_mean(deltas: Sequence[StateDelta], weights: Sequence[float]) -> StateDelta:
    """Combine client deltas into `sum_k (w_k / sum_j w_j) * delta_k`."""
    if not deltas:
        raise ValueError("No client deltas to aggregate.")
    if len(deltas) != len(weights):
        raise ValueError(f"Got {len(deltas)} deltas but {len(weights)} weights.")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError(f"Client weights must sum to a positive value, got {total}.")

    aggregate = {name: torch.zeros_like(value) for name, value in deltas[0].items()}
    for delta, weight in zip(deltas, weights, strict=True):
        scale = float(weight) / total
        for name, value in delta.items():
            aggregate[name].add_(value, alpha=scale)
    return aggregate


class DeltaAccumulator:
    """Running weighted sum of client deltas.

    Equivalent to `weighted_mean` over the same inputs, but holds one aggregate instead of one
    delta per client. That matters for a VLA: with the action expert trainable, each delta is
    hundreds of MB, and keeping `clients_per_round` of them alive at once is the difference
    between fitting in host RAM and not.
    """

    def __init__(self):
        """Start an empty round."""
        self._aggregate: StateDelta = {}
        self._total_weight = 0.0
        self._count = 0

    @torch.no_grad()
    def add(self, delta: StateDelta, weight: float) -> None:
        """Fold one client's delta into the running sum.

        Args:
            delta: The client's `local_weights - global_weights`.
            weight: The client's influence, e.g. its frame count. Must be positive.
        """
        if weight <= 0:
            raise ValueError(f"Client weight must be positive, got {weight}")
        if not self._aggregate:
            self._aggregate = {name: torch.zeros_like(value) for name, value in delta.items()}
        for name, value in delta.items():
            self._aggregate[name].add_(value, alpha=float(weight))
        self._total_weight += float(weight)
        self._count += 1

    @torch.no_grad()
    def result(self) -> StateDelta:
        """The sample-weighted mean of everything added so far."""
        if self._count == 0:
            raise ValueError("No client deltas were accumulated this round.")
        for value in self._aggregate.values():
            value.div_(self._total_weight)
        return self._aggregate


def delta_norm(delta: StateDelta) -> float:
    """Global L2 norm of a delta — the cheapest useful signal that a round did something."""
    return float(torch.sqrt(sum((value.double() ** 2).sum() for value in delta.values())).item())


class ServerOptimizer(ABC):
    """Applies the aggregated client delta to the global weights, in place."""

    @abstractmethod
    def step(self, global_state: StateDelta, aggregate: StateDelta) -> None:
        """Update `global_state` in place from the round's aggregated delta."""


class FedAvg(ServerOptimizer):
    """`global <- global + lr * aggregate` (McMahan et al. 2017; plain averaging at lr=1)."""

    def __init__(self, lr: float = 1.0):
        """Args: lr: Server step size; 1.0 is plain weight averaging."""
        self.lr = lr

    @torch.no_grad()
    def step(self, global_state: StateDelta, aggregate: StateDelta) -> None:
        """Move the global weights along the aggregated delta."""
        for name, value in aggregate.items():
            global_state[name].add_(value, alpha=self.lr)


class FedAvgM(ServerOptimizer):
    """FedAvg with server momentum (Hsu et al. 2019) — cheap damping for client drift."""

    def __init__(self, lr: float = 1.0, beta1: float = 0.9):
        """Args: lr: Server step size. beta1: Momentum decay."""
        self.lr = lr
        self.beta1 = beta1
        self._momentum: StateDelta = {}

    @torch.no_grad()
    def step(self, global_state: StateDelta, aggregate: StateDelta) -> None:
        """Accumulate the delta into the momentum buffer, then apply it."""
        for name, value in aggregate.items():
            buffer = self._momentum.get(name)
            if buffer is None:
                buffer = torch.zeros_like(value)
                self._momentum[name] = buffer
            buffer.mul_(self.beta1).add_(value)
            global_state[name].add_(buffer, alpha=self.lr)


class FedAdam(ServerOptimizer):
    """Adam on the server, over the aggregated delta (Reddi et al., ICLR 2021).

    Follows Algorithm 2 of the paper: the aggregate plays the role of `-gradient`, and there
    is deliberately no bias correction. `eps` is the paper's `tau` and is much larger than a
    local Adam epsilon (1e-3 rather than 1e-8) — it sets the adaptivity floor.
    """

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.99, eps: float = 1e-3):
        """Args: lr: Server step size. beta1/beta2: Moment decays. eps: The paper's `tau`."""
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m: StateDelta = {}
        self._v: StateDelta = {}

    @torch.no_grad()
    def step(self, global_state: StateDelta, aggregate: StateDelta) -> None:
        """Take one Adam step on the server, treating the aggregate as `-gradient`."""
        for name, value in aggregate.items():
            if name not in self._m:
                self._m[name] = torch.zeros_like(value)
                self._v[name] = torch.zeros_like(value)
            m, v = self._m[name], self._v[name]
            m.mul_(self.beta1).add_(value, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(value, value, value=1.0 - self.beta2)
            global_state[name].addcdiv_(m, v.sqrt().add_(self.eps), value=self.lr)


def make_server_optimizer(cfg) -> ServerOptimizer:
    """Build the server optimizer named by `ServerConfig.type`."""
    if cfg.type == "fedavg":
        return FedAvg(lr=cfg.lr)
    if cfg.type == "fedavgm":
        return FedAvgM(lr=cfg.lr, beta1=cfg.beta1)
    if cfg.type == "fedadam":
        return FedAdam(lr=cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2, eps=cfg.eps)
    raise ValueError(f"Unknown server optimizer type: {cfg.type!r}")


class ProximalTerm:
    """FedProx (Li et al., MLSys 2020) as an optimizer pre-step hook.

    Adds `mu * (theta - theta_global)` to the gradient of every trainable parameter, which
    penalizes drifting away from the round's starting point. Implemented as a hook so the
    shared `update_policy` from the centralized trainer can be reused verbatim.

    Note the hook fires *after* `update_policy` has clipped gradients, so the proximal
    contribution itself is unclipped. That is intentional — it is a bounded, well-scaled
    correction, and clipping it would also rescale the data gradient it is meant to correct.
    """

    def __init__(self, policy: PreTrainedPolicy, mu: float):
        """Args: policy: The unwrapped policy. mu: Proximal weight; must be positive."""
        if mu <= 0:
            raise ValueError(f"ProximalTerm requires mu > 0, got {mu}")
        self.mu = mu
        self._params = [(name, param) for name, param in policy.named_parameters() if param.requires_grad]
        self._anchor: dict[str, Tensor] = {}
        self._handle = None

    def set_anchor(self, state: StateDelta) -> None:
        """Pin the penalty to the round's global weights, moved onto the parameters' devices."""
        self._anchor = {
            name: state[name].to(device=param.device, dtype=param.dtype)
            for name, param in self._params
            if name in state
        }

    def attach(self, optimizer: torch.optim.Optimizer) -> None:
        """Hook the *inner* optimizer, which is the one whose `step()` actually runs.

        `accelerator.prepare` returns an `AcceleratedOptimizer` wrapper that delegates to the
        optimizer underneath; registering on the wrapper would be silently ineffective.
        """
        if self._handle is not None:
            raise RuntimeError("ProximalTerm is already attached to an optimizer.")
        self._handle = unwrap_optimizer(optimizer).register_step_pre_hook(self._hook)
        logging.info("FedProx enabled with mu=%g", self.mu)

    def detach(self) -> None:
        """Remove the hook, leaving the optimizer as it was."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @torch.no_grad()
    def _hook(self, optimizer, args, kwargs) -> None:  # noqa: ARG002 - torch hook signature
        if not self._anchor:
            return
        for name, param in self._params:
            if param.grad is not None and name in self._anchor:
                param.grad.add_(param.detach() - self._anchor[name], alpha=self.mu)
