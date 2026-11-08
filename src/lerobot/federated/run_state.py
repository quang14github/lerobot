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
"""Resumable state of a federated run, and stopping cleanly on Ctrl-C or a termination signal.

A federated run is resumed from the end of its last *completed* round. Everything the next round
depends on is restored: the server's trainable weights and optimizer, the round counter, the LR
scheduler, every random generator, each client's position in its own data stream, and (when kept)
each client's optimizer state. A round interrupted by a signal is dropped and replayed on resume,
so a stop request takes effect after the current step instead of after a whole round.

The state is written to a temporary directory and swapped in with renames, so a process killed
mid-write leaves the previous state intact.
"""

import csv
import hashlib
import json
import logging
import os
import random
import shutil
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

STATE_DIR = "fl_state"
EXIT_INTERRUPTED = 75  # EX_TEMPFAIL: stopped on request; resume with the same command.

# Settings that may change between the links of one run without changing what it computes.
_FINGERPRINT_EXCLUDED = (
    "output_dir",
    "job_name",
    "trackio",
    "num_workers",
    "prefetch_factor",
    "persistent_workers",
    "dataloader_multiprocessing_context",
    "log_freq",
    "fl.log_freq_rounds",
    "fl.auto_resume",
    "fl.resume_save_freq_rounds",
    "fl.stop_after_rounds",
    "fl.audit_gradients",
    "fl.allow_config_change",
    "policy.device",
)


class TrainingInterrupted(Exception):  # noqa: N818 - a control-flow signal, not an error
    """Raised once the run state has been saved after a stop request."""


class StopRequest:
    """Turns Ctrl-C (SIGINT) and SIGTERM into a flag the training loop polls between steps.

    The run then saves its state and exits, and rerunning the same command continues it. A second
    Ctrl-C aborts immediately, without saving.
    """

    def __init__(self, signals: tuple[int, ...] = (signal.SIGINT, signal.SIGTERM)):
        """Install the handlers; `restore` puts the previous ones back."""
        self.requested = False
        self.signal_name = ""
        self._previous = {}
        for sig in signals:
            self._previous[sig] = signal.signal(sig, self._handle)

    def _handle(self, signum, frame):  # noqa: ARG002 - signal handler signature
        if self.requested and signum == signal.SIGINT:
            raise KeyboardInterrupt
        if not self.requested:
            hint = " (Ctrl-C again to abort without saving)" if signum == signal.SIGINT else ""
            logging.warning(
                "Received %s: saving and stopping after the current step%s.",
                signal.Signals(signum).name,
                hint,
            )
        self.requested = True
        self.signal_name = signal.Signals(signum).name

    def restore(self) -> None:
        """Reinstall the signal handlers that were active before."""
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)


def capture_rng_state() -> dict[str, Any]:
    """Python, NumPy, torch and (when present) every CUDA device's generator state."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore the generators captured by `capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def optimizer_state_to_cpu(state_dict: dict) -> dict:
    """A CPU copy of an optimizer state dict, so inactive clients' states never occupy the GPU."""

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.detach().to("cpu", copy=True)
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        if isinstance(value, list):
            return [move(v) for v in value]
        return value

    return move(state_dict)


@dataclass
class RoundSnapshot:
    """What changes during a round besides the server state: captured at every round boundary."""

    round: int
    rng: dict[str, Any]
    selection_rng: dict[str, Any]
    clients: dict[int, dict[str, int]] = field(default_factory=dict)
    client_optimizers: dict[int, dict] = field(default_factory=dict)


def config_fingerprint(cfg_dict: dict) -> tuple[str, dict]:
    """Hash of the settings that define a run, and the flattened settings themselves."""
    flat: dict[str, Any] = {}

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(f"{prefix}.{key}" if prefix else str(key), value)
        else:
            flat[prefix] = node

    walk("", cfg_dict)
    kept = {
        k: v
        for k, v in sorted(flat.items())
        if not any(k == ex or k.startswith(ex + ".") for ex in _FINGERPRINT_EXCLUDED)
    }
    digest = hashlib.sha256(json.dumps(kept, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return digest, kept


def state_exists(output_dir: Path) -> bool:
    """Whether `output_dir` holds a resumable run state."""
    return (Path(output_dir) / STATE_DIR / "state.json").is_file()


def save_run_state(
    output_dir: Path,
    *,
    global_state: dict[str, torch.Tensor],
    server_optimizer_state: dict,
    scheduler_state: dict | None,
    snapshot: RoundSnapshot,
    fingerprint: str,
    settings: dict,
    extra: dict | None = None,
) -> Path:
    """Write the run state as of `snapshot.round` atomically to `output_dir/fl_state`."""
    output_dir = Path(output_dir)
    final = output_dir / STATE_DIR
    tmp = output_dir / f"{STATE_DIR}.tmp-{os.getpid()}"
    old = output_dir / f"{STATE_DIR}.old"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    start = time.perf_counter()

    save_file({k: v.contiguous() for k, v in global_state.items()}, tmp / "server_state.safetensors")
    torch.save(server_optimizer_state, tmp / "server_optimizer.pt")
    torch.save({"rng": snapshot.rng}, tmp / "rng.pt")
    if snapshot.client_optimizers:
        torch.save(snapshot.client_optimizers, tmp / "client_optimizers.pt")
    payload = {
        "round": snapshot.round,
        "scheduler": scheduler_state,
        "selection_rng": snapshot.selection_rng,
        "clients": {str(k): v for k, v in snapshot.clients.items()},
        "fingerprint": fingerprint,
        "settings": settings,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **(extra or {}),
    }
    (tmp / "state.json").write_text(json.dumps(payload, indent=1, default=str))

    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    shutil.rmtree(old, ignore_errors=True)
    size_gb = sum(f.stat().st_size for f in final.iterdir()) / 1024**3
    logging.info(
        "Saved run state after round %d (%.2f GB in %.1fs)",
        snapshot.round,
        size_gb,
        time.perf_counter() - start,
    )
    return final


@dataclass
class LoadedState:
    """A run state read back by `load_run_state`."""

    round: int
    global_state: dict[str, torch.Tensor]
    server_optimizer_state: dict
    scheduler_state: dict | None
    rng: dict[str, Any]
    selection_rng: dict[str, Any]
    clients: dict[int, dict[str, int]]
    client_optimizers: dict[int, dict]
    fingerprint: str
    settings: dict


def load_run_state(output_dir: Path) -> LoadedState:
    """Read the run state written by `save_run_state`."""
    root = Path(output_dir) / STATE_DIR
    payload = json.loads((root / "state.json").read_text())
    client_opt_file = root / "client_optimizers.pt"
    return LoadedState(
        round=int(payload["round"]),
        global_state=load_file(root / "server_state.safetensors"),
        server_optimizer_state=torch.load(root / "server_optimizer.pt", weights_only=False),  # nosec B614
        scheduler_state=payload["scheduler"],
        rng=torch.load(root / "rng.pt", weights_only=False)["rng"],  # nosec B614
        selection_rng=payload["selection_rng"],
        clients={int(k): v for k, v in payload["clients"].items()},
        client_optimizers=torch.load(client_opt_file, weights_only=False) if client_opt_file.exists() else {},  # nosec B614
        fingerprint=payload["fingerprint"],
        settings=payload["settings"],
    )


def compare_settings(saved: dict, current: dict) -> list[str]:
    """Human-readable differences between two flattened setting dicts."""
    keys = sorted(set(saved) | set(current))
    return [f"{k}: {saved.get(k)!r} -> {current.get(k)!r}" for k in keys if saved.get(k) != current.get(k)]


class CsvLog:
    """Append-only CSV with a fixed header; on resume, rows past the resumed round are dropped."""

    def __init__(self, path: Path, fields: list[str]):
        """Log to `path` with header `fields`; the file is created on the first append."""
        self.path = Path(path)
        self.fields = fields

    def truncate_after(self, round_: int, key: str = "round") -> None:
        """Drop rows whose `key` is past `round_` (rows of a round that a resume replays)."""
        if not self.path.exists():
            return
        with self.path.open() as f:
            rows = [row for row in csv.DictReader(f) if int(float(row[key])) <= round_]
        with self.path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(rows)

    def append(self, rows: list[dict]) -> None:
        """Append `rows`, writing the header first when the file is new."""
        new = not self.path.exists()
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerows(rows)


def process_memory_gb() -> tuple[float, float]:
    """(this process, this process + its children) resident memory in GB. Linux only; 0 elsewhere."""
    try:
        import psutil

        proc = psutil.Process()
        own = proc.memory_info().rss
        children = sum(c.memory_info().rss for c in proc.children(recursive=True))
        return own / 1024**3, (own + children) / 1024**3
    except Exception:  # noqa: BLE001 - psutil missing or a child exited mid-scan
        try:
            pages = int(Path("/proc/self/statm").read_text().split()[1])
            own = pages * os.sysconf("SC_PAGE_SIZE") / 1024**3
            return own, own
        except Exception:  # noqa: BLE001
            return 0.0, 0.0
