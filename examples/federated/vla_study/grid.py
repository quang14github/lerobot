"""The grid of the federated SmolVLA fine-tuning study, and the commands that train each run.

Settings are environment variables, in the style of examples/federated/noniid.sh. The defaults
reproduce the planned study:

  ROUNDS=200 LOCAL_STEPS=250 NUM_CLIENTS=5 BATCH_SIZE=64 SAVE_EVERY=20
  SEEDS="1000 2000 3000" PARTITION_SEEDS="1 2 3"     (paired: seed 1000 uses partition seed 1, ...)
  BENCHMARKS="libero_object libero_goal libero_spatial metaworld_mt10"
  RUN_LORA=1 RUN_FULL=1 RUN_FEDERATED=1 RUN_CENTRALIZED=1          (switches)
  CLIENT_WEIGHTING=uniform CLIENT_OPT_STATE=reset NUM_WORKERS=3 CEN_NUM_WORKERS=12
  DEVICE=cuda MIXED_PRECISION=bf16
  EVAL_EPISODES=10 EVAL_SEED=1000 EVAL_DEVICE=cuda
  STUDY_ROOT=outputs/fedvla TAG= WANDB_PROJECT=

Run identity (no timestamps, so rerunning a command finds and continues the same run):

  $STUDY_ROOT/studies/<bench>_smolvla_<mode>_<partition>_<K>c_r<R>_ls<L>_bs<B>_<weighting>[_TAG]/
      fed-<mode>-<partition>-s<seed>-p<partition seed>/
  $STUDY_ROOT/studies/<bench>_smolvla_<mode>_pooled_r<R>x<L>x<K>_bs<B>[_TAG]/
      cen-<mode>-s<seed>/          (one per seed: the reference for every partition)
  $STUDY_ROOT/baselines/<bench>_smolvla/   (the untrained pretrained model, evaluated once)
"""

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

from lerobot.federated import vla_study  # noqa: E402
from lerobot.federated.tasks import BENCHMARKS, BenchmarkSpec  # noqa: E402

POLICY = vla_study.POLICY

# Partition name -> (partition.strategy, extra flags).
PARTITIONS = {
    "taskdisjoint": ("task", []),
    "iid": ("iid", []),
    "sharedcore": ("shared_core", ["--partition.num_shared_tasks=5"]),
}
PHASE_PARTITIONS = {1: ("taskdisjoint",), 2: ("iid", "sharedcore")}

WEIGHTING_SHORT = {"uniform": "uni", "frames": "frm", "episodes": "epi"}

# Optimizer steps per second at batch 64, used only for time estimates (throughput.py) until
# `study.py bench-workers` measures them. SmolVLA with only its action expert trainable ran at
# 1.9 (LIBERO) / 2.4 (Meta-World) steps/s on an A100 80GB; LoRA and full backpropagate through the
# whole VLM, hence the lower guesses.
DEFAULT_SPS = {"lora": 1.2, "full": 1.0}


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _flag(name: str, default: str = "1") -> bool:
    return _env(name, default) not in ("0", "false", "no", "")


def _ints(name: str, default: str) -> tuple[int, ...]:
    return tuple(int(x) for x in _env(name, default).split())


@dataclass(frozen=True)
class Settings:
    study_root: Path
    rounds: int
    local_steps: int
    num_clients: int
    batch_size: int
    save_every: int
    seeds: tuple[int, ...]
    partition_seeds: tuple[int, ...]
    benchmarks: tuple[str, ...]
    run_lora: bool
    run_full: bool
    run_federated: bool
    run_centralized: bool
    client_weighting: str
    client_opt_state: str
    num_workers: int
    cen_num_workers: int
    device: str
    mixed_precision: str
    eval_episodes: int
    eval_seed: int
    eval_batch_size: int
    eval_device: str
    tag: str
    wandb_project: str
    wandb_entity: str

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls(
            study_root=Path(_env("STUDY_ROOT", str(REPO_ROOT / "outputs" / "fedvla"))).resolve(),
            rounds=int(_env("ROUNDS", "200")),
            local_steps=int(_env("LOCAL_STEPS", "250")),
            num_clients=int(_env("NUM_CLIENTS", "5")),
            batch_size=int(_env("BATCH_SIZE", "64")),
            save_every=int(_env("SAVE_EVERY", "20")),
            seeds=_ints("SEEDS", "1000 2000 3000"),
            partition_seeds=_ints("PARTITION_SEEDS", "1 2 3"),
            benchmarks=tuple(_env("BENCHMARKS", " ".join(BENCHMARKS)).split()),
            run_lora=_flag("RUN_LORA"),
            run_full=_flag("RUN_FULL"),
            run_federated=_flag("RUN_FEDERATED"),
            run_centralized=_flag("RUN_CENTRALIZED"),
            client_weighting=_env("CLIENT_WEIGHTING", "uniform"),
            client_opt_state=_env("CLIENT_OPT_STATE", "reset"),
            num_workers=int(_env("NUM_WORKERS", "3")),
            cen_num_workers=int(_env("CEN_NUM_WORKERS", "12")),
            device=_env("DEVICE", "cuda"),
            mixed_precision=_env("MIXED_PRECISION", "bf16"),
            eval_episodes=int(_env("EVAL_EPISODES", "10")),
            eval_seed=int(_env("EVAL_SEED", "1000")),
            eval_batch_size=int(_env("EVAL_BATCH_SIZE", "1")),
            eval_device=_env("EVAL_DEVICE", "cuda"),
            tag="".join(c if c.isalnum() or c in ".-" else "-" for c in _env("TAG", "")),
            wandb_project=_env("WANDB_PROJECT", ""),
            wandb_entity=_env("WANDB_ENTITY", ""),
        )
        s.check()
        return s

    def check(self) -> None:
        if len(self.seeds) != len(self.partition_seeds):
            raise ValueError(
                f"SEEDS {self.seeds} and PARTITION_SEEDS {self.partition_seeds} must pair up one to one."
            )
        unknown = [b for b in self.benchmarks if b not in BENCHMARKS]
        if unknown:
            raise ValueError(f"Unknown BENCHMARKS {unknown}; choose from {list(BENCHMARKS)}")
        if self.client_opt_state not in ("reset", "keep"):
            raise ValueError("CLIENT_OPT_STATE must be reset or keep")
        if self.rounds % self.save_every:
            raise ValueError(
                "ROUNDS must be a multiple of SAVE_EVERY so the final round is on the checkpoint grid"
            )

    def modes(self) -> list[str]:
        return [m for m, on in (("lora", self.run_lora), ("full", self.run_full)) if on]

    def as_env(self) -> dict[str, str]:
        out = {}
        for key, value in asdict(self).items():
            out[key.upper()] = " ".join(map(str, value)) if isinstance(value, tuple) else str(value)
        return out


@dataclass(frozen=True)
class RunSpec:
    """One training run: a (mode, benchmark, arm, partition, seed) cell of the grid."""

    mode: str
    bench: str
    arm: str  # federated | centralized
    partition: str | None
    seed: int
    partition_seed: int | None

    @property
    def policy(self) -> str:
        return POLICY

    @property
    def benchmark(self) -> BenchmarkSpec:
        return BENCHMARKS[self.bench]

    def study_id(self, s: Settings) -> str:
        head = f"{self.benchmark.short}_{POLICY}_{self.mode}"
        if self.arm == "federated":
            name = (
                f"{head}_{self.partition}_{s.num_clients}c_r{s.rounds}_ls{s.local_steps}_bs{s.batch_size}"
                f"_{WEIGHTING_SHORT.get(s.client_weighting, s.client_weighting)}"
            )
            if s.client_opt_state == "keep":
                name += "_keepopt"
        else:
            name = f"{head}_pooled_r{s.rounds}x{s.local_steps}x{s.num_clients}_bs{s.batch_size}"
        return name + (f"_{s.tag}" if s.tag else "")

    def run_id(self) -> str:
        if self.arm == "federated":
            return f"fed-{self.mode}-{self.partition}-s{self.seed}-p{self.partition_seed}"
        return f"cen-{self.mode}-s{self.seed}"

    def study_dir(self, s: Settings) -> Path:
        return s.study_root / "studies" / self.study_id(s)

    def run_dir(self, s: Settings) -> Path:
        return self.study_dir(s) / self.run_id()

    def total_steps(self, s: Settings) -> int:
        return s.rounds * s.local_steps * s.num_clients

    def round_steps(self, s: Settings) -> int:
        """Optimizer steps (of this arm's checkpoint numbering) per round-equivalent."""
        return 1 if self.arm == "federated" else s.local_steps * s.num_clients

    def label(self) -> str:
        return f"{self.mode}/{self.bench}/{self.run_id()}"

    def to_json(self) -> dict:
        return asdict(self)


def baseline_dir(s: Settings, bench: str) -> Path:
    return s.study_root / "baselines" / f"{BENCHMARKS[bench].short}_{POLICY}"


def prep_dir(s: Settings, bench: str) -> Path:
    return s.study_root / "prep" / bench


def task_map_path(s: Settings, bench: str) -> Path:
    return prep_dir(s, bench) / "task_map.json"


def benchmark_episodes(s: Settings, bench: str) -> list[int]:
    path = task_map_path(s, bench)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: run `study.py prepare` first.")
    return sorted(int(ep) for ep in json.loads(path.read_text())["episode_tasks"])


def runs_for(mode: str, phase: int, s: Settings) -> list[RunSpec]:
    """Runs of one mode and phase, in the order that finishes complete comparisons first:
    for each seed, for each benchmark, the federated run(s), then the centralized reference."""
    runs = []
    for seed, partition_seed in zip(s.seeds, s.partition_seeds, strict=True):
        for bench in s.benchmarks:
            if s.run_federated:
                for partition in PHASE_PARTITIONS[phase]:
                    runs.append(RunSpec(mode, bench, "federated", partition, seed, partition_seed))
            if s.run_centralized and phase == 1:
                runs.append(RunSpec(mode, bench, "centralized", None, seed, None))
    return runs


def _apply_overrides(args: list[str], overrides: list[str] | None) -> list[str]:
    """Replace flags of `args` by the same flags in `overrides` (and append the new ones)."""
    if not overrides:
        return args
    keys = {o.split("=", 1)[0] for o in overrides}
    return [a for a in args if a.split("=", 1)[0] not in keys] + list(overrides)


def _wandb_args(s: Settings) -> list[str]:
    if not s.wandb_project:
        return ["--wandb.enable=false"]
    args = ["--wandb.enable=true", f"--wandb.project={s.wandb_project}", "--wandb.disable_artifact=true"]
    if s.wandb_entity:
        args.append(f"--wandb.entity={s.wandb_entity}")
    return args


def _dataset_args(s: Settings, bench_name: str) -> list[str]:
    eps = benchmark_episodes(s, bench_name)
    return [
        f"--dataset.repo_id={BENCHMARKS[bench_name].dataset}",
        f"--dataset.episodes=[{','.join(map(str, eps))}]",
    ]


def federated_command(
    spec: RunSpec, s: Settings, output_dir: Path, overrides: list[str] | None = None
) -> list[str]:
    strategy, extra = PARTITIONS[spec.partition]
    args = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_fl_train",
        *_dataset_args(s, spec.bench),
        f"--partition.strategy={strategy}",
        f"--partition.num_clients={s.num_clients}",
        f"--partition.seed={spec.partition_seed}",
        f"--partition.task_map={task_map_path(s, spec.bench)}",
        *extra,
        f"--fl.rounds={s.rounds}",
        f"--fl.local_steps={s.local_steps}",
        f"--fl.save_freq_rounds={s.save_every}",
        f"--fl.client_weighting={s.client_weighting}",
        f"--fl.reset_optimizer_each_round={'true' if s.client_opt_state == 'reset' else 'false'}",
        "--fl.auto_resume=true",
        "--fl.resume_save_freq_rounds=1",
        "--server.type=fedavg",
        f"--batch_size={s.batch_size}",
        f"--num_workers={s.num_workers}",
        f"--seed={spec.seed}",
        f"--accelerator.mixed_precision={s.mixed_precision}",
        f"--output_dir={output_dir}",
        f"--job_name={spec.study_id(s)}-{spec.run_id()}",
        *vla_study.policy_args(spec.mode, device=s.device),
        *vla_study.peft_args(spec.mode),
        *vla_study.scheduler_args(s.rounds, s.local_steps, s.num_clients, "federated"),
        *_wandb_args(s),
    ]
    return _apply_overrides(args, overrides)


def centralized_command(
    spec: RunSpec, s: Settings, output_dir: Path, *, resume: bool, overrides: list[str] | None = None
) -> list[str]:
    head = [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    if resume:
        config = output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        return _apply_overrides(
            [*head, "--resume=true", f"--config_path={config}", f"--output_dir={output_dir}"], overrides
        )
    per_round = s.local_steps * s.num_clients
    args = [
        *head,
        *_dataset_args(s, spec.bench),
        f"--steps={s.rounds * per_round}",
        f"--save_freq={s.save_every * per_round}",
        f"--batch_size={s.batch_size}",
        f"--num_workers={s.cen_num_workers}",
        f"--seed={spec.seed}",
        f"--accelerator.mixed_precision={s.mixed_precision}",
        f"--output_dir={output_dir}",
        f"--job_name={spec.study_id(s)}-{spec.run_id()}",
        "--checkpoint_on_signal=true",
        "--prune_checkpoints=true",
        "--env_eval_freq=0",
        "--log_freq=250",
        *vla_study.policy_args(spec.mode, device=s.device),
        *vla_study.peft_args(spec.mode),
        *vla_study.scheduler_args(s.rounds, s.local_steps, s.num_clients, "centralized"),
        *_wandb_args(s),
    ]
    return _apply_overrides(args, overrides)


def baseline_command(bench: str, s: Settings, output_dir: Path, device: str) -> list[str]:
    """Build the untrained model (pretrained weights + this benchmark's normalization) as round 0.

    Built in LoRA mode: its zero-initialized adapters leave the pretrained model unchanged, and the
    checkpoint holds only the adapters, so no copy of the base model is written.
    """
    spec = RunSpec("lora", bench, "federated", "iid", s.seeds[0], s.partition_seeds[0])
    return federated_command(
        spec,
        s,
        output_dir,
        overrides=[
            "--fl.rounds=0",
            "--fl.save_initial_checkpoint=true",
            "--num_workers=0",
            f"--policy.device={device}",
        ],
    )
