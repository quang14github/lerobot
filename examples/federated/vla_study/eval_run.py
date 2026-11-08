"""Evaluate one saved checkpoint of the federated VLA study on its benchmark's tasks.

Every arm, round and policy is evaluated with the same protocol, so that results are comparable:
the same tasks, `--n-episodes` episodes each, one environment at a time (`--batch-size 1`), and
the same `--seed`. LIBERO then replays the same fixed initial states (states 0..n-1 of each task)
and Meta-World seeds its object positions from the episode seed, so every arm meets the same
starting states. Both are recorded in result.json, and collect_results.py refuses to mix protocols.

  python examples/federated/vla_study/eval_run.py \
      --checkpoint <run>/train/checkpoints/000200/pretrained_model --benchmark libero_object \
      --out <run>/eval/round_0200 --device cuda

Writes <out>/eval_info.json (LeRobot's full eval output) and <out>/result.json (per-task success
rates, timing and protocol). Exit code 0 only when both were written.
"""

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path


def _set_render_backend(device: str, benchmark: str) -> None:
    """Pick a MuJoCo backend before any simulator import, unless the caller chose one.

    Headless Linux: EGL on a GPU, OSMesa on a CPU, for both simulators. WSL has no headless EGL
    device, so it renders through its WSLg display instead, and the two simulators need different
    names for that: LIBERO (robosuite) only accepts `glx`, Meta-World (gymnasium) only `glfw`.
    """
    if os.environ.get("MUJOCO_GL"):
        return
    version = Path("/proc/version")
    wsl = version.exists() and "microsoft" in version.read_text().lower()
    if wsl:
        os.environ["MUJOCO_GL"] = "glx" if benchmark.startswith("libero") else "glfw"
        # Render on the NVIDIA GPU through WSL's d3d12 bridge rather than on the CPU.
        os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
        os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")
    else:
        os.environ["MUJOCO_GL"] = "egl" if device.startswith("cuda") else "osmesa"
    if os.environ["MUJOCO_GL"] in ("egl", "osmesa"):
        os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="a checkpoint's pretrained_model directory")
    ap.add_argument(
        "--benchmark", required=True, help="libero_object | libero_goal | libero_spatial | metaworld_mt10"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-episodes", type=int, default=10, help="episodes per task")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--tasks", default="", help="comma-separated task keys (default: all ten)")
    ap.add_argument("--videos", type=int, default=0, help="episodes to render to video")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast for inference (default float32)")
    args = ap.parse_args()

    _set_render_backend(args.device, args.benchmark)
    if args.device == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))

    import contextlib

    import torch

    from lerobot.common.train_utils import verify_pretrained_weights
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
    from lerobot.envs.configs import LiberoEnvConfig, MetaworldEnvConfig
    from lerobot.federated.tasks import BENCHMARKS
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy_all
    from lerobot.utils.random_utils import set_seed

    if args.device == "cpu":
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

    bench = BENCHMARKS[args.benchmark]
    keys = [k for k in args.tasks.split(",") if k] or list(bench.task_keys)
    if bench.env_type == "libero":
        env_cfg = LiberoEnvConfig(task=bench.suite, task_ids=[int(k.rsplit("_", 1)[1]) for k in keys])
    else:
        env_cfg = MetaworldEnvConfig(task=",".join(keys))

    ckpt = Path(args.checkpoint).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policy_cfg = PreTrainedConfig.from_pretrained(ckpt)
    policy_cfg.pretrained_path = ckpt
    policy_cfg.device = args.device
    policy_cfg.n_action_steps = args.n_action_steps

    set_seed(args.seed)
    start = time.time()
    envs = make_env(env_cfg, n_envs=args.batch_size, use_async_envs=False)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    adapter_config = ckpt / "adapter_config.json"
    if adapter_config.exists():
        base = json.loads(adapter_config.read_text())["base_model_name_or_path"]
        verify_pretrained_weights(policy, base)
    else:
        verify_pretrained_weights(policy, ckpt)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=ckpt,
        preprocessor_overrides={
            "device_processor": {"device": str(policy_cfg.device)},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    load_s = time.time() - start

    amp = torch.autocast(device_type=torch.device(args.device).type) if args.amp else contextlib.nullcontext()
    rollout_start = time.time()
    with torch.no_grad(), amp:
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=args.n_episodes,
            max_episodes_rendered=args.videos,
            videos_dir=out / "videos" if args.videos else None,
            start_seed=args.seed,
            max_parallel_tasks=1,
        )
    rollout_s = time.time() - rollout_start
    close_envs(envs)

    tasks = {}
    for entry in info.get("per_task") or []:
        successes = entry["metrics"].get("successes") or []
        if successes:
            key = bench.eval_task_key(entry["task_group"], entry["task_id"])
            tasks[key] = {
                "success_rate": 100.0 * sum(successes) / len(successes),
                "n_episodes": len(successes),
            }
    missing = sorted(set(keys) - set(tasks))
    if missing:
        print(f"!! no results for tasks {missing}", file=sys.stderr)
        return 1
    n = sum(t["n_episodes"] for t in tasks.values())
    result = {
        "checkpoint": str(ckpt),
        "benchmark": args.benchmark,
        "tasks": tasks,
        "overall": {
            "success_rate": sum(t["success_rate"] * t["n_episodes"] for t in tasks.values()) / n,
            "n_episodes": n,
        },
        "protocol": {
            "n_episodes_per_task": args.n_episodes,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "n_action_steps": args.n_action_steps,
            "amp": args.amp,
        },
        "timing": {
            "load_s": round(load_s, 1),
            "rollout_s": round(rollout_s, 1),
            "episodes_per_hour": round(n / rollout_s * 3600, 1),
        },
        "device": args.device,
        "gpu": torch.cuda.get_device_name()
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else None,
        "cpu_threads": torch.get_num_threads(),
        "render_backend": os.environ.get("MUJOCO_GL"),
        "host": socket.gethostname(),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "eval_info.json").write_text(json.dumps(info, indent=1, default=str))
    tmp = out / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(out / "result.json")
    print(
        f"{args.benchmark} {ckpt.parent.name}: {result['overall']['success_rate']:.1f}% over {n} episodes, "
        f"{result['timing']['episodes_per_hour']:.0f} episodes/hour on {args.device}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
