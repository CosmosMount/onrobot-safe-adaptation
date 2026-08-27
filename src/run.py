"""One readable entrypoint for the complete Go2 SQRL workflow."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import (
    DEFAULT_DDS_DOMAIN_ID,
    DEFAULT_DDS_INTERFACE,
    DEFAULT_MUJOCO_SCENE,
    PRETRAIN_RUN_NAME,
    PROJECT_ROOT,
    RUN_PRESETS,
    find_unitree_mujoco_root,
)


@contextmanager
def _runtime_lock(kind: str, domain_id: int, interface: str, directory=None):
    """Enforce one simulator/policy owner for a DDS endpoint.

    The lock is owned by the process file descriptor, so stale files after a
    crash are harmless.  Keeping simulator and policy locks separate permits
    exactly one of each while rejecting the duplicate-publisher situation that
    interleaves LowState ticks and invalidates control-window measurements.
    """

    safe_interface = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(interface))
    lock_directory = Path(directory or tempfile.gettempdir())
    lock_path = lock_directory / (
        f"onrobot-safe-adaptation-{kind}-dds{int(domain_id)}-{safe_interface}.lock"
    )
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown"
            raise RuntimeError(
                f"A {kind} process already owns DDS domain {domain_id} on "
                f"{interface!r} (owner pid {owner}). Stop it before starting "
                "another experiment."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        yield lock_path
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _simulator_command(args) -> int:
    root = find_unitree_mujoco_root()
    simulate = root / "simulate"
    executable = simulate / "build" / "unitree_mujoco"
    if not executable.is_file():
        raise FileNotFoundError(
            f"Simulator executable not found at {executable}; build unitree_mujoco first."
        )
    environment = os.environ.copy()
    library_directory = simulate / "mujoco" / "lib"
    existing = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        f"{library_directory}:{existing}" if existing else str(library_directory)
    )
    command = [
        str(executable),
        "-r",
        "go2",
        "-s",
        args.scene,
        "-i",
        str(args.domain_id),
        "-n",
        args.interface,
    ]
    print("Starting:", " ".join(command), flush=True)
    return subprocess.run(command, cwd=simulate, env=environment, check=False).returncode


def _artifact_flags(command: str, checkpoint: str | None) -> list[str]:
    if checkpoint is None and command in ("zero-shot", "finetune"):
        checkpoint_path = (
            PROJECT_ROOT
            / f"runs/go2_sqrl/pretrain/{PRETRAIN_RUN_NAME}/models"
        )
    elif checkpoint is None and command == "isaac-eval":
        checkpoint_path = (
            PROJECT_ROOT
            / f"runs/go2_sqrl/pretrain/{PRETRAIN_RUN_NAME}/models"
        )
    elif checkpoint is None and command == "eval":
        checkpoint_path = PROJECT_ROOT / "runs/go2_sqrl/finetune/mujoco/models"
    elif checkpoint is None:
        return []
    else:
        checkpoint_path = Path(checkpoint).expanduser().resolve()

    if command in ("pretrain", "isaac-eval", "eval"):
        if checkpoint_path.is_dir():
            preferred = checkpoint_path / "final.model"
            if not preferred.exists():
                preferred = checkpoint_path / "best.model"
            checkpoint_path = preferred
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. Pass --checkpoint <file-or-models-directory>."
            )
        return [f"--runner.load_model={checkpoint_path}"]

    directory = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    policy = directory / "policy.model"
    qsafe = directory / "qsafe.model"
    if not policy.exists() or not qsafe.exists():
        raise FileNotFoundError(
            "Fine-tune/zero-shot requires policy.model and qsafe.model in "
            f"{directory}. Pass --checkpoint <models-directory>."
        )
    flags = [
        f"--algorithm.pretrained_policy_path={policy}",
        f"--algorithm.qsafe.checkpoint_path={qsafe}",
    ]
    if command in ("zero-shot", "finetune"):
        task_checkpoint = directory / "final.model"
        if not task_checkpoint.exists():
            raise FileNotFoundError(
                "Target-task transfer requires final.model so the task critics and entropy "
                f"coefficient are transferred with the actor: {task_checkpoint}"
            )
        flags.append(
            f"--algorithm.pretrained_task_checkpoint_path={task_checkpoint}"
        )
    return flags


def _run_rlx(args, remaining: list[str]) -> int:
    if args.command in ("zero-shot", "finetune", "eval"):
        # This must be set before importing JAX. The SDK environment has a host
        # replay/transport boundary and should not reserve nearly all GPU memory
        # merely to probe whether the runtime is available.
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        try:
            importlib.import_module("jax")
        except Exception as exc:
            raise RuntimeError(
                "MuJoCo presets use the Flax/JAX SAC-QSafe implementation, but "
                "JAX cannot be imported by this Python interpreter. Install the "
                "project's pinned MuJoCo dependencies (pip install -e '.[mujoco]') "
                "or run the command from a compatible JAX environment."
            ) from exc
    flags = list(RUN_PRESETS[args.command])
    flags.append(f"--environment.seed={args.seed}")
    if args.command in ("zero-shot", "finetune", "eval"):
        flags.extend(
            (
                f"--environment.domain_id={args.domain_id}",
                f"--environment.interface={args.interface}",
            )
        )
    flags.extend(_artifact_flags(args.command, args.checkpoint))
    flags.extend(remaining)
    sys.argv = [sys.argv[0], *flags]

    from rl_x.runner.runner import Runner

    Runner(implementation_package_names=["rl_x", "src"]).run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.run",
        description="Go2 SQRL pre-training and SDK2/MuJoCo adaptation",
    )
    parser.add_argument(
        "command",
        choices=(
            "sim",
            "pretrain",
            "isaac-eval",
            "zero-shot",
            "finetune",
            "eval",
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--domain-id", type=int, default=DEFAULT_DDS_DOMAIN_ID)
    parser.add_argument("--interface", default=DEFAULT_DDS_INTERFACE)
    parser.add_argument("--scene", default=DEFAULT_MUJOCO_SCENE)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)
    if args.command == "sim":
        if remaining:
            parser.error(f"unknown simulator arguments: {' '.join(remaining)}")
        with _runtime_lock("simulator", args.domain_id, args.interface):
            return _simulator_command(args)
    if args.command in ("zero-shot", "finetune", "eval"):
        with _runtime_lock("policy", args.domain_id, args.interface):
            return _run_rlx(args, remaining)
    return _run_rlx(args, remaining)


if __name__ == "__main__":
    raise SystemExit(main())
