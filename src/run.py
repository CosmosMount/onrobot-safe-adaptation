"""One readable entrypoint for the complete Go2 SQRL workflow."""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

from .config import (
    DEFAULT_DDS_DOMAIN_ID,
    DEFAULT_DDS_INTERFACE,
    DEFAULT_MUJOCO_SCENE,
    PROJECT_ROOT,
    RUN_PRESETS,
    find_unitree_mujoco_root,
)


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


def _artifact_flags(
    command: str,
    checkpoint: str | None,
    qsafe_checkpoint: str | None = None,
) -> list[str]:
    if checkpoint is None and command in ("zero-shot", "finetune"):
        checkpoint_path = (
            PROJECT_ROOT
            / "runs/go2_sqrl/pretrain/isaac_sqrl_height_dr_v1/models"
        )
    elif checkpoint is None and command == "finetune-sac":
        checkpoint_path = (
            PROJECT_ROOT / "runs/go2_sqrl/pretrain/isaac_sac_height_dr_v1/models"
        )
    elif checkpoint is None and command == "isaac-eval":
        checkpoint_path = (
            PROJECT_ROOT
            / "runs/go2_sqrl/pretrain/isaac_sqrl_height_dr_v1/models"
        )
    elif checkpoint is None and command == "isaac-collect-qsafe":
        checkpoint_path = (
            PROJECT_ROOT
            / "runs/go2_sqrl/pretrain/isaac_sac_flat_action_v2_legacy_v1/models"
        )
    elif checkpoint is None and command == "eval":
        checkpoint_path = PROJECT_ROOT / "runs/go2_sqrl/finetune/mujoco/models"
    elif checkpoint is None:
        return []
    else:
        checkpoint_path = Path(checkpoint).expanduser().resolve()

    if command == "isaac-finetune-sac":
        if checkpoint_path.is_file():
            raise ValueError(
                "Isaac gait fine-tuning requires a transfer bundle directory "
                "containing policy.model."
            )
        policy = checkpoint_path / "policy.model"
        if not policy.is_file():
            raise FileNotFoundError(f"Transfer policy not found: {policy}")
        return [f"--algorithm.pretrained_policy_path={policy}"]

    if command == "isaac-collect-qsafe":
        if checkpoint_path.is_file():
            raise ValueError(
                "Universal-QSafe collection requires a models directory "
                "containing the policy.model transfer sidecar."
            )
        policy = checkpoint_path / "policy.model"
        if not policy.is_file():
            raise FileNotFoundError(f"Behavior policy not found: {policy}")
        return [f"--algorithm.pretrained_policy_path={policy}"]

    if command == "isaac-finetune":
        if checkpoint_path.is_file():
            raise ValueError(
                "Isaac universal-QSafe fine-tuning requires an actor models "
                "directory containing policy.model."
            )
        policy = checkpoint_path / "policy.model"
        if not policy.is_file():
            raise FileNotFoundError(f"Transfer policy not found: {policy}")
        if not qsafe_checkpoint:
            raise ValueError("isaac-finetune requires --qsafe-checkpoint <qsafe.model>.")
        qsafe = Path(qsafe_checkpoint).expanduser().resolve()
        if not qsafe.is_file():
            raise FileNotFoundError(f"Universal QSafe checkpoint not found: {qsafe}")
        flags = [
            f"--algorithm.pretrained_policy_path={policy}",
            f"--algorithm.qsafe.checkpoint_path={qsafe}",
        ]
        return flags

    if command in (
        "pretrain",
        "pretrain-sac",
        "isaac-eval",
        "eval",
    ):
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

    if checkpoint_path.is_file():
        raise ValueError(
            "Fine-tune/zero-shot requires an exact transfer bundle directory; "
            "passing one combined .model file would silently mix it with sidecars "
            "from a different training step."
        )
    directory = checkpoint_path
    policy = directory / "policy.model"
    if not policy.exists():
        policy = directory / "policy.msgpack"
    qsafe = (
        Path(qsafe_checkpoint).expanduser().resolve()
        if qsafe_checkpoint
        else directory / "qsafe.model"
    )
    needs_qsafe = command != "finetune-sac"
    if not policy.exists() or (needs_qsafe and not qsafe.exists()):
        raise FileNotFoundError(
            "Transfer requires policy.model"
            + (" and qsafe.model" if needs_qsafe else "")
            + " in "
            f"{directory}. Pass --checkpoint <models-directory>."
        )
    flags = [f"--algorithm.pretrained_policy_path={policy}"]
    if needs_qsafe:
        flags.append(f"--algorithm.qsafe.checkpoint_path={qsafe}")
    return flags


def _deduplicate_config_flags(flags: list[str]) -> list[str]:
    """Apply last-value-wins semantics to ``--section.key=value`` flags."""

    deduplicated: dict[str, str] = {}
    ordered_keys: list[str] = []
    positional: list[str] = []
    for flag in flags:
        if flag.startswith("--") and "=" in flag:
            key = flag.split("=", 1)[0]
            if key not in deduplicated:
                ordered_keys.append(key)
            deduplicated[key] = flag
        else:
            positional.append(flag)
    return [deduplicated[key] for key in ordered_keys] + positional


def _run_rlx(args, remaining: list[str]) -> int:
    if args.command in ("zero-shot", "finetune", "finetune-sac", "eval"):
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
    if args.command in ("zero-shot", "finetune", "finetune-sac", "eval"):
        flags.extend(
            (
                f"--environment.domain_id={args.domain_id}",
                f"--environment.interface={args.interface}",
            )
        )
    flags.extend(
        _artifact_flags(
            args.command,
            args.checkpoint,
            args.qsafe_checkpoint,
        )
    )
    flags.extend(remaining)
    # Presets are defaults.  A flag written explicitly on the command line
    # must win, including the few values (runner mode, implementation and
    # launcher settings) that RL-X reads before absl parses the config dicts.
    # Keeping duplicate ``--section.key=...`` entries made those early reads
    # select the preset's first value while the later config parser selected a
    # different value.  Besides being surprising, that could turn a requested
    # frozen evaluation into training.  Collapse duplicates here using the
    # conventional last-value-wins rule.
    flags = _deduplicate_config_flags(flags)
    if "--algorithm.compile_policy=false" in flags:
        # Short Isaac evaluation/collection processes must not spend minutes
        # starting Inductor workers for modules that are never trained.
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        import torch

        torch._dynamo.config.disable = True
        torch.compile = lambda model, *args, **kwargs: model
        torch.compiler.cudagraph_mark_step_begin = lambda: None
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
            "pretrain-sac",
            "isaac-eval",
            "isaac-collect-qsafe",
            "isaac-finetune",
            "isaac-finetune-sac",
            "zero-shot",
            "finetune",
            "finetune-sac",
            "eval",
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--qsafe-checkpoint")
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
        return _simulator_command(args)
    return _run_rlx(args, remaining)


if __name__ == "__main__":
    raise SystemExit(main())
