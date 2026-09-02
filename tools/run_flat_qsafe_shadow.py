#!/usr/bin/env python3
"""Run frozen QSafe beside, but never in control of, flat SAC fine-tuning."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_flat_safe_adaptation import (
    DEFAULT_ACTOR,
    DEFAULT_QSAFE,
    DEFAULT_SCENE,
    ROOT,
    _shared_train_flags,
    _start_simulator,
    _stop_simulator,
    _validate_flat_qsafe_metadata,
    _validate_legacy_flat_actor_manifest,
)


def _command(args, qsafe_metadata: dict) -> list[str]:
    shared = SimpleNamespace(
        actor=args.actor,
        qsafe=args.qsafe,
        steps=args.steps,
        checkpoint_frequency=args.steps,
        logging_frequency=args.logging_frequency,
        qsafe_version=int(qsafe_metadata.get("qsafe_version", 1)),
        qsafe_gamma=float(qsafe_metadata["gamma"]),
        qsafe_epsilon=float(qsafe_metadata["epsilon"]),
        interface=args.interface,
    )
    output = (
        ROOT
        / "runs/go2_sqrl/finetune"
        / args.run_name
        / "qsafe_shadow.npz"
    )
    return [
        sys.executable,
        "-m",
        "src.run",
        "finetune",
        *_shared_train_flags(
            shared, args.seed, args.domain_id, args.run_name
        ),
        "--algorithm.qsafe.enabled=false",
        "--algorithm.qsafe.shadow_enabled=true",
        f"--algorithm.qsafe.shadow_output_path={output}",
    ]


def _load_and_validate(args) -> dict:
    args.actor = args.actor.expanduser().resolve()
    args.qsafe = args.qsafe.expanduser().resolve()
    args.scene = args.scene.expanduser().resolve()
    policy_path = args.actor / "policy.model"
    for path in (policy_path, args.qsafe, args.scene):
        if not path.is_file():
            raise FileNotFoundError(path)
    actor = torch.load(policy_path, map_location="cpu", weights_only=False)
    _validate_legacy_flat_actor_manifest(dict(actor["environment_manifest"]))
    qsafe = torch.load(args.qsafe, map_location="cpu", weights_only=False)
    metadata = dict(qsafe["metadata"])
    _validate_flat_qsafe_metadata(
        metadata,
        dict(qsafe.get("calibration_report", {})),
        allow_diagnostic_near_pass=args.allow_diagnostic_qsafe_near_pass,
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--qsafe", type=Path, default=DEFAULT_QSAFE)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--run-name", default="flat_qsafe_shadow_s0_10k")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domain-id", type=int, default=41)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--logging-frequency", type=int, default=1_000)
    parser.add_argument("--simulator-startup-seconds", type=float, default=5.0)
    parser.add_argument("--no-start-simulator", action="store_true")
    parser.add_argument("--allow-diagnostic-qsafe-near-pass", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    metadata = _load_and_validate(args)
    run_dir = ROOT / "runs/go2_sqrl/finetune" / args.run_name
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    command = _command(args, metadata)
    simulator = simulator_log = None
    diagnostic_dir = ROOT / "runs/go2_sqrl/diagnostics" / args.run_name
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    return_code = None
    try:
        if not args.no_start_simulator:
            simulator, simulator_log, _ = _start_simulator(
                args,
                args.domain_id,
                diagnostic_dir / "simulator.log",
                "shadow",
            )
        print("[shadow] Running: " + " ".join(command), flush=True)
        with (diagnostic_dir / "train.log").open(
            "w", buffering=1, encoding="utf-8"
        ) as log_file:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                if (
                    "steps/nr_env_steps" in line
                    or " WARNING " in line
                    or " ERROR " in line
                    or "Traceback" in line
                    or "QSafe shadow diagnostic" in line
                ):
                    print("[shadow] " + line.rstrip(), flush=True)
            return_code = process.wait()
    finally:
        _stop_simulator(simulator, simulator_log)

    report_path = run_dir / "qsafe_shadow.report.json"
    if not report_path.is_file():
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
        raise FileNotFoundError(f"Shadow report missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["run_completion"] = (
        "complete" if return_code == 0 else f"runtime_stopped_exit_{return_code}"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if return_code == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
