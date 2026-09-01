#!/usr/bin/env python3
"""Collect and retrain the flat action-v1 QSafe v2 without touching SAC."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTORS = ROOT / "configs/flat_qsafe_v2_actors.json"
DEFAULT_DATASET = ROOT / "runs/go2_sqrl/qsafe_datasets/flat_action_v1_v2"
DEFAULT_OUTPUT = ROOT / "runs/go2_sqrl/qsafe_offline/flat_action_v1_v2"


def _run(command):
    print(" ".join(map(str, command)), flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def _statistics(dataset_path):
    from rl_x.algorithms.qsafe.dataset import SafetyTrajectoryDataset

    return SafetyTrajectoryDataset(dataset_path).statistics()


def _data_ready(statistics, minimums):
    return all(
        statistics["by_split"][split][outcome] >= required
        for split, required in minimums.items()
        for outcome in ("fall", "success")
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("collect", "train", "all"), default="all")
    parser.add_argument("--actors", type=Path, default=DEFAULT_ACTORS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes-per-cell", type=int, default=60)
    parser.add_argument("--parallel-envs", type=int, default=20)
    parser.add_argument("--candidate-actions", type=int, default=20)
    # Fast engineering gate. The publication-scale command can raise these to
    # 500/100/100 without changing code or split semantics.
    parser.add_argument("--min-train-outcomes", type=int, default=100)
    parser.add_argument("--min-validation-outcomes", type=int, default=30)
    parser.add_argument("--min-test-outcomes", type=int, default=30)
    parser.add_argument("--updates-per-gamma", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gammas", default="0.9,0.97")
    parser.add_argument("--horizons", default="5,10,25")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if any(
        value < 1
        for value in (
            args.episodes_per_cell,
            args.parallel_envs,
            args.candidate_actions,
            args.min_train_outcomes,
            args.min_validation_outcomes,
            args.min_test_outcomes,
            args.updates_per_gamma,
            args.batch_size,
        )
    ):
        raise ValueError("Episode, environment, outcome, update and batch counts must be positive.")
    actors = args.actors.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.stage in ("collect", "all"):
        command = [
            sys.executable,
            str(ROOT / "tools/collect_universal_qsafe_dataset.py"),
            "--actors",
            str(actors),
            "--dataset",
            str(dataset),
            "--profile",
            "flat_legacy_v1",
            "--episodes-per-run",
            str(args.episodes_per_cell),
            "--parallel-envs",
            str(args.parallel_envs),
            "--candidate-actions",
            str(args.candidate_actions),
        ]
        if args.dry_run:
            command.append("--dry-run")
        code = _run(command)
        if code:
            return code
        if args.dry_run:
            return 0

    if not (dataset / "manifest.json").is_file():
        raise FileNotFoundError(
            f"Dataset manifest not found: {dataset / 'manifest.json'}"
        )
    statistics = _statistics(dataset)
    print(json.dumps(statistics, indent=2, sort_keys=True), flush=True)
    minimums = {
        "train": int(args.min_train_outcomes),
        "validation": int(args.min_validation_outcomes),
        "test": int(args.min_test_outcomes),
    }
    if not _data_ready(statistics, minimums):
        print(
            "[HARD STOP] Dataset does not yet contain enough fall and success "
            f"episodes for {minimums}. Re-run --stage collect with a larger "
            "--episodes-per-cell; collection resumes existing cells.",
            file=sys.stderr,
            flush=True,
        )
        return 3
    if args.stage == "collect":
        return 0

    command = [
        sys.executable,
        "-m",
        "rl_x.algorithms.qsafe.offline_train",
        "--dataset",
        str(dataset),
        "--output",
        str(output),
        "--updates",
        str(args.updates_per_gamma),
        "--batch-size",
        str(args.batch_size),
        "--gammas",
        args.gammas,
        "--horizons",
        args.horizons,
        "--min-train-outcomes",
        str(args.min_train_outcomes),
        "--min-validation-outcomes",
        str(args.min_validation_outcomes),
        "--min-test-outcomes",
        str(args.min_test_outcomes),
        "--seed",
        str(args.seed),
        "--checkpoint-name",
        "qsafe_flat_action_v1_v2.model",
    ]
    return _run(command)


if __name__ == "__main__":
    raise SystemExit(main())
