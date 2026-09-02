#!/usr/bin/env python3
"""Aggregate non-intervening flat QSafe diagnostics across seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_x.algorithms.qsafe.shadow_diagnostics import build_shadow_report


def aggregate(paths: list[Path], epsilon: float) -> dict:
    per_seed = []
    arrays_by_key: dict[str, list[np.ndarray]] = {}
    env_offset = 0
    for seed_index, path in enumerate(paths):
        with np.load(path) as source:
            arrays = {key: source[key] for key in source.files}
        local_envs = np.asarray(arrays["env_index"], dtype=np.int64)
        arrays["env_index"] = local_envs + env_offset
        env_offset += int(np.max(local_envs)) + 1
        arrays["seed_index"] = np.full(
            local_envs.shape, seed_index, dtype=np.int32
        )
        for key, value in arrays.items():
            arrays_by_key.setdefault(key, []).append(value)
        seed_report = build_shadow_report(arrays, epsilon)
        per_seed.append(
            {
                "path": str(path),
                "transitions": seed_report["transitions"],
                "fall_episodes": seed_report["fall_episodes"],
                "complete_safe_episodes": seed_report["complete_safe_episodes"],
                "fallback_fraction": seed_report["candidate_pool"][
                    "fallback_fraction"
                ],
                "intervention_opportunity_fraction": seed_report[
                    "candidate_pool"
                ]["intervention_opportunity_fraction"],
            }
        )
    merged = {
        key: np.concatenate(parts, axis=0)
        for key, parts in arrays_by_key.items()
    }
    return {
        "inputs": per_seed,
        "aggregate": build_shadow_report(merged, epsilon),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = aggregate([path.resolve() for path in args.paths], args.epsilon)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
