#!/usr/bin/env python3
"""Convert complete MuJoCo shadow episodes into a diagnostic QSafe dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_x.algorithms.qsafe.dataset import SafetyTrajectoryDatasetWriter


DEFAULT_CONTRACT = (
    ROOT / "runs/go2_sqrl/qsafe_datasets/flat_action_v1_v2/manifest.json"
)


def _split_episode_keys(
    arrays: dict[str, np.ndarray],
) -> dict[str, list[tuple[int, int]]]:
    keys = np.stack((arrays["env_index"], arrays["episode_id"]), axis=1)
    outcomes: dict[bool, list[tuple[int, int]]] = {False: [], True: []}
    for raw_key in np.unique(keys, axis=0):
        key = (int(raw_key[0]), int(raw_key[1]))
        indices = np.flatnonzero(np.all(keys == raw_key, axis=1))
        if not bool(arrays["done"][indices[-1]]):
            continue
        fell = bool(np.any(arrays["failure"][indices]))
        outcomes[fell].append(key)
    if any(len(values) < 3 for values in outcomes.values()):
        raise RuntimeError(
            "Target diagnostic requires at least three complete fall and three "
            "complete non-fall episodes."
        )

    result = {"train": [], "validation": [], "test": []}
    for values in outcomes.values():
        values = sorted(values)
        nr_validation = max(1, len(values) // 5)
        nr_test = max(1, len(values) // 5)
        nr_train = len(values) - nr_validation - nr_test
        result["train"].extend(values[:nr_train])
        result["validation"].extend(values[nr_train : nr_train + nr_validation])
        result["test"].extend(values[nr_train + nr_validation :])
    return result


def convert(shadow: Path, destination: Path, contract_manifest: Path, actor_id: str):
    required = {
        "env_index",
        "episode_id",
        "state",
        "next_state",
        "applied_action",
        "candidate_actions",
        "failure",
        "done",
    }
    with np.load(shadow) as archive:
        arrays = {key: archive[key] for key in archive.files}
    missing = required.difference(arrays)
    if missing:
        raise ValueError(
            f"Shadow archive is not training-complete; missing {sorted(missing)}"
        )
    if destination.exists():
        raise FileExistsError(
            f"Refusing to append to existing dataset: {destination}"
        )
    contract_payload = json.loads(contract_manifest.read_text(encoding="utf-8"))
    writer = SafetyTrajectoryDatasetWriter(destination, contract_payload["contract"])
    split_keys = _split_episode_keys(arrays)
    keys = np.stack((arrays["env_index"], arrays["episode_id"]), axis=1)
    counts = {}
    for split, episode_keys in split_keys.items():
        for env_index, episode_id in episode_keys:
            indices = np.flatnonzero(
                (keys[:, 0] == env_index) & (keys[:, 1] == episode_id)
            )
            failures = arrays["failure"][indices].astype(bool)
            done = arrays["done"][indices].astype(bool)
            terminations = failures.copy()
            truncations = done & ~terminations
            trajectory = list(
                zip(
                    arrays["state"][indices],
                    arrays["next_state"][indices],
                    arrays["applied_action"][indices],
                    failures,
                    terminations,
                    truncations,
                )
            )
            writer.append(
                trajectory,
                actor_id=actor_id,
                map_seed=0,
                episode_id=f"{env_index}:{episode_id}",
                split=split,
                terrain="mujoco_flat_target_diagnostic",
                action_noise=0.0,
                success=not bool(np.any(failures)),
                candidate_actions=arrays["candidate_actions"][indices],
            )
        counts[split] = len(episode_keys)
    return counts


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--contract-manifest", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--actor-id", default="mujoco_fixed_target_actor")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    counts = convert(
        args.shadow.expanduser().resolve(),
        args.dataset.expanduser().resolve(),
        args.contract_manifest.expanduser().resolve(),
        args.actor_id,
    )
    print(json.dumps({"dataset": str(args.dataset), "episodes": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
