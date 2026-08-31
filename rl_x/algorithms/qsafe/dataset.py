"""Disk-backed, trajectory-atomic datasets for universal QSafe training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


DATASET_VERSION = 2
SPLITS = ("train", "validation", "test")


def deterministic_split(actor_id, map_seed, *, test_actors=(), validation_actors=()):
    """Assign whole actors first, then whole actor/map groups deterministically."""

    actor_id = str(actor_id)
    if actor_id in set(map(str, test_actors)):
        return "test"
    if actor_id in set(map(str, validation_actors)):
        return "validation"
    digest = hashlib.sha256(f"{actor_id}:{int(map_seed)}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def _trajectory_arrays(trajectory):
    if not trajectory:
        raise ValueError("Cannot save an empty safety trajectory.")
    fields = list(zip(*trajectory))
    if len(fields) != 6:
        raise ValueError("Safety trajectories must have six transition fields.")
    states, next_states, actions, failures, terminations, truncations = fields
    arrays = {
        "states": np.asarray(states, dtype=np.float32),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "failures": np.asarray(failures, dtype=np.float32).reshape(-1),
        "terminations": np.asarray(terminations, dtype=np.float32).reshape(-1),
        "truncations": np.asarray(truncations, dtype=np.float32).reshape(-1),
    }
    length = arrays["states"].shape[0]
    if any(value.shape[0] != length for value in arrays.values()):
        raise ValueError("Trajectory fields have inconsistent lengths.")
    if not np.all((arrays["failures"] == 0) | (arrays["failures"] == 1)):
        raise ValueError("Failure labels must be binary.")
    done = arrays["terminations"].astype(bool) | arrays["truncations"].astype(bool)
    if not done[-1] or np.any(done[:-1]):
        raise ValueError("Only the final transition may terminate a trajectory.")
    if np.any(arrays["failures"].astype(bool) & ~arrays["terminations"].astype(bool)):
        raise ValueError("Every failure label must also terminate the episode.")
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("Trajectory contains NaN or infinity.")
    next_actions = np.concatenate(
        [arrays["actions"][1:], arrays["actions"][-1:]], axis=0
    )
    arrays["next_actions"] = next_actions
    return arrays


class SafetyTrajectoryDatasetWriter:
    """Append complete episodes without retaining only a recent replay window."""

    def __init__(self, directory, contract):
        self.directory = Path(directory).resolve()
        self.trajectory_directory = self.directory / "trajectories"
        self.trajectory_directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "manifest.json"
        # JSON-normalize tuples and NumPy scalars before either comparison or
        # persistence, so separate collection processes see one exact contract.
        contract = json.loads(json.dumps(contract))
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest["contract"] != contract:
                raise ValueError("Dataset contract does not match the existing dataset.")
        else:
            self.manifest = {
                "dataset_version": DATASET_VERSION,
                "contract": contract,
                "trajectories": [],
            }
            self._flush()

    def _flush(self):
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(self.manifest_path)

    def append(
        self,
        trajectory,
        *,
        actor_id,
        map_seed,
        episode_id,
        split,
        terrain,
        action_noise,
        success=False,
        stuck=False,
        candidate_actions=None,
        next_actions=None,
    ):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}.")
        arrays = _trajectory_arrays(trajectory)
        if next_actions is not None:
            next_actions = np.asarray(next_actions, dtype=np.float32)
            if next_actions.shape != arrays["actions"].shape:
                raise ValueError(
                    "next_actions must match the executed action array shape."
                )
            if not np.all(np.isfinite(next_actions)):
                raise ValueError("Next actions contain NaN or infinity.")
            arrays["next_actions"] = next_actions
        if candidate_actions is not None:
            candidates = np.asarray(candidate_actions, dtype=np.float32)
            expected_prefix = (arrays["states"].shape[0],)
            if candidates.ndim != 3 or candidates.shape[:1] != expected_prefix:
                raise ValueError(
                    "candidate_actions must have shape [trajectory, candidates, action]."
                )
            if candidates.shape[-1] != arrays["actions"].shape[-1]:
                raise ValueError("Candidate and executed action dimensions differ.")
            if not np.all(np.isfinite(candidates)):
                raise ValueError("Candidate actions contain NaN or infinity.")
            arrays["candidate_actions"] = candidates
        identity = f"{actor_id}:{int(map_seed)}:{episode_id}"
        trajectory_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
        if any(
            item["trajectory_id"] == trajectory_id
            for item in self.manifest["trajectories"]
        ):
            raise ValueError(f"Duplicate trajectory identity: {identity}")
        file_name = f"{trajectory_id}.npz"
        destination = self.trajectory_directory / file_name
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as output:
            np.savez_compressed(output, **arrays)
        temporary.replace(destination)
        item = {
            "trajectory_id": trajectory_id,
            "file": f"trajectories/{file_name}",
            "actor_id": str(actor_id),
            "map_seed": int(map_seed),
            "episode_id": str(episode_id),
            "split": split,
            "terrain": str(terrain),
            "action_noise": float(action_noise),
            "candidate_actions": bool(candidate_actions is not None),
            "length": int(arrays["states"].shape[0]),
            "fall": bool(np.any(arrays["failures"])),
            "success": bool(success),
            # Stuck remains a safe label and is diagnostic metadata only.
            "stuck": bool(stuck),
        }
        self.manifest["trajectories"].append(item)
        self._flush()
        return item


class SafetyTrajectoryDataset:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads(
            (self.directory / "manifest.json").read_text(encoding="utf-8")
        )
        if int(self.manifest.get("dataset_version", -1)) != DATASET_VERSION:
            raise ValueError("Unsupported universal QSafe dataset version.")

    def entries(self, split=None):
        entries = self.manifest["trajectories"]
        return [item for item in entries if split is None or item["split"] == split]

    def load(self, entry):
        with np.load(self.directory / entry["file"]) as archive:
            return {key: archive[key] for key in archive.files}

    def statistics(self):
        entries = self.entries()
        result = {
            "trajectories": len(entries),
            "transitions": sum(int(item["length"]) for item in entries),
        }
        for outcome in ("fall", "success", "stuck"):
            result[outcome] = sum(bool(item[outcome]) for item in entries)
        result["by_split"] = {}
        for split in SPLITS:
            split_entries = self.entries(split)
            result["by_split"][split] = {
                "trajectories": len(split_entries),
                "transitions": sum(int(item["length"]) for item in split_entries),
                **{
                    outcome: sum(bool(item[outcome]) for item in split_entries)
                    for outcome in ("fall", "success", "stuck")
                },
            }
        return result

    def validate_isolation(self):
        """Reject actor or map-seed leakage between train/validation/test."""

        for field in ("actor_id", "map_seed"):
            owners = {}
            for item in self.entries():
                value = item[field]
                previous = owners.setdefault(value, item["split"])
                if previous != item["split"]:
                    raise ValueError(
                        f"Dataset leakage: {field}={value} occurs in both "
                        f"{previous} and {item['split']}."
                    )
        return True
