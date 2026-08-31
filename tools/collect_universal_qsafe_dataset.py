"""Run the actor/noise/terrain matrix for the universal-QSafe v2 dataset.

The actor JSON is a list of objects with ``id``, ``checkpoint``, ``split`` and
optional ``environment_overrides`` (a mapping without the ``environment.``
prefix).  Actor IDs must belong to exactly one split.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch


NOISE_LEVELS = (0.0, 0.05, 0.10, 0.20)
TERRAINS = (
    ("flat", {"terrain_mode": "flat", "terrain_profile": "mixed"}),
    ("step_2cm", {"terrain_mode": "rough", "terrain_profile": "single_step_up", "step_height": 0.02}),
    ("step_4cm", {"terrain_mode": "rough", "terrain_profile": "single_step_up", "step_height": 0.04}),
    ("step_6cm", {"terrain_mode": "rough", "terrain_profile": "single_step_up", "step_height": 0.06}),
    ("boxes_2cm", {"terrain_mode": "rough", "terrain_profile": "boxes_4cm", "boxes_max_adjacent_height_difference": 0.02}),
    ("boxes_4cm", {"terrain_mode": "rough", "terrain_profile": "boxes_4cm", "boxes_max_adjacent_height_difference": 0.04}),
    ("boxes_6cm", {"terrain_mode": "rough", "terrain_profile": "boxes_4cm", "boxes_max_adjacent_height_difference": 0.06}),
    ("boxes_8cm", {"terrain_mode": "rough", "terrain_profile": "boxes_4cm", "boxes_max_adjacent_height_difference": 0.08}),
    # Includes slopes, random roughness, waves, stairs, boxes and flat patches.
    ("mixed", {"terrain_mode": "rough", "terrain_profile": "mixed"}),
    *tuple(
        (
            f"stairs_up_l{level}",
            {
                "terrain_mode": "rough",
                "terrain_profile": "mixed",
                "playback_terrain_type": "small_stairs_up",
                "playback_terrain_level": level,
                "terrain_num_rows": 3,
                "terrain_num_cols": 10,
            },
        )
        for level in range(3)
    ),
)
TERRAIN_BY_NAME = dict(TERRAINS)

FORMAL_ROLES = {
    "flat_seed0": "train",
    "rough_seed0": "train",
    "step_seed0": "train",
    "high_clearance": "train",
    "flat_seed1": "validation",
    "rough_seed1": "validation",
    "step_seed1": "validation",
    "heldout_flat": "test",
}
FRS_DIAGNOSTIC_ROLES = {
    "flat_seed0": "train",
    "rough_seed0": "train",
    "step_seed0": "train",
    "flat_seed1": "validation",
    "rough_seed1": "validation",
    "step_seed1": "validation",
    "heldout_flat": "test",
}


def profile_cells(profile):
    if profile in ("legacy_full", "formal_v2"):
        return [
            (terrain_name, noise)
            for terrain_name, _ in TERRAINS
            if not terrain_name.startswith("stairs_up_l")
            for noise in NOISE_LEVELS
        ]
    if profile != "frs_diagnostic_v1":
        raise ValueError(f"Unknown collection profile: {profile}")
    cells = [
        ("flat", 0.0),
        ("flat", 0.10),
        ("mixed", 0.0),
        ("mixed", 0.10),
    ]
    cells.extend(
        (f"stairs_up_l{level}", noise)
        for level in range(3)
        for noise in NOISE_LEVELS
    )
    cells.extend(
        (
            ("step_2cm", 0.0),
            ("step_4cm", 0.05),
            ("step_4cm", 0.10),
            ("step_6cm", 0.10),
            ("step_6cm", 0.20),
            ("boxes_2cm", 0.0),
            ("boxes_4cm", 0.05),
            ("boxes_4cm", 0.10),
            ("boxes_8cm", 0.10),
            ("boxes_8cm", 0.20),
        )
    )
    return cells


def parse_cell(value):
    try:
        terrain, noise_text = value.rsplit(":", 1)
        noise = float(noise_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "cells must use TERRAIN:NOISE, for example step_6cm:0.20"
        ) from exc
    if terrain not in TERRAIN_BY_NAME:
        raise argparse.ArgumentTypeError(f"Unknown terrain: {terrain}")
    if noise < 0.0:
        raise argparse.ArgumentTypeError("Action noise must be non-negative")
    return terrain, noise


def flag(section, key, value):
    if isinstance(value, bool):
        value = str(value).lower()
    return f"--{section}.{key}={value}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", required=True, help="Actor inventory JSON")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episodes-per-run", type=int, default=20)
    parser.add_argument("--map-seed-start", type=int, default=10000)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument(
        "--profile",
        choices=("legacy_full", "formal_v2", "frs_diagnostic_v1"),
        default="legacy_full",
    )
    parser.add_argument(
        "--cell",
        action="append",
        type=parse_cell,
        help="Override profile cells; may be repeated as TERRAIN:NOISE.",
    )
    parser.add_argument(
        "--actor-id",
        action="append",
        help="Collect only selected inventory actor IDs; may be repeated.",
    )
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    actor_inventory_path = Path(args.actors).expanduser().resolve()
    actors = json.loads(actor_inventory_path.read_text(encoding="utf-8"))
    if not actors:
        raise ValueError("Actor inventory is empty.")
    allowed_splits = {"train", "validation", "test"}
    seen_ids = set()
    resolved_actors = []
    for actor in actors:
        actor_id = str(actor["id"])
        if actor_id in seen_ids:
            raise ValueError(f"Duplicate actor id: {actor_id}")
        seen_ids.add(actor_id)
        split = str(actor["split"])
        if split not in allowed_splits:
            raise ValueError(f"Invalid split for {actor_id}: {split}")
        checkpoint = Path(actor["checkpoint"]).expanduser().resolve()
        policy = checkpoint / "policy.model"
        if not policy.is_file():
            raise FileNotFoundError(policy)
        policy_artifact = torch.load(policy, map_location="cpu", weights_only=False)
        manifest = policy_artifact.get("environment_manifest")
        if not isinstance(manifest, dict) or int(manifest.get("manifest_version", -1)) != 11:
            raise ValueError(f"{actor_id} does not use checkpoint manifest v11.")
        if manifest.get("observation", {}).get("size") != 46:
            raise ValueError(f"{actor_id} does not use the 46D actor observation.")
        if manifest.get("action", {}).get("size") != 12:
            raise ValueError(f"{actor_id} does not use the 12D action contract.")
        resolved_actors.append(
            {
                **actor,
                "id": actor_id,
                "split": split,
                "checkpoint": str(checkpoint),
                "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
                "environment_manifest": manifest,
            }
        )
    required_roles = (
        FRS_DIAGNOSTIC_ROLES
        if args.profile == "frs_diagnostic_v1"
        else FORMAL_ROLES
    )
    if args.formal or args.profile in ("formal_v2", "frs_diagnostic_v1"):
        actual_roles = {actor["id"]: actor["split"] for actor in resolved_actors}
        if actual_roles != required_roles:
            raise ValueError(
                "Formal collection requires exactly the planned actor roles and "
                f"splits. Expected {required_roles}, found {actual_roles}."
            )
    selected_actor_ids = set(args.actor_id or ())
    if selected_actor_ids:
        missing = selected_actor_ids - {actor["id"] for actor in resolved_actors}
        if missing:
            raise ValueError(f"Unknown selected actor IDs: {sorted(missing)}")
        resolved_actors = [
            actor for actor in resolved_actors if actor["id"] in selected_actor_ids
        ]
    cells = list(args.cell) if args.cell else profile_cells(args.profile)

    dataset_directory = Path(args.dataset).expanduser().resolve()
    dataset_directory.mkdir(parents=True, exist_ok=True)
    run_directory = dataset_directory / "collection_runs"
    run_directory.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    json_summary_path = run_directory / f"{timestamp}.json"
    csv_summary_path = run_directory / f"{timestamp}.csv"
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_status = subprocess.run(
            ["git", "status", "--short"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except subprocess.CalledProcessError:
        git_commit = "unavailable"
        git_status = ["unavailable"]
    summary = {
        "started_utc": timestamp,
        "command": [sys.executable, *sys.argv],
        "git_commit": git_commit,
        "git_status": git_status,
        "actor_inventory": str(actor_inventory_path),
        "actors": resolved_actors,
        "episodes_per_run": int(args.episodes_per_run),
        "map_seed_start": int(args.map_seed_start),
        "episode_offset": int(args.episode_offset),
        "formal": bool(args.formal),
        "profile": args.profile,
        "cells": [
            {"terrain": terrain, "action_noise": noise}
            for terrain, noise in cells
        ],
        "runs": [],
    }
    run_index = 0
    for actor in resolved_actors:
        actor_id = actor["id"]
        split = actor["split"]
        checkpoint = Path(actor["checkpoint"])
        actor_overrides = dict(actor.get("environment_overrides", {}))
        for terrain_name, noise in cells:
                terrain_overrides = TERRAIN_BY_NAME[terrain_name]
                map_seed = int(args.map_seed_start) + run_index
                base_episode_offset = int(args.episode_offset) + (
                    run_index * int(args.episodes_per_run)
                )
                existing_entries = []
                manifest_path = dataset_directory / "manifest.json"
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    existing_entries = [
                        entry
                        for entry in manifest.get("trajectories", [])
                        if entry["actor_id"] == actor_id
                        and entry["split"] == split
                        and int(entry["map_seed"]) == map_seed
                        and entry["terrain"] == terrain_name
                        and float(entry["action_noise"]) == float(noise)
                    ]
                if existing_entries and args.no_resume:
                    raise RuntimeError(
                        f"Collection cell already contains {len(existing_entries)} "
                        f"episodes: {actor_id}/{terrain_name}/{noise:.2f}."
                    )
                if len(existing_entries) > int(args.episodes_per_run):
                    raise RuntimeError(
                        f"Collection cell exceeds requested size: {actor_id}/"
                        f"{terrain_name}/{noise:.2f}."
                    )
                remaining_episodes = int(args.episodes_per_run) - len(existing_entries)
                command = [
                    sys.executable,
                    "-m",
                    "src.run",
                    "isaac-collect-qsafe",
                    "--checkpoint",
                    str(checkpoint),
                    "--seed",
                    str(map_seed),
                    f"--runner.nr_test_episodes={remaining_episodes}",
                    flag("algorithm", "qsafe.dataset.directory", dataset_directory),
                    flag("algorithm", "qsafe.dataset.actor_id", actor_id),
                    flag("algorithm", "qsafe.dataset.map_seed", map_seed),
                    flag(
                        "algorithm",
                        "qsafe.dataset.episode_offset",
                        base_episode_offset + len(existing_entries),
                    ),
                    flag("algorithm", "qsafe.dataset.split", split),
                    flag("algorithm", "qsafe.dataset.terrain", terrain_name),
                    flag("algorithm", "qsafe.dataset.action_noise", noise),
                ]
                merged = {**terrain_overrides, **actor_overrides}
                command.extend(
                    flag("environment", key, value)
                    for key, value in merged.items()
                )
                print(" ".join(map(str, command)), flush=True)
                record = {
                    "actor_id": actor_id,
                    "split": split,
                    "terrain": terrain_name,
                    "action_noise": float(noise),
                    "map_seed": map_seed,
                    "existing_episodes": len(existing_entries),
                    "requested_episodes": remaining_episodes,
                    "command": list(map(str, command)),
                    "status": "dry_run" if args.dry_run else "pending",
                }
                if remaining_episodes == 0:
                    record["status"] = "skipped_complete"
                elif not args.dry_run:
                    completed_process = subprocess.run(command, check=False)
                    record["return_code"] = int(completed_process.returncode)
                    record["status"] = (
                        "completed" if completed_process.returncode == 0 else "failed"
                    )
                summary["runs"].append(record)
                temporary = json_summary_path.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
                )
                temporary.replace(json_summary_path)
                with csv_summary_path.open("w", newline="", encoding="utf-8") as output:
                    writer = csv.DictWriter(
                        output,
                        fieldnames=(
                            "actor_id",
                            "split",
                            "terrain",
                            "action_noise",
                            "map_seed",
                            "existing_episodes",
                            "requested_episodes",
                            "status",
                            "return_code",
                        ),
                        extrasaction="ignore",
                    )
                    writer.writeheader()
                    writer.writerows(summary["runs"])
                if record["status"] == "failed":
                    raise subprocess.CalledProcessError(
                        record["return_code"], command
                    )
                run_index += 1
    summary["finished_utc"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    json_summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
