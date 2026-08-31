"""End-to-end flat/rough/step QSafe diagnostic with resumable stages."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from rl_x.algorithms.qsafe.dataset import SafetyTrajectoryDataset
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "runs" / "go2_sqrl"
ACTORS = {
    "flat_seed0": RUN_ROOT / "pretrain" / "isaac_sac_flat_action_v2_legacy_v1" / "models",
    "flat_seed1": RUN_ROOT / "pretrain" / "universal_flat_seed1_v11" / "models",
    "heldout_flat": RUN_ROOT / "pretrain" / "universal_flat_seed2_v11" / "models",
    "rough_seed0": RUN_ROOT / "pretrain" / "universal_rough_seed0_v11" / "models",
    "rough_seed1": RUN_ROOT / "pretrain" / "universal_rough_seed1_v11" / "models",
    "step_seed0": RUN_ROOT / "gait_finetune" / "gait_h07_p03_50k_s0" / "models",
    "step_seed1": RUN_ROOT / "gait_finetune" / "gait_h07_p03_50k_s2" / "models",
}
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "universal_qsafe" / "frs_diagnostic_v1"
DEFAULT_DATASET = PROJECT_ROOT / "runs" / "go2_sqrl" / "qsafe_datasets" / "frs_diagnostic_v1"
DEFAULT_QSAFE_OUTPUT = PROJECT_ROOT / "runs" / "go2_sqrl" / "qsafe_offline" / "frs_diagnostic_v1"
ROLE_SPLITS = {
    "flat_seed0": "train", "rough_seed0": "train", "step_seed0": "train",
    "flat_seed1": "validation", "rough_seed1": "validation",
    "step_seed1": "validation", "heldout_flat": "test",
}
REFERENCE_COUNTS = {
    "train": {"fall": 500, "success": 500},
    "validation": {"fall": 100, "success": 100},
    "test": {"fall": 100, "success": 100},
}


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _run(command, log_path, dry_run=False, accept=(0,)):
    print(" ".join(map(str, command)), flush=True)
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            list(map(str, command)), cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            output.write(line)
            output.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code not in accept:
        raise subprocess.CalledProcessError(code, command)
    return code


def actor_inventory(output):
    inventory = [
        {"id": actor_id, "checkpoint": str(ACTORS[actor_id]), "split": split}
        for actor_id, split in ROLE_SPLITS.items()
    ]
    _write_json(output / "actor_inventory.json", inventory)
    return inventory


def audit_actors(inventory, output):
    records = []
    contract = None
    for actor in inventory:
        directory = Path(actor["checkpoint"])
        policy_path = directory / "policy.model"
        task_path = directory / "final.model"
        if not policy_path.is_file() or not task_path.is_file():
            raise FileNotFoundError(f"Incomplete actor bundle: {directory}")
        policy = torch.load(policy_path, map_location="cpu", weights_only=False)
        task = torch.load(task_path, map_location="cpu", weights_only=False)
        manifest = policy.get("environment_manifest", {})
        current_contract = {
            "manifest_version": manifest.get("manifest_version"),
            "observation": manifest.get("observation"),
            "action": manifest.get("action"),
            "failure": manifest.get("failure"),
        }
        if current_contract["manifest_version"] != 11:
            raise ValueError(f"{actor['id']} is not manifest v11")
        if current_contract["observation"].get("size") != 46:
            raise ValueError(f"{actor['id']} is not a 46D actor")
        if current_contract["action"].get("size") != 12:
            raise ValueError(f"{actor['id']} is not a 12D actor")
        invariant = {
            "observation": current_contract["observation"],
            "action": current_contract["action"],
            "failure": current_contract["failure"],
        }
        if contract is None:
            contract = invariant
        elif invariant != contract:
            raise ValueError(f"Contract mismatch for {actor['id']}")
        config = task.get("config_algorithm", {})
        records.append({
            **actor,
            "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            "task_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
            "manifest_version": 11, "observation_size": 46, "action_size": 12,
            "sac": {
                key: config.get(key) for key in (
                    "learning_rate", "batch_size", "gamma", "tau",
                    "task_utd_ratio", "learning_starts",
                    "finetune_actor_warmup_steps", "finetune_actor_handoff_steps",
                    "finetune_actor_update_interval",
                )
            },
            "alpha": float(torch.as_tensor(task["log_alpha"]).exp()),
        })
    report = {"status": "PASS", "actors": records, "shared_contract": contract}
    _write_json(output / "actor_audit.json", report)
    return report


def _challenge_conditions():
    conditions = {}
    for level in range(3):
        conditions[f"stairs_up_l{level}"] = {
            "terrain_mode": "rough", "terrain_profile": "mixed",
            "playback_terrain_type": "small_stairs_up",
            "playback_terrain_level": level,
            "terrain_num_rows": 3, "terrain_num_cols": 10,
        }
    conditions["step_4cm"] = {
        "terrain_mode": "rough", "terrain_profile": "single_step_up",
        "step_height": 0.04, "terrain_num_rows": 1, "terrain_num_cols": 1,
    }
    return conditions


def _episode_summary(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload["episodes"]
    velocities = [item.get("mean_forward_velocity", float("nan")) for item in episodes]
    velocities = [float(value) for value in velocities if math.isfinite(float(value))]
    return {
        "episodes": len(episodes),
        "fall": sum(bool(item["fall"]) for item in episodes),
        "success": sum(bool(item["success"]) for item in episodes),
        "stable_success": sum(bool(item["stable_success"]) for item in episodes),
        "stuck": sum(bool(item["stuck"]) for item in episodes),
        "mean_forward_velocity": float(np.mean(velocities)) if velocities else float("nan"),
    }


def select_difficulty(python, output, dry_run=False):
    results = {}
    for index, (name, overrides) in enumerate(_challenge_conditions().items()):
        # Evaluate the same stochastic policy distribution that the later
        # paired protection experiment uses.  A deterministic SAC mean action
        # makes all fixed-terrain vector environments exact duplicates and
        # cannot produce a meaningful fall-rate estimate.
        result_path = output / "difficulty" / f"{name}_paired.json"
        command = [
            python, "-m", "src.run", "isaac-finetune-sac",
            "--checkpoint", str(ACTORS["heldout_flat"]),
            "--seed", str(41000 + index), "--runner.mode=test",
            "--runner.save_model=false", "--runner.nr_test_episodes=20",
            "--environment.nr_envs=20",
            "--environment.nr_task_envs=20", "--environment.nr_safety_envs=0",
            "--environment.domain_randomization=false",
            "--environment.foot_clearance_target=0.07",
            "--environment.clearance_reward_mode=swing_weighted",
            "--environment.phase_reward_scale=0.3",
            "--algorithm.compile_policy=false",
            "--algorithm.qsafe.paired_candidate_evaluation=true",
            "--algorithm.qsafe.eval_candidate_seed=73000",
            f"--algorithm.evaluation_results_path={result_path}",
        ]
        command.extend(_flag("environment", key, value) for key, value in overrides.items())
        if not result_path.is_file() or dry_run:
            _run(command, output / "difficulty" / f"{name}_paired.log", dry_run)
        if not dry_run and not result_path.is_file():
            raise RuntimeError(
                f"Isaac evaluation did not produce {result_path}; inspect "
                f"{output / 'difficulty' / f'{name}_paired.log'}"
            )
        if result_path.is_file():
            results[name] = _episode_summary(result_path)
    if dry_run:
        return None
    eligible = [
        name for name, metrics in results.items()
        if 0.20 <= metrics["fall"] / metrics["episodes"] <= 0.70
        and metrics["success"] > 0 and metrics["stuck"] < metrics["episodes"]
    ]
    pool = eligible or [
        name for name, metrics in results.items()
        if metrics["success"] > 0 and metrics["stuck"] < metrics["episodes"]
    ] or list(results)
    selected_name = min(
        pool,
        key=lambda name: abs(results[name]["fall"] / results[name]["episodes"] - 0.45),
    )
    selected_metrics = results[selected_name]
    challenge = {
        "name": selected_name,
        "environment_overrides": _challenge_conditions()[selected_name],
        "baseline": selected_metrics,
        "eligible_20_to_70_percent_fall": selected_name in eligible,
        "status": "PASS" if selected_name in eligible else "WARN",
        "all_conditions": results,
    }
    _write_json(output / "challenge.json", challenge)
    return challenge


def _flag(section, key, value):
    if isinstance(value, bool):
        value = str(value).lower()
    return f"--{section}.{key}={value}"


def collect_initial(python, inventory_path, dataset, output, dry_run=False):
    command = [
        python, "tools/collect_universal_qsafe_dataset.py",
        "--actors", str(inventory_path), "--dataset", str(dataset),
        "--profile", "frs_diagnostic_v1", "--episodes-per-run", "20",
        "--map-seed-start", "10000",
    ]
    if dry_run:
        command.append("--dry-run")
    _run(command, output / "collection_initial.log", dry_run=False)


def supplement_dataset(python, inventory_path, dataset, output, dry_run=False):
    if dry_run or not (dataset / "manifest.json").is_file():
        return {"status": "DRY_RUN", "rounds": []}
    rounds = []
    next_seed = 50000
    next_episode_offset = 2_000_000
    for round_index in range(2):
        data = SafetyTrajectoryDataset(dataset)
        statistics = data.statistics()
        deficits = {
            split: {
                outcome: max(0, required - statistics["by_split"][split][outcome])
                for outcome, required in REFERENCE_COUNTS[split].items()
            }
            for split in REFERENCE_COUNTS
        }
        if all(value == 0 for values in deficits.values() for value in values.values()):
            break
        grouped = defaultdict(list)
        for entry in data.entries():
            grouped[(entry["split"], entry["actor_id"], entry["terrain"],
                     float(entry["action_noise"]))].append(entry)
        selected = []
        for split in REFERENCE_COUNTS:
            need_fall = deficits[split]["fall"] > 0
            need_success = deficits[split]["success"] > 0
            if not (need_fall or need_success):
                continue
            scored = []
            for key, entries in grouped.items():
                if key[0] != split:
                    continue
                fall_rate = np.mean([entry["fall"] for entry in entries])
                success_rate = np.mean([entry["success"] for entry in entries])
                if need_fall and 0.20 <= fall_rate <= 0.80:
                    score = 2.0 - abs(fall_rate - 0.50)
                elif need_success and success_rate > 0.0:
                    score = 1.0 + success_rate
                else:
                    score = fall_rate if need_fall else success_rate
                scored.append((score, key, fall_rate, success_rate))
            scored.sort(reverse=True, key=lambda item: item[0])
            selected.extend(scored[:10])
        round_record = {"round": round_index + 1, "deficits_before": deficits, "cells": []}
        for _, key, fall_rate, success_rate in selected:
            split, actor_id, terrain, noise = key
            command = [
                python, "tools/collect_universal_qsafe_dataset.py",
                "--actors", str(inventory_path), "--dataset", str(dataset),
                "--profile", "frs_diagnostic_v1", "--episodes-per-run", "20",
                "--actor-id", actor_id, "--cell", f"{terrain}:{noise}",
                "--map-seed-start", str(next_seed),
                "--episode-offset", str(next_episode_offset),
            ]
            _run(command, output / f"supplement_{round_index + 1}_{next_seed}.log")
            round_record["cells"].append({
                "split": split, "actor_id": actor_id, "terrain": terrain,
                "action_noise": noise, "prior_fall_rate": float(fall_rate),
                "prior_success_rate": float(success_rate), "map_seed": next_seed,
            })
            next_seed += 1
            next_episode_offset += 20
        rounds.append(round_record)
    final_statistics = SafetyTrajectoryDataset(dataset).statistics()
    status = "PASS" if all(
        final_statistics["by_split"][split][outcome] >= required
        for split, outcomes in REFERENCE_COUNTS.items()
        for outcome, required in outcomes.items()
    ) else "WARN"
    report = {"status": status, "rounds": rounds, "statistics": final_statistics}
    _write_json(output / "collection_report.json", report)
    return report


def _completed_ab_summary(finetune):
    """Summarize only fully completed paired seeds after an intentional stop."""

    steps = int(finetune.get("steps", 100_000))
    milestone = str(steps)
    completed = {}
    for run in finetune.get("runs", []):
        training = run.get("training", {})
        evaluation = run.get("milestones", {}).get(milestone)
        if int(training.get("global_step", 0)) == steps and evaluation:
            completed[(int(run["seed"]), str(run["group"]))] = run
    paired_seeds = sorted(
        seed for seed, group in completed
        if group == "A" and (seed, "B") in completed
    )
    if not paired_seeds:
        return {"status": "INCOMPLETE", "completed_seeds": []}

    def group_total(group, source, key):
        return sum(
            completed[(seed, group)][source][key] for seed in paired_seeds
        )

    training = {}
    evaluation = {}
    for group in ("A", "B"):
        episodes = group_total(group, "training", "task_episodes")
        falls = group_total(group, "training", "task_failures")
        training[group] = {
            "episodes": episodes,
            "falls": falls,
            "fall_rate": falls / episodes if episodes else float("nan"),
            "stable_successes": group_total(
                group, "training", "task_stable_successes"
            ),
            "stuck": group_total(group, "training", "task_stuck"),
        }
        evaluation[group] = {
            "episodes": sum(
                completed[(seed, group)]["milestones"][milestone]["episodes"]
                for seed in paired_seeds
            ),
            "falls": sum(
                completed[(seed, group)]["milestones"][milestone]["fall"]
                for seed in paired_seeds
            ),
            "stable_successes": sum(
                completed[(seed, group)]["milestones"][milestone]["stable_success"]
                for seed in paired_seeds
            ),
            "stuck": sum(
                completed[(seed, group)]["milestones"][milestone]["stuck"]
                for seed in paired_seeds
            ),
        }
    return {
        "status": "PARTIAL_USER_STOP",
        "completed_seeds": paired_seeds,
        "requested_seeds": finetune.get("seeds", []),
        "training": training,
        "evaluation_100k": evaluation,
        "per_seed": [
            {
                "seed": seed,
                "training_fall_delta_B_minus_A": (
                    completed[(seed, "B")]["training"]["task_failures"]
                    - completed[(seed, "A")]["training"]["task_failures"]
                ),
                "evaluation_fall_delta_B_minus_A": (
                    completed[(seed, "B")]["milestones"][milestone]["fall"]
                    - completed[(seed, "A")]["milestones"][milestone]["fall"]
                ),
            }
            for seed in paired_seeds
        ],
    }


def final_report(output, qsafe_output):
    calibration = json.loads((qsafe_output / "calibration_report.json").read_text(
        encoding="utf-8"))
    protection = json.loads((output / "protection" / "protection_report.json").read_text(
        encoding="utf-8"))
    finetune = json.loads((output / "finetune" / "finetune_report.json").read_text(
        encoding="utf-8"))
    selected_test = calibration["selected"]["test"]
    offline_support = bool(
        selected_test["recall_future_failure"] >= 0.70
        and selected_test["safe_action_false_rejection_rate"] <= 0.30
        and selected_test["brier_improvement"] > 0.0
    )
    protection_support = protection.get("comparison", {}).get("diagnostic_status") == "PASS"
    primary_evidence = finetune.get("comparison_100k") or _completed_ab_summary(
        finetune
    )
    ab_status = primary_evidence.get("status", "INCOMPLETE")
    if ab_status == "PASS" and (offline_support or protection_support):
        conclusion = "CLEARLY_USEFUL"
    elif ab_status in ("PASS", "WARN"):
        conclusion = "PROMISING_NOT_PROVEN"
    else:
        conclusion = "NOT_USEFUL_FOR_CURRENT_TASK"
    report = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "profile": "frs_diagnostic_v1", "conclusion": conclusion,
        "primary_evidence": primary_evidence,
        "supporting_evidence": {
            "offline_support": offline_support,
            "offline_strict_status": calibration.get("status"),
            "protection_support": protection_support,
            "protection": protection.get("comparison", {}),
        },
        "interpretation": (
            "The requested 100k five-seed A/B was stopped by the user after two "
            "complete paired seeds; this partial result is diagnostic only. "
            "Offline calibration and frozen protection explain the result but "
            "are not conjunctive gates."
            if ab_status == "PARTIAL_USER_STOP"
            else "The 100k five-seed A/B result is primary. Offline calibration "
            "and frozen protection explain the result but are not conjunctive gates."
        ),
        "post_run_action": "REPORT_BEFORE_ANY_MODIFICATION",
    }
    _write_json(output / "final_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("frs_diagnostic_v1",),
                        default="frs_diagnostic_v1")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--qsafe-output", default=str(DEFAULT_QSAFE_OUTPUT))
    parser.add_argument("--stage", choices=(
        "all", "audit", "difficulty", "collect", "calibrate",
        "protection", "finetune", "report"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    python = str(Path(args.python).expanduser().resolve())
    output = Path(args.output).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()
    qsafe_output = Path(args.qsafe_output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory = actor_inventory(output)
    inventory_path = output / "actor_inventory.json"
    stages = ({"audit", "difficulty", "collect", "calibrate", "protection",
               "finetune", "report"} if args.stage == "all" else {args.stage})
    if "audit" in stages:
        audit_actors(inventory, output)
    if "difficulty" in stages:
        select_difficulty(python, output, args.dry_run)
    if "collect" in stages:
        collect_initial(python, inventory_path, dataset, output, args.dry_run)
        supplement_dataset(python, inventory_path, dataset, output, args.dry_run)
    if "calibrate" in stages:
        command = [
            python, "-m", "rl_x.algorithms.qsafe.offline_train",
            "--dataset", str(dataset), "--output", str(qsafe_output),
            "--updates", "100000", "--batch-size", "256",
            "--allow-incomplete-data", "--diagnostic-continue",
            "--checkpoint-name", "qsafe_flat_rough_step_v1_candidate.model",
        ]
        _run(command, output / "offline_train.log", args.dry_run)
    challenge = output / "challenge.json"
    qsafe = qsafe_output / "qsafe_flat_rough_step_v1_candidate.model"
    if "protection" in stages:
        command = [
            python, "tools/evaluate_universal_qsafe_protection.py",
            "--actor", str(ACTORS["heldout_flat"]), "--qsafe", str(qsafe),
            "--challenge", str(challenge), "--output", str(output / "protection"),
            "--episodes", "100", "--allow-diagnostic-candidate",
        ]
        _run(command, output / "protection_stage.log", args.dry_run)
    if "finetune" in stages:
        command = [
            python, "tools/run_universal_qsafe_finetune.py",
            "--actor", str(ACTORS["heldout_flat"]), "--qsafe", str(qsafe),
            "--challenge", str(challenge), "--output", str(output / "finetune"),
            "--steps", "100000", "--seeds", "0,1,2,3,4",
            "--groups", "A,B", "--allow-diagnostic-candidate",
        ]
        _run(command, output / "finetune_stage.log", args.dry_run)
    if "report" in stages and not args.dry_run:
        report = final_report(output, qsafe_output)
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
