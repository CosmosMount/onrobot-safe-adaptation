"""Run the controlled 100k five-seed SAC versus SAC+QSafe diagnostic."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import STABLE_ISAAC_SAC_FINETUNE_FLAGS


RUN_ROOT = PROJECT_ROOT / "runs" / "go2_sqrl" / "qsafe_frs_diagnostic"
GROUPS = {"A": False, "B": True}
MILESTONES = (10_000, 50_000, 100_000)


def _flag(section, key, value):
    if isinstance(value, bool):
        value = str(value).lower()
    return f"--{section}.{key}={value}"


def _run(command, log_path, dry_run):
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
        return process.wait()


def _mean(episodes, key):
    values = [float(item[key]) for item in episodes if key in item]
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def _evaluation_summary(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload["episodes"]
    return {
        "episodes": len(episodes),
        "fall": sum(bool(item["fall"]) for item in episodes),
        "stable_success": sum(bool(item["stable_success"]) for item in episodes),
        "stuck": sum(bool(item["stuck"]) for item in episodes),
        "mean_forward_velocity": _mean(episodes, "mean_forward_velocity"),
        "last_100_velocity": _mean(episodes, "last_100_velocity"),
        "action_saturation": _mean(episodes, "action_saturation_ratio"),
        "reject_rate": float(payload.get("qsafe", {}).get(
            "qsafe/rejected_fraction", float("nan"))),
        "fallback_rate": float(payload.get("qsafe", {}).get(
            "qsafe/fallback_fraction", float("nan"))),
    }


def _aggregate(report, episodes_per_eval):
    final = {group: [] for group in GROUPS}
    for record in report["runs"]:
        metrics = record.get("milestones", {}).get("100000")
        if metrics:
            final[record["group"]].append((record["seed"], metrics, record["training"]))
    if any(len(values) != 5 for values in final.values()):
        return {"status": "INCOMPLETE"}
    pairs = []
    for seed in range(5):
        a = next(value for value in final["A"] if value[0] == seed)
        b = next(value for value in final["B"] if value[0] == seed)
        pairs.append({
            "seed": seed,
            "training_fall_delta": b[2]["task_failures"] - a[2]["task_failures"],
            "evaluation_fall_delta": b[1]["fall"] - a[1]["fall"],
        })
    a_training_falls = sum(value[2]["task_failures"] for value in final["A"])
    b_training_falls = sum(value[2]["task_failures"] for value in final["B"])
    fall_reduction = ((a_training_falls - b_training_falls) / a_training_falls
                      if a_training_falls else float("nan"))
    seeds_improved = sum(item["training_fall_delta"] < 0 for item in pairs)
    a_success = np.mean([value[1]["stable_success"] / episodes_per_eval
                         for value in final["A"]])
    b_success = np.mean([value[1]["stable_success"] / episodes_per_eval
                         for value in final["B"]])
    b_stuck = np.mean([value[1]["stuck"] / episodes_per_eval for value in final["B"]])
    a_velocity = np.mean([value[1]["mean_forward_velocity"] for value in final["A"]])
    b_velocity = np.mean([value[1]["mean_forward_velocity"] for value in final["B"]])
    velocity_retention = b_velocity / a_velocity if a_velocity else float("nan")
    strong = bool(
        fall_reduction >= 0.30 and seeds_improved >= 3
        and b_success >= a_success - 0.10 and b_stuck <= 0.20
        and velocity_retention >= 0.85
    )
    promising = bool(
        fall_reduction >= 0.10 and seeds_improved >= 3
        and b_success >= a_success - 0.20 and b_stuck <= 0.30
    )
    return {
        "status": "PASS" if strong else ("WARN" if promising else "FAIL"),
        "training_falls_A": a_training_falls,
        "training_falls_B": b_training_falls,
        "fall_reduction": fall_reduction,
        "seeds_with_fewer_falls": seeds_improved,
        "stable_success_A": float(a_success),
        "stable_success_B": float(b_success),
        "stable_success_delta": float(b_success - a_success),
        "stuck_rate_B": float(b_stuck),
        "velocity_retention": float(velocity_retention),
        "per_seed": pairs,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--qsafe", required=True)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--groups", default="A,B")
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--allow-diagnostic-candidate", action="store_true")
    parser.add_argument("--allow-uncalibrated-smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    actor = Path(args.actor).expanduser().resolve()
    qsafe = Path(args.qsafe).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    challenge = json.loads(Path(args.challenge).expanduser().resolve().read_text(
        encoding="utf-8"))
    if not (actor / "policy.model").is_file():
        raise FileNotFoundError("A/B requires the actor-only policy.model sidecar")
    artifact = torch.load(qsafe, map_location="cpu", weights_only=False)
    calibration = artifact.get("calibration_report", {})
    calibrated = bool(calibration.get("universal_qsafe_v2_pass", False))
    if not calibrated and not (
        args.allow_diagnostic_candidate or args.allow_uncalibrated_smoke
    ):
        raise RuntimeError("Use --allow-diagnostic-candidate for an uncalibrated QSafe")
    metadata = artifact["metadata"]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    groups = [value.strip().upper() for value in args.groups.split(",") if value]
    if set(groups) - set(GROUPS):
        raise ValueError(f"Unknown groups: {sorted(set(groups) - set(GROUPS))}")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "finetune_report.json"
    report = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "actor": str(actor), "qsafe": str(qsafe), "calibrated": calibrated,
        "challenge": challenge, "steps": int(args.steps), "seeds": seeds,
        "groups": groups,
        "controlled_transfer": {
            "shared": "actor_and_actor_normalizer",
            "fresh": (
                "task_critics_targets_temperature_replay_all_optimizers_"
                "training_rng"
            ),
            "stable_sac_flags": list(STABLE_ISAAC_SAC_FINETUNE_FLAGS),
            "only_difference": "QSafe_Eq3_Eq4_and_fresh_nu",
        },
        "runs": [],
    }
    for seed in seeds:
        for group in groups:
            enabled = GROUPS[group]
            run_name = f"frs_ab_{group.lower()}_s{seed}_100k"
            models = RUN_ROOT / run_name / "models"
            record = {"seed": seed, "group": group, "run_name": run_name}
            report["runs"].append(record)
            train_command = [
                sys.executable, "-m", "src.run", "isaac-finetune",
                "--checkpoint", str(actor), "--qsafe-checkpoint", str(qsafe),
                "--seed", str(seed),
                *STABLE_ISAAC_SAC_FINETUNE_FLAGS,
                f"--algorithm.total_timesteps={args.steps}",
                "--algorithm.checkpoint_frequency=10000",
                f"--algorithm.qsafe.enabled={str(enabled).lower()}",
                f"--algorithm.eval_policy={'safe' if enabled else 'task'}",
                f"--algorithm.qsafe.gamma={metadata['gamma']}",
                f"--algorithm.qsafe.epsilon={metadata['epsilon']}",
                "--runner.exp_name=qsafe_frs_diagnostic",
                f"--runner.run_name={run_name}",
                "--environment.nr_envs=20", "--environment.nr_task_envs=20",
                "--environment.nr_safety_envs=0",
                "--environment.domain_randomization=false",
                "--environment.foot_clearance_target=0.07",
                "--environment.clearance_reward_mode=swing_weighted",
                "--environment.phase_reward_scale=0.3",
            ]
            train_command.extend(_flag("environment", key, value)
                                 for key, value in challenge["environment_overrides"].items())
            record["train_command"] = list(map(str, train_command))
            final_model = models / "final.model"
            if not final_model.is_file() or args.dry_run:
                code = _run(train_command, output / f"{run_name}.train.log", args.dry_run)
                if code:
                    raise subprocess.CalledProcessError(code, train_command)
            record["training"] = {}
            metrics_path = models / "final.metrics.json"
            if metrics_path.is_file():
                record["training"] = json.loads(metrics_path.read_text(encoding="utf-8"))
            record["milestones"] = {}
            for milestone in (value for value in MILESTONES if value <= args.steps):
                model = models / f"step_{milestone:09d}.model"
                # Deterministic mean-action evaluation with domain randomization
                # disabled makes all parallel Isaac environments exact copies.
                # Use the same paired stochastic candidate protocol as the frozen
                # protection experiment so 20 environments are 20 real trials,
                # while A/B receive identical candidate pools for each seed.
                result_path = output / f"{run_name}_{milestone}_paired.json"
                eval_command = [
                    sys.executable, "-m", "src.run", "isaac-eval",
                    "--checkpoint", str(model), "--seed", str(seed + 200000 + milestone),
                    f"--runner.nr_test_episodes={args.evaluation_episodes}",
                    "--environment.nr_envs=20", "--environment.nr_task_envs=20",
                    "--environment.nr_safety_envs=0",
                    "--environment.domain_randomization=false",
                    "--environment.foot_clearance_target=0.07",
                    "--environment.clearance_reward_mode=swing_weighted",
                    "--environment.phase_reward_scale=0.3",
                    "--algorithm.compile_policy=false",
                    "--algorithm.qsafe.paired_candidate_evaluation=true",
                    f"--algorithm.qsafe.eval_candidate_seed={seed + 400000 + milestone}",
                    f"--algorithm.qsafe.enabled={str(enabled).lower()}",
                    f"--algorithm.eval_policy={'safe' if enabled else 'task'}",
                    f"--algorithm.evaluation_results_path={result_path}",
                ]
                eval_command.extend(_flag("environment", key, value)
                                    for key, value in challenge["environment_overrides"].items())
                if not result_path.is_file() or args.dry_run:
                    code = _run(eval_command,
                                output / f"{run_name}_{milestone}.eval.log", args.dry_run)
                    if code:
                        raise subprocess.CalledProcessError(code, eval_command)
                if result_path.is_file():
                    record["milestones"][str(milestone)] = _evaluation_summary(result_path)
            temporary = report_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            temporary.replace(report_path)
    if not args.dry_run:
        report["comparison_100k"] = _aggregate(report, args.evaluation_episodes)
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
