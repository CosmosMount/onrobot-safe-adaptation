"""Run paired frozen-actor protection with and without a frozen QSafe."""

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


def summarize(payload):
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--qsafe", required=True)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-diagnostic-candidate", action="store_true")
    parser.add_argument("--allow-uncalibrated-smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    actor = Path(args.actor).expanduser().resolve()
    qsafe = Path(args.qsafe).expanduser().resolve()
    challenge_path = Path(args.challenge).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not (actor / "policy.model").is_file():
        raise FileNotFoundError(actor / "policy.model")
    if not qsafe.is_file():
        raise FileNotFoundError(qsafe)
    challenge = json.loads(challenge_path.read_text(encoding="utf-8"))
    artifact = torch.load(qsafe, map_location="cpu", weights_only=False)
    metadata = artifact["metadata"]
    calibration = artifact.get("calibration_report", {})
    calibrated = bool(calibration.get("universal_qsafe_v2_pass", False))
    allow_candidate = bool(args.allow_diagnostic_candidate or args.allow_uncalibrated_smoke)
    if not calibrated and not allow_candidate:
        raise RuntimeError(
            "The selected QSafe is diagnostic-only; pass "
            "--allow-diagnostic-candidate to continue without a calibration claim."
        )
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for enabled in (False, True):
        name = "with_qsafe" if enabled else "without_qsafe"
        result_path = output / f"{name}.json"
        log_path = output / f"{name}.log"
        command = [
            sys.executable, "-m", "src.run", "isaac-finetune",
            "--checkpoint", str(actor), "--qsafe-checkpoint", str(qsafe),
            "--seed", str(args.seed), "--runner.mode=test",
            "--runner.save_model=false",
            f"--runner.nr_test_episodes={args.episodes}",
            f"--runner.run_name=frs_protection_{name}_s{args.seed}",
            "--environment.nr_envs=20", "--environment.nr_task_envs=20",
            "--environment.nr_safety_envs=0",
            "--environment.domain_randomization=false",
            "--environment.foot_clearance_target=0.07",
            "--environment.clearance_reward_mode=swing_weighted",
            "--environment.phase_reward_scale=0.3",
            "--algorithm.compile_policy=false",
            f"--algorithm.qsafe.enabled={str(enabled).lower()}",
            f"--algorithm.eval_policy={'safe' if enabled else 'task'}",
            f"--algorithm.qsafe.gamma={metadata['gamma']}",
            f"--algorithm.qsafe.epsilon={metadata['epsilon']}",
            "--algorithm.qsafe.paired_candidate_evaluation=true",
            f"--algorithm.qsafe.eval_candidate_seed={args.seed + 100000}",
            f"--algorithm.evaluation_results_path={result_path}",
        ]
        command.extend(_flag("environment", key, value)
                       for key, value in challenge["environment_overrides"].items())
        return_code = _run(command, log_path, args.dry_run)
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
        records[name] = {
            "command": list(map(str, command)), "log": str(log_path),
            "result": str(result_path),
        }
        if not args.dry_run:
            records[name]["metrics"] = summarize(
                json.loads(result_path.read_text(encoding="utf-8")))
    report = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "actor": str(actor), "qsafe": str(qsafe), "calibrated": calibrated,
        "challenge": challenge,
        "paired_candidate_protocol": {
            "baseline": "candidate_0",
            "protected": "SQRL_Eq3_with_min_risk_fallback",
            "candidate_seed": args.seed + 100000,
        },
        "arms": records,
    }
    if not args.dry_run:
        baseline = records["without_qsafe"]["metrics"]
        protected = records["with_qsafe"]["metrics"]
        baseline_falls = baseline["fall"]
        reduction = ((baseline_falls - protected["fall"]) / baseline_falls
                     if baseline_falls else float("nan"))
        report["comparison"] = {
            "fall_reduction": reduction,
            "stable_success_delta": (
                protected["stable_success"] - baseline["stable_success"]
            ) / args.episodes,
            "stuck_rate": protected["stuck"] / args.episodes,
            "velocity_retention": (
                protected["mean_forward_velocity"] / baseline["mean_forward_velocity"]
                if baseline["mean_forward_velocity"] else float("nan")
            ),
            "reference_pass": bool(
                reduction >= 0.50 and protected["stuck"] / args.episodes <= 0.10
                and protected["fallback_rate"] <= 0.05),
            "diagnostic_status": (
                "PASS" if reduction >= 0.30
                and protected["stuck"] / args.episodes <= 0.20
                and protected["fallback_rate"] <= 0.10 else "WARN"),
        }
    destination = output / "protection_report.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
