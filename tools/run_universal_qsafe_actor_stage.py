from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

from src.config import STABLE_ISAAC_SAC_FINETUNE_FLAGS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "runs" / "go2_sqrl"
FLAT_SEED0 = RUN_ROOT / "pretrain" / "isaac_sac_flat_action_v2_legacy_v1" / "models"
STEP_SEED0 = RUN_ROOT / "gait_finetune" / "gait_h07_p03_50k_s0" / "models"
STEP_SEED1 = RUN_ROOT / "gait_finetune" / "gait_h07_p03_50k_s2" / "models"
HIGH_V2_DIRECTORY = RUN_ROOT / "universal_high_clearance_v2_stable_sac"
HIGH_V2_RUN_PREFIX = "universal_high_clearance_v2_stable_sac"
ACTORS = {
    "flat_seed0": FLAT_SEED0,
    "flat_seed1": RUN_ROOT / "pretrain" / "universal_flat_seed1_v11" / "models",
    "heldout_flat": RUN_ROOT / "pretrain" / "universal_flat_seed2_v11" / "models",
    "rough_seed0": RUN_ROOT / "pretrain" / "universal_rough_seed0_v11" / "models",
    "rough_seed1": RUN_ROOT / "pretrain" / "universal_rough_seed1_v11" / "models",
    "step_seed0": STEP_SEED0,
    "step_seed1": STEP_SEED1,
}


def high_v2_training_command(python, source, seed, stage, timesteps, run_name):
    command = [
        str(python),
        "-m",
        "src.run",
        "isaac-finetune-sac",
        "--checkpoint",
        str(source),
        "--seed",
        str(seed),
        f"--algorithm.total_timesteps={timesteps}",
        *STABLE_ISAAC_SAC_FINETUNE_FLAGS,
        f"--runner.run_name={run_name}",
        "--environment.nr_envs=20",
        "--environment.nr_task_envs=20",
        "--environment.nr_safety_envs=0",
        "--environment.terrain_mode=rough",
        "--environment.terrain_profile=high_clearance_mix",
        f"--environment.high_clearance_stage={stage}",
        "--environment.terrain_num_rows=1",
        "--environment.terrain_num_cols=20",
        "--environment.target_velocity_x=0.5",
        "--environment.domain_randomization=false",
        *high_reward_flags(),
    ]
    # Every stage transfers only the previous actor and its normalizer. Task
    # critics, targets, replay, optimizers, and alpha always start fresh.
    return command


def run_command(command, log_path, record):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record["command"] = list(map(str, command))
    record["started_utc"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    with log_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            list(map(str, command)),
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            output.write(line)
            output.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    record["finished_utc"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    record["return_code"] = int(return_code)
    record["status"] = "completed" if return_code == 0 else "failed"
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def evaluation_result(log_path):
    text = log_path.read_text(encoding="utf-8")
    summaries = [
        line for line in text.splitlines() if "Gait benchmark (" in line
    ]
    if not summaries:
        raise RuntimeError(f"No gait benchmark summary found in {log_path}.")
    summary = summaries[-1]

    def values(pattern, count):
        match = re.search(pattern, summary)
        if match is None or len(match.groups()) != count:
            return None
        return match.groups()

    headline = values(
        r"Gait benchmark \((\d+) episodes\) - Falls: (\d+).*?"
        r"Successes: (\d+)",
        3,
    )
    stuck = values(r"Stuck: (\d+)", 1)
    clearance = values(
        r"Mean/max foot clearance: ([0-9.]+)/([0-9.]+) m",
        2,
    )
    saturation = values(
        r"Action/torque saturation: ([0-9.]+)/([0-9.]+)",
        2,
    )
    if None in (headline, stuck, clearance, saturation):
        raise RuntimeError(f"Malformed gait benchmark summary in {log_path}.")
    stable = values(r"Stable successes: (\d+)", 1)
    swing_clearance = values(
        r"Swing clearance mean/P95: ([0-9.]+)/([0-9.]+) m",
        2,
    )
    swing_legs = values(
        r"Swing clearance FR/FL/RR/RL: "
        r"([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)",
        4,
    )
    swing_ratio = values(
        r"Swing ratio FR/FL/RR/RL: "
        r"([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)",
        4,
    )
    velocities = [
        float(value)
        for value in re.findall(
            r"Mean simulator forward velocity: (-?[0-9.]+)", text
        )
    ]
    return {
        "episodes": int(headline[0]),
        "fall": int(headline[1]),
        "success": int(headline[2]),
        "stable_success": (
            int(stable[0]) if stable is not None else None
        ),
        "stuck": int(stuck[0]),
        "mean_foot_clearance": float(clearance[0]),
        "max_foot_clearance": float(clearance[1]),
        "mean_swing_clearance": (
            float(swing_clearance[0]) if swing_clearance is not None else None
        ),
        "p95_swing_clearance": (
            float(swing_clearance[1]) if swing_clearance is not None else None
        ),
        "swing_clearance_per_leg": (
            list(map(float, swing_legs)) if swing_legs is not None else None
        ),
        "swing_ratio_per_leg": (
            list(map(float, swing_ratio)) if swing_ratio is not None else None
        ),
        "action_saturation": float(saturation[0]),
        "torque_saturation": float(saturation[1]),
        "mean_forward_velocity": (
            float(sum(velocities) / len(velocities))
            if velocities
            else float("nan")
        ),
    }


def high_reward_flags():
    return [
        "--environment.foot_clearance_target=0.08",
        "--environment.clearance_reward_mode=swing_weighted",
        "--environment.phase_reward_scale=0.3",
        "--environment.foot_clearance_upper_target=0.12",
        "--environment.foot_clearance_overshoot_scale=-10.0",
        "--environment.phase_velocity_gate_start=0.10",
        "--environment.phase_velocity_gate_full=0.20",
        "--environment.stable_progress_start=1.0",
        "--environment.stable_progress_min_base_clearance=0.22",
        "--environment.stable_progress_scale=0.5",
    ]


def write_high_v2_artifacts(manifest, manifest_path):
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    rows = []
    for record in manifest.get("runs", []):
        metrics = record.get("metrics")
        if metrics is None:
            continue
        rows.append(
            {
                "role": record["role"],
                "seed": record.get("seed", ""),
                "gate_pass": record.get("gate_pass", ""),
                **metrics,
            }
        )
    csv_path = manifest_path.with_name("summary.csv")
    if rows:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def run_high_v2(args):
    HIGH_V2_DIRECTORY.mkdir(parents=True, exist_ok=True)
    manifest_path = HIGH_V2_DIRECTORY / "manifest.json"
    if manifest_path.is_file() and not args.force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_status = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        manifest = {
            "started_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "git_commit": git_commit,
            "git_status": git_status,
            "python": str(Path(args.python).expanduser().resolve()),
            "stage": "high_v2",
            "source": str(STEP_SEED1),
            "sac_recipe": {
                "id": "gait_h07_p03_50k_s2",
                "transfer": "actor_normalizer_critics_targets_temperature",
                "flags": list(STABLE_ISAAC_SAC_FINETUNE_FLAGS),
                "nr_envs": 20,
            },
            "runs": [],
        }

    def find_record(role):
        return next(
            (
                record
                for record in manifest["runs"]
                if record.get("role") == role
            ),
            None,
        )

    def evaluate(role, checkpoint, seed, terrain_mode, step_height, high_reward):
        existing = find_record(role)
        log_path = HIGH_V2_DIRECTORY / f"{role}.log"
        if (
            existing is not None
            and existing.get("status") == "completed"
            and log_path.is_file()
            and not args.force
        ):
            existing["metrics"] = evaluation_result(log_path)
            return existing["metrics"]
        record = {
            "role": role,
            "kind": "deterministic_20_episode_evaluation",
            "seed": seed,
        }
        manifest["runs"].append(record)
        command = [
            args.python,
            "-m",
            "src.run",
            "isaac-eval",
            "--checkpoint",
            str(checkpoint),
            "--seed",
            str(seed),
            "--runner.nr_test_episodes=20",
            "--environment.nr_envs=20",
            "--environment.nr_task_envs=20",
            "--environment.nr_safety_envs=0",
            f"--environment.terrain_mode={terrain_mode}",
            "--environment.domain_randomization=false",
        ]
        if terrain_mode == "rough":
            command.extend(
                [
                    "--environment.terrain_profile=single_step_up",
                    f"--environment.step_height={step_height:.2f}",
                    "--environment.terrain_num_rows=1",
                    "--environment.terrain_num_cols=1",
                ]
            )
        if high_reward:
            command.extend(high_reward_flags())
        else:
            command.extend(
                [
                    "--environment.foot_clearance_target=0.07",
                    "--environment.clearance_reward_mode=swing_weighted",
                    "--environment.phase_reward_scale=0.3",
                ]
            )
        write_high_v2_artifacts(manifest, manifest_path)
        run_command(command, log_path, record)
        record["metrics"] = evaluation_result(log_path)
        write_high_v2_artifacts(manifest, manifest_path)
        return record["metrics"]

    source_results = {}
    for index, (name, terrain_mode, height) in enumerate(
        (
            ("flat", "flat", 0.0),
            ("step2cm", "rough", 0.02),
            ("step3cm", "rough", 0.03),
            ("step4cm", "rough", 0.04),
        )
    ):
        source_results[name] = evaluate(
            f"source_{name}",
            STEP_SEED1,
            2600 + index,
            terrain_mode,
            height,
            False,
        )
    manifest["source_metrics"] = source_results
    write_high_v2_artifacts(manifest, manifest_path)

    stages = (
        ("easy", 50000, 0.02, 18),
        ("medium", 100000, 0.03, 17),
        ("hard", 150000, 0.04, 16),
    )
    selected = None
    for seed in (6, 7, 8):
        source = STEP_SEED1
        seed_failed = False
        for stage_index, (
            stage,
            timesteps,
            evaluation_height,
            required_success,
        ) in enumerate(stages):
            run_name = f"{HIGH_V2_RUN_PREFIX}_s{seed}_{stage}"
            models = RUN_ROOT / "gait_finetune" / run_name / "models"
            train_role = f"seed{seed}_{stage}_train"
            existing = find_record(train_role)
            if not (
                existing is not None
                and existing.get("status") == "completed"
                and (models / "policy.model").is_file()
                and (models / "final.model").is_file()
                and not args.force
            ):
                record = {
                    "role": train_role,
                    "kind": "high_clearance_curriculum",
                    "seed": seed,
                    "curriculum_stage": stage,
                    "source": str(source),
                }
                manifest["runs"].append(record)
                command = high_v2_training_command(
                    args.python,
                    source,
                    seed,
                    stage,
                    timesteps,
                    run_name,
                )
                write_high_v2_artifacts(manifest, manifest_path)
                run_command(
                    command,
                    HIGH_V2_DIRECTORY / f"{train_role}.log",
                    record,
                )
                write_high_v2_artifacts(manifest, manifest_path)
            source = models
            eval_role = f"seed{seed}_{stage}_step{int(evaluation_height * 100)}cm"
            metrics = evaluate(
                eval_role,
                models,
                3600 + 10 * seed + stage_index,
                "rough",
                evaluation_height,
                True,
            )
            evaluation_record = find_record(eval_role)
            evaluation_record["gate_pass"] = bool(
                metrics["stable_success"] >= required_success
                and metrics["fall"] <= 2
                and metrics["stuck"] <= 2
            )
            write_high_v2_artifacts(manifest, manifest_path)
            if not evaluation_record["gate_pass"]:
                seed_failed = True
                manifest.setdefault("seed_failures", {})[str(seed)] = (
                    f"{stage} gate failed"
                )
                write_high_v2_artifacts(manifest, manifest_path)
                break
        if seed_failed:
            continue
        flat_metrics = evaluate(
            f"seed{seed}_hard_flat",
            source,
            4600 + seed,
            "flat",
            0.0,
            True,
        )
        step_metrics = find_record(f"seed{seed}_hard_step4cm")["metrics"]
        source_flat_velocity = source_results["flat"]["mean_forward_velocity"]
        source_clearance = source_results["step4cm"]["mean_swing_clearance"]
        clearance_threshold = max(float(source_clearance) + 0.01, 0.06)
        swing_ratios = step_metrics["swing_ratio_per_leg"]
        final_gate = bool(
            step_metrics["stable_success"] >= 16
            and step_metrics["fall"] <= 2
            and step_metrics["stuck"] <= 2
            and flat_metrics["mean_forward_velocity"]
            >= 0.9 * source_flat_velocity
            and step_metrics["mean_swing_clearance"]
            >= clearance_threshold
            and step_metrics["action_saturation"] <= 0.20
            and min(swing_ratios) >= 0.10
        )
        manifest.setdefault("final_gates", {})[str(seed)] = {
            "pass": final_gate,
            "clearance_threshold": clearance_threshold,
            "flat_velocity_threshold": 0.9 * source_flat_velocity,
        }
        if final_gate:
            selected = source
            manifest["selected_high_clearance"] = str(source)
            manifest["playback_command"] = [
                args.python,
                "-m",
                "src.run",
                "isaac-eval",
                "--checkpoint",
                str(source),
                "--environment.render=true",
                "--environment.terrain_mode=rough",
                "--environment.terrain_profile=single_step_up",
                "--environment.step_height=0.04",
                *high_reward_flags(),
            ]
            break
        manifest.setdefault("seed_failures", {})[str(seed)] = (
            "final flat/clearance/action gate failed"
        )
        write_high_v2_artifacts(manifest, manifest_path)

    manifest["finished_utc"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    manifest["gate_pass"] = selected is not None
    if selected is None:
        manifest["stop_reason"] = (
            "Seeds 6, 7, and 8 failed before producing a valid high-clearance actor."
        )
    write_high_v2_artifacts(manifest, manifest_path)
    return 0 if selected is not None else 2


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("all", "flat", "rough", "high", "high_v2", "evaluate"),
        default="all",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "high_v2":
        return run_high_v2(args)
    experiment_directory = RUN_ROOT / "universal_qsafe_actor_stage"
    experiment_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = experiment_directory / "manifest.json"
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_status = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except subprocess.CalledProcessError:
        git_commit = "unavailable"
        git_status = ["unavailable"]
    manifest = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_status": git_status,
        "python": str(Path(args.python).expanduser().resolve()),
        "stage": args.stage,
        "runs": [],
        "evaluations": {},
    }

    flat_roles = (("flat_seed1", 1), ("heldout_flat", 2))
    if args.stage in ("all", "flat"):
        for role, seed in flat_roles:
            models = ACTORS[role]
            if (models / "policy.model").is_file() and not args.force:
                manifest["runs"].append({"role": role, "status": "skipped_existing"})
                continue
            record = {"role": role, "kind": "flat_sac_300k", "seed": seed}
            manifest["runs"].append(record)
            run_command(
                [
                    args.python,
                    "-m",
                    "src.run",
                    "pretrain-sac",
                    "--seed",
                    str(seed),
                    "--algorithm.total_timesteps=300000",
                    f"--runner.run_name={models.parent.name}",
                    "--environment.terrain_mode=flat",
                    "--environment.target_velocity_x=0.5",
                    "--environment.foot_clearance_target=0.1",
                    "--environment.clearance_reward_mode=legacy_mean",
                    "--environment.phase_reward_scale=0.0",
                ],
                experiment_directory / f"train_{role}.log",
                record,
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )

    rough_roles = (("rough_seed0", 0), ("rough_seed1", 1))
    if args.stage in ("all", "rough"):
        for role, seed in rough_roles:
            models = ACTORS[role]
            if (models / "policy.model").is_file() and not args.force:
                manifest["runs"].append({"role": role, "status": "skipped_existing"})
                continue
            record = {"role": role, "kind": "rough_sac_300k", "seed": seed}
            manifest["runs"].append(record)
            run_command(
                [
                    args.python,
                    "-m",
                    "src.run",
                    "pretrain-sac",
                    "--seed",
                    str(seed),
                    "--algorithm.total_timesteps=300000",
                    f"--runner.run_name={models.parent.name}",
                    "--environment.terrain_mode=rough",
                    "--environment.terrain_profile=mixed",
                    "--environment.target_velocity_x=0.5",
                    "--environment.foot_clearance_target=0.07",
                    "--environment.clearance_reward_mode=swing_weighted",
                    "--environment.phase_reward_scale=0.3",
                ],
                experiment_directory / f"train_{role}.log",
                record,
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )

    if args.stage in ("all", "high"):
        high_clearance = None
        for seed in (3, 4, 5):
            source = FLAT_SEED0
            for centimetres in (2, 3, 4):
                run_name = f"universal_high_clearance_s{seed}_step{centimetres}cm"
                models = RUN_ROOT / "gait_finetune" / run_name / "models"
                if not (models / "policy.model").is_file() or args.force:
                    record = {
                        "role": f"high_clearance_seed{seed}_step{centimetres}cm",
                        "kind": "step_curriculum_30k",
                        "seed": seed,
                    }
                    manifest["runs"].append(record)
                    run_command(
                        [
                            args.python,
                            "-m",
                            "src.run",
                            "isaac-finetune-sac",
                            "--checkpoint",
                            str(source),
                            "--seed",
                            str(seed),
                            "--algorithm.total_timesteps=30000",
                            "--algorithm.finetune_actor_warmup_steps=1000",
                            f"--runner.run_name={run_name}",
                            "--environment.nr_envs=20",
                            "--environment.nr_task_envs=20",
                            "--environment.nr_safety_envs=0",
                            "--environment.terrain_mode=rough",
                            "--environment.terrain_profile=single_step_up",
                            f"--environment.step_height={centimetres / 100:.2f}",
                            "--environment.target_velocity_x=0.5",
                            "--environment.foot_clearance_target=0.08",
                            "--environment.clearance_reward_mode=swing_weighted",
                            "--environment.phase_reward_scale=0.3",
                        ],
                        experiment_directory
                        / f"train_high_s{seed}_step{centimetres}cm.log",
                        record,
                    )
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                source = models
            evaluation_record = {
                "role": f"high_clearance_seed{seed}",
                "kind": "step_4cm_evaluation",
            }
            log_path = experiment_directory / f"eval_high_s{seed}_step4cm.log"
            run_command(
                [
                    args.python,
                    "-m",
                    "src.run",
                    "isaac-eval",
                    "--checkpoint",
                    str(source),
                    "--seed",
                    str(seed + 1000),
                    "--runner.nr_test_episodes=20",
                    "--environment.nr_envs=20",
                    "--environment.nr_task_envs=20",
                    "--environment.nr_safety_envs=0",
                    "--environment.terrain_mode=rough",
                    "--environment.terrain_profile=single_step_up",
                    "--environment.step_height=0.04",
                    "--environment.domain_randomization=false",
                    "--environment.foot_clearance_target=0.08",
                    "--environment.clearance_reward_mode=swing_weighted",
                    "--environment.phase_reward_scale=0.3",
                ],
                log_path,
                evaluation_record,
            )
            result = evaluation_result(log_path)
            evaluation_record["metrics"] = result
            evaluation_record["gate_pass"] = bool(
                result["success"] >= 16
                and result["fall"] <= 2
                and result["stuck"] <= 2
            )
            manifest["runs"].append(evaluation_record)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            if evaluation_record["gate_pass"]:
                ACTORS["high_clearance"] = source
                high_clearance = source
                break
        if high_clearance is None:
            manifest["gate_pass"] = False
            manifest["stop_reason"] = "All three high-clearance seeds failed the 4 cm gate."
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            return 2

    if "high_clearance" not in ACTORS:
        passing_high = [
            entry
            for entry in manifest.get("runs", [])
            if entry.get("role", "").startswith("high_clearance_seed")
            and entry.get("gate_pass")
        ]
        if passing_high:
            seed = int(re.search(r"seed(\d+)", passing_high[-1]["role"]).group(1))
            ACTORS["high_clearance"] = (
                RUN_ROOT
                / "gait_finetune"
                / f"universal_high_clearance_s{seed}_step4cm"
                / "models"
            )

    if args.stage in ("all", "evaluate"):
        conditions = {
            "flat_seed0_flat": (ACTORS["flat_seed0"], "flat", "mixed", 0.0, 0.1, "legacy_mean", 0.0),
            "flat_seed1_flat": (ACTORS["flat_seed1"], "flat", "mixed", 0.0, 0.1, "legacy_mean", 0.0),
            "heldout_flat_flat": (ACTORS["heldout_flat"], "flat", "mixed", 0.0, 0.1, "legacy_mean", 0.0),
            "rough_seed0_flat": (ACTORS["rough_seed0"], "flat", "mixed", 0.0, 0.07, "swing_weighted", 0.3),
            "rough_seed0_mixed": (ACTORS["rough_seed0"], "rough", "mixed", 0.0, 0.07, "swing_weighted", 0.3),
            "rough_seed1_flat": (ACTORS["rough_seed1"], "flat", "mixed", 0.0, 0.07, "swing_weighted", 0.3),
            "rough_seed1_mixed": (ACTORS["rough_seed1"], "rough", "mixed", 0.0, 0.07, "swing_weighted", 0.3),
            "step_seed0_flat": (ACTORS["step_seed0"], "flat", "mixed", 0.0, 0.07, "swing_weighted", 0.3),
            "step_seed0_step4cm": (ACTORS["step_seed0"], "rough", "single_step_up", 0.04, 0.07, "swing_weighted", 0.3),
            "step_seed1_flat": (ACTORS["step_seed1"], "flat", "mixed", 0.0, 0.07, "swing_weighted", 0.3),
            "step_seed1_step4cm": (ACTORS["step_seed1"], "rough", "single_step_up", 0.04, 0.07, "swing_weighted", 0.3),
        }
        if "high_clearance" in ACTORS:
            conditions["high_clearance_flat"] = (ACTORS["high_clearance"], "flat", "mixed", 0.0, 0.08, "swing_weighted", 0.3)
            conditions["high_clearance_step4cm"] = (ACTORS["high_clearance"], "rough", "single_step_up", 0.04, 0.08, "swing_weighted", 0.3)
        for name, condition in conditions.items():
            checkpoint, terrain_mode, terrain_profile, step_height, clearance, aggregation, phase = condition
            if not (checkpoint / "policy.model").is_file():
                raise FileNotFoundError(checkpoint / "policy.model")
            record = {"role": name, "kind": "actor_gate_evaluation"}
            log_path = experiment_directory / f"eval_{name}.log"
            run_command(
                [
                    args.python,
                    "-m",
                    "src.run",
                    "isaac-eval",
                    "--checkpoint",
                    str(checkpoint),
                    "--seed",
                    "2026",
                    "--runner.nr_test_episodes=20",
                    "--environment.nr_envs=20",
                    "--environment.nr_task_envs=20",
                    "--environment.nr_safety_envs=0",
                    f"--environment.terrain_mode={terrain_mode}",
                    f"--environment.terrain_profile={terrain_profile}",
                    f"--environment.step_height={step_height}",
                    "--environment.domain_randomization=false",
                    f"--environment.foot_clearance_target={clearance}",
                    f"--environment.clearance_reward_mode={aggregation}",
                    f"--environment.phase_reward_scale={phase}",
                ],
                log_path,
                record,
            )
            manifest["evaluations"][name] = evaluation_result(log_path)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )

        baseline_speed = manifest["evaluations"]["flat_seed0_flat"]["mean_forward_velocity"]
        gate_results = {}
        for role in ("flat_seed0", "flat_seed1", "heldout_flat"):
            result = manifest["evaluations"][f"{role}_flat"]
            gate_results[role] = bool(
                result["fall"] == 0
                and result["stuck"] <= 2
                and result["mean_forward_velocity"] >= 0.43
            )
        for role in ("rough_seed0", "rough_seed1"):
            flat = manifest["evaluations"][f"{role}_flat"]
            mixed = manifest["evaluations"][f"{role}_mixed"]
            gate_results[role] = bool(
                flat["mean_forward_velocity"] >= 0.43
                and mixed["success"] >= 10
                and mixed["stuck"] <= 4
                and mixed["fall"] <= 14
                and mixed["action_saturation"] <= 0.20
            )
        for role in ("step_seed0", "step_seed1"):
            flat = manifest["evaluations"][f"{role}_flat"]
            step = manifest["evaluations"][f"{role}_step4cm"]
            gate_results[role] = bool(
                step["success"] >= 16
                and step["fall"] <= 2
                and step["stuck"] <= 2
                and flat["mean_forward_velocity"] >= 0.9 * baseline_speed
            )
        if "high_clearance" in ACTORS:
            flat = manifest["evaluations"]["high_clearance_flat"]
            step = manifest["evaluations"]["high_clearance_step4cm"]
            gate_results["high_clearance"] = bool(
                step["success"] >= 16
                and step["fall"] <= 2
                and step["stuck"] <= 2
                and flat["mean_forward_velocity"] >= 0.9 * baseline_speed
            )
        manifest["actor_gates"] = gate_results
        manifest["gate_pass"] = all(gate_results.values())
        if not manifest["gate_pass"]:
            manifest["stop_reason"] = "At least one actor failed its acceptance gate."

    manifest["finished_utc"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    if manifest.get("gate_pass") is False:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
