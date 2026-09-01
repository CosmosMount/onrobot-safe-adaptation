#!/usr/bin/env python3
"""Run a strict paired flat-ground Safe Adaptation experiment.

Both arms transfer the same actor and actor normalizer, initialize the target
task critic/target/temperature from the same seed, and use the same SAC update
schedule.  The only behavioral treatment is ``algorithm.qsafe.enabled``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTOR = (
    ROOT
    / "runs/go2_sqrl/pretrain/isaac_sac_height_dr_v1/models"
)
DEFAULT_QSAFE = (
    ROOT
    / "runs/go2_sqrl/pretrain/isaac_sqrl_height_dr_v1/models/qsafe.model"
)
DEFAULT_SCENE = ROOT / "assets/robots/go2/mjcf/scene.xml"
DEFAULT_DATA = ROOT / "runs/go2_sqrl/ablation"
ARMS = ("no_qsafe", "qsafe")
PRINT_LOCK = threading.Lock()
STOP_EVENT = threading.Event()
STOP_REASON = {"message": ""}


class TrainingGateError(RuntimeError):
    """Raised when continuing the experiment would no longer be informative."""


def _request_stop(message: str) -> None:
    with PRINT_LOCK:
        if not STOP_EVENT.is_set():
            STOP_REASON["message"] = message
            print(f"[HARD STOP] {message}", flush=True)
        STOP_EVENT.set()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_legacy_flat_actor_manifest(manifest: dict) -> None:
    """Bind the formal flat A/B to the deployment-verified v9 actor contract."""

    action = dict(manifest.get("action", {}))
    expected = {
        "manifest_version": 9,
        "observation_version": "go2-observation-v3-body-velocity",
        "action_version": "go2-action-v1",
        "action_pipeline": "sdk-absolute-position-v2",
        "action_scale": 0.25,
    }
    actual = {
        "manifest_version": int(manifest.get("manifest_version", -1)),
        "observation_version": manifest.get("observation", {}).get("version"),
        "action_version": action.get("version"),
        "action_pipeline": action.get("pipeline_version"),
        "action_scale": float(action.get("scale", float("nan"))),
    }
    if actual != expected:
        raise ValueError(
            "Flat Safe Adaptation requires the verified manifest-v9/action-v1 "
            f"source actor contract; expected {expected}, got {actual}"
        )


def _validate_flat_qsafe_metadata(
    metadata: dict, calibration_report: dict | None = None
) -> None:
    """Accept the frozen baseline or a calibrated flat action-v1 QSafe v2."""

    version = int(metadata.get("qsafe_version", 1))
    common = {
        "observation_shape": list(metadata.get("observation_shape", [])),
        "action_shape": list(metadata.get("action_shape", [])),
    }
    if version == 1:
        expected = {
            "observation_shape": [46],
            "action_shape": [12],
            "gamma": 0.7,
            "epsilon": 0.1,
        }
        actual = {
            **common,
            "gamma": float(metadata.get("gamma", float("nan"))),
            "epsilon": float(metadata.get("epsilon", float("nan"))),
        }
        if actual != expected:
            raise ValueError(
                "Incompatible legacy flat QSafe contract; "
                f"expected {expected}, got {actual}"
            )
        return

    if version != 2:
        raise ValueError(f"Unsupported flat QSafe version: {version}")
    environment = metadata.get("environment_contract", {})
    action = environment.get("action", {})
    scale = action.get("scale")
    expected = {
        "observation_shape": [230],
        "base_observation_shape": [46],
        "action_shape": [12],
        "history_length": 5,
        "control_dt": 0.02,
        "observation_version": "go2-observation-v3-body-velocity",
        "action_version": "go2-action-v1",
        "action_pipeline": "sdk-absolute-position-v2",
        "action_scale": 0.25,
        "failure_version": "tilt-or-low-terrain-clearance-sustained-v3",
    }
    actual = {
        **common,
        "base_observation_shape": list(
            metadata.get("base_observation_shape", [])
        ),
        "history_length": int(metadata.get("history_length", -1)),
        "control_dt": float(metadata.get("control_dt", float("nan"))),
        "observation_version": environment.get("observation", {}).get("version"),
        "action_version": action.get("version"),
        "action_pipeline": action.get("pipeline_version"),
        "action_scale": float(scale) if isinstance(scale, (int, float)) else scale,
        "failure_version": environment.get("failure", {}).get("version"),
    }
    if actual != expected:
        raise ValueError(
            "Incompatible calibrated flat QSafe v2 contract; "
            f"expected {expected}, got {actual}"
        )
    gamma = float(metadata.get("gamma", float("nan")))
    epsilon = float(metadata.get("epsilon", float("nan")))
    if not 0.0 < gamma < 1.0 or not 0.0 < epsilon < 1.0:
        raise ValueError(
            f"Calibrated QSafe requires gamma and epsilon in (0, 1), got "
            f"gamma={gamma}, epsilon={epsilon}."
        )
    report = dict(calibration_report or {})
    selected = report.get("selected", {})
    if not (
        report.get("universal_qsafe_v2_pass") is True
        and report.get("horizons") == [5, 10, 25]
        and selected.get("validation_pass") is True
        and selected.get("test_pass") is True
        and abs(float(selected.get("gamma_safe", float("nan"))) - gamma) <= 1e-9
        and abs(float(selected.get("epsilon", float("nan"))) - epsilon) <= 1e-9
    ):
        raise ValueError(
            "QSafe v2 has not passed the required held-out 5/10/25-step "
            "calibration gate; refusing to spend time on SAC fine-tuning."
        )


# Kept for older imports while the flat experiment now accepts both contracts.
_validate_legacy_flat_qsafe_metadata = _validate_flat_qsafe_metadata


def _git_metadata() -> dict:
    def output(*command: str) -> str:
        return subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()

    return {
        "commit": output("git", "rev-parse", "HEAD"),
        "status": output("git", "status", "--short"),
        "diff_stat": output("git", "diff", "--stat"),
    }


def _print(label: str, message: str) -> None:
    with PRINT_LOCK:
        print(f"[{label}] {message}", flush=True)


def _run(
    command: list[str],
    log_path: Path,
    label: str,
    gate=None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _print(label, "Running: " + " ".join(command))
    with log_path.open("w", buffering=1, encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        current_metrics = {}
        gate_error = None
        for line in process.stdout:
            log_file.write(line)
            if "┌" in line and "┬" in line:
                current_metrics = {}
            match = re.search(r"│\s*([^│]+?)\s*│\s*([^│]+?)\s*│", line)
            if match:
                try:
                    current_metrics[match.group(1).strip()] = float(
                        match.group(2).strip()
                    )
                except ValueError:
                    pass
            if (
                "steps/nr_env_steps" in line
                or " ERROR " in line
                or " WARNING " in line
                or "Traceback" in line
                or "Episode " in line
            ):
                _print(label, line.rstrip())
            fatal_runtime = next(
                (
                    marker
                    for marker in (
                        "CUDA out of memory",
                        "RESOURCE_EXHAUSTED",
                        "FloatingPointError",
                        "Check failed",
                    )
                    if marker in line
                ),
                None,
            )
            if fatal_runtime is not None:
                gate_error = (
                    f"{label} encountered fatal runtime condition: {fatal_runtime}"
                )
                _request_stop(gate_error)
            if gate is not None and "└" in line and "┴" in line:
                try:
                    gate(current_metrics)
                except TrainingGateError as exc:
                    gate_error = str(exc)
                    _request_stop(gate_error)
            if STOP_EVENT.is_set():
                if process.poll() is None:
                    process.terminate()
                break
        return_code = process.wait()
        if gate_error is not None or STOP_EVENT.is_set():
            raise TrainingGateError(
                gate_error or STOP_REASON["message"] or "Experiment cancelled"
            )
        return return_code


def _start_simulator(args, domain_id: int, log_path: Path, label: str):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", buffering=1, encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "src.run",
        "sim",
        "--domain-id",
        str(domain_id),
        "--interface",
        args.interface,
        "--scene",
        str(args.scene),
    ]
    _print(label, "Starting simulator: " + " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    time.sleep(args.simulator_startup_seconds)
    if process.poll() is not None:
        log_file.close()
        raise RuntimeError(
            f"Simulator exited with code {process.returncode}; see {log_path}"
        )
    return process, log_file, command


def _stop_simulator(process, log_file) -> None:
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    if log_file is not None:
        log_file.close()


def _task_flags() -> list[str]:
    # Exact flat reward/action contract retained by yesterday's successful
    # standard-SAC deployment regression.
    return [
        "--environment.target_velocity_x=0.6",
        "--environment.terrain_profile=flat",
        "--environment.action_profile=legacy_v1",
        "--environment.step_success_distance=2.0",
        "--environment.foot_clearance_reward_scale=0.0",
        # The target must remain positive because the environment computes the
        # diagnostic even when its reward scale is zero. It has no reward
        # effect in this flat regression.
        "--environment.foot_clearance_target=0.07",
        "--environment.clearance_reward_mode=legacy_mean",
        "--environment.phase_reward_scale=0.0",
        "--environment.phase_velocity_gate_start=0.0",
        "--environment.phase_velocity_gate_full=0.0",
        "--environment.stable_progress_scale=0.0",
        "--environment.terminal_failure_penalty=0.0",
    ]


def _shared_train_flags(args, seed: int, domain_id: int, run_name: str) -> list[str]:
    return [
        "--checkpoint",
        str(args.actor),
        "--qsafe-checkpoint",
        str(args.qsafe),
        "--seed",
        str(seed),
        "--domain-id",
        str(domain_id),
        "--interface",
        args.interface,
        f"--algorithm.total_timesteps={args.steps}",
        "--algorithm.learning_starts=1000",
        "--algorithm.finetune_actor_warmup_steps=10000",
        "--algorithm.finetune_actor_update_interval=10",
        "--algorithm.task_utd_ratio=1.0",
        "--algorithm.alpha_init=0.0002",
        "--algorithm.initial_nu=0.0",
        f"--algorithm.checkpoint_frequency={args.checkpoint_frequency}",
        f"--algorithm.logging_frequency={args.logging_frequency}",
        f"--algorithm.qsafe.version={args.qsafe_version}",
        f"--algorithm.qsafe.gamma={args.qsafe_gamma}",
        f"--algorithm.qsafe.epsilon={args.qsafe_epsilon}",
        "--algorithm.eval_policy=task",
        "--runner.track_tb=true",
        f"--runner.run_name={run_name}",
        *_task_flags(),
    ]


def _train_command(
    args, seed: int, domain_id: int, arm: str, run_name: str
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.run",
        "finetune",
        *_shared_train_flags(args, seed, domain_id, run_name),
        f"--algorithm.qsafe.enabled={'true' if arm == 'qsafe' else 'false'}",
    ]


def _eval_command(
    args,
    checkpoint: Path,
    seed: int,
    domain_id: int,
    result_path: Path,
    *,
    eval_policy: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.run",
        "eval",
        "--checkpoint",
        str(checkpoint),
        "--seed",
        str(seed),
        "--domain-id",
        str(domain_id),
        "--interface",
        args.interface,
        "--runner.mode=test",
        "--runner.save_model=false",
        f"--runner.nr_test_episodes={args.evaluation_episodes}",
        f"--algorithm.eval_policy={eval_policy}",
        f"--algorithm.evaluation_results_path={result_path}",
        *_task_flags(),
    ]


def _event_accumulator(run_dir: Path) -> EventAccumulator:
    events = sorted(run_dir.glob("events.out.tfevents.*"))
    if not events:
        raise FileNotFoundError(f"No TensorBoard event file in {run_dir}")
    accumulator = EventAccumulator(
        str(events[-1]), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    return accumulator


def _series(accumulator: EventAccumulator, tag: str) -> list[tuple[int, float]]:
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return [(int(item.step), float(item.value)) for item in accumulator.Scalars(tag)]


def _last_at(series: list[tuple[int, float]], step: int) -> float:
    values = [value for event_step, value in series if event_step <= step]
    return values[-1] if values else 0.0


def _mean_between(
    series: list[tuple[int, float]], start: int, end: int
) -> float:
    values = [
        value for step, value in series if start < step <= end and math.isfinite(value)
    ]
    return float(np.mean(values)) if values else float("nan")


def _training_summary(run_dir: Path, steps: int) -> dict:
    accumulator = _event_accumulator(run_dir)
    failures = _series(accumulator, "steps/nr_failures")
    episodes = _series(accumulator, "steps/nr_episodes")
    critic_updates = _series(accumulator, "steps/nr_critic_updates")
    actor_updates = _series(accumulator, "steps/nr_actor_updates")
    alpha_updates = _series(accumulator, "steps/nr_alpha_updates")
    velocity = _series(accumulator, "env_info/forward_velocity")
    alpha = _series(accumulator, "entropy/alpha")
    intervention_tags = (
        "qsafe/rejected_fraction",
        "qsafe/fallback_fraction",
        "qsafe/action_change_fraction",
        "qsafe/action_change_l2",
        "qsafe/candidate0_rejected_fraction",
        "qsafe/safety_intervention_fraction",
    )
    intervention = {
        tag.split("/", 1)[1]: _series(accumulator, tag)
        for tag in intervention_tags
    }

    windows = []
    for start in range(0, steps, 100_000):
        end = min(start + 100_000, steps)
        window_falls = int(round(_last_at(failures, end) - _last_at(failures, start)))
        window_episodes = int(
            round(_last_at(episodes, end) - _last_at(episodes, start))
        )
        windows.append(
            {
                "start_step": start,
                "end_step": end,
                "falls": window_falls,
                "falls_per_100k_steps": window_falls * 100_000 / (end - start),
                "episodes": window_episodes,
                "episode_fall_probability": (
                    window_falls / window_episodes if window_episodes else float("nan")
                ),
                "mean_forward_velocity": _mean_between(velocity, start, end),
                **{
                    name: _mean_between(values, start, end)
                    for name, values in intervention.items()
                },
            }
        )

    total_falls = int(round(_last_at(failures, steps)))
    total_episodes = int(round(_last_at(episodes, steps)))
    return {
        "steps": steps,
        "falls": total_falls,
        "falls_per_100k_steps": total_falls * 100_000 / steps,
        "episodes": total_episodes,
        "episode_fall_probability": (
            total_falls / total_episodes if total_episodes else float("nan")
        ),
        "critic_updates": int(round(_last_at(critic_updates, steps))),
        "actor_updates": int(round(_last_at(actor_updates, steps))),
        "alpha_updates": int(round(_last_at(alpha_updates, steps))),
        "final_logged_velocity": velocity[-1][1] if velocity else float("nan"),
        "last_10k_mean_velocity": _mean_between(
            velocity, max(0, steps - 10_000), steps
        ),
        "final_logged_alpha": alpha[-1][1] if alpha else 2e-4,
        **{
            name: _mean_between(values, 0, steps)
            for name, values in intervention.items()
        },
        "windows": windows,
    }


def _evaluation_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload["episodes"]

    def mean(key: str) -> float:
        values = [float(item[key]) for item in episodes if key in item]
        return float(np.mean(values)) if values else float("nan")

    return {
        "episodes": len(episodes),
        "success": sum(bool(item["stable_success"]) for item in episodes),
        "fall": sum(bool(item["fall"]) for item in episodes),
        "stuck": sum(bool(item["stuck"]) for item in episodes),
        "success_probability": (
            sum(bool(item["stable_success"]) for item in episodes) / len(episodes)
            if episodes
            else float("nan")
        ),
        "fall_probability": (
            sum(bool(item["fall"]) for item in episodes) / len(episodes)
            if episodes
            else float("nan")
        ),
        "stuck_probability": (
            sum(bool(item["stuck"]) for item in episodes) / len(episodes)
            if episodes
            else float("nan")
        ),
        "mean_forward_velocity": mean("mean_forward_velocity"),
        "mean_last_100_velocity": mean("last_100_velocity"),
        "qsafe": payload.get("qsafe", {}),
    }


def _load_initialization_contract(run_dir: Path) -> dict:
    path = run_dir / "initialization_contract.json"
    if not path.is_file():
        raise FileNotFoundError(f"Initialization audit missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _live_training_gate(
    arm: str,
    run_dir: Path,
    *,
    baseline_run_dir: Path | None = None,
    reference_fingerprints: dict | None = None,
):
    velocity_history = []
    intervention_history = []
    fallback_history = []
    rejected_history = []
    initialization_checked = False

    def baseline_value(tag: str, step: int) -> float:
        if baseline_run_dir is None:
            return float("nan")
        accumulator = _event_accumulator(baseline_run_dir)
        return _last_at(_series(accumulator, tag), step)

    def gate(metrics: dict[str, float]) -> None:
        nonlocal initialization_checked
        step = int(metrics.get("steps/nr_env_steps", 0))
        if step <= 0:
            return

        if reference_fingerprints is not None and not initialization_checked:
            actual = _load_initialization_contract(run_dir)["fingerprints"]
            mismatches = {
                key: (reference_fingerprints[key], actual.get(key))
                for key in reference_fingerprints
                if reference_fingerprints[key] != actual.get(key)
            }
            if mismatches:
                raise TrainingGateError(
                    f"{arm} initialization differs from its paired baseline: "
                    f"{sorted(mismatches)}"
                )
            initialization_checked = True

        finite_keys = (
            "loss/q_loss",
            "loss/policy_loss",
            "entropy/alpha",
            "q_value/q_value",
            "q_value/bellman_target",
            "env_info/forward_velocity",
        )
        invalid = [
            key
            for key in finite_keys
            if key in metrics and not math.isfinite(metrics[key])
        ]
        if invalid:
            raise TrainingGateError(
                f"{arm} produced non-finite training metrics at step {step}: {invalid}"
            )

        critic_updates = int(metrics.get("steps/nr_critic_updates", 0))
        actor_updates = int(metrics.get("steps/nr_actor_updates", 0))
        alpha_updates = int(metrics.get("steps/nr_alpha_updates", actor_updates))
        expected_critic = max(0, step - 1_000)
        expected_actor = max(0, (step - 10_000) // 10 + (step >= 10_000))
        if abs(critic_updates - expected_critic) > 2:
            raise TrainingGateError(
                f"{arm} critic update schedule drifted at step {step}: "
                f"observed {critic_updates}, expected {expected_critic}"
            )
        if abs(actor_updates - expected_actor) > 2 or alpha_updates != actor_updates:
            raise TrainingGateError(
                f"{arm} actor/alpha schedule drifted at step {step}: "
                f"actor={actor_updates}, alpha={alpha_updates}, "
                f"expected actor≈{expected_actor}"
            )

        velocity = metrics.get("env_info/forward_velocity", float("nan"))
        if math.isfinite(velocity):
            velocity_history.append((step, velocity))
        if step >= 30_000:
            recent_velocity = [value for _, value in velocity_history[-3:]]
            if len(recent_velocity) == 3 and max(recent_velocity) < 0.10:
                raise TrainingGateError(
                    f"{arm} locomotion collapsed: the last three logged velocities "
                    f"at/after 30k are {recent_velocity} m/s"
                )

        falls = int(metrics.get("steps/nr_failures", 0))
        episodes = int(metrics.get("steps/nr_episodes", 0))
        if (
            step >= 30_000
            and episodes >= 20
            and falls / episodes > 0.95
            and math.isfinite(velocity)
            and velocity < 0.15
        ):
            raise TrainingGateError(
                f"{arm} is catastrophically unsafe and immobile at step {step}: "
                f"falls/episodes={falls}/{episodes}, velocity={velocity:.3f} m/s"
            )

        if arm != "qsafe":
            return
        required = (
            "qsafe/rejected_fraction",
            "qsafe/fallback_fraction",
            "qsafe/action_change_fraction",
            "qsafe/safety_intervention_fraction",
        )
        if step >= 5_000 and any(key not in metrics for key in required):
            raise TrainingGateError(
                f"QSafe intervention diagnostics are missing at step {step}"
            )
        intervention = metrics.get("qsafe/safety_intervention_fraction")
        fallback = metrics.get("qsafe/fallback_fraction")
        rejected = metrics.get("qsafe/rejected_fraction")
        if intervention is not None:
            intervention_history.append((step, intervention))
        if fallback is not None:
            fallback_history.append((step, fallback))
        if rejected is not None:
            rejected_history.append((step, rejected))

        if step >= 10_000:
            recent_fallback = [value for _, value in fallback_history[-5:]]
            recent_rejected = [value for _, value in rejected_history[-5:]]
            if len(recent_fallback) == 5 and float(np.mean(recent_fallback)) > 0.50:
                raise TrainingGateError(
                    f"QSafe fallback rate is pathological at step {step}: "
                    f"recent mean={np.mean(recent_fallback):.3f}"
                )
            if len(recent_rejected) == 5 and float(np.mean(recent_rejected)) > 0.95:
                raise TrainingGateError(
                    f"QSafe rejects nearly every candidate at step {step}: "
                    f"recent mean={np.mean(recent_rejected):.3f}"
                )

        if step < 30_000 or baseline_run_dir is None:
            return
        baseline_falls = int(
            round(baseline_value("steps/nr_failures", step))
        )
        baseline_velocity = baseline_value("env_info/forward_velocity", step)
        recent_intervention = [
            value for history_step, value in intervention_history
            if history_step > max(10_000, step - 10_000)
        ]
        if (
            baseline_falls >= 5
            and recent_intervention
            and float(np.mean(recent_intervention)) < 0.001
        ):
            raise TrainingGateError(
                f"QSafe has no meaningful treatment dosage at step {step}: "
                f"baseline already has {baseline_falls} falls but recent safety "
                f"intervention rate is {np.mean(recent_intervention):.6f}"
            )
        materially_more_falls = falls >= baseline_falls + max(
            10, int(math.ceil(0.5 * max(baseline_falls, 1)))
        )
        materially_slower = (
            math.isfinite(velocity)
            and math.isfinite(baseline_velocity)
            and velocity < 0.8 * baseline_velocity
        )
        if materially_more_falls and materially_slower:
            raise TrainingGateError(
                f"QSafe is materially worse at step {step}: falls={falls} vs "
                f"paired baseline={baseline_falls}, velocity={velocity:.3f} vs "
                f"{baseline_velocity:.3f} m/s"
            )

    return gate


def _run_arm(
    args,
    seed: int,
    domain_id: int,
    arm: str,
    baseline: dict | None = None,
) -> dict:
    if STOP_EVENT.is_set():
        raise TrainingGateError(STOP_REASON["message"] or "Experiment cancelled")
    label = f"s{seed}/{arm}"
    run_name = f"{args.experiment}_{arm}_s{seed}_{args.steps // 1000}k"
    run_dir = ROOT / "runs/go2_sqrl/finetune" / run_name
    models = run_dir / "models"
    seed_dir = args.data_root / args.experiment / f"seed_{seed}"
    arm_dir = seed_dir / arm
    final_model = models / "final.model"
    train_command = _train_command(args, seed, domain_id, arm, run_name)
    task_result = arm_dir / "final_deterministic_task.json"
    protected_result = arm_dir / "final_protected_policy.json"
    commands = {"train": train_command}
    simulator = simulator_log = None
    try:
        if not args.no_start_simulator:
            simulator, simulator_log, simulator_command = _start_simulator(
                args, domain_id, arm_dir / "simulator.log", label
            )
            commands["simulator"] = simulator_command

        if not final_model.is_file():
            if run_dir.exists() and not args.resume:
                raise FileExistsError(
                    f"Run directory already exists without final.model: {run_dir}. "
                    "Use --resume only after checking its contents."
                )
            gate = _live_training_gate(
                arm,
                run_dir,
                baseline_run_dir=(
                    Path(baseline["run_dir"]) if baseline is not None else None
                ),
                reference_fingerprints=(
                    baseline["initialization"]["fingerprints"]
                    if baseline is not None
                    else None
                ),
            )
            return_code = _run(
                train_command, arm_dir / "train.log", label, gate=gate
            )
            if return_code:
                raise subprocess.CalledProcessError(return_code, train_command)
        else:
            _print(label, f"Reusing completed model {final_model}")

        task_command = _eval_command(
            args,
            final_model,
            seed + 10_000,
            domain_id,
            task_result,
            eval_policy="task",
        )
        commands["deterministic_task_eval"] = task_command
        if not task_result.is_file() or not args.resume:
            return_code = _run(
                task_command, arm_dir / "final_deterministic_task.log", label
            )
            if return_code:
                raise subprocess.CalledProcessError(return_code, task_command)

        protected_summary = None
        if arm == "qsafe":
            protected_command = _eval_command(
                args,
                final_model,
                seed + 20_000,
                domain_id,
                protected_result,
                eval_policy="safe",
            )
            commands["protected_eval"] = protected_command
            if not protected_result.is_file() or not args.resume:
                return_code = _run(
                    protected_command,
                    arm_dir / "final_protected_policy.log",
                    label,
                )
                if return_code:
                    raise subprocess.CalledProcessError(
                        return_code, protected_command
                    )
            protected_summary = _evaluation_summary(protected_result)
    finally:
        _stop_simulator(simulator, simulator_log)

    result = {
        "arm": arm,
        "seed": seed,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "commands": commands,
        "initialization": _load_initialization_contract(run_dir),
        "training": _training_summary(run_dir, args.steps),
        "final_deterministic_task": _evaluation_summary(task_result),
        "final_protected_policy": protected_summary,
    }
    _write_json(arm_dir / "summary.json", result)
    return result


def _run_seed(args, seed: int) -> dict:
    if STOP_EVENT.is_set():
        raise TrainingGateError(STOP_REASON["message"] or "Experiment cancelled")
    domain_id = args.domain_id_base + seed
    # Baseline first permits an online paired harm/no-treatment gate for the
    # protected arm.  This operational ordering does not change either arm's
    # seed, initialization, simulator scene, or training configuration.
    order = ARMS
    arms = {"no_qsafe": _run_arm(args, seed, domain_id, "no_qsafe")}
    arms["qsafe"] = _run_arm(
        args, seed, domain_id, "qsafe", baseline=arms["no_qsafe"]
    )

    baseline_fingerprints = arms["no_qsafe"]["initialization"]["fingerprints"]
    protected_fingerprints = arms["qsafe"]["initialization"]["fingerprints"]
    initialization_match = {
        key: baseline_fingerprints[key] == protected_fingerprints[key]
        for key in baseline_fingerprints
    }
    shared_initialization_valid = all(initialization_match.values())
    if not shared_initialization_valid:
        raise RuntimeError(
            f"Seed {seed} A/B initialization mismatch: {initialization_match}"
        )

    result = {
        "seed": seed,
        "domain_id": domain_id,
        "arm_order": list(order),
        "initialization_match": initialization_match,
        "shared_initialization_valid": shared_initialization_valid,
        "arms": arms,
    }
    _write_json(
        args.data_root / args.experiment / f"seed_{seed}" / "paired_summary.json",
        result,
    )
    return result


def _aggregate(args, results: list[dict], metadata: dict) -> dict:
    aggregate = {}
    for arm in ARMS:
        records = [item["arms"][arm] for item in results]
        training = [item["training"] for item in records]
        deterministic = [item["final_deterministic_task"] for item in records]
        aggregate[arm] = {
            "seeds": len(records),
            "training_falls_total": sum(item["falls"] for item in training),
            "training_episodes_total": sum(item["episodes"] for item in training),
            "mean_falls_per_100k_steps": float(
                np.mean([item["falls_per_100k_steps"] for item in training])
            ),
            "std_falls_per_100k_steps": float(
                np.std(
                    [item["falls_per_100k_steps"] for item in training], ddof=1
                )
            ) if len(training) > 1 else 0.0,
            "pooled_episode_fall_probability": (
                sum(item["falls"] for item in training)
                / max(sum(item["episodes"] for item in training), 1)
            ),
            "mean_final_deterministic_success_probability": float(
                np.mean([item["success_probability"] for item in deterministic])
            ),
            "mean_final_deterministic_fall_probability": float(
                np.mean([item["fall_probability"] for item in deterministic])
            ),
            "mean_final_deterministic_stuck_probability": float(
                np.mean([item["stuck_probability"] for item in deterministic])
            ),
            "mean_final_deterministic_velocity": float(
                np.mean([item["mean_forward_velocity"] for item in deterministic])
            ),
            "mean_rejected_fraction": float(
                np.nanmean([item["rejected_fraction"] for item in training])
            ) if arm == "qsafe" else float("nan"),
            "mean_fallback_fraction": float(
                np.nanmean([item["fallback_fraction"] for item in training])
            ) if arm == "qsafe" else float("nan"),
            "mean_action_change_fraction": float(
                np.nanmean([item["action_change_fraction"] for item in training])
            ) if arm == "qsafe" else float("nan"),
            "mean_safety_intervention_fraction": float(
                np.nanmean(
                    [item["safety_intervention_fraction"] for item in training]
                )
            ) if arm == "qsafe" else float("nan"),
        }

    baseline = aggregate["no_qsafe"]
    protected = aggregate["qsafe"]
    paired_fall_differences = [
        item["arms"]["no_qsafe"]["training"]["falls_per_100k_steps"]
        - item["arms"]["qsafe"]["training"]["falls_per_100k_steps"]
        for item in results
    ]
    baseline_rate = baseline["mean_falls_per_100k_steps"]
    comparison = {
        "paired_falls_per_100k_differences": paired_fall_differences,
        "mean_paired_falls_per_100k_difference": float(
            np.mean(paired_fall_differences)
        ),
        "relative_fall_reduction": (
            (baseline_rate - protected["mean_falls_per_100k_steps"])
            / baseline_rate
            if baseline_rate
            else float("nan")
        ),
        "final_velocity_ratio": (
            protected["mean_final_deterministic_velocity"]
            / baseline["mean_final_deterministic_velocity"]
            if baseline["mean_final_deterministic_velocity"]
            else float("nan")
        ),
        "all_initialization_fingerprints_match": all(
            item["shared_initialization_valid"] for item in results
        ),
    }
    return {
        **metadata,
        "status": "complete",
        "per_seed": results,
        "aggregate": aggregate,
        "comparison": comparison,
    }


def _write_csv(path: Path, report: dict) -> None:
    rows = []
    for seed_result in report["per_seed"]:
        for arm in ARMS:
            record = seed_result["arms"][arm]
            train = record["training"]
            final = record["final_deterministic_task"]
            rows.append(
                {
                    "seed": seed_result["seed"],
                    "arm": arm,
                    "falls_per_100k_steps": train["falls_per_100k_steps"],
                    "episode_fall_probability": train["episode_fall_probability"],
                    "final_success_probability": final["success_probability"],
                    "final_fall_probability": final["fall_probability"],
                    "final_stuck_probability": final["stuck_probability"],
                    "final_velocity": final["mean_forward_velocity"],
                    "reject_rate": train["rejected_fraction"],
                    "fallback_rate": train["fallback_fraction"],
                    "action_change_rate": train["action_change_fraction"],
                    "safety_intervention_rate": train[
                        "safety_intervention_fraction"
                    ],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_and_enrich(args) -> dict:
    args.actor = args.actor.expanduser().resolve()
    args.qsafe = args.qsafe.expanduser().resolve()
    args.scene = args.scene.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    policy = args.actor / "policy.model"
    for path in (policy, args.qsafe, args.scene):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.steps < 1 or args.steps > 300_000:
        raise ValueError("--steps must be in [1, 300000]")
    if args.checkpoint_frequency < 1:
        raise ValueError("--checkpoint-frequency must be positive")

    qsafe_payload = torch.load(args.qsafe, map_location="cpu", weights_only=False)
    calibration = dict(qsafe_payload.get("calibration_report", {}))
    qsafe_metadata = dict(qsafe_payload["metadata"])
    _validate_flat_qsafe_metadata(qsafe_metadata, calibration)
    if int(qsafe_metadata.get("qsafe_version", 1)) == 2:
        for key in (
            "safety_observation_normalizer_state_dict",
            "safety_observation_normalizer_metadata",
        ):
            if key not in qsafe_payload:
                raise ValueError(f"QSafe v2 checkpoint is missing {key}.")
    args.qsafe_version = int(qsafe_metadata.get("qsafe_version", 1))
    args.qsafe_gamma = float(qsafe_metadata["gamma"])
    args.qsafe_epsilon = float(qsafe_metadata["epsilon"])
    actor_payload = torch.load(policy, map_location="cpu", weights_only=False)
    actor_manifest = dict(actor_payload["environment_manifest"])
    _validate_legacy_flat_actor_manifest(actor_manifest)
    return {
        "experiment": args.experiment,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": _git_metadata(),
        "protocol": {
            "steps": args.steps,
            "seeds": args.seeds,
            "paired_seeds": True,
            "only_treatment": "algorithm.qsafe.enabled",
            "source_actor": str(args.actor),
            "source_policy_sha256": _sha256(policy),
            "source_actor_manifest_version": actor_manifest["manifest_version"],
            "qsafe": str(args.qsafe),
            "qsafe_sha256": _sha256(args.qsafe),
            "qsafe_version": args.qsafe_version,
            "qsafe_gamma": args.qsafe_gamma,
            "qsafe_epsilon": args.qsafe_epsilon,
            "qsafe_calibration_pass": bool(
                calibration.get("universal_qsafe_v2_pass", False)
            ),
            "task_critic": "fresh from the same seed in both arms",
            "target_task_critic": "fresh copy of the same initialized online critic",
            "target_alpha": "fresh 2e-4 in both arms; frozen through 10k",
            "actor_normalizer": "transferred and frozen; fingerprinted",
            "critic_updates": "one per transition after learning_starts=1k",
            "actor_and_alpha_updates": "frozen through 10k, then one per ten critic updates",
            "final_primary_evaluation": "deterministic task actor, 20 episodes per seed by default",
            "protected_secondary_evaluation": "stochastic QSafe candidate selection",
        },
        "status": "running",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="flat_safe_adaptation_v1")
    parser.add_argument("--actor", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--qsafe", type=Path, default=DEFAULT_QSAFE)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--checkpoint-frequency", type=int, default=100_000)
    parser.add_argument("--logging-frequency", type=int, default=1_000)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--domain-id-base", type=int, default=31)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--parallel-seeds", type=int, default=1)
    parser.add_argument("--simulator-startup-seconds", type=float, default=5.0)
    parser.add_argument("--no-start-simulator", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(args.seeds) < 1:
        raise ValueError("At least one seed is required")
    metadata = _validate_and_enrich(args)
    output = args.data_root / args.experiment
    _write_json(output / "protocol.json", metadata)

    try:
        if args.parallel_seeds == 1:
            results = [_run_seed(args, seed) for seed in args.seeds]
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.parallel_seeds
            ) as executor:
                futures = {
                    executor.submit(_run_seed, args, seed): seed
                    for seed in args.seeds
                }
                by_seed = {}
                for future in concurrent.futures.as_completed(futures):
                    seed = futures[future]
                    by_seed[seed] = future.result()
                results = [by_seed[seed] for seed in args.seeds]
    except TrainingGateError as exc:
        metadata.update(
            status="stopped_early",
            stop_reason=str(exc),
            stopped_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        _write_json(output / "stopped_report.json", metadata)
        _print("stopped", str(exc))
        return 2
    except Exception as exc:
        _request_stop(f"Experiment failed: {type(exc).__name__}: {exc}")
        metadata.update(
            status="failed",
            stop_reason=STOP_REASON["message"],
            stopped_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        _write_json(output / "failed_report.json", metadata)
        _print("failed", STOP_REASON["message"])
        return 1

    report = _aggregate(args, results, metadata)
    _write_json(output / "final_report.json", report)
    _write_csv(output / "results.csv", report)
    _print("complete", f"Report: {output / 'final_report.json'}")
    print(json.dumps(report["comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
