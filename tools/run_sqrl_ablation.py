#!/usr/bin/env python3
"""Run and archive the paper-aligned SQRL versus standard-SAC ablation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQRL = ROOT / "runs/go2_sqrl/pretrain/isaac_sqrl_height_dr_v1/models"
DEFAULT_SAC = ROOT / "runs/go2_sqrl/pretrain/isaac_sac_height_dr_v1/models"
DEFAULT_HELDOUT_SAC = DEFAULT_SAC
DEFAULT_FLAT_SCENE = ROOT / "assets/robots/go2/mjcf/scene.xml"
DEFAULT_DATA = ROOT / "runs/go2_sqrl/ablation"
SUMMARY_TAGS = (
    "steps/nr_env_steps",
    "steps/nr_episodes",
    "steps/nr_failures",
    "steps/nr_critic_updates",
    "steps/nr_actor_updates",
    "finetune/actor_frozen",
    "finetune/actor_update_interval",
    "env_info/forward_velocity",
    "env_info/estimated_forward_velocity",
    "env_info/target_velocity_error",
    "env_info/velocity_estimation_error",
    "rollout/episode_return",
    "rollout/episode_length",
    "loss/q_loss",
    "loss/policy_loss",
    "q_value/q_value",
    "q_value/bellman_target",
    "entropy/alpha",
    "qsafe/actor_value",
    "qsafe/nu",
    "qsafe/rejected_fraction",
    "qsafe/fallback_fraction",
)


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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Running:", " ".join(command), flush=True)
    with log_path.open("w", buffering=1) as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return process.wait()


def _base_manifest(args, commands: list[list[str]]) -> dict:
    return {
        "experiment": args.experiment,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": _git_metadata(),
        "commands": commands,
        "status": "running",
    }


def _pretrain_sac(args) -> int:
    data_dir = args.data_root / args.experiment / f"pretrain_seed_{args.seed}"
    run_name = args.run_name
    command = [
        sys.executable,
        "-m",
        "src.run",
        "pretrain-sac",
        "--seed",
        str(args.seed),
        f"--algorithm.total_timesteps={args.steps}",
        f"--algorithm.logging_frequency={args.logging_frequency}",
        f"--algorithm.checkpoint_frequency={args.checkpoint_frequency}",
        "--runner.track_tb=true",
        f"--runner.run_name={run_name}",
    ]
    manifest = _base_manifest(args, [command])
    manifest_path = data_dir / "pretrain_sac_manifest.json"
    _write_json(manifest_path, manifest)
    return_code = _run(command, data_dir / "pretrain_sac_console.log")
    manifest["status"] = "complete" if return_code == 0 else "failed"
    manifest["return_code"] = return_code
    manifest["model_directory"] = str(
        ROOT / "runs/go2_sqrl/pretrain" / run_name / "models"
    )
    _write_json(manifest_path, manifest)
    return return_code


def _finetune_command(args, group: str, checkpoint: Path) -> list[str]:
    run_name = f"{args.experiment}_{group}_seed{args.seed}"
    command_name = "finetune" if group == "sqrl" else "finetune-sac"
    return [
        sys.executable,
        "-m",
        "src.run",
        command_name,
        "--checkpoint",
        str(checkpoint),
        "--seed",
        str(args.seed),
        "--domain-id",
        str(args.domain_id),
        "--interface",
        args.interface,
        f"--algorithm.total_timesteps={args.steps}",
        f"--algorithm.finetune_actor_warmup_steps={args.warmup_steps}",
        f"--algorithm.finetune_actor_update_interval={args.actor_update_interval}",
        f"--algorithm.alpha_init={args.alpha_init}",
        f"--algorithm.logging_frequency={args.logging_frequency}",
        "--environment.target_velocity_x=0.6",
        "--runner.track_tb=true",
        f"--runner.run_name={run_name}",
    ]


def _start_simulator(args, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", buffering=1)
    command = [
        sys.executable,
        "-m",
        "src.run",
        "sim",
        "--domain-id",
        str(args.domain_id),
        "--interface",
        args.interface,
    ]
    scene = getattr(args, "scene", None)
    if scene is not None:
        command.extend(("--scene", str(Path(scene).resolve())))
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
            f"MuJoCo simulator exited during startup with code {process.returncode}; "
            f"see {log_path}"
        )
    return process, log_file, command


def _mujoco_task_flags(args) -> list[str]:
    terrain_profile = getattr(args, "terrain_profile", "flat")
    if terrain_profile == "flat":
        # Keep the mobility regression free of step-specific phase and
        # clearance shaping. The target task asks only for flat-ground command
        # tracking at 0.6 m/s.
        return [
            "--environment.target_velocity_x=0.6",
            "--environment.terrain_profile=flat",
            "--environment.action_profile=legacy_v1",
            "--environment.step_success_distance=2.0",
            "--environment.foot_clearance_reward_scale=0.0",
            "--environment.foot_clearance_target=0.07",
            "--environment.clearance_reward_mode=legacy_mean",
            "--environment.phase_reward_scale=0.0",
            "--environment.phase_velocity_gate_start=0.0",
            "--environment.phase_velocity_gate_full=0.0",
            "--environment.stable_progress_scale=0.0",
            "--environment.terminal_failure_penalty=0.0",
        ]
    if terrain_profile != "single_step_up":
        raise ValueError(f"Unsupported MuJoCo terrain profile: {terrain_profile}")
    return [
        "--environment.target_velocity_x=0.6",
        "--environment.terrain_profile=single_step_up",
        f"--environment.step_height={args.step_height}",
        "--environment.step_start_x=1.0",
        "--environment.step_success_distance=2.0",
        "--environment.foot_clearance_target=0.07",
        "--environment.clearance_reward_mode=swing_weighted",
        "--environment.phase_reward_scale=0.3",
        "--environment.phase_velocity_gate_start=0.1",
        "--environment.phase_velocity_gate_full=0.2",
        "--environment.stable_progress_scale=0.5",
        f"--environment.terminal_failure_penalty={args.failure_penalty}",
    ]


def _mujoco_target_reward_contract(args) -> dict:
    if getattr(args, "terrain_profile", "flat") == "flat":
        return {
            "terrain_profile": "flat",
            "source_contract_matched": True,
            "only_task_change": "target_velocity_x: 0.5 -> 0.6 m/s",
            "action_profile": "legacy_v1",
            "clearance_reward_scale": 0.0,
            "clearance_target": 0.07,
            "clearance_mode": "legacy_mean",
            "phase_scale": 0.0,
            "phase_velocity_gate": [0.0, 0.0],
            "stable_progress_scale": 0.0,
            "terminal_failure_penalty": 0.0,
        }
    return {
        "terrain_profile": "single_step_up",
        "source_contract_matched": False,
        "clearance_target": 0.07,
        "clearance_mode": "swing_weighted",
        "phase_scale": 0.3,
        "phase_velocity_gate": [0.1, 0.2],
        "stable_progress_scale": 0.5,
        "terminal_failure_penalty": args.failure_penalty,
    }


def _mujoco_eval_command(
    args, checkpoint: Path, result_path: Path, alpha: float, source: bool
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.run",
        "finetune-sac" if source else "eval",
        "--checkpoint",
        str(checkpoint),
        "--seed",
        str(args.seed),
        "--domain-id",
        str(args.domain_id),
        "--interface",
        args.interface,
        "--runner.mode=test",
        "--runner.save_model=false",
        f"--runner.nr_test_episodes={args.evaluation_episodes}",
        "--algorithm.eval_policy=task",
        f"--algorithm.alpha_init={alpha:.12g}",
        f"--algorithm.evaluation_results_path={result_path}",
    ]
    command.extend(_mujoco_task_flags(args))
    return command


def _evaluation_summary(path: Path) -> dict:
    episodes = json.loads(path.read_text())["episodes"]

    def mean(name: str) -> float:
        values = [float(item[name]) for item in episodes]
        return sum(values) / len(values) if values else float("nan")

    return {
        "episodes": len(episodes),
        "falls": sum(bool(item["fall"]) for item in episodes),
        "stable_successes": sum(
            bool(item["stable_success"]) for item in episodes
        ),
        "stuck": sum(bool(item["stuck"]) for item in episodes),
        "mean_forward_velocity": mean("mean_forward_velocity"),
        "mean_last_100_velocity": mean("last_100_velocity"),
        "mean_action_saturation_ratio": mean("action_saturation_ratio"),
    }


def mujoco_sac_train_command(args, checkpoint: Path, run_name: str) -> list[str]:
    """Build the formal target command with fresh task-learning state."""

    command = [
        sys.executable,
        "-m",
        "src.run",
        "finetune-sac",
        "--checkpoint",
        str(checkpoint),
        "--seed",
        str(args.seed),
        "--domain-id",
        str(args.domain_id),
        "--interface",
        args.interface,
        f"--algorithm.total_timesteps={args.steps}",
        "--algorithm.learning_starts=1000",
        "--algorithm.finetune_actor_warmup_steps=10000",
        "--algorithm.finetune_actor_update_interval=10",
        "--algorithm.task_utd_ratio=1.0",
        f"--algorithm.alpha_init={float(args.alpha_init):.12g}",
        f"--algorithm.checkpoint_frequency={args.checkpoint_frequency}",
        f"--algorithm.logging_frequency={args.logging_frequency}",
        "--runner.track_tb=true",
        f"--runner.run_name={run_name}",
    ]
    command.extend(_mujoco_task_flags(args))
    return command


def _mujoco_sac_baseline(args) -> int:
    """Run the bounded, verified flat no-QSafe fine-tune in MuJoCo."""

    checkpoint = args.checkpoint.resolve()
    if not any(
        (checkpoint / name).is_file() for name in ("policy.model", "policy.msgpack")
    ):
        raise FileNotFoundError(f"Held-out SAC actor not found: {checkpoint}")
    if not args.scene.resolve().is_file():
        raise FileNotFoundError(f"MuJoCo target scene not found: {args.scene}")

    alpha = float(args.alpha_init)
    run_name = args.run_name or (
        f"mujoco_sac_flat_v11_s{args.seed}_{args.steps // 1000}k"
    )
    run_models = ROOT / "runs/go2_sqrl/finetune" / run_name / "models"
    data_dir = args.data_root / args.experiment / f"seed_{args.seed}"
    evaluations_dir = data_dir / "evaluations"
    train_command = mujoco_sac_train_command(args, checkpoint, run_name)

    milestones = [0] + list(
        range(args.checkpoint_frequency, args.steps + 1, args.checkpoint_frequency)
    )
    if milestones[-1] != args.steps:
        # The learner always writes final.model.  Include a non-multiple final
        # step (for example the 15k unlock diagnostic with 10k checkpoints)
        # instead of accidentally evaluating only the frozen-actor checkpoint.
        milestones.append(args.steps)
    result_paths = {
        step: evaluations_dir / f"step_{step:09d}.json" for step in milestones
    }
    commands = [
        _mujoco_eval_command(args, checkpoint, result_paths[0], alpha, source=True),
        train_command,
    ]
    commands.extend(
        _mujoco_eval_command(
            args,
            (
                run_models / "final.model"
                if step == args.steps
                and step % args.checkpoint_frequency != 0
                else run_models / f"step_{step:09d}.model"
            ),
            result_paths[step],
            alpha,
            source=False,
        )
        for step in milestones[1:]
    )

    manifest = _base_manifest(args, commands)
    manifest.update(
        {
            "checkpoint": str(checkpoint),
            "target_alpha_init": alpha,
            "scene": str(args.scene.resolve()),
            "run_name": run_name,
            "run_models": str(run_models),
            "contract": {
                "backend": "MuJoCo SDK2 + Flax/JAX SAC",
                "qsafe": False,
                "critic_utd": 1,
                "actor_warmup_steps": 10000,
                "actor_to_critic_after_warmup": "1:10",
                "actor_learning_rate_after_warmup": "configured base rate",
                "task_action": "projected/applied normalized action",
                "temperature": "fresh from target_alpha_init",
                "temperature_update": "frozen with actor, then 1:10 with actor updates",
                "task_critic": "fresh random initialization",
                "target_critic": "initialized from fresh online critic",
                "source_task_state_transferred": False,
                "replay": "fresh",
                "optimizers": "fresh",
                "target_reward": _mujoco_target_reward_contract(args),
            },
            "evaluations": {},
        }
    )
    manifest_path = data_dir / "mujoco_sac_flat_manifest.json"
    _write_json(manifest_path, manifest)

    simulator = simulator_log = None
    try:
        if not args.no_start_simulator:
            simulator, simulator_log, simulator_command = _start_simulator(
                args, data_dir / "simulator.log"
            )
            manifest["simulator_command"] = simulator_command
            _write_json(manifest_path, manifest)

        for index, command in enumerate(commands):
            if index == 0:
                label, step = "eval_step_000000000", 0
            elif index == 1:
                label, step = "train", None
            else:
                step = milestones[index - 1]
                label = f"eval_step_{step:09d}"
            return_code = _run(command, data_dir / f"{label}.log")
            if return_code:
                manifest.update(
                    status="failed", failed_command=label, return_code=return_code
                )
                _write_json(manifest_path, manifest)
                return return_code
            if step is not None:
                manifest["evaluations"][str(step)] = _evaluation_summary(
                    result_paths[step]
                )
                _write_json(manifest_path, manifest)
    finally:
        _stop_simulator(simulator, simulator_log)

    final = manifest["evaluations"][str(milestones[-1])]
    source = manifest["evaluations"]["0"]
    warmup_reference_step = max(
        step
        for step in milestones
        if step <= int(manifest["contract"]["actor_warmup_steps"])
    )
    warmup_reference = manifest["evaluations"][str(warmup_reference_step)]
    velocity_floor = max(0.3, 0.9 * source["mean_forward_velocity"])
    checks = {
        "forward_velocity": final["mean_forward_velocity"] >= velocity_floor,
        "last_100_velocity": final["mean_last_100_velocity"] >= 0.1,
        "stuck": final["stuck"] <= max(1, int(0.2 * final["episodes"])),
        # A fast fall is not locomotion even if its short-horizon velocity is
        # high.  Compare directly with the last frozen-actor checkpoint.
        "fall_no_jump": final["falls"] <= warmup_reference["falls"],
        "stable_success_no_collapse": (
            final["stable_successes"]
            >= max(0, warmup_reference["stable_successes"] - 1)
        ),
        "action_saturation": final["mean_action_saturation_ratio"] <= 0.20,
    }
    passed = all(checks.values())
    manifest["mobility_gate"] = {
        "passed": passed,
        "velocity_floor": velocity_floor,
        "warmup_reference_step": warmup_reference_step,
        "checks": checks,
        "decision": "extend_to_100k" if passed else "stop_and_diagnose",
    }
    manifest["status"] = "complete"
    _write_json(manifest_path, manifest)
    _write_json(data_dir / "mujoco_sac_flat_summary.json", manifest)
    return 0


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


def _finetune_pair(args) -> int:
    sqrl_checkpoint = args.sqrl_checkpoint.resolve()
    sac_checkpoint = args.sac_checkpoint.resolve()
    required = {
        sqrl_checkpoint / "policy.model",
        sqrl_checkpoint / "qsafe.model",
        sac_checkpoint / "policy.model",
    }
    missing = sorted(str(path) for path in required if not path.is_file())
    if missing:
        raise FileNotFoundError("Missing transfer artifacts: " + ", ".join(missing))

    data_dir = args.data_root / args.experiment / f"seed_{args.seed}"
    commands = [
        _finetune_command(args, "sqrl", sqrl_checkpoint),
        _finetune_command(args, "sac", sac_checkpoint),
    ]
    manifest = _base_manifest(args, commands)
    manifest.update(
        {
            "sqrl_checkpoint": str(sqrl_checkpoint),
            "sac_checkpoint": str(sac_checkpoint),
            "target_velocity_x": 0.6,
            "warmup_steps": args.warmup_steps,
            "actor_update_interval": args.actor_update_interval,
            "alpha_init": args.alpha_init,
            "seed": args.seed,
            "results": {},
        }
    )
    manifest_path = data_dir / "finetune_manifest.json"
    _write_json(manifest_path, manifest)

    for group, command in zip(("sqrl", "sac"), commands):
        simulator = None
        simulator_log = None
        try:
            if not args.no_start_simulator:
                simulator, simulator_log, simulator_command = _start_simulator(
                    args, data_dir / f"simulator_{group}.log"
                )
                manifest.setdefault("simulator_commands", {})[group] = (
                    simulator_command
                )
                _write_json(manifest_path, manifest)
            return_code = _run(command, data_dir / f"finetune_{group}_console.log")
            manifest["results"][group] = {"return_code": return_code}
            _write_json(manifest_path, manifest)
        finally:
            _stop_simulator(simulator, simulator_log)

    failed = any(value["return_code"] for value in manifest["results"].values())
    manifest["status"] = "failed" if failed else "complete"
    _write_json(manifest_path, manifest)
    if not failed:
        summarize(args.experiment, args.data_root)
    return int(failed)


def _event_values(run_dir: Path) -> dict[str, list]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    return {
        tag: accumulator.Scalars(tag)
        for tag in SUMMARY_TAGS
        if tag in available
    }


def summarize(experiment: str, data_root: Path) -> None:
    data_dir = data_root / experiment
    rows = []
    summary = {}
    for group in ("sqrl", "sac"):
        matches = sorted(
            (ROOT / "runs/go2_sqrl/finetune").glob(
                f"{experiment}_{group}_seed*"
            )
        )
        for run_dir in matches:
            values = _event_values(run_dir)
            run_summary = {"run_directory": str(run_dir)}
            for tag, events in values.items():
                for event in events:
                    rows.append(
                        {
                            "group": group,
                            "run": run_dir.name,
                            "tag": tag,
                            "step": event.step,
                            "value": event.value,
                            "wall_time": event.wall_time,
                        }
                    )
                run_summary[tag] = events[-1].value
                if tag in (
                    "env_info/forward_velocity",
                    "env_info/target_velocity_error",
                    "rollout/episode_return",
                ):
                    tail = events[-10:]
                    run_summary[f"{tag}_last10_mean"] = sum(
                        event.value for event in tail
                    ) / len(tail)
            steps = run_summary.get("steps/nr_env_steps", 0.0)
            falls = run_summary.get("steps/nr_failures", 0.0)
            run_summary["falls_per_1000_steps"] = (
                1000.0 * falls / steps if steps else None
            )
            summary[run_dir.name] = run_summary

    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "metrics.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("group", "run", "tag", "step", "value", "wall_time"),
        )
        writer.writeheader()
        writer.writerows(rows)
    _write_json(data_dir / "summary.json", summary)
    print(f"Saved {len(rows)} scalar samples to {data_dir / 'metrics.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain = subparsers.add_parser("pretrain-sac")
    pretrain.add_argument("--experiment", default="sqrl_vs_sac_v1")
    pretrain.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    pretrain.add_argument("--steps", type=int, default=300000)
    pretrain.add_argument("--seed", type=int, default=0)
    pretrain.add_argument("--run-name", default="isaac_sac_height_dr_v1")
    pretrain.add_argument("--logging-frequency", type=int, default=10000)
    pretrain.add_argument("--checkpoint-frequency", type=int, default=50000)
    pretrain.set_defaults(handler=_pretrain_sac)

    pair = subparsers.add_parser("finetune-pair")
    pair.add_argument("--experiment", default="sqrl_vs_sac_v1")
    pair.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    pair.add_argument("--sqrl-checkpoint", type=Path, default=DEFAULT_SQRL)
    pair.add_argument("--sac-checkpoint", type=Path, default=DEFAULT_SAC)
    pair.add_argument("--steps", type=int, default=100000)
    pair.add_argument("--warmup-steps", type=int, default=10000)
    pair.add_argument("--actor-update-interval", type=int, default=10)
    pair.add_argument("--alpha-init", type=float, default=2e-4)
    pair.add_argument("--logging-frequency", type=int, default=1000)
    pair.add_argument("--seed", type=int, default=0)
    pair.add_argument("--domain-id", type=int, default=1)
    pair.add_argument("--interface", default="lo")
    pair.add_argument("--scene", type=Path, default=None)
    pair.add_argument("--no-start-simulator", action="store_true")
    pair.add_argument("--simulator-startup-seconds", type=float, default=5.0)
    pair.set_defaults(handler=_finetune_pair)

    mobility = subparsers.add_parser("mujoco-sac-baseline")
    mobility.add_argument("--experiment", default="mujoco_sac_flat_v11")
    mobility.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    mobility.add_argument("--checkpoint", type=Path, default=DEFAULT_HELDOUT_SAC)
    mobility.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_FLAT_SCENE,
        help=f"MuJoCo scene (default: flat {DEFAULT_FLAT_SCENE})",
    )
    mobility.add_argument(
        "--terrain-profile",
        choices=("flat", "single_step_up"),
        default="flat",
    )
    mobility.add_argument("--step-height", type=float, default=0.04)
    mobility.add_argument("--failure-penalty", type=float, default=-10.0)
    mobility.add_argument("--alpha-init", type=float, default=2e-4)
    mobility.add_argument("--steps", type=int, default=30000)
    mobility.add_argument("--checkpoint-frequency", type=int, default=10000)
    mobility.add_argument("--logging-frequency", type=int, default=1000)
    mobility.add_argument("--evaluation-episodes", type=int, default=10)
    mobility.add_argument("--run-name", default="")
    mobility.add_argument("--seed", type=int, default=0)
    mobility.add_argument("--domain-id", type=int, default=1)
    mobility.add_argument("--interface", default="lo")
    mobility.add_argument("--no-start-simulator", action="store_true")
    mobility.add_argument("--simulator-startup-seconds", type=float, default=5.0)
    mobility.set_defaults(handler=_mujoco_sac_baseline)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--experiment", default="sqrl_vs_sac_v1")
    summary_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    summary_parser.set_defaults(
        handler=lambda args: (summarize(args.experiment, args.data_root) or 0)
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
