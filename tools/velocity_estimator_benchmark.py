"""Compare the legacy and robust velocity estimators against MuJoCo truth.

Run this while ``unitree_mujoco`` and a walking evaluation policy are active.
The tool only subscribes to SDK2 state topics; it never publishes a command.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environments.go2_sqrl.common.estimation.velocity import (
    DEFAULT_VELOCITY_ESTIMATOR_CONFIG,
    VelocityEstimator,
    quaternion_rotation_matrix_wxyz,
)
from src.environments.go2_sqrl.common.estimation.kinematics import (
    foot_position_velocity_body,
)
from src.environments.go2_sqrl.common.specs import PHYSICS_DT
from src.environments.go2_sqrl.common.types import RobotState
from src.environments.go2_sqrl.sdk2_mujoco.sdk_client import SDKClient
from src.environments.go2_sqrl.sdk2_mujoco.state_buffer import SimulatorRestarted


class LegacyVelocityEstimator:
    """The previous fixed-gain estimator, retained only for A/B evaluation."""

    def __init__(
        self,
        dt: float = 0.02,
        support_gain: float = 0.35,
        no_support_damping: float = 0.995,
    ):
        self.dt = float(dt)
        self.support_gain = float(support_gain)
        self.no_support_damping = float(no_support_damping)
        self.world_velocity = np.zeros(3, dtype=np.float64)

    def reset(self, world_velocity: np.ndarray | None = None) -> None:
        self.world_velocity.fill(0.0)
        if world_velocity is not None:
            self.world_velocity[:] = np.asarray(world_velocity, dtype=np.float64)

    def update(self, state: RobotState) -> np.ndarray:
        rotation = quaternion_rotation_matrix_wxyz(state.imu_quat)
        if state.imu_accelerometer is not None:
            acceleration_world = rotation @ np.asarray(
                state.imu_accelerometer, dtype=np.float64
            )
            self.world_velocity += (
                acceleration_world + np.asarray([0.0, 0.0, -9.81])
            ) * self.dt

        positions, relative_velocity = foot_position_velocity_body(
            state.joint_q, state.joint_dq
        )
        support = (
            positions[:, 2] <= np.min(positions[:, 2]) + 0.035
        ) & (np.abs(relative_velocity[:, 2]) < 0.35)
        if np.any(support):
            candidates = -(
                relative_velocity
                + np.cross(np.asarray(state.imu_gyro)[None, :], positions)
            )
            observed_world_velocity = rotation @ np.mean(candidates[support], axis=0)
            self.world_velocity = (
                (1.0 - self.support_gain) * self.world_velocity
                + self.support_gain * observed_world_velocity
            )
            self.world_velocity[2] = observed_world_velocity[2]
        else:
            self.world_velocity *= self.no_support_damping
        return (rotation.T @ self.world_velocity).astype(np.float32)


def summarize_errors(errors: np.ndarray) -> dict[str, list[float] | float | int]:
    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[1] != 3 or errors.shape[0] == 0:
        raise ValueError("errors must have shape (samples, 3) with at least one sample")
    absolute = np.abs(errors)
    norm = np.linalg.norm(errors, axis=1)
    return {
        "samples": int(errors.shape[0]),
        "bias_xyz": np.mean(errors, axis=0).tolist(),
        "rmse_xyz": np.sqrt(np.mean(np.square(errors), axis=0)).tolist(),
        "p95_abs_xyz": np.percentile(absolute, 95.0, axis=0).tolist(),
        "rmse_3d": float(np.sqrt(np.mean(np.square(norm)))),
        "p95_3d": float(np.percentile(norm, 95.0)),
    }


def parse_speed_edges(value: str) -> tuple[float, ...]:
    """Parse monotonically increasing absolute forward-speed bin edges."""

    try:
        edges = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "speed bin edges must be comma-separated numbers"
        ) from exc
    if not edges or not np.all(np.isfinite(edges)):
        raise argparse.ArgumentTypeError("speed bin edges must be finite")
    if edges[0] < 0.0 or any(
        right <= left for left, right in zip(edges, edges[1:])
    ):
        raise argparse.ArgumentTypeError(
            "speed bin edges must be non-negative and strictly increasing"
        )
    return edges


def summarize_speed_segments(
    errors: np.ndarray,
    truth_body_velocity: np.ndarray,
    speed_edges: tuple[float, ...],
) -> dict[str, dict[str, object]]:
    """Summarize errors in bins of absolute SportModeState forward speed."""

    errors = np.asarray(errors, dtype=np.float64)
    truth = np.asarray(truth_body_velocity, dtype=np.float64)
    if errors.shape != truth.shape or errors.ndim != 2 or errors.shape[1] != 3:
        raise ValueError("errors and truth_body_velocity must both have shape (N, 3)")
    absolute_forward_speed = np.abs(truth[:, 0])
    segments: dict[str, dict[str, object]] = {}
    for lower, upper in zip(speed_edges, (*speed_edges[1:], np.inf)):
        mask = absolute_forward_speed >= lower
        if np.isfinite(upper):
            mask &= absolute_forward_speed < upper
            label = f"{lower:g}-{upper:g}"
            upper_value: float | None = float(upper)
        else:
            label = f"{lower:g}+"
            upper_value = None
        count = int(np.sum(mask))
        segment: dict[str, object] = {
            "speed_min": float(lower),
            "speed_max": upper_value,
            "samples": count,
        }
        if count:
            segment.update(summarize_errors(errors[mask]))
            segment["truth_forward_mean"] = float(np.mean(truth[mask, 0]))
            segment["truth_abs_forward_mean"] = float(
                np.mean(absolute_forward_speed[mask])
            )
        segments[label] = segment
    return segments


def leg_diagnostics(
    state: RobotState, truth_body_velocity: np.ndarray, config
) -> tuple[np.ndarray, np.ndarray]:
    positions, relative_velocity = foot_position_velocity_body(
        state.joint_q, state.joint_dq
    )
    candidates = -(
        relative_velocity
        + np.cross(np.asarray(state.imu_gyro)[None, :], positions)
    )
    height_delta = positions[:, 2] - np.min(positions[:, 2])
    confidence = np.exp(
        -0.5 * np.square(height_delta / config.height_scale)
        -0.5
        * np.square(relative_velocity[:, 2] / config.vertical_velocity_scale)
    )
    return candidates - truth_body_velocity[None, :], confidence


def _recording_arrays(recording: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.asarray(values) for name, values in recording.items()}


def evaluate_recording(
    recording: dict[str, np.ndarray],
    config,
    speed_edges: tuple[float, ...] = (0.0, 0.15, 0.3, 0.45, 0.6),
) -> dict[str, object]:
    robust = VelocityEstimator(dt=PHYSICS_DT, config=config)
    legacy = LegacyVelocityEstimator(
        dt=PHYSICS_DT * recording["joint_q"].shape[1]
    )
    robust_errors = []
    legacy_errors = []
    for sample in range(len(recording["truth_body_velocity"])):
        if recording.get(
            "reset_mask", np.zeros(len(recording["truth_body_velocity"]), dtype=bool)
        )[sample]:
            robust.reset()
            legacy.reset()
        for frame in range(recording["joint_q"].shape[1]):
            state = RobotState(
                joint_q=recording["joint_q"][sample, frame],
                joint_dq=recording["joint_dq"][sample, frame],
                imu_gyro=recording["imu_gyro"][sample, frame],
                imu_quat=recording["imu_quat"][sample, frame],
                imu_accelerometer=recording["imu_accelerometer"][sample, frame],
            )
            robust_velocity = robust.update(state)
        legacy_velocity = legacy.update(state)
        truth = recording["truth_body_velocity"][sample]
        robust_errors.append(robust_velocity - truth)
        legacy_errors.append(legacy_velocity - truth)
    evaluation_mask = recording.get(
        "evaluation_mask",
        np.ones(len(robust_errors), dtype=bool),
    ).astype(bool)
    truth = np.asarray(recording["truth_body_velocity"])[evaluation_mask]
    robust_errors = np.asarray(robust_errors)[evaluation_mask]
    legacy_errors = np.asarray(legacy_errors)[evaluation_mask]
    return {
        "legacy": summarize_errors(legacy_errors),
        "robust": summarize_errors(robust_errors),
        "legacy_speed_segments": summarize_speed_segments(
            legacy_errors, truth, speed_edges
        ),
        "robust_speed_segments": summarize_speed_segments(
            robust_errors, truth, speed_edges
        ),
    }


def estimator_config_from_args(args):
    return replace(
        DEFAULT_VELOCITY_ESTIMATOR_CONFIG,
        process_variance=args.process_variance,
        leg_variance=args.leg_variance,
        height_scale=args.height_scale,
        vertical_velocity_scale=args.vertical_velocity_scale,
        huber_delta=args.huber_delta,
        prior_temperature=args.prior_temperature,
        innovation_gate=args.innovation_gate,
        minimum_total_confidence=args.minimum_total_confidence,
    )


def collect(args) -> dict[str, object]:
    client = SDKClient(args.domain_id, args.interface)
    robust_config = estimator_config_from_args(args)
    robust = VelocityEstimator(
        dt=PHYSICS_DT,
        config=robust_config,
    )
    legacy = LegacyVelocityEstimator(dt=PHYSICS_DT * args.policy_frames)
    robust_errors: list[np.ndarray] = []
    legacy_errors: list[np.ndarray] = []
    rejected_updates = 0
    measured_updates = 0
    confidence_samples: list[np.ndarray] = []
    candidate_errors: list[np.ndarray] = []
    truth_velocities: list[np.ndarray] = []
    recording: dict[str, list[np.ndarray]] = {
        "joint_q": [],
        "joint_dq": [],
        "imu_gyro": [],
        "imu_quat": [],
        "imu_accelerometer": [],
        "truth_body_velocity": [],
        "evaluation_mask": [],
        "reset_mask": [],
    }
    last_tick = None
    generation = 0
    pending_reset = False
    started = time.monotonic()

    try:
        client.start()
        generation = client.state_buffer.generation
        while time.monotonic() - started < args.warmup_seconds + args.duration_seconds:
            try:
                frames = client.state_buffer.wait_for_frames(
                    count=args.policy_frames,
                    after_tick=last_tick,
                    timeout=args.state_timeout,
                    generation=generation,
                )
            except SimulatorRestarted:
                generation = client.state_buffer.generation
                client.state_buffer.clear_error()
                last_tick = None
                robust.reset()
                legacy.reset()
                pending_reset = True
                continue
            last_tick = int(frames[-1].tick)
            collecting = time.monotonic() - started >= args.warmup_seconds
            for frame in frames:
                robust_velocity = robust.update(frame)
                if collecting and robust.last_innovation_squared is not None:
                    measured_updates += 1
                    if not robust.last_measurement_accepted:
                        rejected_updates += 1
            legacy_velocity = legacy.update(frames[-1])

            truth = client.latest_training_state()
            if truth is None:
                continue
            rotation = quaternion_rotation_matrix_wxyz(frames[-1].imu_quat)
            truth_body_velocity = rotation.T @ np.asarray(
                truth.world_velocity, dtype=np.float64
            )
            if args.save_samples:
                for name in (
                    "joint_q",
                    "joint_dq",
                    "imu_gyro",
                    "imu_quat",
                    "imu_accelerometer",
                ):
                    recording[name].append(
                        np.stack([np.asarray(getattr(frame, name)) for frame in frames])
                    )
                recording["truth_body_velocity"].append(truth_body_velocity)
                recording["evaluation_mask"].append(collecting)
                recording["reset_mask"].append(pending_reset)
            pending_reset = False
            if not collecting:
                continue
            robust_errors.append(
                np.asarray(robust_velocity, dtype=np.float64) - truth_body_velocity
            )
            legacy_errors.append(
                np.asarray(legacy_velocity, dtype=np.float64) - truth_body_velocity
            )
            leg_error, confidence = leg_diagnostics(
                frames[-1], truth_body_velocity, robust_config
            )
            candidate_errors.append(leg_error)
            confidence_samples.append(confidence)
            truth_velocities.append(truth_body_velocity)
    finally:
        client.close()

    if not robust_errors:
        raise RuntimeError(
            "No synchronized SportModeState truth samples were received; "
            "ensure unitree_mujoco is running on the selected DDS domain/interface."
        )
    robust_summary = summarize_errors(np.asarray(robust_errors))
    legacy_summary = summarize_errors(np.asarray(legacy_errors))
    truth_velocities = np.asarray(truth_velocities)
    moving = np.abs(truth_velocities[:, 0]) >= args.moving_speed_threshold
    confidence_samples = np.asarray(confidence_samples)
    candidate_errors = np.asarray(candidate_errors)
    if args.save_samples:
        output_path = Path(args.save_samples).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **_recording_arrays(recording))
    return {
        "settings": {
            "duration_seconds": args.duration_seconds,
            "warmup_seconds": args.warmup_seconds,
            "policy_frames": args.policy_frames,
            "process_variance": args.process_variance,
            "leg_variance": args.leg_variance,
            "height_scale": args.height_scale,
            "vertical_velocity_scale": args.vertical_velocity_scale,
            "huber_delta": args.huber_delta,
            "prior_temperature": args.prior_temperature,
            "innovation_gate": args.innovation_gate,
            "minimum_total_confidence": args.minimum_total_confidence,
            "moving_speed_threshold": args.moving_speed_threshold,
            "speed_bin_edges": list(args.speed_bin_edges),
            "saved_samples": str(output_path) if args.save_samples else None,
        },
        "legacy": legacy_summary,
        "robust": robust_summary,
        "delta": {
            "rmse_x": (
                robust_summary["rmse_xyz"][0] - legacy_summary["rmse_xyz"][0]
            ),
            "rmse_3d": robust_summary["rmse_3d"] - legacy_summary["rmse_3d"],
            "p95_3d": robust_summary["p95_3d"] - legacy_summary["p95_3d"],
        },
        "robust_measurement_updates": measured_updates,
        "robust_rejected_updates": rejected_updates,
        "moving_samples": int(np.sum(moving)),
        "moving_legacy": (
            summarize_errors(np.asarray(legacy_errors)[moving]) if np.any(moving) else None
        ),
        "moving_robust": (
            summarize_errors(np.asarray(robust_errors)[moving]) if np.any(moving) else None
        ),
        "legacy_speed_segments": summarize_speed_segments(
            np.asarray(legacy_errors), truth_velocities, args.speed_bin_edges
        ),
        "robust_speed_segments": summarize_speed_segments(
            np.asarray(robust_errors), truth_velocities, args.speed_bin_edges
        ),
        "support_confidence_mean": np.mean(confidence_samples, axis=0).tolist(),
        "support_confidence_active_rate": np.mean(
            confidence_samples > 0.0, axis=0
        ).tolist(),
        "leg_candidate_rmse_3d": np.sqrt(
            np.mean(np.sum(np.square(candidate_errors), axis=2), axis=0)
        ).tolist(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A/B benchmark Go2 velocity estimators against SportModeState truth"
    )
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--policy-frames", type=int, default=10)
    parser.add_argument("--state-timeout", type=float, default=1.0)
    parser.add_argument(
        "--process-variance",
        type=float,
        default=DEFAULT_VELOCITY_ESTIMATOR_CONFIG.process_variance,
    )
    parser.add_argument(
        "--leg-variance",
        type=float,
        default=DEFAULT_VELOCITY_ESTIMATOR_CONFIG.leg_variance,
    )
    parser.add_argument(
        "--height-scale",
        type=float,
        default=DEFAULT_VELOCITY_ESTIMATOR_CONFIG.height_scale,
    )
    parser.add_argument(
        "--vertical-velocity-scale",
        type=float,
        default=DEFAULT_VELOCITY_ESTIMATOR_CONFIG.vertical_velocity_scale,
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=DEFAULT_VELOCITY_ESTIMATOR_CONFIG.huber_delta,
    )
    parser.add_argument(
        "--prior-temperature",
        type=float,
        default=DEFAULT_VELOCITY_ESTIMATOR_CONFIG.prior_temperature,
    )
    parser.add_argument(
        "--innovation-gate",
        type=float,
        default=DEFAULT_VELOCITY_ESTIMATOR_CONFIG.innovation_gate,
    )
    parser.add_argument(
        "--minimum-total-confidence",
        type=float,
        default=DEFAULT_VELOCITY_ESTIMATOR_CONFIG.minimum_total_confidence,
    )
    parser.add_argument("--moving-speed-threshold", type=float, default=0.1)
    parser.add_argument(
        "--speed-bin-edges",
        type=parse_speed_edges,
        default=parse_speed_edges("0,0.15,0.3,0.45,0.6"),
        help="Absolute SportModeState forward-speed bin edges in m/s",
    )
    parser.add_argument("--save-samples")
    parser.add_argument(
        "--load-samples",
        help="Replay a previously saved NPZ without DDS or a running simulator",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.duration_seconds <= 0.0 or args.warmup_seconds < 0.0:
        raise ValueError("duration must be positive and warm-up must be non-negative")
    if args.policy_frames <= 0:
        raise ValueError("policy-frames must be positive")
    if args.load_samples:
        if args.save_samples:
            raise ValueError("--load-samples and --save-samples are mutually exclusive")
        sample_path = Path(args.load_samples).expanduser().resolve()
        with np.load(sample_path) as loaded:
            recording = {name: loaded[name] for name in loaded.files}
        result = {
            "settings": {
                "loaded_samples": str(sample_path),
                "speed_bin_edges": list(args.speed_bin_edges),
                **asdict(estimator_config_from_args(args)),
            },
            **evaluate_recording(
                recording,
                estimator_config_from_args(args),
                args.speed_bin_edges,
            ),
        }
    else:
        result = collect(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
