"""Command-following FlashSAC Go2 reward shared by both simulators."""

from __future__ import annotations

import math

import numpy as np

from .types import RewardTerms


REWARD_VERSION = "flashsac-go2-walk-easy-command-v6-state-trot-phase"
HIGH_CLEARANCE_REWARD_VERSION = (
    "flashsac-go2-walk-easy-command-v7-high-clearance"
)
REWARD_DT = 0.02
TRACKING_SIGMA = 0.25
BASE_HEIGHT_TARGET = 0.3
FOOT_CLEARANCE_TARGET = 0.07
PHASE_REFERENCE_FREQUENCY = 2.0
PHASE_EPSILON = 1e-6
SWING_SPEED_START = 0.15
SWING_SPEED_FULL = 0.50
REWARD_SCALES = {
    "tracking_lin_vel": 1.0,
    # The source exponential still pays 36.8% of its maximum at zero speed
    # for a 0.5 m/s command.  This explicit error cost makes standing still a
    # negative-return local optimum while leaving exact tracking unchanged.
    "velocity_error": -3.0,
    "tracking_ang_vel": 0.2,
    "lin_vel_z": -1.0,
    "base_height": -50.0,
    "foot_clearance": -20.0,
    # This remains configurable so the unchanged reward (scale=0) and the
    # three planned phase-reward candidates share one implementation.
    "phase": 0.3,
    "action_rate": -0.005,
    "similar_to_default": -0.1,
}
REWARD_DEFAULT_JOINT_POSITION = np.asarray(
    [
        0.0, 0.8, -1.5,
        0.0, 0.8, -1.5,
        0.0, 1.0, -1.5,
        0.0, 1.0, -1.5,
    ],
    dtype=np.float32,
)


def add_terminal_failure_penalty(reward, failure, penalty: float):
    """Add a one-shot task penalty to the transition that ends in a fall.

    This deliberately supports Python/NumPy scalars and Torch tensors so both
    simulator adapters use the same operation.  A zero value preserves the
    legacy Isaac pre-training reward exactly.
    """

    return reward + failure * float(penalty)


def local_base_clearance(base_world_height, local_ground_world_height):
    """Return base height measured from the local terrain surface."""

    return base_world_height - local_ground_world_height


def deterministic_ground_height(
    world_x,
    *,
    terrain_profile="flat",
    step_start_x=1.0,
    step_height=0.04,
):
    """Ground truth for the deterministic MuJoCo flat/one-step scenes.

    This function is training/evaluation truth only.  Its result is never
    appended to the actor or QSafe observation.
    """

    profile = str(terrain_profile).lower()
    x = np.asarray(world_x, dtype=np.float64)
    if profile == "flat":
        result = np.zeros_like(x)
    elif profile == "single_step_up":
        height = float(step_height)
        if not 0.0 < height <= 0.10:
            raise ValueError("step_height must be in (0, 0.10] m")
        result = np.where(x >= float(step_start_x), height, 0.0)
    else:
        raise ValueError(
            "MuJoCo terrain_profile must be 'flat' or 'single_step_up'."
        )
    return float(result) if result.ndim == 0 else result


def swing_weights(foot_horizontal_speed):
    """Return the continuous, contact-free swing activity of every foot."""

    speed = np.asarray(foot_horizontal_speed, dtype=np.float64)
    return np.clip(
        (speed - SWING_SPEED_START) / (SWING_SPEED_FULL - SWING_SPEED_START),
        0.0,
        1.0,
    )


def swing_foot_clearance_error(
    foot_clearance,
    foot_horizontal_speed,
    *,
    target: float = FOOT_CLEARANCE_TARGET,
    aggregation: str = "swing_weighted",
):
    """Return swing-weighted clearance error without diluting over four feet."""

    clearance = np.asarray(foot_clearance, dtype=np.float64)
    weight = swing_weights(foot_horizontal_speed)
    deficit = np.maximum(float(target) - clearance, 0.0)
    if aggregation == "legacy_mean":
        return float(np.mean(weight * np.square(deficit)))
    if aggregation != "swing_weighted":
        raise ValueError(
            "clearance aggregation must be 'legacy_mean' or 'swing_weighted'"
        )
    return float(np.sum(weight * np.square(deficit)) / max(float(weight.sum()), 1.0))


def swing_foot_clearance_overshoot_error(
    foot_clearance,
    foot_horizontal_speed,
    *,
    upper_target: float,
):
    """Penalize only excessive active-swing clearance."""

    clearance = np.asarray(foot_clearance, dtype=np.float64)
    weight = swing_weights(foot_horizontal_speed)
    overshoot = np.maximum(clearance - float(upper_target), 0.0)
    return float(
        np.sum(weight * np.square(overshoot))
        / max(float(weight.sum()), 1.0)
    )


def movement_reward_gate(
    forward_velocity,
    *,
    start: float,
    full: float,
):
    """Return one when disabled, otherwise a smooth forward-motion gate."""

    start = float(start)
    full = float(full)
    if full <= start:
        return np.ones_like(np.asarray(forward_velocity, dtype=np.float64))
    return np.clip(
        (np.asarray(forward_velocity, dtype=np.float64) - start)
        / (full - start),
        0.0,
        1.0,
    )


def state_estimated_trot_phase_reward(
    foot_clearance,
    foot_vertical_velocity,
    foot_horizontal_speed,
    *,
    target: float = FOOT_CLEARANCE_TARGET,
    reference_frequency: float = PHASE_REFERENCE_FREQUENCY,
    epsilon: float = PHASE_EPSILON,
):
    """Estimate diagonal-trot phase from current foot motion only.

    Leg order is the shared SDK order ``FR, FL, RR, RL``.  No clock, contact
    sensor, phase observation, or hidden controller state enters this score.
    """

    target = float(target)
    reference_frequency = float(reference_frequency)
    if target <= 0.0:
        raise ValueError("foot clearance target must be positive")
    if reference_frequency <= 0.0:
        raise ValueError("phase reference frequency must be positive")
    clearance = np.asarray(foot_clearance, dtype=np.float64)
    vertical_velocity = np.asarray(foot_vertical_velocity, dtype=np.float64)
    if clearance.shape[-1] != 4 or vertical_velocity.shape[-1] != 4:
        raise ValueError("phase reward expects four feet in FR, FL, RR, RL order")
    x = (
        np.clip(clearance, 0.0, target) - 0.5 * target
    ) / (0.5 * target)
    y = vertical_velocity / (math.pi * reference_frequency * target)
    phase = np.stack((x, y), axis=-1)
    phase /= np.linalg.norm(phase, axis=-1, keepdims=True) + float(epsilon)
    fr, fl, rr, rl = np.moveaxis(phase, -2, 0)
    same_fr_rl = 0.5 * (1.0 + np.sum(fr * rl, axis=-1))
    same_fl_rr = 0.5 * (1.0 + np.sum(fl * rr, axis=-1))
    opposite_fr_fl = 0.5 * (1.0 - np.sum(fr * fl, axis=-1))
    opposite_rl_rr = 0.5 * (1.0 - np.sum(rl * rr, axis=-1))
    phase_score = 0.25 * (
        same_fr_rl + same_fl_rr + opposite_fr_fl + opposite_rl_rr
    )
    activity = np.clip(np.sum(swing_weights(foot_horizontal_speed), axis=-1) / 2.0, 0.0, 1.0)
    return activity * phase_score


def quaternion_to_rpy_wxyz(
    quaternion: np.ndarray,
) -> tuple[float, float, float]:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def compute_reward(
    world_velocity: np.ndarray,
    imu_quat: np.ndarray,
    imu_gyro: np.ndarray,
    joint_q: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
    target_velocity_x: float,
    *,
    base_clearance: float,
    foot_clearance_error: float = 0.0,
    foot_clearance_scale: float = REWARD_SCALES["foot_clearance"],
    foot_clearance_overshoot_error: float = 0.0,
    foot_clearance_overshoot_scale: float = 0.0,
    phase_reward: float = 0.0,
    phase_reward_scale: float = REWARD_SCALES["phase"],
    phase_movement_gate: float = 1.0,
    stable_progress_reward: float = 0.0,
    stable_progress_scale: float = 0.0,
    target_velocity_y: float = 0.0,
    target_angular_velocity_z: float = 0.0,
) -> RewardTerms:
    """Compute the source reward plus a continuous command-error penalty."""

    quaternion = np.asarray(imu_quat, dtype=np.float64)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-8)
    w, x, y, z = quaternion
    rotation_body_to_world = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    body_velocity = rotation_body_to_world.T @ np.asarray(
        world_velocity, dtype=np.float64
    )[:3]
    command_xy = np.asarray(
        [target_velocity_x, target_velocity_y], dtype=np.float64
    )
    linear_error = float(np.square(command_xy - body_velocity[:2]).sum())
    angular_error = (
        float(target_angular_velocity_z) - float(np.asarray(imu_gyro)[2])
    ) ** 2
    tracking_lin_vel = math.exp(-linear_error / TRACKING_SIGMA)
    tracking_ang_vel = math.exp(-angular_error / TRACKING_SIGMA)
    lin_vel_z = float(body_velocity[2] ** 2)
    base_height_error = (float(base_clearance) - BASE_HEIGHT_TARGET) ** 2
    action_rate = float(
        np.square(
            np.asarray(previous_action, dtype=np.float64)
            - np.asarray(action, dtype=np.float64)
        ).sum()
    )
    similar_to_default = float(
        np.abs(
            np.asarray(joint_q, dtype=np.float64)
            - np.asarray(REWARD_DEFAULT_JOINT_POSITION, dtype=np.float64)
        ).sum()
    )
    weighted = {
        "tracking_lin_vel": REWARD_DT
        * REWARD_SCALES["tracking_lin_vel"]
        * tracking_lin_vel,
        "velocity_error": REWARD_DT
        * REWARD_SCALES["velocity_error"]
        * linear_error,
        "tracking_ang_vel": REWARD_DT
        * REWARD_SCALES["tracking_ang_vel"]
        * tracking_ang_vel,
        "lin_vel_z": REWARD_DT * REWARD_SCALES["lin_vel_z"] * lin_vel_z,
        "base_height": REWARD_DT
        * REWARD_SCALES["base_height"]
        * base_height_error,
        "foot_clearance": REWARD_DT
        * float(foot_clearance_scale)
        * float(foot_clearance_error),
        "foot_clearance_overshoot": REWARD_DT
        * float(foot_clearance_overshoot_scale)
        * float(foot_clearance_overshoot_error),
        "phase": REWARD_DT
        * float(phase_reward_scale)
        * float(phase_movement_gate)
        * float(phase_reward),
        "stable_progress": REWARD_DT
        * float(stable_progress_scale)
        * float(stable_progress_reward),
        "action_rate": REWARD_DT
        * REWARD_SCALES["action_rate"]
        * action_rate,
        "similar_to_default": REWARD_DT
        * REWARD_SCALES["similar_to_default"]
        * similar_to_default,
    }
    return RewardTerms(**weighted, total=sum(weighted.values()))
