"""Decode Unitree SDK messages into dependency-free state records."""

from __future__ import annotations

from typing import Any

import numpy as np

from train.core.base import ACTION_SIZE, RobotState

from .buffers import SynchronizedTrainingState, TrainingStateSyncError


SIMULATOR_TICK_SECONDS = 1.0e-3


TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_HIGHSTATE = "rt/sportmodestate"


def _read(value: Any) -> Any:
    return value() if callable(value) else value


def _field(value: Any, name: str) -> Any | None:
    if not hasattr(value, name):
        return None
    return _read(getattr(value, name))


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(_read(value), dtype=np.float32).reshape(-1)
    if result.size < int(size):
        raise ValueError(f"{name} must contain at least {size} values")
    result = result[:size].copy()
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def decode_simulator_tick(message: Any) -> int:
    """Decode the bridge's simulation tick from an explicit field or stamp.

    The stock bridge leaves ``SportModeState.stamp`` unset. Strict per-frame
    transition semantics require the lockstep bridge patch shipped with the
    robot asset, which timestamps root truth at the same post-step snapshot.
    """

    for name in ("simulation_tick", "tick"):
        value = _field(message, name)
        if value is not None:
            tick = int(value)
            if tick < 0:
                raise TrainingStateSyncError(
                    f"SportModeState.{name} must be non-negative, got {tick}"
                )
            return tick

    stamp = _field(message, "stamp")
    if stamp is None:
        raise TrainingStateSyncError(
            "SportModeState has no simulation tick or stamp. Apply "
            "assets/robots/go2/unitree_mujoco_bridge_lockstep.patch."
        )
    seconds = _field(stamp, "sec")
    nanoseconds = _field(stamp, "nanosec")
    if nanoseconds is None:
        nanoseconds = _field(stamp, "nsec")
    if seconds is None or nanoseconds is None:
        raise TrainingStateSyncError(
            "SportModeState.stamp must expose sec and nanosec fields"
        )
    seconds = int(seconds)
    nanoseconds = int(nanoseconds)
    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        raise TrainingStateSyncError(
            "SportModeState.stamp is not a valid non-negative simulation time"
        )
    simulation_seconds = seconds + nanoseconds * 1.0e-9
    tick = int(round(simulation_seconds / SIMULATOR_TICK_SECONDS))
    if not np.isclose(
        simulation_seconds,
        tick * SIMULATOR_TICK_SECONDS,
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise TrainingStateSyncError(
            "SportModeState.stamp is not aligned to the 1 ms bridge tick"
        )
    return tick


def decode_low_state(message: Any) -> RobotState:
    motors = list(_read(message.motor_state))
    if len(motors) < ACTION_SIZE:
        raise ValueError(
            f"LowState must expose {ACTION_SIZE} SDK-ordered motors, got {len(motors)}"
        )
    motors = motors[:ACTION_SIZE]
    imu = _read(message.imu_state)
    reserve = _field(message, "reserve")
    if reserve is None:
        raise ValueError(
            "LowState has no command-sequence reserve field; the strict "
            "ORSA lockstep bridge is required"
        )
    command_sequence = int(reserve)
    if not 0 <= command_sequence <= np.iinfo(np.uint32).max:
        raise ValueError(
            "LowState command sequence must fit uint32, got "
            f"{command_sequence}"
        )
    state = RobotState(
        joint_q=np.asarray([float(_read(motor.q)) for motor in motors], dtype=np.float32),
        joint_dq=np.asarray(
            [float(_read(motor.dq)) for motor in motors], dtype=np.float32
        ),
        imu_gyro=_finite_vector(imu.gyroscope, 3, "LowState IMU gyroscope"),
        imu_quat=_finite_vector(imu.quaternion, 4, "LowState IMU quaternion"),
        imu_accelerometer=_finite_vector(
            imu.accelerometer, 3, "LowState IMU accelerometer"
        ),
        actuator_torque=np.asarray(
            [float(_read(motor.tau_est)) for motor in motors], dtype=np.float32
        ),
        tick=int(_read(message.tick)),
        command_sequence=command_sequence,
    )
    for name in ("joint_q", "joint_dq", "actuator_torque"):
        if not np.all(np.isfinite(getattr(state, name))):
            raise ValueError(f"LowState {name} contains NaN or infinity")
    if state.tick < 0:
        raise ValueError(f"LowState tick must be non-negative, got {state.tick}")
    return state


def validate_low_state_crc(message: Any, crc: Any) -> None:
    """Reject a LowState whose CRC does not cover the received payload."""

    received = _field(message, "crc")
    if received is None:
        raise ValueError("LowState has no CRC field")
    received = int(received)
    expected = int(crc.Crc(message))
    if received != expected:
        raise ValueError(
            f"LowState CRC mismatch: received {received:#010x}, "
            f"expected {expected:#010x}"
        )


def decode_training_state(
    message: Any,
    torque: np.ndarray,
) -> SynchronizedTrainingState:
    command_sequence = _field(message, "error_code")
    if command_sequence is None:
        raise TrainingStateSyncError(
            "SportModeState has no applied-command tag. Apply "
            "assets/robots/go2/unitree_mujoco_bridge_lockstep.patch."
        )
    command_sequence = int(command_sequence)
    if not 0 <= command_sequence <= np.iinfo(np.uint32).max:
        raise TrainingStateSyncError(
            "SportModeState applied-command tag must fit uint32"
        )
    return SynchronizedTrainingState(
        world_velocity=_finite_vector(
            message.velocity, 3, "SportModeState root velocity"
        ),
        base_position=_finite_vector(
            message.position, 3, "SportModeState root position"
        ),
        actuator_torque=np.asarray(torque, dtype=np.float32).copy(),
        tick=decode_simulator_tick(message),
        command_sequence=command_sequence,
    )

