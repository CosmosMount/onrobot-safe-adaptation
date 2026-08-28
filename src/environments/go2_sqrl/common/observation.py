"""Construction of the 46D Walk-in-the-Park-style policy observation."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .estimation import VelocityEstimator
from .specs import DEFAULT_JOINT_POSITION, OBSERVATION_SPEC
from .types import RobotState


def continuous_quaternion_wxyz(
    quaternion: np.ndarray, previous: np.ndarray | None
) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32).copy()
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Invalid IMU quaternion")
    quaternion /= norm
    if previous is None:
        if quaternion[0] < 0:
            quaternion *= -1
    elif float(np.dot(quaternion, previous)) < 0:
        quaternion *= -1
    return quaternion


def build_observation(
    state: RobotState,
    estimated_body_velocity: np.ndarray,
    previous_q_target: np.ndarray,
    previous_quaternion: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    quaternion = continuous_quaternion_wxyz(state.imu_quat, previous_quaternion)
    observation = np.empty(OBSERVATION_SPEC.size, dtype=np.float32)
    observation[OBSERVATION_SPEC.joint_q] = np.asarray(state.joint_q, dtype=np.float32)
    observation[OBSERVATION_SPEC.joint_dq] = np.asarray(state.joint_dq, dtype=np.float32)
    observation[OBSERVATION_SPEC.imu_gyro] = np.asarray(
        state.imu_gyro, dtype=np.float32
    )
    observation[OBSERVATION_SPEC.body_velocity] = np.asarray(
        estimated_body_velocity, dtype=np.float32
    )
    observation[OBSERVATION_SPEC.imu_quat] = quaternion
    observation[OBSERVATION_SPEC.previous_action_q_target] = np.asarray(
        previous_q_target, dtype=np.float32
    )
    if not np.all(np.isfinite(observation)):
        raise ValueError("Observation contains NaN or infinity")
    return observation, quaternion


class BodyVelocityEstimator(Protocol):
    def reset(self) -> None: ...

    def update(self, state: RobotState) -> np.ndarray: ...


class ObservationBuilder:
    def __init__(
        self, velocity_estimator: BodyVelocityEstimator | None = None
    ):
        self.velocity_estimator = velocity_estimator or VelocityEstimator()
        self.previous_quaternion: np.ndarray | None = None
        self.previous_q_target = DEFAULT_JOINT_POSITION.copy()

    def reset(self, previous_q_target: np.ndarray | None = None) -> None:
        self.velocity_estimator.reset()
        self.previous_quaternion = None
        self.previous_q_target = np.asarray(
            DEFAULT_JOINT_POSITION if previous_q_target is None else previous_q_target,
            dtype=np.float32,
        ).copy()

    def set_previous_q_target(self, q_target: np.ndarray) -> None:
        self.previous_q_target = np.asarray(q_target, dtype=np.float32).copy()

    def _build_with_velocity(
        self, state: RobotState, estimated_body_velocity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        observation, quaternion = build_observation(
            state,
            estimated_body_velocity,
            self.previous_q_target,
            self.previous_quaternion,
        )
        self.previous_quaternion = quaternion
        return observation, estimated_body_velocity

    def build(self, state: RobotState) -> tuple[np.ndarray, np.ndarray]:
        estimated_body_velocity = self.velocity_estimator.update(state)
        return self._build_with_velocity(state, estimated_body_velocity)

    def build_many(
        self, states: Sequence[RobotState]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance every physics frame and build one observation from the last."""

        if len(states) == 0:
            raise ValueError("states must contain at least one physics frame")
        estimated_body_velocity = None
        for state in states:
            estimated_body_velocity = self.velocity_estimator.update(state)
        return self._build_with_velocity(states[-1], estimated_body_velocity)
