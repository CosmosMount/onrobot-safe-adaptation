"""Small analytic/numeric Go2 leg kinematics used by the estimator."""

from __future__ import annotations

import numpy as np


THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213
HIP_X = np.asarray([0.1934, 0.1934, -0.1934, -0.1934], dtype=np.float64)
LEG_SIDE = np.asarray([-1.0, 1.0, -1.0, 1.0], dtype=np.float64)
HIP_ABDUCTION_Y = 0.0465
HIP_LATERAL_OFFSET = 0.0955


def foot_position_velocity_body(
    joint_q: np.ndarray, joint_dq: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact Go2 foot positions and joint-induced velocities."""

    q = np.asarray(joint_q, dtype=np.float64).reshape(4, 3)
    dq = np.asarray(joint_dq, dtype=np.float64).reshape(4, 3)
    abduction, thigh, calf = q.T
    abduction_dq, thigh_dq, calf_dq = dq.T
    total = thigh + calf
    lateral = LEG_SIDE * HIP_LATERAL_OFFSET
    x_plane = -THIGH_LENGTH * np.sin(thigh) - CALF_LENGTH * np.sin(total)
    z_plane = -THIGH_LENGTH * np.cos(thigh) - CALF_LENGTH * np.cos(total)
    positions = np.stack(
        (
            HIP_X + x_plane,
            LEG_SIDE * HIP_ABDUCTION_Y
            + lateral * np.cos(abduction)
            - z_plane * np.sin(abduction),
            lateral * np.sin(abduction) + z_plane * np.cos(abduction),
        ),
        axis=-1,
    )

    dx = (
        (-THIGH_LENGTH * np.cos(thigh) - CALF_LENGTH * np.cos(total))
        * thigh_dq
        - CALF_LENGTH * np.cos(total) * calf_dq
    )
    dz_plane = (
        (THIGH_LENGTH * np.sin(thigh) + CALF_LENGTH * np.sin(total))
        * thigh_dq
        + CALF_LENGTH * np.sin(total) * calf_dq
    )
    velocities = np.stack(
        (
            dx,
            (-lateral * np.sin(abduction) - z_plane * np.cos(abduction))
            * abduction_dq
            - dz_plane * np.sin(abduction),
            (lateral * np.cos(abduction) - z_plane * np.sin(abduction))
            * abduction_dq
            + dz_plane * np.cos(abduction),
        ),
        axis=-1,
    )
    return positions, velocities


def foot_positions_body(joint_q: np.ndarray) -> np.ndarray:
    positions, _ = foot_position_velocity_body(
        joint_q, np.zeros(12, dtype=np.float64)
    )
    return positions


def foot_jacobians_body(joint_q: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """Return four 3x3 Jacobians in SDK leg order."""

    joint_q = np.asarray(joint_q, dtype=np.float64).reshape(4, 3)
    jacobians = np.zeros((4, 3, 3), dtype=np.float64)
    for leg in range(4):
        for joint in range(3):
            positive = joint_q.copy()
            negative = joint_q.copy()
            positive[leg, joint] += epsilon
            negative[leg, joint] -= epsilon
            jacobians[leg, :, joint] = (
                foot_positions_body(positive.reshape(-1))[leg]
                - foot_positions_body(negative.reshape(-1))[leg]
            ) / (2.0 * epsilon)
    return jacobians
