"""Small analytic/numeric Go2 leg kinematics used by the estimator."""

from __future__ import annotations

import numpy as np


THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213
HIP_X = np.asarray([0.1934, 0.1934, -0.1934, -0.1934], dtype=np.float64)
HIP_Y = np.asarray([-0.142, 0.142, -0.142, 0.142], dtype=np.float64)


def foot_positions_body(joint_q: np.ndarray) -> np.ndarray:
    joint_q = np.asarray(joint_q, dtype=np.float64).reshape(4, 3)
    feet = np.zeros((4, 3), dtype=np.float64)
    for leg, (abduction, thigh, calf) in enumerate(joint_q):
        x = HIP_X[leg] - THIGH_LENGTH * np.sin(thigh) - CALF_LENGTH * np.sin(
            thigh + calf
        )
        z_plane = -THIGH_LENGTH * np.cos(thigh) - CALF_LENGTH * np.cos(
            thigh + calf
        )
        y = HIP_Y[leg] - z_plane * np.sin(abduction)
        z = z_plane * np.cos(abduction)
        feet[leg] = (x, y, z)
    return feet


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


def supported_body_velocity(
    joint_q: np.ndarray,
    joint_dq: np.ndarray,
    angular_velocity_body: np.ndarray,
    support_mask: np.ndarray,
) -> np.ndarray | None:
    """Estimate body velocity from legs inferred to be stationary supports."""

    support_mask = np.asarray(support_mask, dtype=bool).reshape(4)
    if not np.any(support_mask):
        return None
    positions = foot_positions_body(joint_q)
    jacobians = foot_jacobians_body(joint_q)
    joint_dq = np.asarray(joint_dq, dtype=np.float64).reshape(4, 3)
    omega = np.asarray(angular_velocity_body, dtype=np.float64)
    estimates = []
    for leg in np.flatnonzero(support_mask):
        foot_velocity_from_joints = jacobians[leg] @ joint_dq[leg]
        estimates.append(
            -(foot_velocity_from_joints + np.cross(omega, positions[leg]))
        )
    return np.mean(estimates, axis=0)
