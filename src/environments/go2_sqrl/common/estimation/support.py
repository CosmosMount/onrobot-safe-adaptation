"""Sensor-free stance-support inference from joint proprioception only."""

from __future__ import annotations

import numpy as np

from .kinematics import foot_jacobians_body, foot_positions_body


def infer_support_mask(joint_q: np.ndarray, joint_dq: np.ndarray) -> np.ndarray:
    """Infer likely supporting legs without a force/contact sensor.

    The heuristic uses only encoder positions and velocities: a leg is treated
    as supporting when its foot is near the lowest kinematic height and its
    vertical velocity is small.  It is an estimator-internal constraint and is
    never exposed as a policy observation.
    """

    positions = foot_positions_body(joint_q)
    jacobians = foot_jacobians_body(joint_q)
    velocities = np.einsum(
        "lij,lj->li", jacobians, np.asarray(joint_dq).reshape(4, 3)
    )
    lowest = positions[:, 2] <= np.min(positions[:, 2]) + 0.035
    slow_vertical = np.abs(velocities[:, 2]) < 0.35
    return lowest & slow_vertical
