"""Torch tensor adapters shared by the Isaac wrapper and contract tests."""

from __future__ import annotations

import torch

from ..common.specs import DEFAULT_JOINT_POSITION, JOINT_NAMES


def sdk_joint_indices(source_names, device=None):
    normalized = [name.removesuffix("_joint") for name in source_names]
    missing = [name for name in JOINT_NAMES if name not in normalized]
    if missing:
        raise ValueError(f"Missing Go2 joints in Isaac articulation: {missing}")
    return torch.tensor(
        [normalized.index(name) for name in JOINT_NAMES],
        dtype=torch.long,
        device=device,
    )


def continuous_quaternion(quaternion, previous=None):
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    if previous is None:
        sign = torch.where(quaternion[..., :1] < 0, -1.0, 1.0)
    else:
        sign = torch.where(
            torch.sum(quaternion * previous, dim=-1, keepdim=True) < 0,
            -1.0,
            1.0,
        )
    return quaternion * sign


def build_observation_tensor(
    joint_q,
    joint_dq,
    imu_gyro,
    body_velocity,
    imu_quat,
    previous_q_target,
    previous_quaternion=None,
):
    quaternion = continuous_quaternion(imu_quat, previous_quaternion)
    observation = torch.cat(
        (
            joint_q,
            joint_dq,
            imu_gyro,
            body_velocity,
            quaternion,
            previous_q_target,
        ),
        dim=-1,
    )
    if observation.shape[-1] != 46:
        raise ValueError(f"Isaac observation must have 46 values, got {observation.shape}")
    return observation, quaternion


def default_joint_target(batch_size, device):
    return torch.as_tensor(
        DEFAULT_JOINT_POSITION, dtype=torch.float32, device=device
    ).expand(batch_size, -1).clone()

