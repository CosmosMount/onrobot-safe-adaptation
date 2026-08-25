"""Torch tensor adapters shared by the Isaac wrapper and contract tests."""

from __future__ import annotations

import numpy as np
import torch

from ..common.specs import ACTION_SPEC, DEFAULT_JOINT_POSITION, JOINT_NAMES


SDK_JOINT_NAMES = tuple(f"{name}_joint" for name in JOINT_NAMES)


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


def validate_action_term_contract(action_term) -> None:
    """Fail fast unless Isaac executes the exact shared SDK action mapping."""

    actual_names = tuple(action_term._joint_names)
    if actual_names != SDK_JOINT_NAMES:
        raise RuntimeError(
            "Isaac action joint order does not match the SDK contract: "
            f"expected {SDK_JOINT_NAMES}, got {actual_names}"
        )

    offset = action_term._offset
    if torch.is_tensor(offset):
        offset = offset[0].detach().cpu().numpy()
    else:
        offset = np.full(len(SDK_JOINT_NAMES), float(offset), dtype=np.float32)
    expected_offset = np.asarray(ACTION_SPEC.default_position, dtype=np.float32)
    if not np.allclose(offset, expected_offset, atol=1e-6, rtol=0.0):
        raise RuntimeError(
            "Isaac action offset does not match DEFAULT_JOINT_POSITION: "
            f"expected {expected_offset.tolist()}, got {np.asarray(offset).tolist()}"
        )

    scale = action_term._scale
    if torch.is_tensor(scale):
        scale = scale.detach().cpu().numpy()
    if not np.allclose(scale, ACTION_SPEC.scale, atol=1e-7, rtol=0.0):
        raise RuntimeError(
            "Isaac action scale does not match the shared action contract: "
            f"expected {ACTION_SPEC.scale}, got {scale}"
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
    velocity_command,
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
            velocity_command,
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
