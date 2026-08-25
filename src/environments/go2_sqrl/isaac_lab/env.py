"""RL-X adapter around Isaac Lab's manager-based Go2 environment."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium.spaces import Box
from rl_x.environments.safety_rollout import RolloutStep

from ..common.specs import (
    ACTION_SPEC,
    ACTION_SIZE,
    DEFAULT_JOINT_POSITION,
    JOINT_LOWER_LIMIT,
    JOINT_UPPER_LIMIT,
    OBSERVATION_SPEC,
    OBSERVATION_SIZE,
    PHYSICS_STEPS_PER_ACTION,
)
from ..common.manifest import build_manifest, validate_manifest
from ..common.reward import (
    BASE_HEIGHT_TARGET,
    REWARD_DT,
    REWARD_DEFAULT_JOINT_POSITION,
    REWARD_SCALES,
    TRACKING_SIGMA,
)
from ..common.estimation import TorchVelocityEstimator
from .general_properties import GeneralProperties
from .mdp import (
    build_observation_tensor,
    default_joint_target,
    sdk_joint_indices,
    validate_action_term_contract,
)


def project_action_targets_tensor(previous_q_target, actions):
    """Torch-native, differentiable counterpart of the common projection."""

    if not torch.is_tensor(actions):
        actions = torch.as_tensor(actions, dtype=torch.float32)
    if not torch.is_tensor(previous_q_target):
        previous_q_target = torch.as_tensor(
            previous_q_target, dtype=actions.dtype, device=actions.device
        )
    else:
        previous_q_target = previous_q_target.to(
            dtype=actions.dtype, device=actions.device
        )
    default = torch.as_tensor(
        DEFAULT_JOINT_POSITION, dtype=actions.dtype, device=actions.device
    )
    lower = torch.as_tensor(
        JOINT_LOWER_LIMIT, dtype=actions.dtype, device=actions.device
    )
    upper = torch.as_tensor(
        JOINT_UPPER_LIMIT, dtype=actions.dtype, device=actions.device
    )
    raw_target = default + ACTION_SPEC.scale * actions.clamp(-1, 1)
    raw_target = raw_target.clamp(lower, upper)
    max_delta = ACTION_SPEC.max_target_rate * ACTION_SPEC.control_dt
    target = torch.maximum(
        torch.minimum(raw_target, previous_q_target + max_delta),
        previous_q_target - max_delta,
    )
    applied = ((target - default) / ACTION_SPEC.scale).clamp(-1, 1)
    return applied, target


class TorchFallDetector:
    """Vectorized sustained IMU roll/pitch detector for Isaac environments."""

    def __init__(
        self,
        nr_envs: int,
        device,
        angle_threshold: float = 0.8,
        consecutive_frames: int = 5,
        samples_per_update: int = 1,
    ):
        if consecutive_frames < 1:
            raise ValueError("consecutive_frames must be at least 1")
        if samples_per_update < 1:
            raise ValueError("samples_per_update must be at least 1")
        self.angle_threshold = float(angle_threshold)
        self.consecutive_frames = int(consecutive_frames)
        self.samples_per_update = int(samples_per_update)
        self.count = torch.zeros(nr_envs, dtype=torch.long, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.count.zero_()
        else:
            self.count[env_ids] = 0

    @torch.no_grad()
    def update(self, quaternion):
        quaternion = quaternion / torch.linalg.vector_norm(
            quaternion, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        w, x, y, z = quaternion.unbind(dim=-1)
        roll = torch.atan2(
            2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square())
        )
        pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
        tilted = (roll.abs() > self.angle_threshold) | (
            pitch.abs() > self.angle_threshold
        )
        self.count = torch.where(
            tilted,
            self.count + self.samples_per_update,
            torch.zeros_like(self.count),
        )
        return self.count >= self.consecutive_frames


class Go2IsaacEnv:
    general_properties = GeneralProperties

    def __init__(self, config, backend=None):
        self._owns_backend = backend is None
        if backend is None:
            from isaaclab.envs import ManagerBasedRLEnv
            from .env_cfg import make_env_cfg
            from .randomization_cfg import format_domain_randomization_report

            backend = ManagerBasedRLEnv(cfg=make_env_cfg(config))
            print(
                format_domain_randomization_report(
                    enabled=bool(config.environment.domain_randomization),
                    friction=float(config.environment.friction),
                ),
                flush=True,
            )
        self.backend = backend
        self.config = config.environment
        self.nr_envs = int(self.config.nr_envs)
        self.num_envs = self.nr_envs
        self.nr_task_envs = int(self.config.nr_task_envs)
        self.nr_safety_envs = int(self.config.nr_safety_envs)
        if self.nr_task_envs + self.nr_safety_envs != self.nr_envs:
            raise ValueError("Isaac task and safety pool sizes must sum to nr_envs")
        self.single_observation_space = Box(
            low=-np.inf, high=np.inf, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )
        self.single_action_space = Box(
            low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32
        )
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self.policy_observation_indices = np.arange(OBSERVATION_SIZE)
        self.critic_observation_indices = np.arange(OBSERVATION_SIZE)
        self.safety_critic_observation_indices = np.arange(OBSERVATION_SIZE)
        robot = self.backend.scene["robot"]
        validate_action_term_contract(
            self.backend.action_manager.get_term("joint_pos")
        )
        self._joint_indices = sdk_joint_indices(robot.joint_names, robot.device)
        self._previous_target = default_joint_target(self.nr_envs, robot.device)
        self._previous_reward_action = torch.zeros(
            (self.nr_envs, ACTION_SIZE), device=robot.device
        )
        self._previous_quaternion = None
        self._velocity_estimator = TorchVelocityEstimator(
            self.nr_envs, robot.device
        )
        self._fall_detector = TorchFallDetector(
            self.nr_envs,
            robot.device,
            angle_threshold=float(self.config.fall_angle_threshold),
            consecutive_frames=int(self.config.fall_consecutive_frames),
            # The adapter observes orientation once after each decimated policy
            # step.  Treating that sample as the latest 10 physics frames is a
            # conservative discrete approximation of SDK's 500 Hz sustained
            # detector and keeps the manifest's frame unit identical.
            samples_per_update=PHYSICS_STEPS_PER_ACTION,
        )
        self._episode_return = torch.zeros(self.nr_envs, device=robot.device)
        self._episode_length = torch.zeros(
            self.nr_envs, device=robot.device, dtype=torch.long
        )

    @staticmethod
    def project_actions(states, actions):
        """Return side-effect-free applied actions in the executable domain."""

        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.float32)
        if not torch.is_tensor(states):
            states = torch.as_tensor(
                states, dtype=actions.dtype, device=actions.device
            )
        else:
            states = states.to(dtype=actions.dtype, device=actions.device)
        if states.shape[-1] != OBSERVATION_SIZE:
            raise ValueError(f"Observation must end in 46 values, got {states.shape}")
        applied, _ = project_action_targets_tensor(
            states[..., OBSERVATION_SPEC.previous_action_q_target], actions
        )
        return applied

    @property
    def _robot(self):
        return self.backend.scene["robot"]

    def _observation(self):
        robot = self._robot
        imu = self.backend.scene["imu"]
        joint_q = robot.data.joint_pos[:, self._joint_indices]
        joint_dq = robot.data.joint_vel[:, self._joint_indices]
        self._latest_estimated_body_velocity = self._velocity_estimator.update(
            joint_q,
            joint_dq,
            imu.data.ang_vel_b,
            imu.data.quat_w,
            imu.data.lin_acc_b,
        )
        velocity_command = torch.zeros(
            (self.nr_envs, 3), dtype=joint_q.dtype, device=joint_q.device
        )
        velocity_command[:, 0] = float(self.config.target_velocity_x)
        observation, quaternion = build_observation_tensor(
            joint_q,
            joint_dq,
            imu.data.ang_vel_b,
            velocity_command,
            imu.data.quat_w,
            self._previous_target,
            self._previous_quaternion,
        )
        self._previous_quaternion = quaternion
        return observation

    def _reset_done_state(self, done):
        env_ids = torch.nonzero(done).flatten()
        if env_ids.numel() == 0:
            return
        self._previous_target[env_ids] = torch.as_tensor(
            DEFAULT_JOINT_POSITION,
            dtype=self._previous_target.dtype,
            device=self._previous_target.device,
        )
        self._previous_reward_action[env_ids] = 0.0
        self._velocity_estimator.reset(env_ids)
        self._fall_detector.reset(env_ids)
        if self._previous_quaternion is not None:
            current = self.backend.scene["imu"].data.quat_w[env_ids].clone()
            current = current / torch.linalg.vector_norm(
                current, dim=-1, keepdim=True
            ).clamp_min(1e-8)
            current = torch.where(current[:, :1] < 0, -current, current)
            self._previous_quaternion[env_ids] = current

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.backend.reset()
        self._previous_target = default_joint_target(
            self.nr_envs, self._robot.device
        )
        self._previous_reward_action.zero_()
        self._previous_quaternion = None
        self._velocity_estimator.reset()
        self._fall_detector.reset()
        self._episode_return.zero_()
        self._episode_length.zero_()
        return self._observation().detach().cpu().numpy(), {}

    def _reset_failed_backend_envs(self, failure, already_reset):
        """Reset only IMU-failed envs not already reset by Isaac's time limit."""

        env_ids = torch.nonzero(failure & ~already_reset).flatten()
        if env_ids.numel() == 0:
            return
        reset_idx = getattr(self.backend, "_reset_idx", None)
        if reset_idx is None:
            raise RuntimeError(
                "Isaac backend does not support indexed reset required by the "
                "adapter IMU failure contract"
            )
        reset_idx(env_ids)

    def step(self, actions):
        action = torch.as_tensor(
            actions, dtype=torch.float32, device=self._robot.device
        ).reshape(self.nr_envs, ACTION_SIZE)
        applied_action, target = project_action_targets_tensor(
            self._previous_target, action
        )
        reward_action = action.clamp(-1.0, 1.0)
        self._previous_target = target

        _, _, backend_terminated, truncated, extras = self.backend.step(applied_action)
        # Clone manager buffers before an adapter-driven indexed reset mutates
        # manager state in place.
        backend_terminated = backend_terminated.clone()
        truncated = truncated.clone()
        robot = self._robot
        del extras

        # Capture all training-only truth before resetting IMU-failed bodies.
        # The policy observes the proprioceptive estimator, not this simulator
        # value, so retain a snapshot for sim-to-sim diagnostics.
        target_velocity = float(self.config.target_velocity_x)
        body_velocity = robot.data.root_lin_vel_b.clone()
        tracking_lin_vel = torch.exp(
            -(
                (target_velocity - body_velocity[:, 0]).square()
                + body_velocity[:, 1].square()
            )
            / TRACKING_SIGMA
        )
        velocity_error = (
            (target_velocity - body_velocity[:, 0]).square()
            + body_velocity[:, 1].square()
        )
        tracking_ang_vel = torch.exp(
            -robot.data.root_ang_vel_b[:, 2].square() / TRACKING_SIGMA
        )
        lin_vel_z = body_velocity[:, 2].square()
        base_height = robot.data.root_pos_w[:, 2]
        base_height_error = (base_height - BASE_HEIGHT_TARGET).square()
        action_rate = (
            reward_action - self._previous_reward_action
        ).square().sum(dim=-1)
        joint_q = robot.data.joint_pos[:, self._joint_indices]
        default_joint_q = torch.as_tensor(
            REWARD_DEFAULT_JOINT_POSITION,
            dtype=joint_q.dtype,
            device=joint_q.device,
        )
        similar_to_default = torch.abs(joint_q - default_joint_q).sum(dim=-1)
        reward_terms = {
            "tracking_lin_vel": REWARD_DT
            * REWARD_SCALES["tracking_lin_vel"]
            * tracking_lin_vel,
            "velocity_error": REWARD_DT
            * REWARD_SCALES["velocity_error"]
            * velocity_error,
            "tracking_ang_vel": REWARD_DT
            * REWARD_SCALES["tracking_ang_vel"]
            * tracking_ang_vel,
            "lin_vel_z": REWARD_DT * REWARD_SCALES["lin_vel_z"] * lin_vel_z,
            "base_height": REWARD_DT
            * REWARD_SCALES["base_height"]
            * base_height_error,
            "action_rate": REWARD_DT
            * REWARD_SCALES["action_rate"]
            * action_rate,
            "similar_to_default": REWARD_DT
            * REWARD_SCALES["similar_to_default"]
            * similar_to_default,
        }
        reward = torch.stack(tuple(reward_terms.values()), dim=0).sum(dim=0)
        self._previous_reward_action.copy_(reward_action)
        failure = self._fall_detector.update(
            self.backend.scene["imu"].data.quat_w
        )
        failure = failure.clone()
        terminated = backend_terminated | failure
        already_reset = backend_terminated | truncated
        self._reset_failed_backend_envs(failure, already_reset)
        done = terminated | truncated
        self._reset_done_state(done)
        observation = self._observation()
        estimated_body_velocity = self._latest_estimated_body_velocity.clone()

        self._episode_return += reward
        self._episode_length += 1
        info = {
            "failure": failure.float().cpu().numpy(),
            "applied_action": applied_action.detach().cpu().numpy(),
            "forward_velocity": body_velocity[:, 0].detach().cpu().numpy(),
            "estimated_forward_velocity": estimated_body_velocity[
                :, 0
            ].detach().cpu().numpy(),
            "target_velocity_error": (
                body_velocity[:, 0] - target_velocity
            ).abs().detach().cpu().numpy(),
            "velocity_estimation_error": torch.linalg.vector_norm(
                estimated_body_velocity - body_velocity, dim=-1
            ).detach().cpu().numpy(),
            **{
                f"reward/{name}": value.detach().cpu().numpy()
                for name, value in reward_terms.items()
            },
            "reward/total": reward.detach().cpu().numpy(),
            "final_observation": [None] * self.nr_envs,
            "final_info": [None] * self.nr_envs,
        }
        for index in torch.nonzero(done).flatten().tolist():
            info["final_observation"][index] = observation[index].detach().cpu().numpy()
            info["final_info"][index] = {
                "episode_return": float(self._episode_return[index]),
                "episode_length": int(self._episode_length[index]),
            }
            self._episode_return[index] = 0
            self._episode_length[index] = 0
        return (
            observation.detach().cpu().numpy(),
            reward.detach().cpu().numpy(),
            terminated.detach().cpu().numpy(),
            truncated.detach().cpu().numpy(),
            info,
        )

    def reset_partitions(self):
        observation, _ = self.reset()
        return (
            observation[: self.nr_task_envs],
            observation[self.nr_task_envs :],
        )

    def step_partitions(self, task_actions, safety_actions):
        actions = np.concatenate((task_actions, safety_actions), axis=0)
        observation, reward, terminated, truncated, info = self.step(actions)

        def slice_info(start, stop):
            sliced = {}
            for key, value in info.items():
                if isinstance(value, list):
                    sliced[key] = value[start:stop]
                else:
                    array = np.asarray(value)
                    sliced[key] = array[start:stop]
            return sliced

        split = self.nr_task_envs
        task_step = RolloutStep(
            observation[:split],
            reward[:split],
            terminated[:split],
            truncated[:split],
            slice_info(0, split),
        )
        safety_step = RolloutStep(
            observation[split:],
            reward[split:],
            terminated[split:],
            truncated[split:],
            slice_info(split, self.nr_envs),
        )
        return task_step, safety_step

    def get_final_observation_at_index(self, info, index):
        return info["final_observation"][index]

    def get_final_info_value_at_index(self, info, key, index):
        return info["final_info"][index][key]

    def get_logging_info_dict(self, info):
        return {
            "velocity_estimation_error": info["velocity_estimation_error"].tolist()
        }

    def close(self):
        if self._owns_backend:
            self.backend.close()

    def checkpoint_manifest(self, normalizer=None):
        return build_manifest(
            normalizer,
            fall_angle_threshold=float(self.config.fall_angle_threshold),
            fall_consecutive_frames=int(self.config.fall_consecutive_frames),
            target_velocity_x=float(self.config.target_velocity_x),
        )

    def validate_checkpoint_manifest(self, manifest, normalizer=None):
        validate_manifest(manifest, self.checkpoint_manifest(normalizer))
