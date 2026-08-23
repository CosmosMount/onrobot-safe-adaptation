"""RL-X vector-environment adapter backed only by Unitree SDK2 topics."""

from __future__ import annotations

import logging

import numpy as np
from gymnasium.spaces import Box

from rl_x.environments.safety_rollout import InvalidTransitionError

from ..common.action import ActionMapper, project_actions_from_observation
from ..common.estimation.velocity import quaternion_rotation_matrix_wxyz
from ..common.observation import ObservationBuilder
from ..common.reward import compute_reward
from ..common.specs import ACTION_SIZE, DEFAULT_JOINT_POSITION, OBSERVATION_SIZE
from ..common.termination import EpisodeTracker
from ..common.manifest import build_manifest, validate_manifest
from .fall_detector import FallDetector
from .general_properties import GeneralProperties
from .sdk_client import SDKClient
from .state_buffer import StateBufferError


logger = logging.getLogger("rl_x")


class Go2SDKMujocoEnv:
    general_properties = GeneralProperties
    policy_observation_indices = np.arange(OBSERVATION_SIZE)
    critic_observation_indices = np.arange(OBSERVATION_SIZE)
    safety_critic_observation_indices = np.arange(OBSERVATION_SIZE)

    def __init__(self, config, client: SDKClient | None = None, role: str = "train"):
        environment = config.environment
        self.config = environment
        self.role = role
        self.nr_envs = 1
        self.num_envs = 1
        self.single_observation_space = Box(
            low=-np.inf, high=np.inf, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )
        self.single_action_space = Box(
            low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32
        )
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self.client = client or SDKClient(environment.domain_id, environment.interface)
        self.action_mapper = ActionMapper()
        self.observation_builder = ObservationBuilder()
        self.fall_detector = FallDetector(
            environment.fall_angle_threshold, environment.fall_consecutive_frames
        )
        self.episode = EpisodeTracker(environment.episode_steps)
        self._last_tick: int | None = None
        self._generation = 0
        self._last_observation: np.ndarray | None = None

    @staticmethod
    def project_actions(states, actions):
        """Project actions using only the previous target encoded in ``states``.

        This hook is side-effect free so an algorithm may apply it to replay,
        policy candidates, and Q queries without advancing the DDS environment.
        NumPy arrays are returned as NumPy; Array-API/JAX inputs retain their
        array namespace when the installed array implementation exposes it.
        """

        return project_actions_from_observation(states, actions)

    def _wait_window(self, count: int | None = None):
        count = int(count or self.config.policy_frames)
        try:
            frames = self.client.state_buffer.wait_for_frames(
                count=count,
                after_tick=self._last_tick,
                timeout=float(self.config.state_timeout),
                generation=self._generation,
            )
        except StateBufferError as exc:
            raise InvalidTransitionError(str(exc)) from exc
        self._last_tick = int(frames[-1].tick)
        return frames

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.client.start()
        self._generation = self.client.state_buffer.generation
        self.action_mapper.reset()
        self.observation_builder.reset()
        self.fall_detector.reset()
        self.episode.reset()
        # Anchor immediately before publishing.  Frames accumulated while the
        # learner was busy belong to the preceding action and must not form the
        # next transition window.
        self._last_tick = self.client.state_buffer.last_tick
        self.client.publish_joint_target(DEFAULT_JOINT_POSITION)
        frames = self._wait_window()
        observation, _ = self.observation_builder.build(frames[-1])
        self._last_observation = observation
        return observation[None, :], {}

    def _manual_failure_reset(self):
        logger.warning(
            "Go2 fall detected. Press Backspace in the unitree_mujoco window to reset."
        )
        timeout = float(self.config.manual_reset_timeout)
        timeout = None if timeout < 0 else timeout
        generation = self.client.state_buffer.wait_for_restart(
            self._generation, timeout=timeout
        )
        self._generation = generation
        self._last_tick = None
        stable = 0
        while stable < int(self.config.stable_reset_frames):
            frames = self._wait_window(count=1)
            if self.fall_detector.is_stable(frames[-1].imu_quat):
                stable += 1
            else:
                stable = 0
        return self.reset()[0][0]

    def _logical_reset(self, state):
        self.action_mapper.reset()
        self.observation_builder.reset()
        self.fall_detector.reset()
        self.episode.reset()
        observation, _ = self.observation_builder.build(state)
        return observation

    def step(self, actions):
        action = np.asarray(actions, dtype=np.float32).reshape(-1, ACTION_SIZE)[0]
        mapped = self.action_mapper.apply(action)
        # JAX updates can take longer than one control interval while the C++
        # simulator continues publishing LowState.  Discard that backlog by
        # anchoring the next ten-frame window at the latest pre-command tick.
        self._last_tick = self.client.state_buffer.last_tick
        self.client.publish_joint_target(mapped.q_target)
        self.observation_builder.set_previous_q_target(mapped.q_target)
        frames = self._wait_window()

        failure = False
        for frame in frames:
            failure = self.fall_detector.update(frame.imu_quat) or failure
        final_state = frames[-1]
        observation, estimated_body_velocity = self.observation_builder.build(final_state)

        truth = self.client.latest_training_state()
        if truth is None:
            rotation = quaternion_rotation_matrix_wxyz(final_state.imu_quat)
            world_velocity = rotation @ estimated_body_velocity
            torque = np.zeros(ACTION_SIZE, dtype=np.float32)
        else:
            world_velocity = np.asarray(truth.world_velocity)
            torque = np.asarray(truth.actuator_torque)
        terms = compute_reward(
            world_velocity,
            final_state.imu_quat,
            final_state.imu_gyro,
            torque,
            float(self.config.target_velocity_x),
        )
        terminated, truncated = self.episode.advance(terms.total, failure)

        rotation = quaternion_rotation_matrix_wxyz(final_state.imu_quat)
        truth_body_velocity = rotation.T @ np.asarray(world_velocity)
        info = {
            "failure": np.asarray([int(failure)], dtype=np.float32),
            "applied_action": mapped.applied_action[None, :],
            "velocity_estimation_error": np.asarray(
                [np.linalg.norm(truth_body_velocity - estimated_body_velocity)],
                dtype=np.float32,
            ),
            **{key: np.asarray([value], dtype=np.float32) for key, value in terms.as_dict().items()},
        }

        terminal_info = None
        final_observation = None
        if terminated or truncated:
            final_observation = observation.copy()
            terminal_info = {
                **{key: float(value[0]) for key, value in info.items() if np.asarray(value).size == 1},
                "episode_return": self.episode.episode_return,
                "episode_length": self.episode.steps,
            }
            if terminated:
                observation = self._manual_failure_reset()
            else:
                observation = self._logical_reset(final_state)
            info["episode_return"] = np.asarray(
                [terminal_info["episode_return"]], dtype=np.float32
            )
            info["episode_length"] = np.asarray(
                [terminal_info["episode_length"]], dtype=np.float32
            )

        info["final_observation"] = [final_observation]
        info["final_info"] = [terminal_info]
        self._last_observation = observation
        return (
            observation[None, :],
            np.asarray([terms.total], dtype=np.float32),
            np.asarray([terminated], dtype=bool),
            np.asarray([truncated], dtype=bool),
            info,
        )

    def get_final_observation_at_index(self, info, index):
        return info["final_observation"][index]

    def get_final_info_value_at_index(self, info, key, index):
        final_info = info["final_info"][index]
        if final_info is None:
            raise KeyError(f"No final info for environment {index}")
        return final_info[key]

    def get_logging_info_dict(self, info):
        ignored = {"failure", "applied_action", "final_observation", "final_info"}
        return {
            key: np.asarray(value).reshape(-1).tolist()
            for key, value in info.items()
            if key not in ignored and not isinstance(value, list)
        }

    def close(self):
        self.client.close()

    def checkpoint_manifest(self, normalizer=None):
        return build_manifest(
            normalizer,
            fall_angle_threshold=float(self.config.fall_angle_threshold),
            fall_consecutive_frames=int(self.config.fall_consecutive_frames),
        )

    def validate_checkpoint_manifest(self, manifest, normalizer=None):
        validate_manifest(manifest, self.checkpoint_manifest(normalizer))
