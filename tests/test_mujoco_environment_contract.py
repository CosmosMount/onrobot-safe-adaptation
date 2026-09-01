from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from ml_collections import config_dict

from train.core.environment import InvalidTransitionError
from train.core.base import DEFAULT_JOINT_POSITION, OBSERVATION_SPEC, RobotState
from train.core.task import EpisodeTracker
from train.mujoco.pytorch.environment import FallDetector, Go2MujocoEnv, get_config
from train.mujoco.pytorch.sdk import SynchronizedTrainingState


def _state(
    tick: int,
    *,
    quaternion=None,
    command_sequence: int | None = None,
) -> RobotState:
    return RobotState(
        joint_q=DEFAULT_JOINT_POSITION.copy(),
        joint_dq=np.zeros(12, dtype=np.float32),
        imu_gyro=np.zeros(3, dtype=np.float32),
        imu_quat=np.asarray(
            [1.0, 0.0, 0.0, 0.0] if quaternion is None else quaternion,
            dtype=np.float32,
        ),
        imu_accelerometer=np.asarray([0.0, 0.0, 9.81], dtype=np.float32),
        actuator_torque=np.zeros(12, dtype=np.float32),
        tick=tick,
        command_sequence=command_sequence,
    )


def _truth(tick: int, *, height: float = 0.289) -> SynchronizedTrainingState:
    return SynchronizedTrainingState(
        world_velocity=np.asarray([0.5, 0.0, 0.0], dtype=np.float32),
        base_position=np.asarray([0.0, 0.0, height], dtype=np.float32),
        actuator_torque=np.zeros(12, dtype=np.float32),
        tick=tick,
    )


class _FakeStateBuffer:
    generation = 0
    last_tick = 0
    latest_state = None


class _FakeClient:
    def __init__(self):
        self.state_buffer = _FakeStateBuffer()
        self.published = []
        self.started = False

    def start(self):
        self.started = True

    def publish_joint_target(self, target):
        self.published.append(np.asarray(target).copy())

    def close(self):
        pass


def _environment() -> Go2MujocoEnv:
    environment = get_config("test.mujoco")
    config = config_dict.ConfigDict(
        {
            "environment": environment,
            "runner": {"mode": "test"},
        }
    )
    return Go2MujocoEnv(
        config,
        client=_FakeClient(),
        role="eval",
        reset_controller=SimpleNamespace(reset=lambda: None),
    )


class MujocoFallDetectorTest(unittest.TestCase):
    def test_tilt_requires_five_consecutive_2ms_frames(self):
        detector = FallDetector()
        tilted = np.asarray(
            [math.cos(0.45), math.sin(0.45), 0.0, 0.0], dtype=np.float32
        )
        for _ in range(4):
            self.assertFalse(detector.update_frame(tilted, 0.289))
        self.assertTrue(detector.update_frame(tilted, 0.289))

    def test_height_requires_five_consecutive_2ms_frames(self):
        detector = FallDetector()
        upright = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(4):
            self.assertFalse(detector.update_frame(upright, 0.17))
        self.assertTrue(detector.update_frame(upright, 0.17))

    def test_one_good_frame_resets_each_consecutive_counter(self):
        detector = FallDetector()
        tilted = np.asarray(
            [math.cos(0.45), math.sin(0.45), 0.0, 0.0], dtype=np.float32
        )
        upright = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(4):
            detector.update_frame(tilted, 0.17)
        self.assertFalse(detector.update_frame(upright, 0.289))
        for _ in range(4):
            self.assertFalse(detector.update_frame(tilted, 0.17))


class MujocoTransitionContractTest(unittest.TestCase):
    def test_eval_role_can_use_a_longer_episode_without_changing_contract(self):
        environment = get_config("test.mujoco")
        environment.evaluation_episode_steps = 3000
        config = config_dict.ConfigDict(
            {"environment": environment, "runner": {"mode": "test"}}
        )

        env = Go2MujocoEnv(
            config,
            client=_FakeClient(),
            role="eval",
            reset_controller=SimpleNamespace(reset=lambda: None),
        )

        self.assertEqual(env.episode.max_steps, 3000)
        self.assertEqual(environment.episode_steps, 500)

    def test_reset_pose_requires_all_four_foot_surfaces_on_ground(self):
        env = _environment()
        state = _state(2)
        env._training_states_for_frames = lambda frames: [_truth(frames[0].tick)]

        self.assertTrue(env._reset_pose_ready(state))

        raised_foot_positions = np.asarray(
            [
                [0.2, -0.1, -0.20],
                [0.2, 0.1, -0.267],
                [-0.2, -0.1, -0.267],
                [-0.2, 0.1, -0.267],
            ],
            dtype=np.float64,
        )
        with mock.patch(
            "train.mujoco.pytorch.environment.foot_position_velocity_body",
            return_value=(raised_foot_positions, np.zeros((4, 3))),
        ):
            self.assertFalse(env._reset_pose_ready(state))

    def test_reset_pose_requires_identity_yaw_not_only_upright_tilt(self):
        env = _environment()
        yawed = _state(
            2,
            quaternion=np.asarray(
                [math.cos(0.05), 0.0, 0.0, math.sin(0.05)],
                dtype=np.float32,
            ),
        )
        env._training_states_for_frames = lambda frames: [_truth(frames[0].tick)]
        self.assertFalse(env._reset_pose_ready(yawed))

    def test_reset_pose_accepts_stable_torque_controlled_pd_sag(self):
        env = _environment()
        settled = _state(
            2,
            quaternion=np.asarray(
                [math.cos(0.0125), math.sin(0.0125), 0.0, 0.0],
                dtype=np.float32,
            ),
        )
        settled = RobotState(
            joint_q=settled.joint_q
            + np.tile(np.asarray([0.02, -0.03, -0.13], dtype=np.float32), 4),
            joint_dq=settled.joint_dq,
            imu_gyro=settled.imu_gyro,
            imu_quat=settled.imu_quat,
            imu_accelerometer=settled.imu_accelerometer,
            actuator_torque=settled.actuator_torque,
            tick=settled.tick,
            command_sequence=settled.command_sequence,
        )
        env._training_states_for_frames = lambda frames: [
            _truth(frames[0].tick, height=0.253)
        ]

        self.assertTrue(env._reset_pose_ready(settled))

    def test_sdk_bridge_rejects_a_fake_vector_width(self):
        environment = get_config("test.mujoco")
        environment.nr_envs = 2
        config = config_dict.ConfigDict(
            {"environment": environment, "runner": {"mode": "test"}}
        )
        with self.assertRaisesRegex(ValueError, "exactly one robot"):
            Go2MujocoEnv(
                config,
                client=_FakeClient(),
                role="eval",
                reset_controller=SimpleNamespace(reset=lambda: None),
            )

    def test_policy_window_requires_every_2ms_tick(self):
        env = _environment()
        env._last_tick = None
        env.client.state_buffer.wait_for_frames = lambda **kwargs: [
            _state(2),
            _state(4),
        ]
        self.assertEqual(
            [frame.tick for frame in env._wait_window(count=2)],
            [2, 4],
        )
        env._last_tick = None
        env.client.state_buffer.wait_for_frames = lambda **kwargs: [
            _state(2),
            _state(6),
        ]
        with self.assertRaisesRegex(InvalidTransitionError, "missing a 2 ms"):
            env._wait_window(count=2)
        env._last_tick = 20
        env.client.state_buffer.wait_for_frames = lambda **kwargs: [
            _state(24),
            _state(26),
        ]
        with self.assertRaisesRegex(InvalidTransitionError, "first 2 ms"):
            env._wait_window(count=2)

    def test_policy_window_requires_one_acknowledged_command_sequence(self):
        env = _environment()
        env._last_tick = 0
        env.client.state_buffer.wait_for_frames = lambda **kwargs: [
            _state(tick, command_sequence=7) for tick in (2, 4)
        ]
        self.assertEqual(
            [frame.tick for frame in env._wait_window(
                count=2, expected_command_sequence=7
            )],
            [2, 4],
        )

        env._last_tick = 0
        env.client.state_buffer.wait_for_frames = lambda **kwargs: [
            _state(2, command_sequence=6),
            _state(4, command_sequence=7),
        ]
        with self.assertRaisesRegex(
            InvalidTransitionError, "acknowledged post-step window"
        ):
            env._wait_window(count=2, expected_command_sequence=7)

    def test_unobserved_learner_gap_fails_before_the_next_command(self):
        env = _environment()
        env._last_tick = 20
        env.client.state_buffer.last_tick = 24
        with self.assertRaisesRegex(
            InvalidTransitionError, "advanced while the policy/learner"
        ):
            env._assert_lockstep_boundary()

    def test_timeout_physically_resets_but_preserves_terminal_observation(self):
        env = _environment()
        frames = [_state(tick) for tick in range(2, 22, 2)]
        final_quaternion = np.asarray(
            [math.cos(0.05), 0.0, 0.0, math.sin(0.05)], dtype=np.float32
        )
        frames[-1] = _state(20, quaternion=final_quaternion)
        truths = [_truth(frame.tick) for frame in frames]
        env._wait_window = lambda count=None: frames
        env._training_states_for_frames = lambda supplied: truths
        env.episode = EpisodeTracker(max_steps=1)
        reset_observation = np.full(46, -7.0, dtype=np.float32)
        reset_reasons = []

        def physical_reset(reason, delay_seconds=0.0):
            reset_reasons.append((reason, delay_seconds))
            return reset_observation.copy()

        env._physical_episode_reset = physical_reset
        observation, _, terminated, truncated, info = env.step(
            np.zeros((1, 12), dtype=np.float32)
        )
        self.assertFalse(bool(terminated[0]))
        self.assertTrue(bool(truncated[0]))
        self.assertEqual(len(reset_reasons), 1)
        np.testing.assert_array_equal(observation[0], reset_observation)
        self.assertIsNotNone(info["final_observation"][0])
        self.assertFalse(
            np.array_equal(info["final_observation"][0], reset_observation)
        )
        np.testing.assert_allclose(
            info["final_observation"][0][OBSERVATION_SPEC.imu_quat],
            final_quaternion,
            atol=1e-6,
        )

    def test_failure_uses_the_first_triggering_physics_frame(self):
        env = _environment()
        tilted = [
            np.asarray(
                [math.cos(angle / 2.0), math.sin(angle / 2.0), 0.0, 0.0],
                dtype=np.float32,
            )
            for angle in np.linspace(0.9, 1.1, 5)
        ]
        frames = [
            _state(tick, quaternion=tilted[index] if index < 5 else None)
            for index, tick in enumerate(range(2, 22, 2))
        ]
        truths = [
            _truth(frame.tick, height=0.17 if index < 5 else 0.289)
            for index, frame in enumerate(frames)
        ]
        env._wait_window = lambda count=None: frames
        env._training_states_for_frames = lambda supplied: truths
        reset_observation = np.zeros(46, dtype=np.float32)
        env._manual_failure_reset = lambda: reset_observation.copy()
        _, _, terminated, truncated, info = env.step(
            np.zeros((1, 12), dtype=np.float32)
        )
        self.assertTrue(bool(terminated[0]))
        self.assertFalse(bool(truncated[0]))
        self.assertEqual(float(info["failure"][0]), 1.0)
        self.assertEqual(float(info["failure/tilt"][0]), 1.0)
        self.assertEqual(float(info["failure/height"][0]), 1.0)
        np.testing.assert_allclose(
            info["final_observation"][0][OBSERVATION_SPEC.imu_quat],
            tilted[4],
            atol=1e-6,
        )

    def test_tilt_counter_crosses_policy_window_boundary(self):
        env = _environment()
        tilted = np.asarray(
            [math.cos(0.45), math.sin(0.45), 0.0, 0.0], dtype=np.float32
        )
        frames = [
            _state(tick, quaternion=tilted if index >= 6 else None)
            for index, tick in enumerate(range(2, 22, 2))
        ]
        truths = [_truth(frame.tick) for frame in frames]
        env._wait_window = lambda count=None: frames
        env._training_states_for_frames = lambda supplied: truths

        _, _, terminated, _, _ = env.step(np.zeros((1, 12), dtype=np.float32))
        self.assertFalse(bool(terminated[0]))

        frames = [
            _state(tick, quaternion=tilted if index == 0 else None)
            for index, tick in enumerate(range(22, 42, 2))
        ]
        truths = [_truth(frame.tick) for frame in frames]
        reset_observation = np.zeros(46, dtype=np.float32)
        env._manual_failure_reset = lambda: reset_observation.copy()
        _, _, terminated, _, info = env.step(
            np.zeros((1, 12), dtype=np.float32)
        )
        self.assertTrue(bool(terminated[0]))
        self.assertEqual(float(info["failure/tilt"][0]), 1.0)
        np.testing.assert_allclose(
            info["final_observation"][0][OBSERVATION_SPEC.imu_quat],
            tilted,
            atol=1e-6,
        )

    def test_eval_reset_uses_physical_simulator_reset(self):
        env = _environment()
        env.client.state_buffer.last_tick = 20
        calls = []
        env._auto_reset_simulator = lambda reason: calls.append(reason)
        expected = np.zeros((1, 46), dtype=np.float32)
        env._finish_physical_reset = lambda: (expected.copy(), {})
        observation, _ = env.reset()
        self.assertTrue(env.client.started)
        self.assertEqual(calls, ["Policy environment reset."])
        np.testing.assert_array_equal(observation, expected)

    def test_physical_episode_reset_invokes_simulator_reset(self):
        env = _environment()
        calls = []
        env._auto_reset_simulator = lambda reason, delay_seconds=0.0: calls.append(
            (reason, delay_seconds)
        )
        expected = np.full(46, 3.0, dtype=np.float32)
        env._finish_physical_reset = lambda: (expected[None, :].copy(), {})
        actual = env._physical_episode_reset("timeout")
        self.assertEqual(calls, [("timeout", 0.0)])
        np.testing.assert_array_equal(actual, expected)

    def test_reset_observation_starts_with_zero_velocity_prior(self):
        env = _environment()
        state = _state(2)
        env.client.state_buffer.latest_state = state
        env._hold_reset_pose = lambda measured: measured
        observation, _ = env._finish_physical_reset()
        np.testing.assert_array_equal(
            observation[0, OBSERVATION_SPEC.body_velocity],
            np.zeros(3, dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
