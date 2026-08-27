import os

import numpy as np
import pytest
from ml_collections import ConfigDict

from src.environments.go2_sqrl.common.specs import DEFAULT_JOINT_POSITION
from src.environments.go2_sqrl.common.types import RobotState, TrainingState
from src.environments.go2_sqrl.sdk2_mujoco.default_config import get_config
from src.environments.go2_sqrl.sdk2_mujoco.env import Go2SDKMujocoEnv
from src.environments.go2_sqrl.sdk2_mujoco.state_buffer import (
    SimulatorRestarted,
    StateBuffer,
)


class FakeSDKClient:
    def __init__(self):
        self.state_buffer = StateBuffer()
        self.tick = 0
        self.last_target = None

    def start(self):
        if self.state_buffer.last_tick is None:
            self.publish_joint_target(DEFAULT_JOINT_POSITION)

    def publish_joint_target(self, target):
        self.last_target = np.asarray(target).copy()
        for _ in range(10):
            self.tick += 2
            self.state_buffer.push(
                RobotState(
                    joint_q=DEFAULT_JOINT_POSITION.copy(),
                    joint_dq=np.zeros(12, dtype=np.float32),
                    imu_gyro=np.zeros(3, dtype=np.float32),
                    imu_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    imu_accelerometer=np.asarray([0.0, 0.0, 9.81], dtype=np.float32),
                    tick=self.tick,
                )
            )

    def latest_training_state(self):
        return TrainingState(
            world_velocity=np.asarray([0.4, 0.0, 0.0], dtype=np.float32),
            base_position=np.asarray([0.0, 0.0, 0.45], dtype=np.float32),
            actuator_torque=np.zeros(12, dtype=np.float32),
        )

    def close(self):
        pass


def test_sdk_environment_body_velocity_observation_and_reward_smoke():
    config = ConfigDict()
    config.environment = get_config("go2_sqrl.sdk2_mujoco")
    environment = Go2SDKMujocoEnv(config, client=FakeSDKClient())
    observation, _ = environment.reset()
    assert observation.shape == (1, 46)
    # The stationary proprioceptive estimator is policy-visible. The 0.6 m/s
    # task target and simulator-only 0.4 m/s truth must not leak into it.
    np.testing.assert_allclose(observation[0, 27:30], [0.0, 0.0, 0.0], atol=1e-8)
    estimator = environment.observation_builder.velocity_estimator
    updates_after_reset = estimator.update_count
    next_observation, reward, terminated, truncated, info = environment.step(
        np.zeros((1, 12), dtype=np.float32)
    )
    assert next_observation.shape == (1, 46)
    np.testing.assert_allclose(
        next_observation[0, 27:30], [0.0, 0.0, 0.0], atol=1e-8
    )
    assert estimator.update_count - updates_after_reset == 10
    assert reward[0] == pytest.approx(-0.0070571201, abs=1e-7)
    assert not terminated[0]
    assert not truncated[0]
    assert info["applied_action"].shape == (1, 12)


def test_sdk_environment_runs_policy_window_callback_once():
    config = ConfigDict()
    config.environment = get_config("go2_sqrl.sdk2_mujoco")
    environment = Go2SDKMujocoEnv(config, client=FakeSDKClient())
    environment.reset()
    calls = []
    environment.set_policy_window_callback(lambda: calls.append("update"))
    with pytest.raises(RuntimeError, match="already pending"):
        environment.set_policy_window_callback(lambda: None)

    environment.step(np.zeros((1, 12), dtype=np.float32))
    environment.step(np.zeros((1, 12), dtype=np.float32))

    assert calls == ["update"]
    assert environment._policy_window_callback is None


def test_time_limit_preserves_controller_and_estimator_continuity():
    config = ConfigDict()
    config.environment = get_config("go2_sqrl.sdk2_mujoco")
    config.environment.episode_steps = 1
    environment = Go2SDKMujocoEnv(config, client=FakeSDKClient())
    environment.reset()
    estimator = environment.observation_builder.velocity_estimator
    updates_before = estimator.update_count

    first = environment.step(np.ones((1, 12), dtype=np.float32))
    first_target = environment.action_mapper.previous_q_target.copy()
    assert first[3][0]
    assert estimator.update_count - updates_before == 10
    np.testing.assert_allclose(
        first[0][0, 34:46], first_target, rtol=0.0, atol=1e-7
    )

    second = environment.step(np.ones((1, 12), dtype=np.float32))
    second_target = environment.action_mapper.previous_q_target.copy()
    assert second[3][0]
    assert estimator.update_count - updates_before == 20
    assert np.max(second_target - first_target) > 0.0
    np.testing.assert_allclose(
        second[0][0, 34:46], second_target, rtol=0.0, atol=1e-7
    )


def test_sdk_environment_rejects_noncanonical_control_window():
    config = ConfigDict()
    config.environment = get_config("go2_sqrl.sdk2_mujoco")
    config.environment.policy_frames = 9
    with pytest.raises(ValueError, match="exactly 10 LowState frames"):
        Go2SDKMujocoEnv(config, client=FakeSDKClient())


def test_time_limit_uses_independent_physical_reset_when_available():
    class ResettableClient(FakeSDKClient):
        def __init__(self):
            super().__init__()
            self.state_buffer = StateBuffer(restart_threshold_ticks=1)

        def reset_simulator(self):
            self.tick = 0
            self.state_buffer.push(
                RobotState(
                    joint_q=DEFAULT_JOINT_POSITION.copy(),
                    joint_dq=np.zeros(12, dtype=np.float32),
                    imu_gyro=np.zeros(3, dtype=np.float32),
                    imu_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    imu_accelerometer=np.asarray([0.0, 0.0, 9.81], dtype=np.float32),
                    tick=self.tick,
                )
            )

    class ResetController:
        def __init__(self, client):
            self.client = client
            self.count = 0

        def reset(self):
            self.count += 1
            self.client.reset_simulator()

    client = ResettableClient()
    controller = ResetController(client)
    config = ConfigDict()
    config.environment = get_config("go2_sqrl.sdk2_mujoco")
    config.environment.episode_steps = 1
    config.environment.reset_post_restart_settle_seconds = 0.0
    config.environment.standup_phase_1_seconds = 0.0
    config.environment.standup_phase_2_seconds = 0.0
    config.environment.standup_hold_seconds = 0.0
    config.environment.reset_sync_timeout_seconds = 0.0
    environment = Go2SDKMujocoEnv(
        config, client=client, reset_controller=controller
    )
    environment.reset()
    assert controller.count == 1

    observation, _, _, truncated, _ = environment.step(
        np.ones((1, 12), dtype=np.float32)
    )

    assert truncated[0]
    assert controller.count == 2
    np.testing.assert_allclose(
        environment.action_mapper.previous_q_target,
        DEFAULT_JOINT_POSITION,
    )
    np.testing.assert_allclose(observation[0, 27:30], 0.0, atol=1e-7)


@pytest.mark.parametrize("restart_count", [1, 5])
def test_reset_pose_resynchronizes_queued_simulator_ticks(restart_count):
    class RestartingBuffer:
        def __init__(self):
            self.generation = 0
            self.last_tick = 100
            self.calls = 0

        def wait_for_frames(self, *, generation, **_):
            self.calls += 1
            if self.calls <= restart_count:
                self.generation += 1
                self.last_tick = 0
                raise SimulatorRestarted("reset")
            self.last_tick = 20
            return [
                RobotState(
                    joint_q=DEFAULT_JOINT_POSITION.copy(),
                    joint_dq=np.zeros(12, dtype=np.float32),
                    imu_gyro=np.zeros(3, dtype=np.float32),
                    imu_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    imu_accelerometer=np.asarray(
                        [0.0, 0.0, 9.81], dtype=np.float32
                    ),
                    tick=20,
                )
            ]

        def clear_error(self):
            pass

    class RestartingClient:
        def __init__(self):
            self.state_buffer = RestartingBuffer()
            self.published = 0

        def publish_joint_target(self, _):
            self.published += 1

    config = ConfigDict()
    config.environment = get_config("go2_sqrl.sdk2_mujoco")
    client = RestartingClient()
    environment = Go2SDKMujocoEnv(config, client=client)
    state = environment._publish_and_wait(DEFAULT_JOINT_POSITION)
    assert state.tick == 20
    assert client.published == restart_count + 1
    assert environment._generation == restart_count


@pytest.mark.live_sdk
@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_SDK") != "1",
    reason="set RUN_LIVE_SDK=1 with unitree_mujoco already running",
)
def test_live_sdk_single_policy_step():
    config = ConfigDict()
    config.environment = get_config("go2_sqrl.sdk2_mujoco")
    environment = Go2SDKMujocoEnv(config)
    try:
        observation, _ = environment.reset()
        result = environment.step(np.zeros((1, 12), dtype=np.float32))
        assert observation.shape == (1, 46)
        assert result[0].shape == (1, 46)
    finally:
        environment.close()
