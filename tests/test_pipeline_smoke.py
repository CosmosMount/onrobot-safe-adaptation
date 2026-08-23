import os

import numpy as np
import pytest
from ml_collections import ConfigDict

from src.environments.go2_sqrl.common.specs import DEFAULT_JOINT_POSITION
from src.environments.go2_sqrl.common.types import RobotState, TrainingState
from src.environments.go2_sqrl.sdk2_mujoco.default_config import get_config
from src.environments.go2_sqrl.sdk2_mujoco.env import Go2SDKMujocoEnv
from src.environments.go2_sqrl.sdk2_mujoco.state_buffer import StateBuffer


class FakeSDKClient:
    def __init__(self):
        self.state_buffer = StateBuffer()
        self.tick = 0
        self.last_target = None

    def start(self):
        pass

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


def test_sdk_environment_observation_command_reward_smoke():
    config = ConfigDict()
    config.environment = get_config("go2_sqrl.sdk2_mujoco")
    environment = Go2SDKMujocoEnv(config, client=FakeSDKClient())
    observation, _ = environment.reset()
    assert observation.shape == (1, 46)
    next_observation, reward, terminated, truncated, info = environment.step(
        np.zeros((1, 12), dtype=np.float32)
    )
    assert next_observation.shape == (1, 46)
    assert reward[0] == 1.0
    assert not terminated[0]
    assert not truncated[0]
    assert info["applied_action"].shape == (1, 12)


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
