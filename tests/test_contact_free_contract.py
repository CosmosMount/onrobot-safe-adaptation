from copy import deepcopy
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
from ml_collections import ConfigDict

from src.environments.go2_sqrl.common.action import (
    ActionMapper,
    project_actions_from_observation,
)
from src.environments.go2_sqrl.common.estimation.velocity import VelocityEstimator
from src.environments.go2_sqrl.common.manifest import (
    FAILURE_CONTRACT_VERSION,
    MANIFEST_VERSION,
    VELOCITY_ESTIMATOR_VERSION,
    build_manifest,
    validate_manifest,
)
from src.environments.go2_sqrl.common.specs import (
    DEFAULT_JOINT_POSITION,
    OBSERVATION_SPEC,
)
from src.environments.go2_sqrl.common.types import RobotState
from src.environments.go2_sqrl.sdk2_mujoco.env import Go2SDKMujocoEnv
from src.environments.go2_sqrl.sdk2_mujoco.default_config import get_config
from src.environments.go2_sqrl.sdk2_mujoco.sdk_client import decode_low_state
from src.environments.go2_sqrl.sdk2_mujoco.state_buffer import StateBuffer


def _fake_low_state():
    motors = [SimpleNamespace(q=0.0, dq=0.0, tau_est=0.0) for _ in range(12)]
    imu = SimpleNamespace(
        quaternion=[1.0, 0.0, 0.0, 0.0],
        gyroscope=[0.0, 0.0, 0.0],
        accelerometer=[0.0, 0.0, 9.81],
    )

    class Message:
        motor_state = motors
        imu_state = imu
        tick = 2

        @property
        def foot_force(self):
            raise AssertionError("contact sensor field must not be read")

    return Message()


def test_robot_and_sdk_contract_never_read_contact_sensor_fields():
    assert "foot_force" not in {field.name for field in fields(RobotState)}
    state = decode_low_state(_fake_low_state())
    assert state.tick == 2
    assert state.joint_q.shape == (12,)


def test_manifest_versions_sensor_free_estimator_and_imu_failure():
    manifest = build_manifest({"observation_size": 46})
    assert manifest["manifest_version"] == MANIFEST_VERSION == 2
    estimator = manifest["observation"]["velocity_estimator"]
    assert estimator["version"] == VELOCITY_ESTIMATOR_VERSION
    assert estimator["external_contact_sensor"] is False
    assert manifest["failure"]["version"] == FAILURE_CONTRACT_VERSION
    assert manifest["failure"]["external_contact_sensor"] is False
    assert manifest["failure"]["frame_unit"] == "physics_frames"
    assert manifest["failure"]["frame_dt"] == pytest.approx(0.002)

    old = deepcopy(manifest)
    del old["observation"]["velocity_estimator"]
    with pytest.raises(ValueError, match="velocity_estimator"):
        validate_manifest(old, manifest)


def test_numpy_sdk_projection_is_side_effect_free_and_matches_mapper():
    states = np.zeros((2, OBSERVATION_SPEC.size), dtype=np.float32)
    previous = np.stack(
        (DEFAULT_JOINT_POSITION, DEFAULT_JOINT_POSITION + 0.05), axis=0
    )
    states[..., OBSERVATION_SPEC.previous_action_q_target] = previous
    actions = np.stack(
        (
            np.full(12, 2.0, dtype=np.float32),
            np.linspace(-1.5, 1.5, 12, dtype=np.float32),
        )
    )
    states_before = states.copy()
    actions_before = actions.copy()

    common = project_actions_from_observation(states, actions)
    sdk = Go2SDKMujocoEnv.project_actions(states, actions)
    expected = []
    for prior, action in zip(previous, actions):
        mapper = ActionMapper()
        mapper.reset(prior)
        expected.append(mapper.apply(action).applied_action)

    np.testing.assert_allclose(common, expected, atol=1e-6)
    np.testing.assert_allclose(sdk, expected, atol=1e-6)
    np.testing.assert_array_equal(states, states_before)
    np.testing.assert_array_equal(actions, actions_before)


def test_sdk_transition_window_discards_pre_command_backlog():
    class BacklogClient:
        def __init__(self):
            self.state_buffer = StateBuffer()
            self.tick = 0
            self.publish_count = 0

        def start(self):
            pass

        def push_frames(self, marker, count=10):
            for _ in range(count):
                self.tick += 2
                self.state_buffer.push(
                    RobotState(
                        joint_q=np.full(12, marker, dtype=np.float32),
                        joint_dq=np.zeros(12, dtype=np.float32),
                        imu_gyro=np.zeros(3, dtype=np.float32),
                        imu_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        imu_accelerometer=np.asarray(
                            [0.0, 0.0, 9.81], dtype=np.float32
                        ),
                        tick=self.tick,
                    )
                )

        def publish_joint_target(self, target):
            del target
            self.publish_count += 1
            self.push_frames(0.1 * self.publish_count)

        def latest_training_state(self):
            return None

        def close(self):
            pass

    client = BacklogClient()
    # Frames already present before reset's stand command must not be consumed.
    client.push_frames(-0.5)
    config = ConfigDict()
    config.environment = get_config("go2_sqrl.sdk2_mujoco")
    environment = Go2SDKMujocoEnv(config, client=client)
    initial, _ = environment.reset()
    np.testing.assert_allclose(initial[0, :12], 0.1)

    # Model updates happen here in a real run while LowState continues at 500Hz.
    client.push_frames(-0.25)
    result = environment.step(np.zeros((1, 12), dtype=np.float32))
    np.testing.assert_allclose(result[0][0, :12], 0.2)
    assert environment._last_tick == client.state_buffer.last_tick


def test_torch_and_numpy_estimators_and_action_projection_match():
    torch = pytest.importorskip("torch")
    from src.environments.go2_sqrl.common.estimation.velocity_torch import (
        TorchVelocityEstimator,
    )
    from src.environments.go2_sqrl.isaac_lab.env import Go2IsaacEnv

    q = DEFAULT_JOINT_POSITION + np.linspace(-0.08, 0.08, 12, dtype=np.float32)
    dq = np.linspace(-0.12, 0.12, 12, dtype=np.float32)
    gyro = np.asarray([0.03, -0.02, 0.05], dtype=np.float32)
    quat = np.asarray([0.99875027, 0.0, 0.0, 0.04997917], dtype=np.float32)
    accel = np.asarray([0.1, -0.05, 9.81], dtype=np.float32)
    state = RobotState(q, dq, gyro, quat, accel)
    numpy_estimator = VelocityEstimator()
    torch_estimator = TorchVelocityEstimator(1, "cpu")
    for _ in range(3):
        numpy_velocity = numpy_estimator.update(state)
        torch_velocity = torch_estimator.update(
            torch.tensor(q)[None],
            torch.tensor(dq)[None],
            torch.tensor(gyro)[None],
            torch.tensor(quat)[None],
            torch.tensor(accel)[None],
        )[0]
        np.testing.assert_allclose(
            numpy_velocity, torch_velocity.numpy(), atol=2e-5, rtol=2e-5
        )

    observations = np.zeros((2, 46), dtype=np.float32)
    observations[:, 34:46] = np.stack(
        (DEFAULT_JOINT_POSITION, DEFAULT_JOINT_POSITION + 0.05)
    )
    actions = np.stack(
        (np.full(12, 1.5, dtype=np.float32), np.linspace(-2, 2, 12, dtype=np.float32))
    )
    expected = project_actions_from_observation(observations, actions)
    torch_actions = torch.tensor(actions, requires_grad=True)
    actual = Go2IsaacEnv.project_actions(torch.tensor(observations), torch_actions)
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-6)
    actual.sum().backward()
    assert torch_actions.grad is not None


def test_isaac_fall_detector_requires_consecutive_imu_frames_and_resets():
    torch = pytest.importorskip("torch")
    from src.environments.go2_sqrl.isaac_lab.env import TorchFallDetector

    detector = TorchFallDetector(
        nr_envs=2, device="cpu", angle_threshold=0.8, consecutive_frames=2
    )
    tilted = torch.tensor(
        [np.cos(0.5), np.sin(0.5), 0.0, 0.0], dtype=torch.float32
    )
    upright = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    quaternions = torch.stack((tilted, upright))

    torch.testing.assert_close(
        detector.update(quaternions), torch.tensor([False, False])
    )

    decimated = TorchFallDetector(
        nr_envs=1,
        device="cpu",
        angle_threshold=0.8,
        consecutive_frames=5,
        samples_per_update=10,
    )
    torch.testing.assert_close(decimated.update(tilted[None]), torch.tensor([True]))
    torch.testing.assert_close(
        detector.update(quaternions), torch.tensor([True, False])
    )
    detector.reset(torch.tensor([0]))
    torch.testing.assert_close(
        detector.update(quaternions), torch.tensor([False, False])
    )
