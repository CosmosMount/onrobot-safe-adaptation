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
    ACTION_PIPELINE_VERSION,
    FAILURE_CONTRACT_VERSION,
    MANIFEST_VERSION,
    VELOCITY_ESTIMATOR_VERSION,
    build_manifest,
    validate_manifest,
)
from src.environments.go2_sqrl.common.specs import (
    DEFAULT_JOINT_POSITION,
    JOINT_NAMES,
    OBSERVATION_SPEC,
)
from src.environments.go2_sqrl.common.types import RobotState, TrainingState
from src.environments.go2_sqrl.sdk2_mujoco.env import Go2SDKMujocoEnv
from src.environments.go2_sqrl.sdk2_mujoco.default_config import get_config
from src.environments.go2_sqrl.sdk2_mujoco.reset_controller import (
    MujocoResetController,
)
from src.environments.go2_sqrl.sdk2_mujoco.sdk_client import decode_low_state
from src.environments.go2_sqrl.sdk2_mujoco.state_buffer import StateBuffer


def test_mujoco_reset_controller_uses_domain_scoped_software_channel(monkeypatch):
    sent = []

    class DatagramClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def sendto(self, payload, path):
            sent.append((payload, path))

    monkeypatch.setattr(
        "src.environments.go2_sqrl.sdk2_mujoco.reset_controller.socket.socket",
        lambda *_args: DatagramClient(),
    )
    controller = MujocoResetController(domain_id=32)
    controller.reset()

    assert sent == [(b"reset", str(controller.socket_path))]


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


def test_mujoco_policy_pd_defaults_match_checkpoint_contract():
    from src.environments.go2_sqrl.common.specs import ACTION_SPEC

    config = get_config("go2_sqrl.sdk2_mujoco")
    assert config.policy_kp == ACTION_SPEC.kp == 25.0
    assert config.policy_kd == ACTION_SPEC.kd == 0.5
    assert config.terminal_failure_penalty == 0.0


def test_mujoco_reset_pd_defaults_settle_before_policy_takeover():
    config = get_config("go2_sqrl.sdk2_mujoco")
    assert config.reset_kp == 60.0
    assert config.reset_kd == 1.0
    assert config.reset_max_joint_velocity == 0.5


def test_manifest_versions_sensor_free_estimator_and_shared_failure():
    manifest = build_manifest({"observation_size": 46})
    assert manifest["manifest_version"] == MANIFEST_VERSION == 11
    assert (
        manifest["reward_version"]
        == "flashsac-go2-walk-easy-command-v6-state-trot-phase"
    )
    assert (
        manifest["reward_contract"]["base_height_reference"]
        == "local_terrain_clearance"
    )
    assert manifest["reward_contract"]["command"]["linear_velocity_x"] == 0.5
    assert manifest["reward_contract"]["foot_clearance_target"] == 0.07
    assert (
        manifest["reward_contract"]["foot_clearance_aggregation"]
        == "swing_weighted"
    )
    assert manifest["reward_contract"]["phase"]["external_clock"] is False
    assert manifest["reward_contract"]["phase"]["policy_observation"] is False
    np.testing.assert_allclose(
        manifest["reward_contract"]["similar_to_default_joint_position"],
        [0.0, 0.8, -1.5] * 2 + [0.0, 1.0, -1.5] * 2,
    )
    estimator = manifest["observation"]["velocity_estimator"]
    assert estimator["version"] == VELOCITY_ESTIMATOR_VERSION
    assert estimator["policy_visible"] is True
    assert estimator["external_contact_sensor"] is False
    assert manifest["observation"]["body_velocity"]["indices"] == [27, 30]
    assert manifest["observation"]["body_velocity"]["frame"] == "body"
    assert manifest["failure"]["version"] == FAILURE_CONTRACT_VERSION
    assert manifest["failure"]["signal"] == [
        "imu_quaternion_roll_pitch",
        "base_clearance_above_local_terrain",
    ]
    assert manifest["failure"]["min_base_clearance"] == pytest.approx(0.18)
    assert manifest["failure"]["external_contact_sensor"] is False
    assert manifest["failure"]["frame_unit"] == "physics_frames"
    assert manifest["failure"]["frame_dt"] == pytest.approx(0.002)
    action = manifest["action"]
    assert action["pipeline_version"] == ACTION_PIPELINE_VERSION
    assert action["joint_order"] == list(JOINT_NAMES)
    np.testing.assert_allclose(action["default_position"], DEFAULT_JOINT_POSITION)
    assert action["target_semantics"] == "absolute_joint_position"

    old = deepcopy(manifest)
    del old["observation"]["velocity_estimator"]
    with pytest.raises(ValueError, match="velocity_estimator"):
        validate_manifest(old, manifest)

    legacy_action_semantics = deepcopy(manifest)
    legacy_action_semantics["manifest_version"] = 4
    del legacy_action_semantics["action"]["pipeline_version"]
    del legacy_action_semantics["action"]["joint_order"]
    del legacy_action_semantics["action"]["default_position"]
    with pytest.raises(ValueError, match="manifest_version"):
        validate_manifest(legacy_action_semantics, manifest)


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
    sdk_environment = object.__new__(Go2SDKMujocoEnv)
    sdk_environment.action_contract = {"scale": ActionMapper().action_scale}
    sdk = sdk_environment.project_actions(states, actions)
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
    # This test uses synthetic marker joint positions to prove backlog
    # ownership; pose convergence is covered separately.
    config.environment.reset_joint_tolerance = 10.0
    config.environment.standup_phase_1_seconds = 0.0
    config.environment.standup_phase_2_seconds = 0.0
    config.environment.standup_hold_seconds = 0.0
    config.environment.reset_sync_timeout_seconds = 0.0
    environment = Go2SDKMujocoEnv(config, client=client)
    initial, _ = environment.reset()
    np.testing.assert_allclose(initial[0, :12], -0.5)

    # Model updates happen here in a real run while LowState continues at 500Hz.
    client.push_frames(-0.25)
    result = environment.step(np.zeros((1, 12), dtype=np.float32))
    np.testing.assert_allclose(result[0][0, :12], 0.1)
    assert result[4]["reward_uses_simulator_truth"][0] == 0.0
    assert "forward_velocity" not in result[4]
    assert "target_velocity_error" not in result[4]
    assert "velocity_estimation_error" not in result[4]
    assert environment._last_tick == client.state_buffer.last_tick


def test_non_home_reset_tries_home_before_crouch_recovery_fallback():
    class StandupClient:
        def __init__(self):
            self.state_buffer = StateBuffer()
            self.tick = 0
            self.targets = []

        def _push(self, joint_q):
            self.tick += 2
            self.state_buffer.push(
                RobotState(
                    np.asarray(joint_q, dtype=np.float32).copy(),
                    np.zeros(12, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    np.asarray([0.0, 0.0, 9.81], dtype=np.float32),
                    tick=self.tick,
                )
            )

        def start(self):
            if self.state_buffer.last_tick is None:
                self._push(np.zeros(12, dtype=np.float32))

        def publish_joint_target(self, target):
            target = np.asarray(target, dtype=np.float32)
            self.targets.append(target.copy())
            # Model a legacy reset whose first 20 ms home command cannot
            # instantly recover the all-zero joint pose.  Subsequent targets
            # track normally, exercising the crouch interpolation fallback.
            measured = (
                np.zeros(12, dtype=np.float32)
                if len(self.targets) == 1
                else target
            )
            for _ in range(10):
                self._push(measured)

        def latest_training_state(self):
            return TrainingState(
                world_velocity=np.zeros(3, dtype=np.float32),
                base_position=np.asarray([0.0, 0.0, 0.45], dtype=np.float32),
            )

    client = StandupClient()
    config = ConfigDict()
    config.environment = get_config("go2_sqrl.sdk2_mujoco")
    config.environment.standup_phase_1_seconds = 0.04
    config.environment.standup_phase_2_seconds = 0.04
    config.environment.standup_hold_seconds = 0.02
    config.environment.reset_sync_timeout_seconds = 0.0
    environment = Go2SDKMujocoEnv(config, client=client)
    environment.reset()

    crouch = np.asarray(config.environment.standup_pose_1, dtype=np.float32)
    assert len(client.targets) == 6
    np.testing.assert_allclose(client.targets[0], DEFAULT_JOINT_POSITION)
    np.testing.assert_allclose(client.targets[1], 0.5 * crouch)
    np.testing.assert_allclose(client.targets[2], crouch)
    np.testing.assert_allclose(
        client.targets[3], 0.5 * (crouch + DEFAULT_JOINT_POSITION)
    )
    np.testing.assert_allclose(client.targets[4], DEFAULT_JOINT_POSITION)
    np.testing.assert_allclose(client.targets[5], DEFAULT_JOINT_POSITION)


def test_policy_directly_takes_over_after_standup():
    class BlendClient:
        def __init__(self):
            self.state_buffer = StateBuffer()
            self.tick = 0

        def start(self):
            if self.state_buffer.last_tick is None:
                self.publish_joint_target(DEFAULT_JOINT_POSITION)

        def publish_joint_target(self, target):
            del target
            for _ in range(10):
                self.tick += 2
                self.state_buffer.push(
                    RobotState(
                        DEFAULT_JOINT_POSITION.copy(),
                        np.zeros(12, dtype=np.float32),
                        np.zeros(3, dtype=np.float32),
                        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        np.asarray([0.0, 0.0, 9.81], dtype=np.float32),
                        tick=self.tick,
                    )
                )

        def latest_training_state(self):
            return TrainingState(
                world_velocity=np.asarray([0.4, 0.0, 0.0], dtype=np.float32),
                base_position=np.asarray([0.0, 0.0, 0.45], dtype=np.float32),
            )

    config = ConfigDict()
    config.environment = get_config("go2_sqrl.sdk2_mujoco")
    config.environment.standup_phase_1_seconds = 0.0
    config.environment.standup_phase_2_seconds = 0.0
    config.environment.standup_hold_seconds = 0.0
    config.environment.reset_sync_timeout_seconds = 0.0
    environment = Go2SDKMujocoEnv(config, client=BlendClient())
    environment.reset()

    requested = np.ones((1, 12), dtype=np.float32)
    first = environment.step(requested)
    assert first[4]["policy_blend_alpha"][0] == pytest.approx(1.0)
    mapper = ActionMapper()
    expected = mapper.apply(requested[0]).applied_action
    np.testing.assert_allclose(first[4]["applied_action"][0], expected)


def test_finetune_auto_resets_on_start_and_one_second_after_fall(monkeypatch):
    class AutoResetClient:
        def __init__(self):
            self.state_buffer = StateBuffer(restart_threshold_ticks=1)
            self.tick = 1000

        def _push(self, count=10):
            for _ in range(count):
                self.state_buffer.push(
                    RobotState(
                        DEFAULT_JOINT_POSITION.copy(),
                        np.zeros(12, dtype=np.float32),
                        np.zeros(3, dtype=np.float32),
                        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        np.asarray([0.0, 0.0, 9.81], dtype=np.float32),
                        tick=self.tick,
                    )
                )
                self.tick += 2

        def start(self):
            if self.state_buffer.last_tick is None:
                self._push(count=1)

        def publish_joint_target(self, target):
            del target
            self._push()

        def reset_simulator(self):
            self.tick = 0
            self._push(count=1)

        def latest_training_state(self):
            return TrainingState(
                world_velocity=np.zeros(3, dtype=np.float32),
                base_position=np.asarray([0.0, 0.0, 0.45], dtype=np.float32),
            )

    class ResetController:
        def __init__(self, client):
            self.client = client
            self.count = 0

        def reset(self):
            self.count += 1
            self.client.reset_simulator()

    client = AutoResetClient()
    controller = ResetController(client)
    config = ConfigDict()
    config.environment = get_config("go2_sqrl.sdk2_mujoco")
    config.environment.standup_phase_1_seconds = 0.0
    config.environment.standup_phase_2_seconds = 0.0
    config.environment.standup_hold_seconds = 0.0
    config.environment.reset_sync_timeout_seconds = 0.0
    sleeps = []
    monkeypatch.setattr(
        "src.environments.go2_sqrl.sdk2_mujoco.env.time.sleep", sleeps.append
    )

    environment = Go2SDKMujocoEnv(
        config,
        client=client,
        role="train",
        reset_controller=controller,
    )
    environment.reset()
    assert controller.count == 1

    # The real simulator continues advancing between resets.
    client._push(count=1)
    environment._manual_failure_reset()
    assert controller.count == 2
    assert sleeps == [pytest.approx(1.0)]


def test_automatic_reset_retries_a_dropped_x11_key():
    class RetryClient:
        def __init__(self):
            self.state_buffer = StateBuffer(restart_threshold_ticks=1)
            self.state_buffer.push(
                RobotState(
                    DEFAULT_JOINT_POSITION.copy(),
                    np.zeros(12, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    np.asarray([0.0, 0.0, 9.81], dtype=np.float32),
                    tick=1000,
                )
            )

        def start(self):
            pass

        def publish_joint_target(self, target):
            del target

        def latest_training_state(self):
            return None

    class DroppedFirstReset:
        def __init__(self, client):
            self.client = client
            self.count = 0

        def reset(self):
            self.count += 1
            if self.count == 2:
                self.client.state_buffer.push(
                    RobotState(
                        DEFAULT_JOINT_POSITION.copy(),
                        np.zeros(12, dtype=np.float32),
                        np.zeros(3, dtype=np.float32),
                        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        np.asarray([0.0, 0.0, 9.81], dtype=np.float32),
                        tick=0,
                    )
                )

    client = RetryClient()
    controller = DroppedFirstReset(client)
    config = ConfigDict()
    config.environment = get_config("go2_sqrl.sdk2_mujoco")
    config.environment.auto_reset_timeout_seconds = 0.001
    config.environment.auto_reset_attempts = 2
    environment = Go2SDKMujocoEnv(
        config,
        client=client,
        role="train",
        reset_controller=controller,
    )

    environment._auto_reset_simulator("Test reset.")

    assert controller.count == 2
    assert environment._generation == 1


def test_stuck_flat_truncation_physically_resets_but_healthy_motion_does_not():
    environment = object.__new__(Go2SDKMujocoEnv)
    environment.config = SimpleNamespace(
        terrain_profile="flat",
        auto_reset_after_stuck=True,
    )
    resets = []
    logical_resets = []
    physical_observation = np.asarray([2.0], dtype=np.float32)

    environment._auto_reset_simulator = lambda reason: resets.append(reason)
    environment.reset = lambda: (physical_observation[None, :], {})
    environment._logical_reset = lambda observation: (
        logical_resets.append(np.asarray(observation).copy())
        or np.asarray(observation).copy()
    )

    stuck_observation = environment._reset_after_truncation(
        np.asarray([1.0], dtype=np.float32),
        stuck=True,
    )
    np.testing.assert_array_equal(stuck_observation, physical_observation)
    assert resets == ["Stuck flat-ground episode completed."]
    assert logical_resets == []

    moving_observation = np.asarray([3.0], dtype=np.float32)
    returned = environment._reset_after_truncation(
        moving_observation,
        stuck=False,
    )
    np.testing.assert_array_equal(returned, moving_observation)
    assert len(logical_resets) == 1


def test_auto_reset_arms_home_before_reset_and_skips_crouch_trajectory():
    events = []

    class HomeResetClient:
        def __init__(self):
            self.state_buffer = StateBuffer(restart_threshold_ticks=1)

        def _push(self, tick):
            self.state_buffer.push(
                RobotState(
                    DEFAULT_JOINT_POSITION.copy(),
                    np.zeros(12, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    np.asarray([0.0, 0.0, 9.81], dtype=np.float32),
                    tick=tick,
                )
            )

        def start(self):
            if self.state_buffer.last_tick is None:
                self._push(100)

        def publish_joint_target(self, target):
            events.append(("target", np.asarray(target).copy()))

        def latest_training_state(self):
            return TrainingState(
                world_velocity=np.zeros(3, dtype=np.float32),
                base_position=np.asarray([0.0, 0.0, 0.289], dtype=np.float32),
            )

    class HomeResetController:
        def __init__(self, client):
            self.client = client

        def reset(self):
            events.append(("reset", None))
            self.client._push(0)

    client = HomeResetClient()
    config = ConfigDict()
    config.environment = get_config("go2_sqrl.sdk2_mujoco")
    config.environment.standup_hold_seconds = 0.0
    config.environment.reset_sync_timeout_seconds = 0.0
    environment = Go2SDKMujocoEnv(
        config,
        client=client,
        reset_controller=HomeResetController(client),
    )

    observation, _ = environment.reset()

    assert [event[0] for event in events] == ["target", "reset"]
    np.testing.assert_allclose(events[0][1], DEFAULT_JOINT_POSITION)
    np.testing.assert_allclose(observation[0, :12], DEFAULT_JOINT_POSITION)


def test_sdk_reset_pose_requires_upright_home_configuration():
    config = ConfigDict()
    config.environment = get_config("go2_sqrl.sdk2_mujoco")
    environment = Go2SDKMujocoEnv(config, client=SimpleNamespace())

    ready = RobotState(
        DEFAULT_JOINT_POSITION.copy(),
        np.zeros(12, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 9.81], dtype=np.float32),
    )
    tilted = RobotState(
        ready.joint_q,
        ready.joint_dq,
        ready.imu_gyro,
        np.asarray([0.5, 0.866, 0.0, 0.0], dtype=np.float32),
        ready.imu_accelerometer,
    )
    displaced = RobotState(
        ready.joint_q + 0.3,
        ready.joint_dq,
        ready.imu_gyro,
        ready.imu_quat,
        ready.imu_accelerometer,
    )
    moving = RobotState(
        ready.joint_q,
        np.full(12, 0.6, dtype=np.float32),
        ready.imu_gyro,
        ready.imu_quat,
        ready.imu_accelerometer,
    )

    assert environment._reset_pose_ready(ready)
    assert not environment._reset_pose_ready(tilted)
    assert not environment._reset_pose_ready(displaced)
    assert not environment._reset_pose_ready(moving)


def test_sdk_reset_pose_rejects_low_base_even_when_level():
    config = ConfigDict()
    config.environment = get_config("go2_sqrl.sdk2_mujoco")
    client = SimpleNamespace(
        latest_training_state=lambda: TrainingState(
            world_velocity=np.zeros(3, dtype=np.float32),
            base_position=np.asarray([0.0, 0.0, 0.1], dtype=np.float32),
        )
    )
    environment = Go2SDKMujocoEnv(config, client=client)
    level_home = RobotState(
        DEFAULT_JOINT_POSITION.copy(),
        np.zeros(12, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 9.81], dtype=np.float32),
    )

    assert not environment._reset_pose_ready(level_home)


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

    low_detector = TorchFallDetector(
        nr_envs=1,
        device="cpu",
        min_base_clearance=0.18,
        consecutive_frames=2,
    )
    torch.testing.assert_close(
        low_detector.update(upright[None], torch.tensor([0.17])),
        torch.tensor([False]),
    )
    torch.testing.assert_close(
        low_detector.update(upright[None], torch.tensor([0.17])),
        torch.tensor([True]),
    )
    torch.testing.assert_close(
        low_detector.last_height_failure, torch.tensor([True])
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
