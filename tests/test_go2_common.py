import numpy as np
import pytest

from src.environments.go2_sqrl.common.action import (
    ActionMapper,
    normalized_action_from_target,
)
from src.environments.go2_sqrl.common.observation import (
    build_observation,
    continuous_quaternion_wxyz,
)
from src.environments.go2_sqrl.common.manifest import (
    build_manifest,
    validate_manifest,
)
from src.environments.go2_sqrl.common.reward import (
    REWARD_DEFAULT_JOINT_POSITION,
    compute_reward,
)
from src.environments.go2_sqrl.common.specs import (
    ACTION_SPEC,
    DEFAULT_JOINT_POSITION,
    OBSERVATION_SPEC,
    joint_order_indices,
)
from src.environments.go2_sqrl.common.termination import EpisodeTracker
from src.environments.go2_sqrl.common.types import RobotState


def robot_state(quaternion=(1.0, 0.0, 0.0, 0.0)):
    return RobotState(
        joint_q=DEFAULT_JOINT_POSITION.copy(),
        joint_dq=np.zeros(12, dtype=np.float32),
        imu_gyro=np.zeros(3, dtype=np.float32),
        imu_quat=np.asarray(quaternion, dtype=np.float32),
        imu_accelerometer=np.asarray([0.0, 0.0, 9.81], dtype=np.float32),
        tick=0,
    )


def test_versioned_observation_layout_and_joint_order():
    assert OBSERVATION_SPEC.size == 46
    assert OBSERVATION_SPEC.velocity_command == slice(27, 30)
    assert ACTION_SPEC.size == 12
    assert OBSERVATION_SPEC.previous_action_q_target == slice(34, 46)
    source = [f"{name}_joint" for name in reversed(OBSERVATION_SPEC.joint_order)]
    np.testing.assert_array_equal(joint_order_indices(source), np.arange(11, -1, -1))


def test_quaternion_continuity_and_observation_layout():
    first = continuous_quaternion_wxyz(np.asarray([-1.0, 0, 0, 0]), None)
    second = continuous_quaternion_wxyz(np.asarray([-1.0, 0, 0, 0]), first)
    np.testing.assert_array_equal(first, np.asarray([1.0, 0, 0, 0]))
    np.testing.assert_array_equal(second, first)
    observation, quaternion = build_observation(
        robot_state(), np.asarray([0.1, 0.2, 0.3]), DEFAULT_JOINT_POSITION
    )
    assert observation.shape == (46,)
    np.testing.assert_allclose(observation[27:30], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(observation[30:34], quaternion)


def test_action_limit_rate_limit_and_inverse_mapping():
    mapper = ActionMapper(max_target_rate=1.0)
    result = mapper.apply(np.full(12, 2.0, dtype=np.float32))
    np.testing.assert_allclose(
        result.q_target, DEFAULT_JOINT_POSITION + 0.02, atol=1e-6
    )
    np.testing.assert_allclose(
        normalized_action_from_target(result.q_target), result.applied_action
    )
    assert np.all(result.applied_action <= 1.0)


def test_reward_and_episode_semantics():
    terms = compute_reward(
        np.asarray([0.5, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        REWARD_DEFAULT_JOINT_POSITION,
        np.zeros(12),
        np.zeros(12),
        target_velocity_x=0.5,
        base_height=0.3,
    )
    assert terms.tracking_lin_vel == pytest.approx(0.02)
    assert terms.velocity_error == pytest.approx(0.0)
    assert terms.tracking_ang_vel == pytest.approx(0.004)
    assert terms.lin_vel_z == pytest.approx(0.0)
    assert terms.base_height == pytest.approx(0.0)
    assert terms.action_rate == pytest.approx(0.0)
    assert terms.similar_to_default == pytest.approx(0.0)
    assert terms.total == pytest.approx(0.024)

    stationary = compute_reward(
        np.zeros(3),
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        REWARD_DEFAULT_JOINT_POSITION,
        np.zeros(12),
        np.zeros(12),
        target_velocity_x=0.5,
        base_height=0.3,
    )
    assert stationary.velocity_error == pytest.approx(-0.015)
    assert stationary.total < 0.0
    tracker = EpisodeTracker(max_steps=2)
    assert tracker.advance(1.0, failure=False) == (False, False)
    assert tracker.advance(1.0, failure=False) == (False, True)
    tracker.reset()
    assert tracker.advance(0.0, failure=True) == (True, False)


def test_checkpoint_manifest_rejects_contract_drift():
    expected = build_manifest({"observation_size": 46})
    validate_manifest(expected, expected)
    incompatible = build_manifest({"observation_size": 46})
    incompatible["observation"]["quaternion_order"] = "XYZW"
    with pytest.raises(ValueError, match="quaternion_order"):
        validate_manifest(incompatible, expected)
    incompatible_command = build_manifest(
        {"observation_size": 46}, target_velocity_x=0.3
    )
    with pytest.raises(ValueError, match="linear_velocity_x"):
        validate_manifest(incompatible_command, expected)
