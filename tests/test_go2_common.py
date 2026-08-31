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
    validate_transfer_manifest,
)
from src.environments.go2_sqrl.common.reward import (
    FOOT_CLEARANCE_TARGET,
    REWARD_DEFAULT_JOINT_POSITION,
    add_terminal_failure_penalty,
    compute_reward,
    deterministic_ground_height,
    local_base_clearance,
    movement_reward_gate,
    swing_foot_clearance_error,
    swing_foot_clearance_overshoot_error,
    state_estimated_trot_phase_reward,
)
from src.environments.go2_sqrl.common.estimation.kinematics import foot_positions_body
from src.environments.go2_sqrl.common.specs import (
    ACTION_SPEC,
    DEFAULT_JOINT_POSITION,
    OBSERVATION_SPEC,
    format_policy_io_contract,
    joint_order_indices,
    policy_observation_rows,
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
    assert OBSERVATION_SPEC.body_velocity == slice(27, 30)
    assert ACTION_SPEC.size == 12
    np.testing.assert_allclose(
        ACTION_SPEC.scale, [0.25, 0.35, 0.45] * 4
    )
    assert OBSERVATION_SPEC.previous_action_q_target == slice(34, 46)
    source = [f"{name}_joint" for name in reversed(OBSERVATION_SPEC.joint_order)]
    np.testing.assert_array_equal(joint_order_indices(source), np.arange(11, -1, -1))


def test_policy_io_report_is_derived_from_the_actual_46d_contract():
    rows = policy_observation_rows()
    assert [(row[1].start, row[1].stop) for row in rows] == [
        (0, 12),
        (12, 24),
        (24, 27),
        (27, 30),
        (30, 34),
        (34, 46),
    ]
    report = format_policy_io_contract(0.6)
    assert "Policy observation: shape (46,)" in report
    assert "[27:30] body_velocity" in report
    assert "robust IMU + leg-odometry estimate" in report
    assert "velocity_commands: not observed by the policy" in report
    assert (
        "simulator base_lin_vel: reward and diagnostics only when available"
        in report
    )
    assert "reward velocity fallback: robust body_velocity estimate" in report
    assert "reward target_velocity_x: 0.6 m/s" in report
    assert "Policy action: shape (12,)" in report


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


def test_sagittal_action_range_can_clear_seven_cm_stairs():
    default_z = foot_positions_body(DEFAULT_JOINT_POSITION)[0, 2]
    high_step_action = np.tile([0.0, -1.0, -1.0], 4)
    high_step_target = DEFAULT_JOINT_POSITION + np.asarray(
        ACTION_SPEC.scale
    ) * high_step_action
    raised_z = foot_positions_body(high_step_target)[0, 2]
    assert raised_z - default_z > 0.09


def test_swing_clearance_cost_is_contact_free_and_speed_gated():
    low_feet = np.full(4, FOOT_CLEARANCE_TARGET - 0.05)
    assert swing_foot_clearance_error(low_feet, np.zeros(4)) == 0.0
    assert swing_foot_clearance_error(low_feet, np.ones(4)) == pytest.approx(
        0.05**2
    )
    # A single swinging leg is no longer divided by all four feet.
    assert swing_foot_clearance_error(
        low_feet, np.asarray([1.0, 0.0, 0.0, 0.0])
    ) == pytest.approx(0.05**2)
    assert swing_foot_clearance_error(
        low_feet,
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        aggregation="legacy_mean",
    ) == pytest.approx(0.05**2 / 4.0)


def test_high_clearance_overshoot_and_forward_gate_are_bounded():
    clearance = np.asarray([0.08, 0.12, 0.14, 0.22])
    speed = np.ones(4)
    expected = (0.02**2 + 0.10**2) / 4.0
    assert swing_foot_clearance_overshoot_error(
        clearance, speed, upper_target=0.12
    ) == pytest.approx(expected)
    np.testing.assert_allclose(
        movement_reward_gate(
            np.asarray([0.05, 0.10, 0.15, 0.20, 0.30]),
            start=0.10,
            full=0.20,
        ),
        [0.0, 0.0, 0.5, 1.0, 1.0],
    )
    assert movement_reward_gate(0.0, start=0.0, full=0.0) == pytest.approx(1.0)


def test_state_estimated_phase_rewards_active_diagonal_trot_only():
    target = 0.07
    diagonal = np.asarray([target, 0.0, 0.0, target])
    active = np.ones(4)
    assert state_estimated_trot_phase_reward(
        diagonal, np.zeros(4), active, target=target
    ) == pytest.approx(1.0, abs=3e-6)
    assert state_estimated_trot_phase_reward(
        np.zeros(4), np.zeros(4), np.zeros(4), target=target
    ) == pytest.approx(0.0)
    all_same = state_estimated_trot_phase_reward(
        np.full(4, target), np.zeros(4), active, target=target
    )
    assert all_same == pytest.approx(0.5, abs=3e-6)


def test_reward_and_episode_semantics():
    assert local_base_clearance(0.37, 0.07) == pytest.approx(0.3)
    np.testing.assert_allclose(
        deterministic_ground_height(
            np.asarray([0.5, 0.999, 1.0, 2.0]),
            terrain_profile="single_step_up",
            step_start_x=1.0,
            step_height=0.04,
        ),
        [0.0, 0.0, 0.04, 0.04],
    )
    terms = compute_reward(
        np.asarray([0.5, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        REWARD_DEFAULT_JOINT_POSITION,
        np.zeros(12),
        np.zeros(12),
        target_velocity_x=0.5,
        base_clearance=0.3,
    )
    assert terms.tracking_lin_vel == pytest.approx(0.02)
    assert terms.velocity_error == pytest.approx(0.0)
    assert terms.tracking_ang_vel == pytest.approx(0.004)
    assert terms.lin_vel_z == pytest.approx(0.0)
    assert terms.base_height == pytest.approx(0.0)
    assert terms.foot_clearance_overshoot == pytest.approx(0.0)
    assert terms.phase == pytest.approx(0.0)
    assert terms.stable_progress == pytest.approx(0.0)
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
        base_clearance=0.3,
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
    validate_transfer_manifest(incompatible_command, expected)
    incompatible_observation = build_manifest({"observation_size": 46})
    incompatible_observation["observation"]["size"] = 48
    with pytest.raises(ValueError, match="observation.size"):
        validate_transfer_manifest(incompatible_observation, expected)


def test_high_clearance_reward_is_an_optional_transfer_task_contract():
    source = build_manifest({"observation_size": 46})
    assert "high_clearance" not in source["reward_contract"]
    target = build_manifest(
        {"observation_size": 46},
        foot_clearance_upper_target=0.12,
        foot_clearance_overshoot_scale=-10.0,
        phase_velocity_gate_start=0.1,
        phase_velocity_gate_full=0.2,
        stable_progress_scale=0.5,
    )
    assert target["reward_version"] != source["reward_version"]
    assert target["reward_contract"]["high_clearance"][
        "policy_observation_changed"
    ] is False
    validate_transfer_manifest(source, target)
    with pytest.raises(ValueError, match="reward_version"):
        validate_manifest(source, target)


def test_terminal_failure_penalty_is_one_shot_and_optional_reward_contract():
    rewards = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    failures = np.asarray([False, True, False])
    np.testing.assert_allclose(
        add_terminal_failure_penalty(rewards, failures, -10.0),
        [1.0, -9.0, 1.0],
    )

    source = build_manifest({"observation_size": 46})
    assert source["reward_contract"]["failure_reward_shaping"] is False
    assert "terminal_failure_penalty" not in source["reward_contract"]

    target = build_manifest(
        {"observation_size": 46}, terminal_failure_penalty=-10.0
    )
    assert target["reward_contract"]["failure_reward_shaping"] is True
    assert target["reward_contract"]["terminal_failure_penalty"] == -10.0
    validate_transfer_manifest(source, target)
    with pytest.raises(ValueError, match="failure_reward_shaping"):
        validate_manifest(source, target)
