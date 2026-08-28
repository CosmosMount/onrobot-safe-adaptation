import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.environments.go2_sqrl.common.observation import build_observation
from src.environments.go2_sqrl.common.observation import ObservationBuilder
from src.environments.go2_sqrl.common.estimation.kinematics import (
    foot_jacobians_body,
    foot_position_velocity_body,
)
from src.environments.go2_sqrl.common.estimation.velocity import VelocityEstimator
from src.environments.go2_sqrl.common.specs import (
    DEFAULT_JOINT_POSITION,
    OBSERVATION_SPEC,
    PHYSICS_DT,
)
from src.environments.go2_sqrl.common.types import RobotState
from src.environments.go2_sqrl.isaac_lab.mdp import build_observation_tensor
from src.environments.go2_sqrl.common.estimation.velocity_torch import (
    TorchVelocityEstimator,
)
from tools.velocity_estimator_benchmark import (
    LegacyVelocityEstimator,
    build_parser,
    summarize_errors,
)


def _state_with_leg_velocity_candidates(
    candidates,
    *,
    quaternion=(1.0, 0.0, 0.0, 0.0),
    angular_velocity=(0.0, 0.0, 0.0),
    accelerometer=(0.0, 0.0, 9.81),
):
    joint_q = DEFAULT_JOINT_POSITION.astype(np.float64)
    positions, _ = foot_position_velocity_body(joint_q, np.zeros(12))
    jacobians = foot_jacobians_body(joint_q)
    candidates = np.broadcast_to(
        np.asarray(candidates, dtype=np.float64), (4, 3)
    )
    angular_velocity = np.asarray(angular_velocity, dtype=np.float64)
    joint_dq = np.empty((4, 3), dtype=np.float64)
    for leg in range(4):
        relative_velocity = -candidates[leg] - np.cross(
            angular_velocity, positions[leg]
        )
        joint_dq[leg] = np.linalg.solve(jacobians[leg], relative_velocity)
    return RobotState(
        joint_q=joint_q.astype(np.float32),
        joint_dq=joint_dq.reshape(-1).astype(np.float32),
        imu_gyro=angular_velocity.astype(np.float32),
        imu_quat=np.asarray(quaternion, dtype=np.float32),
        imu_accelerometer=np.asarray(accelerometer, dtype=np.float32),
    )


def test_numpy_and_isaac_torch_observation_adapters_match():
    joint_q = DEFAULT_JOINT_POSITION + np.linspace(-0.1, 0.1, 12, dtype=np.float32)
    joint_dq = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
    gyro = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    velocity = np.asarray([0.4, -0.1, 0.0], dtype=np.float32)
    quaternion = np.asarray([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    previous_target = DEFAULT_JOINT_POSITION + 0.01
    state = RobotState(joint_q, joint_dq, gyro, quaternion)
    numpy_observation, _ = build_observation(
        state, velocity, previous_target, previous_quaternion=None
    )
    torch_observation, _ = build_observation_tensor(
        torch.tensor(joint_q)[None],
        torch.tensor(joint_dq)[None],
        torch.tensor(gyro)[None],
        torch.tensor(velocity)[None],
        torch.tensor(quaternion)[None],
        torch.tensor(previous_target)[None],
    )
    np.testing.assert_allclose(
        numpy_observation, torch_observation[0].numpy(), atol=1e-6
    )


def test_torch_velocity_estimator_is_stationary_at_nominal_stance():
    estimator = TorchVelocityEstimator(1, "cpu")
    velocity = estimator.update(
        torch.tensor(DEFAULT_JOINT_POSITION)[None],
        torch.zeros((1, 12)),
        torch.zeros((1, 3)),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 9.81]]),
    )
    torch.testing.assert_close(velocity, torch.zeros_like(velocity), atol=1e-6, rtol=0)


def test_robust_velocity_estimator_converges_and_keeps_covariance_psd():
    estimator = VelocityEstimator(dt=PHYSICS_DT)
    state = _state_with_leg_velocity_candidates([0.5, -0.1, 0.0])

    for _ in range(20):
        velocity = estimator.update(state)

    np.testing.assert_allclose(velocity, [0.5, -0.1, 0.0], atol=2e-3)
    assert estimator.last_measurement_accepted
    assert np.all(np.linalg.eigvalsh(estimator.covariance) >= -1e-12)


def test_analytic_foot_velocity_matches_numeric_jacobian():
    joint_q = DEFAULT_JOINT_POSITION + np.linspace(-0.1, 0.1, 12)
    joint_dq = np.linspace(-1.0, 1.0, 12)
    _, analytic = foot_position_velocity_body(joint_q, joint_dq)
    numeric = np.einsum(
        "lij,lj->li", foot_jacobians_body(joint_q), joint_dq.reshape(4, 3)
    )
    np.testing.assert_allclose(analytic, numeric, atol=2e-7)


def test_imu_prediction_respects_world_to_body_orientation():
    estimator = VelocityEstimator(dt=PHYSICS_DT)
    yaw_90 = np.asarray([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    airborne = RobotState(
        joint_q=DEFAULT_JOINT_POSITION.copy(),
        joint_dq=np.tile([0.0, 100.0, -100.0], 4).astype(np.float32),
        imu_gyro=np.zeros(3, dtype=np.float32),
        imu_quat=yaw_90.astype(np.float32),
        imu_accelerometer=np.asarray([0.0, -1.0, 9.81], dtype=np.float32),
    )

    velocity = estimator.update(airborne)

    np.testing.assert_allclose(
        estimator.world_velocity, [PHYSICS_DT, 0.0, 0.0], atol=1e-7
    )
    np.testing.assert_allclose(velocity, [0.0, -PHYSICS_DT, 0.0], atol=1e-7)


def test_huber_and_nis_reject_slip_and_all_leg_outlier():
    estimator = VelocityEstimator(dt=PHYSICS_DT)
    nominal = _state_with_leg_velocity_candidates([0.5, 0.0, 0.0])
    for _ in range(20):
        estimator.update(nominal)

    slipping = _state_with_leg_velocity_candidates(
        [[0.5, 0.0, 0.0]] * 3 + [[-2.0, 0.0, 0.0]]
    )
    slipped_velocity = estimator.update(slipping)
    np.testing.assert_allclose(slipped_velocity, [0.5, 0.0, 0.0], atol=0.03)

    before = estimator.world_velocity.copy()
    outlier = _state_with_leg_velocity_candidates([3.0, 0.0, 0.0])
    estimator.update(outlier)
    assert not estimator.last_measurement_accepted
    assert estimator.last_innovation_squared > estimator.innovation_gate
    np.testing.assert_allclose(estimator.world_velocity, before, atol=1e-8)


def test_repeated_coherent_innovation_recovers_from_stale_covariance():
    estimator = VelocityEstimator(dt=PHYSICS_DT)
    stationary = _state_with_leg_velocity_candidates([0.0, 0.0, 0.0])
    for _ in range(30):
        estimator.update(stationary)

    moving = _state_with_leg_velocity_candidates([0.8, 0.0, 0.0])
    first = estimator.update(moving)
    assert not estimator.last_measurement_accepted
    np.testing.assert_allclose(first, 0.0, atol=1e-6)

    accepted = False
    for _ in range(20):
        velocity = estimator.update(moving)
        accepted = accepted or estimator.last_measurement_accepted

    assert accepted
    assert velocity[0] > 0.5


def test_no_support_keeps_imu_prediction_without_velocity_damping():
    estimator = VelocityEstimator(dt=PHYSICS_DT)
    estimator.reset(np.asarray([0.4, -0.2, 0.1]))
    airborne = RobotState(
        joint_q=DEFAULT_JOINT_POSITION.copy(),
        joint_dq=np.tile([0.0, 100.0, -100.0], 4).astype(np.float32),
        imu_gyro=np.zeros(3, dtype=np.float32),
        imu_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        imu_accelerometer=np.asarray([1.0, 0.0, 9.81], dtype=np.float32),
    )

    velocity = estimator.update(airborne)

    np.testing.assert_allclose(
        velocity, [0.4 + PHYSICS_DT, -0.2, 0.1], atol=1e-6
    )
    assert estimator.last_innovation_squared is None


def test_reset_and_invalid_input_contracts():
    estimator = VelocityEstimator(dt=PHYSICS_DT)
    estimator.update(_state_with_leg_velocity_candidates([0.3, 0.0, 0.0]))
    estimator.reset()
    np.testing.assert_allclose(estimator.world_velocity, 0.0)
    np.testing.assert_allclose(estimator.covariance, np.eye(3) * 0.1)
    assert estimator.update_count == 0
    assert estimator.last_innovation_squared is None

    invalid = _state_with_leg_velocity_candidates([0.0, 0.0, 0.0])
    invalid.joint_dq[0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        estimator.update(invalid)


def test_observation_builder_consumes_ten_frames_once_and_uses_final_state():
    class CountingEstimator:
        def __init__(self):
            self.count = 0

        def reset(self):
            self.count = 0

        def update(self, state):
            self.count += 1
            return np.asarray([self.count, 0.0, 0.0], dtype=np.float32)

    estimator = CountingEstimator()
    builder = ObservationBuilder(estimator)
    frames = []
    for index in range(10):
        state = _state_with_leg_velocity_candidates([0.0, 0.0, 0.0])
        state.joint_q = state.joint_q + index * 1e-3
        frames.append(state)

    observation, velocity = builder.build_many(frames)

    assert estimator.count == 10
    np.testing.assert_allclose(velocity, [10.0, 0.0, 0.0])
    np.testing.assert_allclose(
        observation[OBSERVATION_SPEC.body_velocity], velocity
    )
    np.testing.assert_allclose(observation[:12], frames[-1].joint_q)


def test_numpy_and_torch_estimators_share_the_same_model_contract():
    state = _state_with_leg_velocity_candidates([0.45, -0.08, 0.02])
    numpy_estimator = VelocityEstimator(dt=0.02)
    torch_estimator = TorchVelocityEstimator(1, "cpu", dt=0.02)

    for _ in range(10):
        numpy_velocity = numpy_estimator.update(state)
        torch_velocity = torch_estimator.update(
            torch.tensor(state.joint_q)[None],
            torch.tensor(state.joint_dq)[None],
            torch.tensor(state.imu_gyro)[None],
            torch.tensor(state.imu_quat)[None],
            torch.tensor(state.imu_accelerometer)[None],
        )[0]

    np.testing.assert_allclose(
        numpy_velocity, torch_velocity.numpy(), rtol=2e-5, atol=2e-5
    )


def test_robust_filter_reduces_periodic_slip_rmse_against_legacy_filter():
    robust = VelocityEstimator(dt=PHYSICS_DT)
    legacy = LegacyVelocityEstimator(dt=0.02)
    truth = np.asarray([0.5, 0.0, 0.0])
    robust_errors = []
    legacy_errors = []

    for policy_step in range(80):
        candidates = np.tile(truth, (4, 1))
        if policy_step % 5 == 0:
            candidates[policy_step // 5 % 4] = [-2.0, 0.0, 0.0]
        state = _state_with_leg_velocity_candidates(candidates)
        for _ in range(10):
            robust_velocity = robust.update(state)
        legacy_velocity = legacy.update(state)
        if policy_step >= 10:
            robust_errors.append(robust_velocity - truth)
            legacy_errors.append(legacy_velocity - truth)

    robust_rmse = summarize_errors(np.asarray(robust_errors))["rmse_3d"]
    legacy_rmse = summarize_errors(np.asarray(legacy_errors))["rmse_3d"]
    assert robust_rmse < legacy_rmse


def test_benchmark_defaults_follow_common_estimator_config():
    args = build_parser().parse_args([])
    assert args.process_variance == pytest.approx(0.03059)
    assert args.leg_variance == pytest.approx(0.002)
    summary = summarize_errors(
        np.asarray([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]])
    )
    np.testing.assert_allclose(summary["bias_xyz"], 0.0)
    np.testing.assert_allclose(summary["rmse_xyz"], [1.0, 1.0, 0.0])
