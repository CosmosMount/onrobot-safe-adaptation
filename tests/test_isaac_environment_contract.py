import math
import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np
import torch


def _install_isaac_import_stubs():
    """Allow dependency-light adapter tests without an Isaac Sim installation."""

    try:
        __import__("isaaclab")
        return
    except ModuleNotFoundError:
        pass

    def module(name, package=False):
        result = types.ModuleType(name)
        if package:
            result.__path__ = []
        sys.modules[name] = result
        return result

    module("isaaclab", package=True)
    utils = module("isaaclab.utils")
    utils.configclass = lambda cls: cls
    module("isaaclab.terrains")
    sensors = module("isaaclab.sensors", package=True)
    sensors.Imu = type("Imu", (), {})
    sensors.ImuCfg = type("ImuCfg", (), {})
    sensors.patterns = SimpleNamespace()
    sensor_base = module("isaaclab.sensors.sensor_base")
    sensor_base.SensorBase = type("SensorBase", (), {"update": lambda *args: None})

    module("isaaclab_assets", package=True)
    module("isaaclab_assets.robots", package=True)
    unitree = module("isaaclab_assets.robots.unitree")
    unitree.UNITREE_GO2_CFG = object()

    module("isaaclab_tasks", package=True)
    module("isaaclab_tasks.manager_based", package=True)
    module("isaaclab_tasks.manager_based.locomotion", package=True)
    module("isaaclab_tasks.manager_based.locomotion.velocity", package=True)
    velocity_cfg = module(
        "isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg"
    )
    velocity_cfg.LocomotionVelocityRoughEnvCfg = type(
        "LocomotionVelocityRoughEnvCfg", (), {}
    )


_install_isaac_import_stubs()

from train.core.base import (
    ACTION_SPEC,
    DEFAULT_JOINT_POSITION,
    JOINT_NAMES,
    OBSERVATION_SPEC,
    RobotState,
)
from train.core.estimation import VelocityEstimatorConfig
from train.core.task import (
    build_manifest,
    build_observation,
    compute_reward,
    compute_reward_tensor,
    validate_transfer_manifest,
)
from train.isaac.pytorch.config import get_config
from train.isaac.pytorch.environment import (
    Go2IsaacEnv,
    PhysicsFrameReportingManagerMixin,
)
from train.isaac.pytorch.setup import build_observation_tensor, sdk_joint_indices


class _ActionTerm:
    def __init__(self):
        self._joint_names = tuple(f"{name}_joint" for name in JOINT_NAMES)
        self._offset = torch.as_tensor(DEFAULT_JOINT_POSITION)[None, :]
        self._scale = torch.as_tensor(ACTION_SPEC.scale)


class _ActionManager:
    def __init__(self):
        self.term = _ActionTerm()

    def get_term(self, name):
        if name != "joint_pos":
            raise KeyError(name)
        return self.term


class _RobotData:
    def __init__(self, nr_envs):
        self.joint_pos = torch.zeros(nr_envs, 12)
        self.joint_vel = torch.zeros(nr_envs, 12)
        self.root_lin_vel_b = torch.zeros(nr_envs, 3)
        self.root_pos_w = torch.zeros(nr_envs, 3)
        self.body_pos_w = torch.zeros(nr_envs, 4, 3)
        self.applied_torque = torch.zeros(nr_envs, 12)


class _Robot:
    device = torch.device("cpu")

    def __init__(self, nr_envs):
        self.joint_names = [f"{name}_joint" for name in JOINT_NAMES]
        self.data = _RobotData(nr_envs)

    def find_bodies(self, pattern):
        self.last_body_pattern = pattern
        return [0, 1, 2, 3], [
            "FR_foot", "FL_foot", "RR_foot", "RL_foot"
        ]


class _Scene(dict):
    def __init__(self, robot, imu, scanner):
        super().__init__(robot=robot, imu=imu, height_scanner=scanner)
        self.terrain = SimpleNamespace(terrain_origins=None)


class _FakeIsaacBackend:
    def __init__(self, nr_envs):
        self.nr_envs = nr_envs
        self.robot = _Robot(nr_envs)
        self.imu = SimpleNamespace(
            data=SimpleNamespace(
                ang_vel_b=torch.zeros(nr_envs, 3),
                lin_acc_b=torch.tensor([[0.0, 0.0, 9.81]]).repeat(nr_envs, 1),
                quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(nr_envs, 1),
            )
        )
        ray_xy = torch.tensor(
            [[0.0, 0.0], [0.05, 0.0], [0.0, 0.05], [-0.05, 0.0]]
        )
        rays = torch.zeros(nr_envs, 4, 3)
        rays[:, :, :2] = ray_xy
        self.scanner = SimpleNamespace(data=SimpleNamespace(ray_hits_w=rays))
        self.scene = _Scene(self.robot, self.imu, self.scanner)
        self.action_manager = _ActionManager()
        self.callback = None
        self.queued_frames = None
        self.queued_terminated = torch.zeros(nr_envs, dtype=torch.bool)
        self.queued_truncated = torch.zeros(nr_envs, dtype=torch.bool)
        self.public_reset_calls = []
        self.reset()

    def set_physics_frame_callback(self, callback):
        self.callback = callback

    def _apply_reset(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long).reshape(-1)
        if env_ids.numel() == 0:
            return
        self.robot.data.joint_pos[env_ids] = torch.as_tensor(
            DEFAULT_JOINT_POSITION
        )
        self.robot.data.joint_vel[env_ids] = 0
        self.robot.data.root_lin_vel_b[env_ids] = 0
        self.robot.data.root_pos_w[env_ids] = torch.tensor([0.0, 0.0, 0.289])
        self.robot.data.body_pos_w[env_ids] = torch.tensor(
            [[0.0, 0.0, 0.022]] * 4
        )
        self.robot.data.applied_torque[env_ids] = 0
        self.imu.data.ang_vel_b[env_ids] = 0
        self.imu.data.lin_acc_b[env_ids] = torch.tensor([0.0, 0.0, 9.81])
        self.imu.data.quat_w[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0])

    def reset(self):
        self._apply_reset(torch.arange(self.nr_envs))

    def reset_envs(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long).reshape(-1)
        self.public_reset_calls.append(tuple(env_ids.tolist()))
        self._apply_reset(env_ids)

    def queue(self, frames, *, terminated=None, truncated=None):
        if len(frames) != 10:
            raise ValueError("fake Isaac step requires ten physics frames")
        self.queued_frames = frames
        self.queued_terminated = torch.as_tensor(
            np.zeros(self.nr_envs, dtype=bool) if terminated is None else terminated,
            dtype=torch.bool,
        )
        self.queued_truncated = torch.as_tensor(
            np.zeros(self.nr_envs, dtype=bool) if truncated is None else truncated,
            dtype=torch.bool,
        )

    def _set_frame(self, frame):
        nr_envs = self.nr_envs
        q_delta = torch.as_tensor(frame.get("q_delta", 0.0), dtype=torch.float32)
        if q_delta.ndim == 0:
            q_delta = q_delta.repeat(nr_envs)
        self.robot.data.joint_pos[:] = torch.as_tensor(
            DEFAULT_JOINT_POSITION
        )[None, :] + q_delta[:, None]
        self.robot.data.joint_vel.zero_()
        velocity_x = torch.as_tensor(
            frame.get("velocity_x", [0.5] * nr_envs), dtype=torch.float32
        ).reshape(nr_envs)
        self.robot.data.root_lin_vel_b.zero_()
        self.robot.data.root_lin_vel_b[:, 0] = velocity_x
        base_z = torch.as_tensor(
            frame.get("base_z", [0.289] * nr_envs), dtype=torch.float32
        ).reshape(nr_envs)
        self.robot.data.root_pos_w[:, 2] = base_z
        self.robot.data.body_pos_w[:, :, 2] = 0.022
        quaternion = torch.as_tensor(
            frame.get("quaternion", [[1.0, 0.0, 0.0, 0.0]] * nr_envs),
            dtype=torch.float32,
        ).reshape(nr_envs, 4)
        self.imu.data.quat_w[:] = quaternion
        self.imu.data.ang_vel_b.zero_()
        self.imu.data.lin_acc_b[:] = torch.tensor([0.0, 0.0, 9.81])
        self.robot.data.applied_torque.zero_()

    def step(self, action):
        del action
        if self.queued_frames is None:
            raise RuntimeError("fake backend has no queued physics window")
        for frame in self.queued_frames:
            self._set_frame(frame)
            self.callback()
        terminated = self.queued_terminated.clone()
        truncated = self.queued_truncated.clone()
        auto_reset_ids = torch.nonzero(terminated | truncated).flatten()
        self._apply_reset(auto_reset_ids)
        self.queued_frames = None
        return None, None, terminated, truncated, {}


def _make_config(nr_envs=1, nr_task_envs=1):
    environment = get_config("fake.isaac")
    environment.nr_envs = nr_envs
    environment.nr_task_envs = nr_task_envs
    environment.nr_safety_envs = nr_envs - nr_task_envs
    environment.device = "cpu"
    environment.terrain_mode = "flat"
    environment.domain_randomization = False
    environment.playback_terrain_type = "auto"
    environment.playback_terrain_level = -1
    return SimpleNamespace(environment=environment)


def _roll_quaternion(angle):
    return [math.cos(0.5 * angle), math.sin(0.5 * angle), 0.0, 0.0]


class IsaacEnvironmentContractTest(unittest.TestCase):
    def make_env(self, nr_envs=1, nr_task_envs=1):
        backend = _FakeIsaacBackend(nr_envs)
        env = Go2IsaacEnv(
            _make_config(nr_envs, nr_task_envs), backend=backend
        )
        env.reset()
        backend.public_reset_calls.clear()
        return env, backend

    def test_numpy_and_torch_observation_reward_contracts_are_identical(self):
        joint_q = DEFAULT_JOINT_POSITION + np.linspace(-0.02, 0.02, 12)
        joint_dq = np.linspace(-0.3, 0.3, 12)
        gyro = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
        velocity = np.asarray([0.45, -0.02, 0.01], dtype=np.float32)
        # Deliberately non-unit: both backends must normalize identically.
        quaternion = np.asarray([1.8, 0.2, -0.1, 0.3], dtype=np.float32)
        previous_target = DEFAULT_JOINT_POSITION + 0.01
        torque = np.linspace(-2.0, 2.0, 12, dtype=np.float32)
        state = RobotState(joint_q, joint_dq, gyro, quaternion)

        numpy_observation, _ = build_observation(
            state, velocity, previous_target
        )
        torch_observation, _ = build_observation_tensor(
            torch.as_tensor(joint_q)[None, :],
            torch.as_tensor(joint_dq)[None, :],
            torch.as_tensor(gyro)[None, :],
            torch.as_tensor(velocity)[None, :],
            torch.as_tensor(quaternion)[None, :],
            torch.as_tensor(previous_target)[None, :],
        )
        np.testing.assert_allclose(
            numpy_observation,
            torch_observation[0].numpy(),
            atol=2.0e-7,
            rtol=0.0,
        )

        numpy_reward = compute_reward(
            velocity, quaternion, gyro, torque, 0.5
        )
        torch_terms, torch_total = compute_reward_tensor(
            torch.as_tensor(velocity)[None, :],
            torch.as_tensor(quaternion)[None, :],
            torch.as_tensor(gyro)[None, :],
            torch.as_tensor(torque)[None, :],
            0.5,
        )
        for name, value in numpy_reward.as_dict().items():
            actual = (
                torch_total[0]
                if name == "reward/total"
                else torch_terms[name.removeprefix("reward/")][0]
            )
            self.assertAlmostEqual(value, float(actual), places=6)

    def test_checkpoint_manifest_records_the_complete_observation_and_reset_layout(self):
        manifest = build_manifest()
        self.assertEqual(
            manifest["observation"]["layout"],
            {
                "joint_q": [0, 12],
                "joint_dq": [12, 24],
                "imu_gyro": [24, 27],
                "body_velocity": [27, 30],
                "imu_quat": [30, 34],
                "previous_action_q_target": [34, 46],
            },
        )
        self.assertEqual(manifest["observation"]["size"], OBSERVATION_SPEC.size)
        self.assertEqual(
            manifest["reset"]["nominal_foot_contact"], "all_four_feet"
        )
        self.assertEqual(
            manifest["reset"]["base_quaternion_wxyz"], [1.0, 0.0, 0.0, 0.0]
        )
        self.assertEqual(
            manifest["observation"]["velocity_estimator"]["parameters"],
            {
                "process_variance": 0.03059,
                "leg_variance": 0.002,
                "initial_variance": 0.1,
                "height_scale": 0.05,
                "vertical_velocity_scale": 0.35,
                "huber_delta": 0.25,
                "prior_temperature": 0.05,
                "innovation_gate": 11.34,
                "rejection_covariance_inflation": 2.0,
                "minimum_total_confidence": 0.2,
            },
        )

    def test_transfer_manifest_rejects_velocity_estimator_parameter_drift(self):
        source = build_manifest()
        destination = build_manifest(
            velocity_estimator_config=VelocityEstimatorConfig(
                process_variance=0.04
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "velocity_estimator.parameters.process_variance",
        ):
            validate_transfer_manifest(source, destination)

    def test_deterministic_reset_rejects_nonidentity_base_yaw(self):
        env, backend = self.make_env()
        backend.imu.data.quat_w[0] = torch.tensor(
            [math.cos(0.05), 0.0, 0.0, math.sin(0.05)]
        )
        with self.assertRaisesRegex(RuntimeError, "identity contract"):
            env._validate_reset_contract(torch.tensor([0]))

    def test_public_indexed_reset_flushes_scene_before_post_reset(self):
        events = []

        class Recorder:
            active_terms = ()

            def record_pre_reset(self, env_ids):
                events.append(("pre", tuple(env_ids.tolist())))

            def record_post_reset(self, env_ids):
                events.append(("post", tuple(env_ids.tolist())))

        class Scene:
            def write_data_to_sim(self):
                events.append("write")

        class Sim:
            def forward(self):
                events.append("forward")

            def has_rtx_sensors(self):
                return False

        manager = PhysicsFrameReportingManagerMixin.__new__(
            PhysicsFrameReportingManagerMixin
        )
        manager.device = torch.device("cpu")
        manager.recorder_manager = Recorder()
        manager.scene = Scene()
        manager.sim = Sim()
        manager.cfg = SimpleNamespace(num_rerenders_on_reset=0)
        manager._reset_idx = lambda env_ids: events.append(
            ("reset", tuple(env_ids.tolist()))
        )

        manager.reset_envs([3, 1])

        self.assertEqual(
            events,
            [
                ("pre", (3, 1)),
                ("reset", (3, 1)),
                "write",
                "forward",
                ("post", (3, 1)),
            ],
        )

    def test_timeout_reward_and_final_observation_are_pre_reset(self):
        env, backend = self.make_env()
        frames = [
            {"velocity_x": [0.5], "q_delta": 0.05} for _ in range(10)
        ]
        backend.queue(frames, truncated=[True])

        observation, reward, terminated, truncated, info = env.step(
            np.zeros((1, 12), dtype=np.float32)
        )

        self.assertFalse(terminated[0])
        self.assertTrue(truncated[0])
        self.assertAlmostEqual(float(reward[0]), 1.0, places=6)
        np.testing.assert_allclose(
            info["final_observation"][0][:12],
            DEFAULT_JOINT_POSITION + 0.05,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            observation[0, :12], DEFAULT_JOINT_POSITION, atol=1e-6
        )
        self.assertAlmostEqual(float(info["forward_velocity"][0]), 0.5)
        self.assertEqual(env._velocity_estimator.update_count, 10)
        self.assertAlmostEqual(env._velocity_estimator.dt, 0.002)
        self.assertEqual(backend.public_reset_calls, [])

    def test_joint_mapping_is_exact_one_to_one_sdk_order(self):
        source = [
            f"{name}_joint"
            for name in (
                "FL_hip", "FR_calf", "RL_thigh", "RR_hip",
                "FR_hip", "FL_calf", "RR_thigh", "RL_hip",
                "FR_thigh", "FL_thigh", "RR_calf", "RL_calf",
            )
        ]
        indices = sdk_joint_indices(source).tolist()
        reordered = [source[index].removesuffix("_joint") for index in indices]
        self.assertEqual(reordered, list(JOINT_NAMES))
        with self.assertRaisesRegex(ValueError, "Duplicate joints"):
            sdk_joint_indices(source[:-1] + [source[0]])
        with self.assertRaisesRegex(ValueError, "Missing Go2 joints"):
            sdk_joint_indices(source[:-1] + ["extra_joint"])

    def test_tilt_requires_five_consecutive_physics_frames_across_windows(self):
        env, backend = self.make_env()
        bad = {"quaternion": [_roll_quaternion(1.0)]}
        good = {"quaternion": [[1.0, 0.0, 0.0, 0.0]]}
        backend.queue([good] * 6 + [bad] * 4)
        _, _, terminated, _, _ = env.step(np.zeros((1, 12), dtype=np.float32))
        self.assertFalse(terminated[0])

        backend.queue([bad] + [good] * 9)
        _, _, terminated, _, info = env.step(
            np.zeros((1, 12), dtype=np.float32)
        )
        self.assertTrue(terminated[0])
        self.assertEqual(float(info["failure/tilt"][0]), 1.0)
        self.assertEqual(backend.public_reset_calls, [(0,)])

    def test_height_requires_five_consecutive_physics_frames_across_windows(self):
        env, backend = self.make_env()
        low = {"base_z": [0.17]}
        good = {"base_z": [0.289]}
        backend.queue([good] * 6 + [low] * 4)
        _, _, terminated, _, _ = env.step(np.zeros((1, 12), dtype=np.float32))
        self.assertFalse(terminated[0])

        backend.queue([low] + [good] * 9)
        _, _, terminated, _, info = env.step(
            np.zeros((1, 12), dtype=np.float32)
        )
        self.assertTrue(terminated[0])
        self.assertEqual(float(info["failure/height"][0]), 1.0)

    def test_public_partition_reset_does_not_touch_other_pool(self):
        env, backend = self.make_env(nr_envs=2, nr_task_envs=1)
        backend.robot.data.joint_pos[0] += 0.1
        backend.robot.data.joint_pos[1] += 0.2
        env._previous_target[1] += 0.15
        untouched_target = env._previous_target[1].clone()

        task_observation = env.reset_task_partition()

        self.assertEqual(task_observation.shape, (1, 46))
        np.testing.assert_allclose(
            backend.robot.data.joint_pos[0].numpy(),
            DEFAULT_JOINT_POSITION,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            backend.robot.data.joint_pos[1].numpy(),
            DEFAULT_JOINT_POSITION + 0.2,
            atol=1e-6,
        )
        torch.testing.assert_close(env._previous_target[1], untouched_target)
        self.assertEqual(backend.public_reset_calls, [(0,)])

    def test_partition_step_returns_task_and_safety_slices(self):
        env, backend = self.make_env(nr_envs=2, nr_task_envs=1)
        backend.queue([{}] * 10)

        task_step, safety_step = env.step_partitions(
            task_actions=np.zeros((1, 12), dtype=np.float32)
        )

        self.assertEqual(task_step.observation.shape, (1, 46))
        self.assertEqual(safety_step.observation.shape, (1, 46))
        self.assertEqual(task_step.reward.shape, (1,))
        self.assertEqual(safety_step.reward.shape, (1,))
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            env.step_partitions()

if __name__ == "__main__":
    unittest.main()
