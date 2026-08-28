from pathlib import Path

import numpy as np
import pytest

from src.config import DEFAULT_MUJOCO_SCENE, RUN_PRESETS
from src.environments.go2_sqrl.common.specs import (
    ACTION_SPEC,
    DEFAULT_BASE_HEIGHT,
    JOINT_NAMES,
    PHYSICS_DT,
)
from src.run import _artifact_flags
from rl_x.algorithms.sac_qsafe.pytorch.default_config import get_config


def test_framework_is_selected_by_training_backend():
    for command in ("pretrain", "isaac-eval"):
        assert "--algorithm.name=sac_qsafe.pytorch" in RUN_PRESETS[command]
    for command in ("zero-shot", "finetune", "eval"):
        assert "--algorithm.name=sac_qsafe.flax" in RUN_PRESETS[command]


def test_sqrl_defaults_match_paper_settings():
    config = get_config("sac_qsafe.pytorch")

    assert config.total_timesteps == pytest.approx(5e5)
    assert config.learning_rate == pytest.approx(3e-4)
    assert config.qsafe.epsilon == pytest.approx(0.1)
    assert config.qsafe.gamma == pytest.approx(0.7)


def test_zero_shot_uses_the_sqrl_projected_policy_on_flat_ground():
    scene = Path(DEFAULT_MUJOCO_SCENE)
    assert scene.name == "scene.xml"
    assert scene.is_file()
    assert scene.parent.name == "mjcf"
    assert "--algorithm.eval_policy=safe" in RUN_PRESETS["zero-shot"]


def test_pretrain_parity_checkpoint_uses_flat_ground():
    assert "--environment.terrain_mode=flat" in RUN_PRESETS["pretrain"]
    assert (
        "--runner.run_name=isaac_flashsac_cmd_reward_v3"
        in RUN_PRESETS["pretrain"]
    )


def test_isaac_eval_is_single_environment_deterministic_task_policy():
    preset = RUN_PRESETS["isaac-eval"]
    assert "--runner.mode=test" in preset
    assert "--environment.nr_envs=1" in preset
    assert "--environment.nr_task_envs=1" in preset
    assert "--environment.nr_safety_envs=0" in preset
    assert "--environment.terrain_mode=flat" in preset
    assert "--algorithm.eval_policy=task" in preset


def test_canonical_mujoco_asset_matches_sdk_bridge_contract():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(DEFAULT_MUJOCO_SCENE)

    def names(object_type, count):
        return [
            mujoco.mj_id2name(model, object_type, index)
            for index in range(count)
        ]

    assert model.opt.timestep == pytest.approx(PHYSICS_DT)
    assert model.nu == ACTION_SPEC.size
    assert names(mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu) == list(JOINT_NAMES)
    np.testing.assert_allclose(
        model.actuator_ctrlrange,
        np.tile(
            [-ACTION_SPEC.effort_limit, ACTION_SPEC.effort_limit],
            (ACTION_SPEC.size, 1),
        ),
    )

    # unitree_sdk2_bridge reads the first 3 * nu scalar sensor values as joint
    # position, velocity and actuator force, in SDK motor order.
    expected_sensors = [f"{name}_pos" for name in JOINT_NAMES]
    expected_sensors += [f"{name}_vel" for name in JOINT_NAMES]
    expected_sensors += [f"{name}_torque" for name in JOINT_NAMES]
    sensor_names = names(mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor)
    assert sensor_names[: 3 * ACTION_SPEC.size] == expected_sensors
    assert {"imu_quat", "imu_gyro", "imu_acc", "frame_pos", "frame_vel"} <= set(
        sensor_names
    )

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    assert base_id >= 0
    assert model.body_pos[base_id, 2] == pytest.approx(DEFAULT_BASE_HEIGHT)
    assert model.nq == 7 + ACTION_SPEC.size
    assert model.nv == 6 + ACTION_SPEC.size
    np.testing.assert_allclose(model.dof_damping[6:], ACTION_SPEC.joint_damping)
    np.testing.assert_allclose(model.dof_armature[6:], ACTION_SPEC.armature)
    np.testing.assert_allclose(model.dof_frictionloss[6:], ACTION_SPEC.joint_friction)

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert home_id >= 0
    np.testing.assert_allclose(
        model.key_qpos[home_id, 7:], ACTION_SPEC.default_position, atol=1e-7
    )
    np.testing.assert_allclose(model.key_qvel[home_id], 0.0, atol=1e-12)
    np.testing.assert_allclose(model.key_ctrl[home_id], 0.0, atol=1e-12)

    # The home base height is geometry-derived: all four spherical feet begin
    # in the contact margin, rather than suspended or deeply penetrating.
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)
    foot_bottoms = []
    for name in ("FR", "FL", "RR", "RL"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        foot_bottoms.append(data.geom_xpos[geom_id, 2] - model.geom_size[geom_id, 0])
    np.testing.assert_allclose(foot_bottoms, 0.0, atol=1e-3)

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert floor_id >= 0
    assert model.geom_friction[floor_id, 0] == pytest.approx(0.4)


def test_transfer_commands_use_torch_pretrain_sidecars(tmp_path: Path):
    (tmp_path / "policy.model").touch()
    (tmp_path / "qsafe.model").touch()
    flags = _artifact_flags("finetune", str(tmp_path))
    assert flags == [
        f"--algorithm.pretrained_policy_path={tmp_path / 'policy.model'}",
        f"--algorithm.qsafe.checkpoint_path={tmp_path / 'qsafe.model'}",
    ]


def test_eval_rejects_a_missing_flax_checkpoint(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        _artifact_flags("eval", str(tmp_path))
