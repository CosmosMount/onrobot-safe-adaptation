from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import (
    DEFAULT_MUJOCO_SCENE,
    MUJOCO_SAC_FINETUNE_FLAGS,
    RUN_PRESETS,
    STABLE_ISAAC_SAC_FINETUNE_FLAGS,
)
from src.environments.go2_sqrl.common.specs import (
    ACTION_SPEC,
    DEFAULT_BASE_HEIGHT,
    JOINT_NAMES,
    PHYSICS_DT,
)
from src.run import _artifact_flags, _deduplicate_config_flags
from rl_x.algorithms.qsafe.common import (
    GaitEvaluationMetrics,
    actor_updates_enabled,
    finetune_constraints_enabled,
)
from rl_x.algorithms.sac_qsafe.pytorch.default_config import get_config
from tools.run_universal_qsafe_actor_stage import high_v2_training_command
from tools.run_sqrl_ablation import (
    build_parser as build_ablation_parser,
    mujoco_sac_train_command,
)
from tools.collect_universal_qsafe_dataset import (
    FRS_DIAGNOSTIC_ROLES,
    profile_cells,
)


def test_framework_is_selected_by_training_backend():
    for command in (
        "pretrain",
        "pretrain-sac",
        "isaac-eval",
        "isaac-collect-qsafe",
        "isaac-finetune",
        "isaac-finetune-sac",
    ):
        assert "--algorithm.name=sac_qsafe.pytorch" in RUN_PRESETS[command]
    for command in ("zero-shot", "finetune", "finetune-sac", "eval"):
        assert "--algorithm.name=sac_qsafe.flax" in RUN_PRESETS[command]


def test_explicit_config_flags_override_preset_defaults():
    flags = _deduplicate_config_flags([
        "--runner.mode=train",
        "--environment.nr_envs=20",
        "--runner.mode=test",
        "--environment.nr_envs=4",
    ])
    assert "--runner.mode=train" not in flags
    assert "--environment.nr_envs=20" not in flags
    assert "--runner.mode=test" in flags
    assert "--environment.nr_envs=4" in flags


def test_sqrl_defaults_match_paper_settings():
    config = get_config("sac_qsafe.pytorch")

    assert config.total_timesteps == pytest.approx(5e5)
    assert config.learning_rate == pytest.approx(3e-4)
    assert config.qsafe.enabled is True
    assert config.qsafe.epsilon == pytest.approx(0.1)
    assert config.qsafe.gamma == pytest.approx(0.7)
    assert config.qsafe.paired_candidate_evaluation is False
    assert config.evaluation_results_path == ""


def test_zero_shot_uses_the_sqrl_projected_policy_on_flat_ground():
    scene = Path(DEFAULT_MUJOCO_SCENE)
    assert scene.name == "scene.xml"
    assert scene.is_file()
    assert scene.parent.name == "mjcf"
    assert "--algorithm.eval_policy=safe" in RUN_PRESETS["zero-shot"]


def test_pretrain_parity_checkpoint_uses_flat_ground():
    assert "--environment.terrain_mode=flat" in RUN_PRESETS["pretrain"]
    assert "--environment.domain_randomization=true" in RUN_PRESETS["pretrain"]
    assert (
        "--runner.run_name=isaac_sqrl_height_dr_v1"
        in RUN_PRESETS["pretrain"]
    )


def test_standard_sac_baseline_has_no_safety_partition_or_qsafe():
    pretrain = RUN_PRESETS["pretrain-sac"]
    assert "--environment.nr_envs=256" in pretrain
    assert "--environment.nr_task_envs=256" in pretrain
    assert "--environment.nr_safety_envs=0" in pretrain
    assert "--algorithm.qsafe.enabled=false" in pretrain
    assert "--algorithm.eval_policy=task" in pretrain

    finetune = RUN_PRESETS["finetune-sac"]
    assert "--algorithm.qsafe.enabled=false" in finetune
    assert "--algorithm.eval_policy=task" in finetune
    assert "--environment.target_velocity_x=0.6" in finetune
    assert "--environment.target_velocity_x=0.6" in RUN_PRESETS["finetune"]
    assert set(MUJOCO_SAC_FINETUNE_FLAGS) <= set(finetune)
    assert set(MUJOCO_SAC_FINETUNE_FLAGS) <= set(RUN_PRESETS["finetune"])
    assert "--algorithm.learning_starts=1000" in finetune
    assert "--algorithm.finetune_actor_warmup_steps=10000" in finetune
    assert not any(
        flag.startswith("--algorithm.finetune_actor_handoff_steps=")
        for flag in finetune
    )
    assert "--algorithm.finetune_actor_update_interval=10" in finetune
    assert "--algorithm.task_utd_ratio=1.0" in finetune
    assert "--algorithm.alpha_init=0.0002" in finetune

    isaac_finetune = RUN_PRESETS["isaac-finetune-sac"]
    assert "--algorithm.phase=finetune" in isaac_finetune
    assert "--algorithm.rollout_mode=partitioned" in isaac_finetune
    assert "--algorithm.qsafe.enabled=false" in isaac_finetune
    assert "--environment.terrain_profile=single_step_up" in isaac_finetune
    assert set(STABLE_ISAAC_SAC_FINETUNE_FLAGS) <= set(isaac_finetune)
    assert "--algorithm.learning_starts=2500" in isaac_finetune
    assert "--algorithm.finetune_actor_warmup_steps=2500" in isaac_finetune
    assert "--algorithm.finetune_actor_update_interval=1" in isaac_finetune
    assert "--environment.nr_envs=20" in isaac_finetune
    assert "--environment.nr_task_envs=20" in isaac_finetune


def test_formal_mujoco_finetune_starts_fresh_task_learning_state(tmp_path):
    args = SimpleNamespace(
        seed=0,
        domain_id=1,
        interface="lo",
        steps=30000,
        alpha_init=2e-4,
        checkpoint_frequency=10000,
        logging_frequency=1000,
        step_height=0.02,
        failure_penalty=-10.0,
    )
    command = mujoco_sac_train_command(args, tmp_path, "fresh_task_s0")
    assert "--full-task-transfer" not in command
    assert not any(
        value.startswith("--algorithm.pretrained_task_checkpoint_path=")
        for value in command
    )
    assert "--algorithm.alpha_init=0.0002" in command
    assert "--algorithm.finetune_actor_warmup_steps=10000" in command
    assert not any(
        value.startswith("--algorithm.finetune_actor_handoff_steps=")
        for value in command
    )
    assert "--algorithm.finetune_actor_update_interval=10" in command
    assert "--algorithm.task_utd_ratio=1.0" in command


def test_flat_baseline_disables_optional_clearance_and_phase_shaping(tmp_path):
    args = SimpleNamespace(
        seed=0,
        domain_id=1,
        interface="lo",
        steps=15000,
        alpha_init=2e-4,
        checkpoint_frequency=5000,
        logging_frequency=1000,
        terrain_profile="flat",
    )

    command = mujoco_sac_train_command(args, tmp_path, "flat_speed_s0")

    assert "--environment.target_velocity_x=0.6" in command
    assert "--environment.terrain_profile=flat" in command
    assert "--environment.foot_clearance_target=0.0" in command
    assert "--environment.clearance_reward_mode=legacy_mean" in command
    assert "--environment.phase_reward_scale=0.0" in command
    assert "--environment.stable_progress_scale=0.0" in command
    assert "--environment.terminal_failure_penalty=0.0" in command
    assert not any(value.startswith("--environment.step_height=") for value in command)
    assert "--full-task-transfer" not in command
    assert not any(
        value.startswith("--algorithm.pretrained_task_checkpoint_path=")
        for value in command
    )


def test_mujoco_sac_baseline_defaults_to_v11_flat_contract():
    args = build_ablation_parser().parse_args(["mujoco-sac-baseline"])

    assert args.scene.name == "scene.xml"
    assert args.terrain_profile == "flat"
    assert args.checkpoint.name == "models"
    assert args.checkpoint.parent.name == "isaac_sac_flat_action_v2_legacy_v1"


def test_high_clearance_actor_uses_frozen_stable_sac_recipe():
    easy_command = high_v2_training_command(
        "python",
        "/source/models",
        6,
        "easy",
        50000,
        "stable_high",
    )
    assert set(STABLE_ISAAC_SAC_FINETUNE_FLAGS) <= set(easy_command)
    assert "--actor-only" not in easy_command
    assert "--full-task-transfer" not in easy_command
    assert "--algorithm.finetune_actor_warmup_steps=10000" not in easy_command
    assert "--algorithm.finetune_actor_warmup_steps=0" not in easy_command
    assert "--algorithm.finetune_actor_update_interval=10" not in easy_command
    assert "--algorithm.finetune_actor_update_interval=5" not in easy_command
    assert "--environment.nr_envs=20" in easy_command
    assert "--environment.nr_task_envs=20" in easy_command

    medium_command = high_v2_training_command(
        "python",
        "/target/easy/models",
        6,
        "medium",
        100000,
        "stable_high_medium",
    )
    assert "--full-task-transfer" in medium_command


def test_isaac_eval_is_single_environment_deterministic_task_policy():
    preset = RUN_PRESETS["isaac-eval"]
    assert "--runner.mode=test" in preset
    assert "--environment.nr_envs=1" in preset
    assert "--environment.nr_task_envs=1" in preset
    assert "--environment.nr_safety_envs=0" in preset
    assert "--environment.terrain_mode=flat" in preset
    assert "--environment.viewer_follow_robot=true" in preset
    assert "--environment.terrain_num_rows=3" in preset
    assert "--environment.terrain_num_cols=10" in preset
    assert "--algorithm.eval_policy=task" in preset


def test_universal_qsafe_collection_and_finetune_are_explicitly_separated():
    collect = RUN_PRESETS["isaac-collect-qsafe"]
    assert "--algorithm.phase=finetune" in collect
    assert "--algorithm.qsafe.enabled=false" in collect
    assert "--algorithm.qsafe.dataset.enabled=true" in collect
    assert "--algorithm.eval_policy=task" in collect

    finetune = RUN_PRESETS["isaac-finetune"]
    assert "--algorithm.phase=finetune" in finetune
    assert "--algorithm.finetune_actor_warmup_steps=10000" in finetune
    assert "--algorithm.qsafe.enabled=true" in finetune
    assert "--environment.step_height=0.04" in finetune


def test_canonical_mujoco_asset_matches_sdk_bridge_contract():
    mujoco = pytest.importorskip("mujoco")
    scene = Path(DEFAULT_MUJOCO_SCENE)
    model = mujoco.MjModel.from_xml_path(str(scene))

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

    step_scene = scene.with_name("scene_step_4cm.xml")
    step_model = mujoco.MjModel.from_xml_path(str(step_scene))
    step_id = mujoco.mj_name2id(
        step_model, mujoco.mjtObj.mjOBJ_GEOM, "step_4cm"
    )
    assert step_id >= 0
    assert step_model.geom_pos[step_id, 0] - step_model.geom_size[step_id, 0] == pytest.approx(1.0)
    assert step_model.geom_pos[step_id, 2] + step_model.geom_size[step_id, 2] == pytest.approx(0.04)
    for height_cm in (2, 3):
        curriculum_scene = scene.with_name(f"scene_step_{height_cm}cm.xml")
        curriculum_model = mujoco.MjModel.from_xml_path(str(curriculum_scene))
        curriculum_id = mujoco.mj_name2id(
            curriculum_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"step_{height_cm}cm",
        )
        assert curriculum_id >= 0
        assert (
            curriculum_model.geom_pos[curriculum_id, 2]
            + curriculum_model.geom_size[curriculum_id, 2]
        ) == pytest.approx(height_cm / 100.0)


def test_transfer_commands_use_torch_pretrain_sidecars(tmp_path: Path):
    (tmp_path / "policy.model").touch()
    (tmp_path / "qsafe.model").touch()
    flags = _artifact_flags("finetune", str(tmp_path))
    assert flags == [
        f"--algorithm.pretrained_policy_path={tmp_path / 'policy.model'}",
        f"--algorithm.qsafe.checkpoint_path={tmp_path / 'qsafe.model'}",
    ]


def test_standard_sac_transfer_only_requires_policy(tmp_path: Path):
    (tmp_path / "policy.model").touch()
    assert _artifact_flags("finetune-sac", str(tmp_path)) == [
        f"--algorithm.pretrained_policy_path={tmp_path / 'policy.model'}"
    ]
    (tmp_path / "final.model").touch()
    with pytest.raises(ValueError, match="fresh task critics/targets"):
        _artifact_flags("finetune-sac", str(tmp_path), full_task_transfer=True)
    assert _artifact_flags("isaac-finetune-sac", str(tmp_path)) == [
        f"--algorithm.pretrained_policy_path={tmp_path / 'policy.model'}"
    ]
    assert _artifact_flags(
        "isaac-finetune-sac", str(tmp_path), actor_only=True
    ) == [f"--algorithm.pretrained_policy_path={tmp_path / 'policy.model'}"]
    assert _artifact_flags(
        "isaac-finetune-sac", str(tmp_path), full_task_transfer=True
    ) == [
        f"--algorithm.pretrained_policy_path={tmp_path / 'policy.model'}",
        f"--algorithm.pretrained_task_checkpoint_path={tmp_path / 'final.model'}",
    ]

    native = tmp_path / "native"
    native.mkdir()
    (native / "policy.msgpack").touch()
    (native / "final.model").touch()
    with pytest.raises(ValueError, match="fresh task critics/targets"):
        _artifact_flags("finetune-sac", str(native), full_task_transfer=True)


def test_universal_qsafe_transfer_accepts_an_independent_checkpoint(tmp_path: Path):
    actor = tmp_path / "actor"
    actor.mkdir()
    (actor / "policy.model").touch()
    qsafe = tmp_path / "universal_qsafe_v2.model"
    qsafe.touch()
    assert _artifact_flags(
        "isaac-finetune", str(actor), str(qsafe)
    ) == [
        f"--algorithm.pretrained_policy_path={actor / 'policy.model'}",
        f"--algorithm.qsafe.checkpoint_path={qsafe}",
    ]
    (actor / "final.model").touch()
    with pytest.raises(ValueError, match="fresh task critics/targets"):
        _artifact_flags(
            "isaac-finetune",
            str(actor),
            str(qsafe),
            full_task_transfer=True,
        )


def test_frs_diagnostic_collection_profile_is_targeted_and_actor_isolated():
    assert FRS_DIAGNOSTIC_ROLES == {
        "flat_seed0": "train",
        "rough_seed0": "train",
        "step_seed0": "train",
        "flat_seed1": "validation",
        "rough_seed1": "validation",
        "step_seed1": "validation",
        "heldout_flat": "test",
    }
    cells = profile_cells("frs_diagnostic_v1")
    assert len(cells) == 26
    assert len(set(cells)) == len(cells)
    for level in range(3):
        assert {
            noise for terrain, noise in cells
            if terrain == f"stairs_up_l{level}"
        } == {0.0, 0.05, 0.10, 0.20}
    assert ("step_6cm", 0.20) in cells
    assert ("boxes_8cm", 0.20) in cells


def test_qsafe_ablation_uses_one_gate_for_action_masking_and_eq4():
    assert finetune_constraints_enabled("finetune", True) is True
    assert finetune_constraints_enabled("finetune", False) is False
    assert finetune_constraints_enabled("pretrain", True) is False


def test_finetune_actor_waits_for_fresh_task_critic_warmup():
    assert actor_updates_enabled("finetune", 9999, 10000) is False
    assert actor_updates_enabled("finetune", 10000, 10000) is True
    assert actor_updates_enabled("pretrain", 0, 10000) is True
    with pytest.raises(ValueError, match="non-negative"):
        actor_updates_enabled("finetune", 0, -1)


def test_finetune_actor_update_interval_keeps_critic_ahead():
    assert actor_updates_enabled("finetune", 10000, 10000, 10) is True
    assert actor_updates_enabled("finetune", 10001, 10000, 10) is False
    assert actor_updates_enabled("finetune", 10010, 10000, 10) is True
    assert actor_updates_enabled("pretrain", 1, 10000, 10) is True
    with pytest.raises(ValueError, match="at least 1"):
        actor_updates_enabled("finetune", 10000, 10000, 0)


def test_stable_success_excludes_falls_and_final_stuck_state():
    metrics = GaitEvaluationMetrics()
    metrics.update(
        {
            "terrain/success": np.asarray([1.0]),
            "terrain/stuck": np.asarray([0.0]),
        }
    )
    assert metrics.result(0)["stable_success"] is True
    assert metrics.result(1)["stable_success"] is False
    metrics.update({"terrain/stuck": np.asarray([1.0])})
    assert metrics.result(0)["stable_success"] is False


def test_eval_rejects_a_missing_flax_checkpoint(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        _artifact_flags("eval", str(tmp_path))


def test_transfer_rejects_a_combined_checkpoint_without_matching_sidecars(
    tmp_path: Path,
):
    checkpoint = tmp_path / "step_000300032.model"
    checkpoint.touch()
    with pytest.raises(ValueError, match="transfer bundle directory"):
        _artifact_flags("finetune", str(checkpoint))
