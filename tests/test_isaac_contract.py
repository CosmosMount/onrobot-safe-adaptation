import importlib

import numpy as np
import pytest


def test_isaac_environment_registers_without_importing_isaaclab():
    module = importlib.import_module("src.environments.go2_sqrl.isaac_lab")
    assert module.GO2_SQRL_ISAAC_LAB == "go2_sqrl.isaac_lab"


def test_isaac_launcher_string_defaults_are_not_none():
    from src.environments.go2_sqrl.isaac_lab.default_config import get_config

    config = get_config("go2_sqrl.isaac_lab")
    assert config.experience == ""
    assert config.kit_args == ""
    assert isinstance(config.rendering_mode, str)
    assert config.terrain_mode == "rough"
    assert config.terrain_num_rows == 10
    assert config.terrain_num_cols == 20
    assert config.boxes_max_adjacent_height_difference == 0.14
    assert config.playback_terrain_type == "auto"
    assert config.playback_terrain_level == -1
    assert config.viewer_follow_robot is False


def test_isaac_joint_gather_uses_sdk_order():
    torch = pytest.importorskip("torch")
    from src.environments.go2_sqrl.common.specs import JOINT_NAMES
    from src.environments.go2_sqrl.isaac_lab.mdp import sdk_joint_indices

    source = [f"{name}_joint" for name in reversed(JOINT_NAMES)]
    indices = sdk_joint_indices(source)
    torch.testing.assert_close(indices, torch.arange(11, -1, -1))


def test_isaac_action_term_requires_sdk_order_and_shared_offset():
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from src.environments.go2_sqrl.common.specs import (
        ACTION_SPEC,
        DEFAULT_JOINT_POSITION,
        JOINT_NAMES,
    )
    from src.environments.go2_sqrl.isaac_lab.mdp import (
        validate_action_term_contract,
    )

    valid = SimpleNamespace(
        _joint_names=[f"{name}_joint" for name in JOINT_NAMES],
        _offset=torch.tensor(DEFAULT_JOINT_POSITION)[None, :],
        _scale=ACTION_SPEC.scale,
    )
    validate_action_term_contract(valid)

    wrong_order = SimpleNamespace(
        _joint_names=[
            "FL_hip_joint",
            "FR_hip_joint",
            "RL_hip_joint",
            "RR_hip_joint",
            "FL_thigh_joint",
            "FR_thigh_joint",
            "RL_thigh_joint",
            "RR_thigh_joint",
            "FL_calf_joint",
            "FR_calf_joint",
            "RL_calf_joint",
            "RR_calf_joint",
        ],
        _offset=valid._offset,
        _scale=valid._scale,
    )
    with pytest.raises(RuntimeError, match="joint order"):
        validate_action_term_contract(wrong_order)

    wrong_offset = SimpleNamespace(
        _joint_names=valid._joint_names,
        _offset=torch.zeros_like(valid._offset),
        _scale=valid._scale,
    )
    with pytest.raises(RuntimeError, match="action offset"):
        validate_action_term_contract(wrong_offset)


def test_domain_randomization_report_shows_effective_ranges():
    from src.environments.go2_sqrl.isaac_lab.randomization_cfg import (
        format_domain_randomization_report,
    )

    enabled = format_domain_randomization_report(enabled=True, friction=0.4)
    assert "Domain Randomization: enabled" in enabled
    assert "static/dynamic friction coefficient" in enabled
    assert "uniform [0.34, 0.46]" in enabled
    assert "base mass delta (kg)" in enabled
    assert "uniform [-0.5, 0.5]" in enabled
    assert "joint position noise (rad)" in enabled
    assert "uniform [-0.005, 0.005]" in enabled
    assert "IMU acceleration noise (m/s^2)" in enabled
    assert "leg mass scale" in enabled
    assert "uniform [0.9, 1.1]" in enabled

    disabled = format_domain_randomization_report(enabled=False, friction=0.4)
    assert "Domain Randomization: disabled" in disabled
    assert "fixed 0.4" in disabled
    assert "uniform" not in disabled


def test_isaac_manager_output_labels_internal_discarded_values():
    from src.environments.go2_sqrl.isaac_lab.env import (
        relabel_isaac_backend_manager_output,
    )

    raw = """[INFO] Command Manager: commands
[INFO] Observation Manager: observations
Active Observation Terms in Group: 'policy' (shape: (48,))
[INFO] Reward Manager: rewards
"""
    reported = relabel_isaac_backend_manager_output(raw)
    assert "internal; not the Go2 reward target" in reported
    assert "internal tensor discarded by the Go2 RL-X adapter" in reported
    assert "not model input" in reported
    assert "disabled; Go2 adapter computes the effective reward" in reported


def test_domain_randomization_can_be_enabled_after_fixed_baseline(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    from src.environments.go2_sqrl.isaac_lab.randomization_cfg import (
        configure_existing_events,
    )

    class FakeEventTerm:
        def __init__(self, *, func, mode, params):
            self.func = func
            self.mode = mode
            self.params = params

    class FakeSceneEntityCfg:
        def __init__(self, name, **kwargs):
            self.name = name
            vars(self).update(kwargs)

    mdp = SimpleNamespace(
        randomize_rigid_body_mass=object(),
        randomize_rigid_body_com=object(),
        randomize_actuator_gains=object(),
    )
    isaaclab = ModuleType("isaaclab")
    envs_module = ModuleType("isaaclab.envs")
    envs_module.mdp = mdp
    managers_module = ModuleType("isaaclab.managers")
    managers_module.EventTermCfg = FakeEventTerm
    managers_module.SceneEntityCfg = FakeSceneEntityCfg
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.envs", envs_module)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers_module)

    events = SimpleNamespace(
        push_robot=object(),
        base_external_force_torque=object(),
        physics_material=SimpleNamespace(params={}),
        add_base_mass=object(),
        base_com=object(),
        actuator_gains=object(),
        reset_robot_joints=SimpleNamespace(params={}),
        reset_base=SimpleNamespace(params={}),
    )

    configure_existing_events(events, enabled=False, friction=0.4)
    assert events.add_base_mass is None
    assert events.base_com is None
    assert events.leg_mass is None

    configure_existing_events(events, enabled=True, friction=0.4)
    assert events.add_base_mass.params["mass_distribution_params"] == (-0.5, 0.5)
    assert events.add_base_mass.params["asset_cfg"].body_names == "base"
    assert events.base_com.params["asset_cfg"].body_names == "base"
    assert events.leg_mass.params["mass_distribution_params"] == (0.9, 1.1)
    assert events.leg_mass.params["asset_cfg"].body_names == ".*_(hip|thigh|calf)"
    assert events.actuator_gains.mode == "reset"


def test_flat_terrain_mode_disables_generator_and_curriculum():
    from types import SimpleNamespace

    from src.environments.go2_sqrl.isaac_lab.terrain_cfg import (
        configure_terrain_mode,
    )

    generator = object()
    terrain = SimpleNamespace(
        terrain_type="generator",
        terrain_generator=generator,
    )
    curriculum = SimpleNamespace(terrain_levels=object())

    configure_terrain_mode(terrain, curriculum, "flat")

    assert terrain.terrain_type == "plane"
    assert terrain.terrain_generator is None
    assert curriculum.terrain_levels is None


def test_rough_terrain_mode_preserves_generator_and_rejects_unknown_modes():
    from types import SimpleNamespace

    from src.environments.go2_sqrl.isaac_lab.terrain_cfg import (
        configure_terrain_mode,
    )

    generator = object()
    terrain = SimpleNamespace(
        terrain_type="generator",
        terrain_generator=generator,
    )
    curriculum_term = object()
    curriculum = SimpleNamespace(terrain_levels=curriculum_term)

    configure_terrain_mode(terrain, curriculum, "rough")
    assert terrain.terrain_type == "generator"
    assert terrain.terrain_generator is generator
    assert curriculum.terrain_levels is curriculum_term

    with pytest.raises(ValueError, match="terrain_mode"):
        configure_terrain_mode(terrain, curriculum, "stairs-only")


def test_rough_terrain_distribution_and_column_order_are_explicit():
    from types import SimpleNamespace

    from src.environments.go2_sqrl.isaac_lab.terrain_cfg import configure_terrain

    class TerrainCfg(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    sub_terrains = {
        "pyramid_stairs": TerrainCfg(proportion=1.0),
        "pyramid_stairs_inv": TerrainCfg(proportion=1.0),
        "hf_pyramid_slope": TerrainCfg(proportion=1.0),
        "hf_pyramid_slope_inv": TerrainCfg(proportion=1.0),
        "boxes": TerrainCfg(proportion=0.0),
        "random_rough": TerrainCfg(proportion=0.0),
    }
    terrain_generator = SimpleNamespace(sub_terrains=sub_terrains)
    terrain_gen = SimpleNamespace(
        MeshPlaneTerrainCfg=TerrainCfg,
        MeshRandomGridTerrainCfg=TerrainCfg,
        HfRandomUniformTerrainCfg=TerrainCfg,
        HfPyramidSlopedTerrainCfg=TerrainCfg,
        HfWaveTerrainCfg=TerrainCfg,
        HfPyramidStairsTerrainCfg=TerrainCfg,
        HfInvertedPyramidStairsTerrainCfg=TerrainCfg,
    )

    configure_terrain(terrain_generator, terrain_gen)
    sub_terrains = terrain_generator.sub_terrains

    assert list(sub_terrains) == [
        "flat",
        "boxes",
        "random_rough",
        "slopes",
        "small_bumps",
        "small_stairs",
        "small_stairs_up",
    ]
    assert sub_terrains["boxes"].grid_height_range == (0.0, 0.07)
    assert sub_terrains["random_rough"].noise_range[1] <= 0.07
    assert sub_terrains["small_bumps"].amplitude_range[1] <= 0.07
    assert sub_terrains["small_stairs"].step_height_range == (0.02, 0.07)
    assert sub_terrains["small_stairs_up"].step_height_range == (0.02, 0.07)
    assert sub_terrains["small_stairs_up"].inverted is True
    assert sum(cfg.proportion for cfg in sub_terrains.values()) == pytest.approx(1.0)


def test_playback_terrain_type_selects_a_representative_mixed_map_column():
    from src.environments.go2_sqrl.isaac_lab.terrain_cfg import (
        playback_terrain_column,
    )

    assert playback_terrain_column("auto", 20) is None
    assert playback_terrain_column("flat", 20) == 1
    assert playback_terrain_column("boxes", 20) == 3
    assert playback_terrain_column("random_rough", 20) == 7
    assert playback_terrain_column("slopes", 20) == 10
    assert playback_terrain_column("small_bumps", 20) == 13
    assert playback_terrain_column("small_stairs", 20) == 15
    assert playback_terrain_column("small_stairs_up", 20) == 18
    with pytest.raises(ValueError, match="playback_terrain_type"):
        playback_terrain_column("unknown", 20)


def test_boxes_height_setting_means_true_worst_case_adjacent_difference():
    from src.environments.go2_sqrl.isaac_lab.terrain_cfg import boxes_height_range

    assert boxes_height_range(0.02) == (0.0, 0.01)
    assert boxes_height_range(0.14) == (0.0, 0.07)
    with pytest.raises(ValueError, match="between 0 and 0.30"):
        boxes_height_range(0.31)
