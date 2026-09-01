"""Dependency-light Isaac configuration; safe to import before AppLauncher."""

from ml_collections import config_dict

from train.core.base import configure_environment_contract
from train.core.estimation import configure_velocity_estimator


def get_config(environment_name):
    config = config_dict.ConfigDict()
    config.name = environment_name
    config.seed = 0
    config.nr_envs = 320
    config.nr_task_envs = 256
    config.nr_safety_envs = 64
    config.device = "gpu"
    configure_velocity_estimator(config)
    configure_environment_contract(config)
    # Training keeps Isaac Lab's full 10 x 20 curriculum grid.  Playback can
    # override these independently to avoid surrounding a single robot with a
    # needlessly large terrain field.
    config.terrain_num_rows = 10
    config.terrain_num_cols = 20
    # ``MeshRandomGridTerrainCfg`` samples every box symmetrically in
    # [-amplitude, +amplitude].  Express the public setting as the actual
    # worst-case difference between two adjacent boxes instead of exposing
    # Isaac Lab's easy-to-misread amplitude.
    config.boxes_max_adjacent_height_difference = 0.14
    config.playback_terrain_type = "auto"
    config.playback_terrain_level = -1
    config.viewer_follow_robot = False
    config.render = False
    # Deterministic-reset contract checks.  They are evaluated after physical
    # resets when domain randomization is disabled; randomized resets are
    # intentionally exempt from exact-pose validation.
    config.reset_joint_validation_tolerance = 1.0e-3
    config.reset_base_height_validation_tolerance = 2.0e-3
    config.reset_orientation_validation_tolerance = 1.0e-3
    config.reset_foot_surface_validation_tolerance = 3.0e-3
    config.foot_collision_radius = 0.022

    # AppLauncher-compatible defaults consumed by the local runner.
    config.disable_fabric = False
    config.livestream = -1
    config.enable_cameras = False
    config.xr = False
    config.cpu = False
    config.verbose = False
    config.info = False
    # AppLauncher treats these fields as strings when present.  Empty strings
    # ask Isaac Lab to select its standard headless experience and add no Kit
    # arguments, respectively.
    config.experience = ""
    config.rendering_mode = "balanced"
    config.kit_args = ""
    config.anim_recording_enabled = False
    config.anim_recording_start_time = 0.0
    config.anim_recording_stop_time = 10.0
    return config
