from ml_collections import config_dict

from ..common.estimation.velocity import configure_velocity_estimator
from ..common.specs import configure_failure_detection
from ..common.specs import DEFAULT_ACTION_PROFILE


def get_config(environment_name):
    config = config_dict.ConfigDict()
    config.name = environment_name
    config.seed = 0
    config.nr_envs = 320
    config.nr_task_envs = 256
    config.nr_safety_envs = 64
    config.rollout_mode = "partitioned"
    config.device = "gpu"
    config.target_velocity_x = 0.5
    config.action_profile = DEFAULT_ACTION_PROFILE
    configure_velocity_estimator(config)
    configure_failure_detection(config)
    config.terrain_mode = "rough"
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
    # Deterministic benchmark profiles use one terrain family and a fixed
    # height.  ``mixed`` preserves the seven-family rough training map.
    config.terrain_profile = "mixed"
    config.step_height = 0.04
    config.step_success_distance = 2.0
    config.playback_terrain_type = "auto"
    config.playback_terrain_level = -1
    config.viewer_follow_robot = False
    config.domain_randomization = False
    config.friction = 0.4
    config.foot_clearance_target = 0.07
    config.clearance_reward_mode = "swing_weighted"
    config.phase_reference_frequency = 2.0
    config.phase_reward_scale = 0.0
    config.foot_clearance_upper_target = 0.0
    config.foot_clearance_overshoot_scale = 0.0
    config.phase_velocity_gate_start = 0.0
    config.phase_velocity_gate_full = 0.0
    config.stable_progress_start = 1.0
    config.stable_progress_min_base_clearance = 0.22
    config.stable_progress_scale = 0.0
    config.terminal_failure_penalty = 0.0
    config.high_clearance_stage = "easy"
    config.episode_steps = 500
    config.render = False

    # AppLauncher-compatible defaults consumed by RL-X before environment creation.
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
