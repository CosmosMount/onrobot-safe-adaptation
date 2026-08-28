from ml_collections import config_dict

from ..common.estimation.velocity import configure_velocity_estimator
from ..common.specs import configure_failure_detection


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
    configure_velocity_estimator(config)
    configure_failure_detection(config)
    config.terrain_mode = "rough"
    config.domain_randomization = False
    config.friction = 0.4
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
