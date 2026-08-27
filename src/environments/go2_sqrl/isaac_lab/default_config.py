from ml_collections import config_dict

from ..common.estimation.velocity import configure_velocity_estimator
from ..common.specs import EPISODE_STEPS, PRETRAIN_TARGET_VELOCITY_X


def get_config(environment_name):
    config = config_dict.ConfigDict()
    config.name = environment_name
    config.seed = 0
    # A 300k budget needs at least two 500-step safety horizons so QSafe can be
    # updated before training ends. This raises total parallelism above the old
    # 320-env setup without collapsing the run to a single safety-trajectory
    # batch as a 512-task-env pool would.
    config.nr_envs = 400
    config.nr_task_envs = 300
    config.nr_safety_envs = 100
    config.rollout_mode = "partitioned"
    config.device = "gpu"
    config.target_velocity_x = PRETRAIN_TARGET_VELOCITY_X
    # Robust batched velocity Kalman filter. Isaac exposes one sensor snapshot
    # per 20 ms decimated policy step, while SDK2/MuJoCo consumes all ten 2 ms
    # LowState frames with the same estimator model.
    configure_velocity_estimator(config)
    config.terrain_mode = "rough"
    config.domain_randomization = False
    config.friction = 0.4
    config.episode_steps = EPISODE_STEPS
    config.fall_angle_threshold = 0.8
    config.fall_consecutive_frames = 5
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
