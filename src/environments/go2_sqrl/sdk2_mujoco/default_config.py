from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()
    config.name = environment_name
    config.seed = 0
    config.nr_envs = 1
    config.domain_id = 1
    config.interface = "lo"
    config.policy_frames = 10
    # LowState is published at 500 Hz; one policy/learner iteration consumes
    # ten fresh physical ticks. The Flax learner reports deadline misses while
    # DDS transport itself remains on the host.
    config.policy_period_seconds = 0.02
    config.state_timeout = 1.0
    config.manual_reset_timeout = -1.0
    config.stable_reset_frames = 20
    config.episode_steps = 500
    config.target_velocity_x = 0.4
    config.fall_angle_threshold = 0.8
    config.fall_consecutive_frames = 5
    return config
