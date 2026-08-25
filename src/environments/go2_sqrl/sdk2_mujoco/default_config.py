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
    # Runtime policy PD. Defaults reproduce the Isaac/checkpoint contract;
    # command-line overrides are MuJoCo-only sensitivity experiments.
    config.policy_kp = 25.0
    config.policy_kd = 0.5
    config.state_timeout = 1.0
    config.manual_reset_timeout = -1.0
    config.auto_reset_on_start = True
    config.auto_reset_after_fall = True
    config.fall_auto_reset_delay_seconds = 1.0
    config.auto_reset_timeout_seconds = 5.0
    config.mujoco_window_title = "MuJoCo"
    # Two-stage linear stand-up copied from the proven Go2 controller in the
    # reference repository: current pose -> folded/crouched keyframe -> home.
    config.standup_pose_1 = [
        0.0, 1.36, -2.65,
        0.0, 1.36, -2.65,
        -0.2, 1.36, -2.65,
        0.2, 1.36, -2.65,
    ]
    config.standup_phase_1_seconds = 1.0
    config.standup_phase_2_seconds = 1.0
    config.standup_hold_seconds = 2.0
    config.reset_sync_timeout_seconds = 3.0
    config.reset_kp = 60.0
    # Kd=5 drives the torque-clipped SDK bridge into a persistent limit cycle
    # on this 2 ms MuJoCo model.  Kd=1 settles the same stand-up trajectory
    # before policy takeover while keeping enough damping for foot impacts.
    config.reset_kd = 1.0
    # Stand-up completes before the first policy step, so no second blend is
    # applied to the policy action.
    config.policy_blend_seconds = 0.0
    config.reset_joint_tolerance = 0.20
    # Position alone can look ready while the legs are still oscillating.
    config.reset_max_joint_velocity = 0.5
    config.reset_min_base_height = 0.20
    config.episode_steps = 500
    # The policy has no command input, so evaluation must use the fixed velocity
    # objective on which it was pre-trained.
    config.target_velocity_x = 0.5
    config.fall_angle_threshold = 0.8
    config.fall_consecutive_frames = 5

    config.kp = 25.0
    config.kd = 0.5

    return config
