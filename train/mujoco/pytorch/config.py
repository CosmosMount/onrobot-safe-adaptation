"""MuJoCo environment defaults."""

from ml_collections import config_dict

from train.core.base import configure_environment_contract
from train.core.estimation import configure_velocity_estimator


def get_config(environment_name):
    config = config_dict.ConfigDict()
    config.name = environment_name
    config.seed = 0
    config.nr_envs = 1
    config.domain_id = 1
    config.interface = "lo"
    configure_velocity_estimator(config)
    configure_environment_contract(config)
    config.state_timeout = 1.0
    config.manual_reset_timeout = -1.0
    config.auto_reset_on_start = True
    config.auto_reset_after_fall = True
    config.fall_auto_reset_delay_seconds = 1.0
    config.auto_reset_timeout_seconds = 10.0
    # X11 synthetic key events can occasionally be dropped by the visible
    # simulator window. Re-arm tick rollback detection and retry the shortcut
    # instead of terminating a long online run on one missed event.
    config.auto_reset_attempts = 3
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
    config.standup_hold_seconds = 0.0
    config.reset_sync_timeout_seconds = 3.0
    config.reset_kp = 60.0
    # Kd=5 drives the torque-clipped SDK bridge into a persistent limit cycle
    # on this 2 ms MuJoCo model.  Kd=1 settles the same stand-up trajectory
    # before policy takeover while keeping enough damping for foot impacts.
    config.reset_kd = 1.0
    # Stand-up completes before the first policy step, so no second blend is
    # applied to the policy action.
    config.policy_blend_seconds = 0.0
    # Match Isaac's deterministic flat-ground reset validation.  The velocity
    # and orientation checks are MuJoCo-only settling guards; joint pose, base
    # height, and foot-surface tolerances are shared contract values.
    config.reset_joint_tolerance = 1.0e-3
    config.reset_max_joint_velocity = 0.25
    config.reset_angle_tolerance = 1.0e-3
    config.reset_base_height_tolerance = 2.0e-3
    config.reset_foot_surface_tolerance = 3.0e-3
    config.foot_collision_radius = 0.022
    return config

