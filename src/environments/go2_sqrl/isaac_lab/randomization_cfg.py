"""Transfer randomization values applied to the Isaac event manager."""


TRANSFER_RANDOMIZATION = {
    "friction_range": (0.6, 1.4),
    "base_mass_delta": (-1.0, 3.0),
    "com_offset": (-0.02, 0.02),
    "motor_strength_scale": (0.9, 1.1),
    "kp_scale": (0.9, 1.1),
    "kd_scale": (0.9, 1.1),
    "joint_zero_offset": (-0.02, 0.02),
    "action_delay_steps": (0, 1),
}


def configure_existing_events(events):
    """Apply fields available in the upstream locomotion task defensively."""

    events.push_robot = None
    # The inherited force/torque ranges are all zero.  Keeping the term would
    # still launch a Warp wrench-composer kernel on every reset.
    events.base_external_force_torque = None
    events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
    events.add_base_mass.params["asset_cfg"].body_names = "base"
    events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    events.reset_base.params = {
        "pose_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "yaw": (-3.14, 3.14),
        },
        "velocity_range": {
            key: (0.0, 0.0)
            for key in ("x", "y", "z", "roll", "pitch", "yaw")
        },
    }
