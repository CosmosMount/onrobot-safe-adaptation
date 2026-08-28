"""Transfer randomization values applied to the Isaac event manager."""


TRANSFER_RANDOMIZATION = {
    "friction_scale": (0.85, 1.15),
    "base_mass_delta": (-0.5, 0.5),
    "com_offset": (-0.01, 0.01),
    "kp_scale": (0.95, 1.05),
    "kd_scale": (0.95, 1.05),
    "joint_position_scale": (0.98, 1.02),
    "initial_tilt_rad": (-0.035, 0.035),
    "initial_velocity": (-0.05, 0.05),
}


def _format_range(value):
    return f"[{value[0]:g}, {value[1]:g}]"


def domain_randomization_rows(*, enabled: bool, friction: float):
    """Return the effective transfer-randomization settings for logging."""

    friction_range = (
        friction * TRANSFER_RANDOMIZATION["friction_scale"][0],
        friction * TRANSFER_RANDOMIZATION["friction_scale"][1],
    )

    def uniform(name):
        if not enabled:
            return "disabled"
        return f"uniform {_format_range(TRANSFER_RANDOMIZATION[name])}"

    return (
        (
            "policy",
            "46D policy observation corruption",
            "disabled (common adapter output)",
        ),
        (
            "startup",
            "static/dynamic friction coefficient",
            (
                f"uniform {_format_range(friction_range)}"
                if enabled
                else f"fixed {friction:g}"
            ),
        ),
        ("startup", "base mass delta (kg)", uniform("base_mass_delta")),
        ("startup", "base COM offset xyz (m)", uniform("com_offset")),
        ("reset", "actuator stiffness scale", uniform("kp_scale")),
        ("reset", "actuator damping scale", uniform("kd_scale")),
        ("reset", "joint position scale", uniform("joint_position_scale")),
        ("reset", "base roll/pitch (rad)", uniform("initial_tilt_rad")),
        (
            "reset",
            "base linear velocity xyz (m/s)",
            uniform("initial_velocity"),
        ),
        (
            "reset",
            "base angular velocity xyz (rad/s)",
            uniform("initial_velocity"),
        ),
    )


def format_domain_randomization_report(*, enabled: bool, friction: float):
    """Format effective settings in the same compact style as manager output."""

    headers = ("Mode", "Parameter", "Applied value")
    rows = domain_randomization_rows(enabled=enabled, friction=friction)
    widths = tuple(
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    )
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def format_row(row):
        cells = (f" {value:<{width}} " for value, width in zip(row, widths))
        return "|" + "|".join(cells) + "|"

    state = "enabled" if enabled else "disabled"
    table = [separator, format_row(headers), separator]
    table.extend(format_row(row) for row in rows)
    table.append(separator)
    return "\n".join((f"[INFO] Domain Randomization: {state}", *table))


def configure_existing_events(events, *, enabled: bool, friction: float):
    """Configure a fixed paper baseline or the optional transfer randomization."""

    events.push_robot = None
    # The inherited force/torque ranges are all zero.  Keeping the term would
    # still launch a Warp wrench-composer kernel on every reset.
    events.base_external_force_torque = None
    friction_range = (
        friction * TRANSFER_RANDOMIZATION["friction_scale"][0],
        friction * TRANSFER_RANDOMIZATION["friction_scale"][1],
    )
    events.physics_material.params["static_friction_range"] = (
        friction_range if enabled else (friction, friction)
    )
    events.physics_material.params["dynamic_friction_range"] = (
        friction_range if enabled else (friction, friction)
    )
    if enabled:
        from isaaclab.envs import mdp
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg

        # Recreate startup terms instead of mutating the inherited instances.
        # The fixed baseline removes them by setting the fields to ``None``;
        # make_env_cfg may subsequently enable randomization on the same config.
        events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": TRANSFER_RANDOMIZATION[
                    "base_mass_delta"
                ],
                "operation": "add",
            },
        )
        events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {
                    axis: TRANSFER_RANDOMIZATION["com_offset"]
                    for axis in ("x", "y", "z")
                },
            },
        )
        events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": TRANSFER_RANDOMIZATION[
                    "kp_scale"
                ],
                "damping_distribution_params": TRANSFER_RANDOMIZATION[
                    "kd_scale"
                ],
                "operation": "scale",
                "distribution": "uniform",
            },
        )
    else:
        events.add_base_mass = None
        events.base_com = None
        events.actuator_gains = None
    events.reset_robot_joints.params["position_range"] = (
        TRANSFER_RANDOMIZATION["joint_position_scale"]
        if enabled
        else (1.0, 1.0)
    )
    tilt = (
        TRANSFER_RANDOMIZATION["initial_tilt_rad"]
        if enabled
        else (0.0, 0.0)
    )
    velocity = (
        TRANSFER_RANDOMIZATION["initial_velocity"]
        if enabled
        else (0.0, 0.0)
    )
    events.reset_base.params = {
        "pose_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "roll": tilt,
            "pitch": tilt,
            "yaw": (0.0, 0.0),
        },
        "velocity_range": {
            key: velocity
            for key in ("x", "y", "z", "roll", "pitch", "yaw")
        },
    }
