"""Isaac Lab configuration and tensor adapters, imported after AppLauncher."""
from __future__ import annotations

from train.config import PROJECT_ROOT
from train.common.base import (
    ACTION_SPEC, DEFAULT_BASE_HEIGHT, DEFAULT_JOINT_POSITION, JOINT_NAMES,
)

import numpy as np
import torch

# Torch adapters for the shared contract.
SDK_JOINT_NAMES = tuple(f"{name}_joint" for name in JOINT_NAMES)


def sdk_joint_indices(source_names, device=None):
    normalized = [name.removesuffix("_joint") for name in source_names]
    missing = [name for name in JOINT_NAMES if name not in normalized]
    if missing:
        raise ValueError(f"Missing Go2 joints in Isaac articulation: {missing}")
    return torch.tensor(
        [normalized.index(name) for name in JOINT_NAMES],
        dtype=torch.long,
        device=device,
    )


def validate_action_term_contract(action_term) -> None:
    """Fail fast unless Isaac executes the exact shared SDK action mapping."""

    actual_names = tuple(action_term._joint_names)
    if actual_names != SDK_JOINT_NAMES:
        raise RuntimeError(
            "Isaac action joint order does not match the SDK contract: "
            f"expected {SDK_JOINT_NAMES}, got {actual_names}"
        )

    offset = action_term._offset
    if torch.is_tensor(offset):
        offset = offset[0].detach().cpu().numpy()
    else:
        offset = np.full(len(SDK_JOINT_NAMES), float(offset), dtype=np.float32)
    expected_offset = np.asarray(ACTION_SPEC.default_position, dtype=np.float32)
    if not np.allclose(offset, expected_offset, atol=1e-6, rtol=0.0):
        raise RuntimeError(
            "Isaac action offset does not match DEFAULT_JOINT_POSITION: "
            f"expected {expected_offset.tolist()}, got {np.asarray(offset).tolist()}"
        )

    scale = action_term._scale
    if torch.is_tensor(scale):
        scale = scale.detach().cpu().numpy()
    if not np.allclose(scale, ACTION_SPEC.scale, atol=1e-7, rtol=0.0):
        raise RuntimeError(
            "Isaac action scale does not match the shared action contract: "
            f"expected {ACTION_SPEC.scale}, got {scale}"
        )


def continuous_quaternion(quaternion, previous=None):
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    if previous is None:
        sign = torch.where(quaternion[..., :1] < 0, -1.0, 1.0)
    else:
        sign = torch.where(
            torch.sum(quaternion * previous, dim=-1, keepdim=True) < 0,
            -1.0,
            1.0,
        )
    return quaternion * sign


def build_observation_tensor(
    joint_q,
    joint_dq,
    imu_gyro,
    estimated_body_velocity,
    imu_quat,
    previous_q_target,
    previous_quaternion=None,
):
    quaternion = continuous_quaternion(imu_quat, previous_quaternion)
    observation = torch.cat(
        (
            joint_q,
            joint_dq,
            imu_gyro,
            estimated_body_velocity,
            quaternion,
            previous_q_target,
        ),
        dim=-1,
    )
    if observation.shape[-1] != 46:
        raise ValueError(f"Isaac observation must have 46 values, got {observation.shape}")
    return observation, quaternion


def default_joint_target(batch_size, device):
    return torch.as_tensor(
        DEFAULT_JOINT_POSITION, dtype=torch.float32, device=device
    ).expand(batch_size, -1).clone()

# Midpoints of the cumulative column intervals configured below.  Selecting a
# representative column keeps the full mixed terrain visible while placing a
# single playback robot on the requested terrain family.
# Procedural terrain and transfer randomization.
PLAYBACK_TERRAIN_COLUMN_FRACTIONS = {
    "flat": 0.05,
    "boxes": 0.175,
    "random_rough": 0.35,
    "slopes": 0.525,
    "small_bumps": 0.65,
    "small_stairs": 0.75,
    "small_stairs_up": 0.90,
}


def boxes_height_range(max_adjacent_height_difference):
    """Convert a true adjacent-cell limit to Isaac's symmetric amplitude."""

    difference = float(max_adjacent_height_difference)
    if not 0.0 <= difference <= 0.30:
        raise ValueError(
            "boxes_max_adjacent_height_difference must be between 0 and 0.30 m"
        )
    return (0.0, 0.5 * difference)


def playback_terrain_column(terrain_type, num_cols):
    """Return a representative generator column, or ``None`` for auto."""

    terrain_type = str(terrain_type).lower()
    if terrain_type == "auto":
        return None
    if terrain_type not in PLAYBACK_TERRAIN_COLUMN_FRACTIONS:
        choices = ", ".join(("auto", *PLAYBACK_TERRAIN_COLUMN_FRACTIONS))
        raise ValueError(
            f"Unsupported playback_terrain_type {terrain_type!r}; "
            f"expected one of: {choices}."
        )
    num_cols = int(num_cols)
    if num_cols < 1:
        raise ValueError("terrain_num_cols must be positive")
    fraction = PLAYBACK_TERRAIN_COLUMN_FRACTIONS[terrain_type]
    return min(int(fraction * num_cols), num_cols - 1)


def configure_terrain_mode(terrain, curriculum, mode):
    """Select the generated rough map or Isaac Lab's infinite ground plane."""

    mode = str(mode).lower()
    if mode == "rough":
        return
    if mode != "flat":
        raise ValueError(
            f"Unsupported terrain_mode {mode!r}; expected 'rough' or 'flat'."
        )
    terrain.terrain_type = "plane"
    terrain.terrain_generator = None
    curriculum.terrain_levels = None


def configure_terrain(terrain_generator, terrain_gen):
    # Rebuild the mapping instead of mutating Isaac Lab's inherited mapping.
    # TerrainGenerator assigns families to columns in dictionary insertion
    # order, and playback_terrain_column relies on this exact order.
    terrain_generator.sub_terrains = {
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_height_range=(0.0, 0.07),
            grid_width=0.45,
            platform_width=2.0,
            holes=False,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "slopes": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.05, 0.25),
            platform_width=2.0,
            border_width=0.25,
        ),
        "small_bumps": terrain_gen.HfWaveTerrainCfg(
            proportion=0.10,
            amplitude_range=(0.005, 0.015),
            num_waves=8,
            border_width=0.25,
        ),
        "small_stairs": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.02, 0.07),
            step_width=0.35,
            platform_width=2.0,
            border_width=0.25,
        ),
        "small_stairs_up": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.02, 0.07),
            step_width=0.35,
            platform_width=2.0,
            border_width=0.25,
            inverted=True,
        ),
    }

TRANSFER_RANDOMIZATION = {
    "friction_scale": (0.85, 1.15),
    "base_mass_delta": (-0.5, 0.5),
    "leg_mass_scale": (0.9, 1.1),
    "com_offset": (-0.01, 0.01),
    "kp_scale": (0.95, 1.05),
    "kd_scale": (0.95, 1.05),
    "joint_position_scale": (0.98, 1.02),
    "initial_tilt_rad": (-0.035, 0.035),
    "initial_velocity": (-0.05, 0.05),
    # Modest sensor errors cover the residual Isaac/MuJoCo estimator gap while
    # preserving the shared 46D tensor meaning.
    "joint_position_noise": (-0.005, 0.005),
    "joint_velocity_noise": (-0.05, 0.05),
    "gyro_noise": (-0.01, 0.01),
    "accelerometer_noise": (-0.05, 0.05),
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
            "joint position noise (rad)",
            uniform("joint_position_noise"),
        ),
        ("policy", "joint velocity noise (rad/s)", uniform("joint_velocity_noise")),
        ("policy", "IMU gyro noise (rad/s)", uniform("gyro_noise")),
        ("policy", "IMU acceleration noise (m/s^2)", uniform("accelerometer_noise")),
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
        ("startup", "leg mass scale", uniform("leg_mass_scale")),
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
        events.leg_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=".*_(hip|thigh|calf)"
                ),
                "mass_distribution_params": TRANSFER_RANDOMIZATION[
                    "leg_mass_scale"
                ],
                "operation": "scale",
                "distribution": "uniform",
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
        events.leg_mass = None
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

from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.sensors import Imu, ImuCfg, patterns
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)


# Isaac scene configuration; imported only after AppLauncher starts.
GO2_USD_PATH = PROJECT_ROOT / "assets" / "robots" / "go2" / "usd" / "go2.usd"


class PolicyIntervalImu(Imu):
    """Use the configured capture interval for the IMU velocity derivative."""

    def update(self, dt: float, force_recompute: bool = False):
        self._dt = self.cfg.update_period or dt
        SensorBase.update(self, dt, force_recompute)


@configclass
class Go2SQRLIsaacEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UNITREE_GO2_CFG.spawn.replace(usd_path=str(GO2_USD_PATH)),
        )
        self.scene.robot.init_state.pos = (0.0, 0.0, DEFAULT_BASE_HEIGHT)
        self.scene.robot.init_state.joint_pos = {
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.9,
            ".*_calf_joint": -1.8,
        }
        self.scene.robot.init_state.joint_vel = {".*": 0.0}
        actuator = self.scene.robot.actuators["base_legs"]
        actuator.effort_limit = ACTION_SPEC.effort_limit
        actuator.saturation_effort = ACTION_SPEC.effort_limit
        actuator.effort_limit_sim = ACTION_SPEC.effort_limit
        # The canonical MuJoCo bridge applies a torque-clipped PD controller,
        # not a speed-dependent motor envelope.  Keep the solver velocity limit
        # effectively inactive so both backends implement the same controller.
        actuator.velocity_limit = ACTION_SPEC.velocity_limit
        actuator.velocity_limit_sim = ACTION_SPEC.velocity_limit
        actuator.stiffness = ACTION_SPEC.kp
        actuator.damping = ACTION_SPEC.kd
        actuator.armature = ACTION_SPEC.armature
        actuator.friction = ACTION_SPEC.joint_friction
        actuator.dynamic_friction = ACTION_SPEC.joint_friction
        actuator.viscous_friction = ACTION_SPEC.joint_damping
        self.scene.imu = ImuCfg(
            class_type=PolicyIntervalImu,
            prim_path="{ENV_REGEX_NS}/Robot/base",
            update_period=ACTION_SPEC.control_dt,
            history_length=0,
            debug_vis=False,
        )
        # Terrain height is training-only truth for clearance reward/failure.
        # The scan remains training-only truth and never enters the 46D policy
        # input.  Its footprint covers all four feet so reward/failure heights
        # can be referenced to the local terrain instead of world z.
        self.scene.height_scanner.update_period = ACTION_SPEC.control_dt
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(
            resolution=0.1,
            size=(0.8, 0.6),
        )
        self.observations.policy.height_scan = None
        self.decimation = 10
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0
        self.commands.base_velocity.debug_vis = False
        configure_terrain(self.scene.terrain.terrain_generator, terrain_gen)
        configure_existing_events(
            self.events,
            enabled=False,
            friction=0.4,
        )
        # ManagerBasedRLEnv requires a reward manager, but its output is not
        # part of the Go2 SQRL contract.  The adapter computes the shared
        # FlashSAC walk-easy reward itself, so disable every inherited term.
        for reward_name in (
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "lin_vel_z_l2",
            "ang_vel_xy_l2",
            "dof_torques_l2",
            "dof_acc_l2",
            "action_rate_l2",
            "feet_air_time",
            "undesired_contacts",
            "flat_orientation_l2",
            "dof_pos_limits",
        ):
            setattr(self.rewards, reward_name, None)
        # The policy/action contract is Unitree SDK order.  Isaac's articulation
        # order is grouped by joint type (all hips, then thighs, then calves),
        # so relying on the inherited ``[".*"]`` order silently permutes policy
        # outputs.  The inherited action term also offsets actions by the
        # articulation's default pose, which differs from the shared SDK home
        # pose.  Make both parts of the physical target mapping explicit:
        #
        #   q_target = DEFAULT_JOINT_POSITION + ACTION_SPEC.scale * action
        self.actions.joint_pos.joint_names = [
            f"{joint_name}_joint" for joint_name in JOINT_NAMES
        ]
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.use_default_offset = False
        self.actions.joint_pos.scale = {
            f"{joint_name}_joint": float(scale)
            for joint_name, scale in zip(JOINT_NAMES, ACTION_SPEC.scale)
        }
        self.actions.joint_pos.offset = {
            f"{joint_name}_joint": float(default_position)
            for joint_name, default_position in zip(
                JOINT_NAMES, ACTION_SPEC.default_position
            )
        }
        # Failure is detected from the IMU by the Go2 adapter.  Neither the
        # inherited contact sensor nor its base-contact termination is needed.
        self.terminations.base_contact = None
        self.scene.contact_forces = None


def make_env_cfg(config, num_envs=None):
    cfg = Go2SQRLIsaacEnvCfg()
    configure_terrain_mode(
        cfg.scene.terrain,
        cfg.curriculum,
        config.environment.terrain_mode,
    )
    # A requested playback row is a frozen evaluation condition.  Otherwise
    # the curriculum manager updates the row again during the first reset.
    if int(config.environment.playback_terrain_level) >= 0:
        cfg.curriculum.terrain_levels = None
    terrain_generator = cfg.scene.terrain.terrain_generator
    if terrain_generator is not None:
        terrain_rows = int(config.environment.terrain_num_rows)
        terrain_cols = int(config.environment.terrain_num_cols)
        if terrain_rows < 1 or terrain_cols < 1:
            raise ValueError("terrain_num_rows and terrain_num_cols must be positive")
        terrain_generator.num_rows = terrain_rows
        terrain_generator.num_cols = terrain_cols
        # Isaac's random-grid implementation samples cell tops uniformly from
        # [-amplitude, +amplitude], hence adjacent cells can differ by twice
        # the configured amplitude.  At the highest curriculum row this makes
        # the public value below the true worst-case adjacent height change.
        terrain_generator.sub_terrains["boxes"].grid_height_range = boxes_height_range(
            config.environment.boxes_max_adjacent_height_difference
        )

    if bool(config.environment.viewer_follow_robot):
        # Anchor the viewport to env 0's robot.  Besides centering the robot at
        # startup, asset_root makes the camera follow it throughout playback.
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.env_index = 0
        cfg.viewer.asset_name = "robot"
        cfg.viewer.eye = (2.6, 2.6, 1.6)
        cfg.viewer.lookat = (0.0, 0.0, 0.15)
    randomization = bool(config.environment.domain_randomization)
    # The manager observation is an internal Isaac tensor discarded by the
    # Go2 adapter.  Corrupting it wastes work and does not perturb the actual
    # common 46D policy observation.
    cfg.observations.policy.enable_corruption = False
    configure_existing_events(
        cfg.events,
        enabled=randomization,
        friction=float(config.environment.friction),
    )
    cfg.scene.num_envs = int(num_envs or config.environment.nr_envs)
    cfg.seed = int(config.environment.seed)
    configured_device = str(config.environment.device)
    cfg.sim.device = "cuda:0" if configured_device == "gpu" else configured_device
    return cfg

def relabel_isaac_backend_manager_output(output: str) -> str:
    """Make clear which Isaac manager outputs the Go2 adapter discards."""

    replacements = (
        (
            "[INFO] Command Manager:",
            "[INFO] IsaacLab backend Command Manager "
            "(internal; not the Go2 reward target):",
        ),
        (
            "[INFO] Observation Manager:",
            "[INFO] IsaacLab backend Observation Manager "
            "(internal tensor discarded by the shared Go2 adapter):",
        ),
        (
            "Active Observation Terms in Group: 'policy'",
            "Active internal backend terms [not model input] in Group: 'policy'",
        ),
        (
            "[INFO] Reward Manager:",
            "[INFO] IsaacLab backend Reward Manager "
            "(disabled; Go2 adapter computes the effective reward):",
        ),
    )
    for original, replacement in replacements:
        output = output.replace(original, replacement)
    return output
