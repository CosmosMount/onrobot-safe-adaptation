"""Isaac Lab manager environment configuration.

This module is imported only after :class:`isaaclab.app.AppLauncher` starts.
"""

from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.sensors import Imu, ImuCfg, patterns
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from train.config import PROJECT_ROOT

from ..common.specs import ACTION_SPEC, DEFAULT_BASE_HEIGHT, JOINT_NAMES
from .randomization_cfg import configure_existing_events
from .terrain_cfg import boxes_height_range, configure_terrain, configure_terrain_mode


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
