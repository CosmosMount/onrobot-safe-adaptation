"""Isaac Lab manager environment configuration.

This module is imported only after :class:`isaaclab.app.AppLauncher` starts.
"""

from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.sensors import Imu, ImuCfg
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from src.config import PROJECT_ROOT

from ..common.specs import ACTION_SPEC, DEFAULT_BASE_HEIGHT, JOINT_NAMES
from .randomization_cfg import configure_existing_events
from .terrain_cfg import configure_terrain, configure_terrain_mode


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
        # The shared 46D contract has no terrain-height observation.  Removing
        # the inherited scanner also avoids unnecessary Warp ray-cast kernels.
        self.scene.height_scanner = None
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
        self.actions.joint_pos.scale = ACTION_SPEC.scale
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
    randomization = bool(config.environment.domain_randomization)
    cfg.observations.policy.enable_corruption = randomization
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
