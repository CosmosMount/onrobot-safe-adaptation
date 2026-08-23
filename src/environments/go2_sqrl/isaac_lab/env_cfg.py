"""Isaac Lab manager environment configuration.

This module is imported only after :class:`isaaclab.app.AppLauncher` starts.
"""

from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.sensors import ImuCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from src.config import PROJECT_ROOT

from .randomization_cfg import configure_existing_events
from .terrain_cfg import configure_terrain


GO2_USD_PATH = PROJECT_ROOT / "assets" / "robots" / "go2" / "usd" / "go2.usd"


@configclass
class Go2SQRLIsaacEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UNITREE_GO2_CFG.spawn.replace(usd_path=str(GO2_USD_PATH)),
        )
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            update_period=0.0,
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
        configure_existing_events(self.events)
        # ManagerBasedRLEnv requires a reward manager, but its output is not
        # part of the Go2 SQRL contract.  The adapter computes the shared
        # total-track-xy reward itself, so disable every inherited term.
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
        self.actions.joint_pos.scale = 0.25
        # Failure is detected from the IMU by the Go2 adapter.  Neither the
        # inherited contact sensor nor its base-contact termination is needed.
        self.terminations.base_contact = None
        self.scene.contact_forces = None


def make_env_cfg(config, num_envs=None):
    cfg = Go2SQRLIsaacEnvCfg()
    cfg.scene.num_envs = int(num_envs or config.environment.nr_envs)
    cfg.seed = int(config.environment.seed)
    configured_device = str(config.environment.device)
    cfg.sim.device = "cuda:0" if configured_device == "gpu" else configured_device
    return cfg
