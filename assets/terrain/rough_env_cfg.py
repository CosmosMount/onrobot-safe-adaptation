# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import isaaclab.terrains as terrain_gen

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        


        # ------------------------------------------------------------
        # Terrain configuration 1：
        # ------------------------------------------------------------

        # terrain_generator = self.scene.terrain.terrain_generator
        # sub_terrains = terrain_generator.sub_terrains

        # # Disable unused terrains
        # for terrain_name in (
        #     "pyramid_stairs",
        #     "pyramid_stairs_inv",
        #     "hf_pyramid_slope",
        #     "hf_pyramid_slope_inv",
        # ):
        #     if terrain_name in sub_terrains:
        #         sub_terrains[terrain_name].proportion = 0.0

        # # Random box terrain
        # #
        # # Every box height is sampled from [0.0, 0.10] meters.
        # # Therefore, the maximum theoretical height difference
        # # between two neighboring boxes is no more than 0.10 meters.
        # sub_terrains["boxes"].proportion = 0.4
        # sub_terrains["boxes"].grid_height_range = (0.0, 0.10)
        # sub_terrains["boxes"].grid_width = 0.45
        # sub_terrains["boxes"].platform_width = 2.0
        # sub_terrains["boxes"].holes = False

        # # Ordinary random rough terrain
        # sub_terrains["random_rough"].proportion = 0.4
        # sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        # sub_terrains["random_rough"].noise_step = 0.01

        # # Flat terrain
        # sub_terrains["flat"] = terrain_gen.MeshPlaneTerrainCfg(
        #     proportion=0.2,
        # )

        # ------------------------------------------------------------
        # Terrain configuration 2：
        # ------------------------------------------------------------

        terrain_generator = self.scene.terrain.terrain_generator
        sub_terrains = terrain_generator.sub_terrains
        # Disable unused terrains inherited from ROUGH_TERRAINS_CFG
        for terrain_name in (
            "pyramid_stairs",
            "pyramid_stairs_inv",
            "hf_pyramid_slope",
            "hf_pyramid_slope_inv",
        ):
            if terrain_name in sub_terrains:
                sub_terrains[terrain_name].proportion = 0.0

        # Flat terrain: 15%
        sub_terrains["flat"] = terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.15,
        )

        # Random box terrain: 20%
        sub_terrains["boxes"].proportion = 0.20
        sub_terrains["boxes"].grid_height_range = (0.0, 0.10)
        sub_terrains["boxes"].grid_width = 0.45
        sub_terrains["boxes"].platform_width = 2.0
        sub_terrains["boxes"].holes = False

        # Ordinary random rough terrain: 25%
        sub_terrains["random_rough"].proportion = 0.25
        sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        sub_terrains["random_rough"].noise_step = 0.01

        # Sloped terrain: 15%
        sub_terrains["slopes"] = terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.05, 0.25),
            platform_width=2.0,
            border_width=0.25,
        )

        # Continuous wave / small bump terrain: 15%
        sub_terrains["small_bumps"] = terrain_gen.HfWaveTerrainCfg(
            proportion=0.15,
            amplitude_range=(0.005, 0.015),
            num_waves=8,
            border_width=0.25,
        )

        # Low stairs: 10%
        sub_terrains["small_stairs"] = terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.02, 0.08),
            step_width=0.35,
            platform_width=2.0,
            border_width=0.25,
        )

        # ------------------------------------------------------------
        # Terrain configuration 3：
        # ------------------------------------------------------------
        # terrain_generator = self.scene.terrain.terrain_generator
        # sub_terrains = terrain_generator.sub_terrains

        # # Disable unused inherited terrains
        # for terrain_name in (
        #     "pyramid_stairs",
        #     "pyramid_stairs_inv",
        #     "hf_pyramid_slope",
        #     "hf_pyramid_slope_inv",
        # ):
        #     if terrain_name in sub_terrains:
        #         sub_terrains[terrain_name].proportion = 0.0

        # # Flat terrain: 10%
        # sub_terrains["flat"] = terrain_gen.MeshPlaneTerrainCfg(
        #     proportion=0.10,
        # )

        # # Random box terrain: 15%
        # sub_terrains["boxes"].proportion = 0.15
        # sub_terrains["boxes"].grid_height_range = (0.0, 0.10)
        # sub_terrains["boxes"].grid_width = 0.45
        # sub_terrains["boxes"].platform_width = 2.0
        # sub_terrains["boxes"].holes = False

        # # Ordinary random rough terrain: 20%
        # sub_terrains["random_rough"].proportion = 0.20
        # sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        # sub_terrains["random_rough"].noise_step = 0.01

        # # Sloped terrain: 15%
        # sub_terrains["slopes"] = terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.15,
        #     slope_range=(0.05, 0.25),
        #     platform_width=2.0,
        #     border_width=0.25,
        # )

        # # Continuous wave / small bump terrain: 10%
        # sub_terrains["small_bumps"] = terrain_gen.HfWaveTerrainCfg(
        #     proportion=0.10,
        #     amplitude_range=(0.005, 0.015),
        #     num_waves=8,
        #     border_width=0.25,
        # )

        # # Low stairs: 10%
        # sub_terrains["small_stairs"] = terrain_gen.HfPyramidStairsTerrainCfg(
        #     proportion=0.10,
        #     step_height_range=(0.02, 0.08),
        #     step_width=0.35,
        #     platform_width=2.0,
        #     border_width=0.25,
        #     holes=False,
        # )

        # # Stepping stones: 10%
        # sub_terrains["stepping_stones"] = (
        #     terrain_gen.HfSteppingStonesTerrainCfg(
        #         proportion=0.10,
        #         stone_height_max=0.04,
        #         stone_width_range=(0.25, 0.45),
        #         stone_distance_range=(0.02, 0.08),
        #         holes_depth=0.03,
        #         platform_width=2.0,
        #         border_width=0.25,
        #     )
        # )

        # # Small discrete obstacles: 5%
        # sub_terrains["small_obstacles"] = (
        #     terrain_gen.HfDiscreteObstaclesTerrainCfg(
        #         proportion=0.05,
        #         obstacle_width_range=(0.15, 0.30),
        #         obstacle_height_range=(0.02, 0.06),
        #         num_obstacles=10,
        #         platform_width=2.0,
        #         border_width=0.25,
        #     )
        # )

        # # Small gaps around the central platform: 5%
        # sub_terrains["small_gaps"] = terrain_gen.MeshGapTerrainCfg(
        #     proportion=0.05,
        #     gap_width_range=(0.03, 0.08),
        #     platform_width=2.0,
        # )

        

        #------------------------------------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None