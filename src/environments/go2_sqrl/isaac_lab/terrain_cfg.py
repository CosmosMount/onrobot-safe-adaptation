"""The selected six-terrain distribution, colocated with the Isaac env."""


def configure_terrain(terrain_generator, terrain_gen):
    sub_terrains = terrain_generator.sub_terrains
    for name in (
        "pyramid_stairs",
        "pyramid_stairs_inv",
        "hf_pyramid_slope",
        "hf_pyramid_slope_inv",
    ):
        if name in sub_terrains:
            sub_terrains[name].proportion = 0.0

    sub_terrains["flat"] = terrain_gen.MeshPlaneTerrainCfg(proportion=0.15)
    sub_terrains["boxes"].proportion = 0.20
    sub_terrains["boxes"].grid_height_range = (0.0, 0.10)
    sub_terrains["boxes"].grid_width = 0.45
    sub_terrains["boxes"].platform_width = 2.0
    sub_terrains["boxes"].holes = False
    sub_terrains["random_rough"].proportion = 0.25
    sub_terrains["random_rough"].noise_range = (0.01, 0.06)
    sub_terrains["random_rough"].noise_step = 0.01
    sub_terrains["slopes"] = terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=0.15,
        slope_range=(0.05, 0.25),
        platform_width=2.0,
        border_width=0.25,
    )
    sub_terrains["small_bumps"] = terrain_gen.HfWaveTerrainCfg(
        proportion=0.15,
        amplitude_range=(0.005, 0.015),
        num_waves=8,
        border_width=0.25,
    )
    sub_terrains["small_stairs"] = terrain_gen.HfPyramidStairsTerrainCfg(
        proportion=0.10,
        step_height_range=(0.02, 0.08),
        step_width=0.35,
        platform_width=2.0,
        border_width=0.25,
    )

