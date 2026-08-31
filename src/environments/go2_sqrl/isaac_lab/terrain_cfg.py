"""The selected seven-terrain distribution, colocated with the Isaac env."""


# Midpoints of the cumulative column intervals configured below.  Selecting a
# representative column keeps the full mixed terrain visible while placing a
# single playback robot on the requested terrain family.
PLAYBACK_TERRAIN_COLUMN_FRACTIONS = {
    "flat": 0.05,
    "boxes": 0.175,
    "random_rough": 0.35,
    "slopes": 0.525,
    "small_bumps": 0.65,
    "small_stairs": 0.75,
    "small_stairs_up": 0.90,
}

HIGH_CLEARANCE_STAGE_PROPORTIONS = {
    "easy": (0.20, 0.50, 0.25, 0.05),
    "medium": (0.10, 0.25, 0.45, 0.20),
    "hard": (0.10, 0.10, 0.30, 0.50),
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


def configure_terrain_profile(
    terrain_generator,
    curriculum,
    terrain_gen,
    profile,
    *,
    step_height=0.04,
    boxes_max_adjacent_height_difference=0.04,
    high_clearance_stage="easy",
):
    """Select a deterministic benchmark family inside generator terrain.

    ``single_step_up`` starts the robot on the low central platform of a
    one-ring inverted stair and places the high surface one metre ahead.
    With a three-metre step width, the standard eight-metre patch contains
    exactly one four-sided ledge rather than a staircase curriculum.
    """

    profile = str(profile).lower()
    if profile == "mixed":
        return
    if terrain_generator is None:
        raise ValueError("deterministic terrain profiles require terrain_mode=rough")
    curriculum.terrain_levels = None
    if profile == "single_step_up":
        height = float(step_height)
        if not 0.0 < height <= 0.10:
            raise ValueError("step_height must be in (0, 0.10] m")
        terrain_generator.sub_terrains = {
            "single_step_up": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(height, height),
                step_width=3.0,
                platform_width=2.0,
                border_width=0.25,
                inverted=True,
            )
        }
        return
    if profile == "high_clearance_mix":
        stage = str(high_clearance_stage).lower()
        if stage not in HIGH_CLEARANCE_STAGE_PROPORTIONS:
            raise ValueError(
                "high_clearance_stage must be easy, medium, or hard"
            )
        flat, step_2cm, step_3cm, step_4cm = (
            HIGH_CLEARANCE_STAGE_PROPORTIONS[stage]
        )

        def repeated_step(proportion, height):
            return terrain_gen.HfInvertedPyramidStairsTerrainCfg(
                proportion=proportion,
                step_height_range=(height, height),
                step_width=1.0,
                platform_width=2.0,
                border_width=0.25,
                inverted=True,
            )

        terrain_generator.sub_terrains = {
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=flat),
            "repeated_step_2cm": repeated_step(step_2cm, 0.02),
            "repeated_step_3cm": repeated_step(step_3cm, 0.03),
            "repeated_step_4cm": repeated_step(step_4cm, 0.04),
        }
        return
    if profile == "boxes_4cm":
        terrain_generator.sub_terrains = {
            "boxes_4cm": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=1.0,
                grid_height_range=boxes_height_range(
                    boxes_max_adjacent_height_difference
                ),
                grid_width=0.45,
                platform_width=2.0,
                holes=False,
            )
        }
        return
    raise ValueError(
        f"Unsupported terrain_profile {profile!r}; expected mixed, "
        "single_step_up, high_clearance_mix, or boxes_4cm."
    )


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
