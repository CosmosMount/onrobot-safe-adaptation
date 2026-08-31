"""Isaac terrain layout and playback selection."""

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

