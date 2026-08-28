"""Small, centralized defaults used by :mod:`src.run`.

Environment-specific configuration remains next to each environment.  This
module only owns paths and the human-facing launch presets.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DDS_DOMAIN_ID = 1
DEFAULT_DDS_INTERFACE = "lo"
# Keep the default transfer gate on the flat scene.  Terrain evaluation remains
# available explicitly through ``--scene scene_terrain.xml`` after flat
# zero-shot locomotion passes.
DEFAULT_MUJOCO_SCENE = str(
    PROJECT_ROOT / "assets" / "robots" / "go2" / "mjcf" / "scene.xml"
)


def find_unitree_mujoco_root() -> Path:
    """Resolve the existing checkout without modifying it."""

    override = os.environ.get("UNITREE_MUJOCO_ROOT")
    candidates = [
        Path(override).expanduser() if override else None,
        PROJECT_ROOT.parent / "modules" / "unitree_mujoco",
        PROJECT_ROOT.parent.parent / "modules" / "unitree_mujoco",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "simulate").is_dir():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(
        "unitree_mujoco checkout was not found. Set UNITREE_MUJOCO_ROOT or "
        f"place it in one of: {searched}"
    )


COMMON_RUNNER_FLAGS = (
    "--runner.project_name=go2_sqrl",
    "--runner.track_console=true",
)

TORCH_PRETRAIN_FLAGS = COMMON_RUNNER_FLAGS + (
    "--algorithm.name=sac_qsafe.pytorch",
)

# DDS remains a NumPy/host boundary.  The policy, critics and optimizer used
# by the online MuJoCo stage are nevertheless Flax/JAX implementations.
JAX_ONLINE_FLAGS = COMMON_RUNNER_FLAGS + (
    "--algorithm.name=sac_qsafe.flax",
)

RUN_PRESETS = {
    "pretrain": TORCH_PRETRAIN_FLAGS
    + (
        "--runner.mode=train",
        "--runner.exp_name=pretrain",
        "--runner.run_name=isaac_flashsac_cmd_reward_v3",
        "--runner.save_model=true",
        "--environment.name=go2_sqrl.isaac_lab",
        # Train the first transfer checkpoint on the same flat-ground task as
        # the canonical MuJoCo scene.  Rough-terrain robustness is a later
        # training/evaluation gate, not part of the simulator parity check.
        "--environment.terrain_mode=flat",
        "--algorithm.phase=pretrain",
        "--algorithm.rollout_mode=partitioned",
    ),
    "isaac-eval": TORCH_PRETRAIN_FLAGS
    + (
        "--runner.mode=test",
        "--runner.exp_name=evaluation",
        "--runner.run_name=isaac_flashsac_cmd_reward_v3",
        "--runner.nr_test_episodes=5",
        "--environment.name=go2_sqrl.isaac_lab",
        "--environment.nr_envs=1",
        "--environment.nr_task_envs=1",
        "--environment.nr_safety_envs=0",
        "--environment.terrain_mode=flat",
        "--algorithm.phase=pretrain",
        "--algorithm.eval_policy=task",
    ),
    "zero-shot": JAX_ONLINE_FLAGS
    + (
        "--runner.mode=test",
        "--runner.exp_name=zero_shot",
        "--runner.run_name=mujoco",
        "--environment.name=go2_sqrl.sdk2_mujoco",
        "--algorithm.phase=finetune",
        # Algorithm 2 always executes the QSafe-projected policy in the target
        # environment, including the pre-finetune zero-shot evaluation.
        "--algorithm.eval_policy=safe",
    ),
    "finetune": JAX_ONLINE_FLAGS
    + (
        "--runner.mode=train",
        "--runner.exp_name=finetune",
        "--runner.run_name=mujoco",
        "--runner.save_model=true",
        "--environment.name=go2_sqrl.sdk2_mujoco",
        "--algorithm.phase=finetune",
    ),
    "eval": JAX_ONLINE_FLAGS
    + (
        "--runner.mode=test",
        "--runner.exp_name=evaluation",
        "--runner.run_name=mujoco",
        "--environment.name=go2_sqrl.sdk2_mujoco",
        "--algorithm.phase=finetune",
    ),
}
