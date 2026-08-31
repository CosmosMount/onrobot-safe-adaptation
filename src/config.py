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

# Frozen policy-training recipe recovered from the successful
# gait_h07_p03_50k_s2 checkpoint.  Data-producing SAC actors use this recipe;
# task-specific terrain and reward flags may change, but optimization settings
# must not drift between actor seeds or actor roles.
STABLE_ISAAC_SAC_FINETUNE_FLAGS = (
    "--algorithm.learning_rate=0.0003",
    "--algorithm.anneal_learning_rate=false",
    "--algorithm.buffer_size=1000000",
    "--algorithm.learning_starts=2500",
    "--algorithm.batch_size=256",
    "--algorithm.tau=0.005",
    "--algorithm.gamma=0.99",
    "--algorithm.target_entropy=auto",
    "--algorithm.alpha_init=1.0",
    "--algorithm.finetune_actor_warmup_steps=2500",
    "--algorithm.finetune_actor_update_interval=1",
    "--algorithm.log_std_min=-20",
    "--algorithm.log_std_max=2",
    "--algorithm.nr_hidden_units=256",
    "--algorithm.enable_observation_normalization=true",
    "--algorithm.normalizer_epsilon=1e-8",
    "--algorithm.task_utd_ratio=1.0",
    "--algorithm.rollout_mode=partitioned",
    "--algorithm.qsafe.enabled=false",
    "--algorithm.eval_policy=task",
)

# DDS remains a NumPy/host boundary.  The policy, critics and optimizer used
# by the online MuJoCo stage are nevertheless Flax/JAX implementations.
JAX_ONLINE_FLAGS = COMMON_RUNNER_FLAGS + (
    "--algorithm.name=sac_qsafe.flax",
)

# Target-domain SAC recipe. MuJoCo is intentionally serial (one transition per
# learner iteration), so UTD=1 and an actor interval of ten mean exactly ten
# critic updates per actor/temperature update. Target fine-tuning starts with a
# fresh task critic, target critic, replay, optimizers, and entropy temperature.
# The transferred actor stays frozen for 10k transitions and then resumes at
# the normal learning rate; there is no additional learning-rate handoff. Task
# critics and replay use the projected action that was actually executed.
# Keep these flags shared by the SAC and SAC+QSafe target stages; QSafe must be
# their only algorithmic difference in a formal comparison.
MUJOCO_SAC_FINETUNE_FLAGS = (
    "--algorithm.learning_rate=0.0003",
    "--algorithm.anneal_learning_rate=false",
    "--algorithm.buffer_size=1000000",
    "--algorithm.learning_starts=1000",
    "--algorithm.batch_size=256",
    "--algorithm.tau=0.005",
    "--algorithm.gamma=0.99",
    "--algorithm.target_entropy=auto",
    # Target-task temperature is intentionally fresh: source alpha is not
    # transferred together with the actor. Actor and alpha stay fixed while
    # the critic warms up, then update together at the 1:10 policy cadence.
    "--algorithm.alpha_init=0.0002",
    "--algorithm.finetune_actor_warmup_steps=10000",
    "--algorithm.finetune_actor_update_interval=10",
    "--algorithm.task_utd_ratio=1.0",
    "--algorithm.log_std_min=-20",
    "--algorithm.log_std_max=2",
    "--algorithm.nr_hidden_units=256",
    "--algorithm.enable_observation_normalization=true",
    "--algorithm.normalizer_epsilon=1e-8",
    # Match the manifest-v9 actor used by the verified flat regression.
    "--environment.action_profile=legacy_v1",
    "--environment.foot_clearance_reward_scale=0.0",
    "--environment.clearance_reward_mode=legacy_mean",
    "--environment.phase_reward_scale=0.0",
    "--environment.stable_progress_scale=0.0",
    "--environment.terminal_failure_penalty=0.0",
)

RUN_PRESETS = {
    "pretrain": TORCH_PRETRAIN_FLAGS
    + (
        "--runner.mode=train",
        "--runner.exp_name=pretrain",
        "--runner.run_name=isaac_sqrl_height_dr_v1",
        "--runner.save_model=true",
        "--environment.name=go2_sqrl.isaac_lab",
        # Train the first transfer checkpoint on the same flat-ground task as
        # the canonical MuJoCo scene.  Rough-terrain robustness is a later
        # training/evaluation gate, not part of the simulator parity check.
        "--environment.terrain_mode=flat",
        "--environment.domain_randomization=true",
        "--algorithm.phase=pretrain",
        "--algorithm.rollout_mode=partitioned",
    ),
    "pretrain-sac": TORCH_PRETRAIN_FLAGS
    + (
        "--runner.mode=train",
        "--runner.exp_name=pretrain",
        "--runner.run_name=isaac_sac_height_dr_v1",
        "--runner.save_model=true",
        "--environment.name=go2_sqrl.isaac_lab",
        "--environment.nr_envs=256",
        "--environment.nr_task_envs=256",
        "--environment.nr_safety_envs=0",
        "--environment.terrain_mode=flat",
        "--environment.domain_randomization=true",
        "--algorithm.phase=pretrain",
        "--algorithm.rollout_mode=partitioned",
        "--algorithm.qsafe.enabled=false",
        "--algorithm.eval_policy=task",
    ),
    "isaac-eval": TORCH_PRETRAIN_FLAGS
    + (
        "--runner.mode=test",
        "--runner.exp_name=evaluation",
        "--runner.run_name=isaac_sqrl_height_dr_v1",
        "--runner.nr_test_episodes=5",
        "--environment.name=go2_sqrl.isaac_lab",
        "--environment.nr_envs=1",
        "--environment.nr_task_envs=1",
        "--environment.nr_safety_envs=0",
        "--environment.terrain_mode=flat",
        # Playback follows env 0's robot and builds only 30 rough-terrain
        # patches when terrain_mode is overridden to rough.  Training retains
        # the environment defaults (10 x 20 patches).
        "--environment.viewer_follow_robot=true",
        "--environment.terrain_num_rows=3",
        "--environment.terrain_num_cols=10",
        "--algorithm.phase=pretrain",
        "--algorithm.eval_policy=task",
    ),
    "isaac-collect-qsafe": TORCH_PRETRAIN_FLAGS
    + (
        "--runner.mode=test",
        "--runner.exp_name=qsafe_dataset",
        "--runner.run_name=universal_qsafe_v2_collection",
        "--runner.nr_test_episodes=20",
        "--environment.name=go2_sqrl.isaac_lab",
        "--environment.nr_envs=20",
        "--environment.nr_task_envs=20",
        "--environment.nr_safety_envs=0",
        "--environment.terrain_mode=rough",
        "--environment.terrain_profile=single_step_up",
        "--environment.domain_randomization=true",
        # Transfer only the actor and its own normalizer.  Reward/task-critic
        # differences are expected across the behavior-policy inventory.
        "--algorithm.phase=finetune",
        "--algorithm.compile_policy=false",
        "--algorithm.qsafe.enabled=false",
        "--algorithm.qsafe.version=2",
        "--algorithm.qsafe.dataset.enabled=true",
        "--algorithm.eval_policy=task",
    ),
    "isaac-finetune": TORCH_PRETRAIN_FLAGS
    + (
        "--runner.mode=train",
        "--runner.exp_name=universal_qsafe_finetune",
        "--runner.run_name=isaac_step4cm_qsafe",
        "--runner.save_model=true",
        "--environment.name=go2_sqrl.isaac_lab",
        "--environment.nr_envs=20",
        "--environment.nr_task_envs=20",
        "--environment.nr_safety_envs=0",
        "--environment.terrain_mode=rough",
        "--environment.terrain_profile=single_step_up",
        "--environment.step_height=0.04",
        "--environment.domain_randomization=true",
        "--algorithm.phase=finetune",
        "--algorithm.rollout_mode=partitioned",
        "--algorithm.finetune_actor_warmup_steps=10000",
        "--algorithm.qsafe.enabled=true",
        "--algorithm.qsafe.version=2",
        "--algorithm.eval_policy=safe",
    ),
    "isaac-finetune-sac": TORCH_PRETRAIN_FLAGS
    + STABLE_ISAAC_SAC_FINETUNE_FLAGS
    + (
        "--runner.mode=train",
        "--runner.exp_name=gait_finetune",
        "--runner.run_name=isaac_step_phase_screen",
        "--runner.save_model=true",
        "--environment.name=go2_sqrl.isaac_lab",
        # A 10k screening budget must still expose every environment to one
        # complete 500-step ledge episode: 10_000 / 20 = 500 control steps.
        "--environment.nr_envs=20",
        "--environment.nr_task_envs=20",
        "--environment.nr_safety_envs=0",
        "--environment.terrain_mode=rough",
        "--environment.terrain_profile=single_step_up",
        "--environment.domain_randomization=false",
        "--algorithm.phase=finetune",
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
    + MUJOCO_SAC_FINETUNE_FLAGS
    + (
        "--runner.mode=train",
        "--runner.exp_name=finetune",
        "--runner.run_name=mujoco",
        "--runner.save_model=true",
        "--environment.name=go2_sqrl.sdk2_mujoco",
        "--environment.target_velocity_x=0.6",
        "--algorithm.phase=finetune",
    ),
    "finetune-sac": JAX_ONLINE_FLAGS
    + MUJOCO_SAC_FINETUNE_FLAGS
    + (
        "--runner.mode=train",
        "--runner.exp_name=finetune",
        "--runner.run_name=mujoco_sac",
        "--runner.save_model=true",
        "--environment.name=go2_sqrl.sdk2_mujoco",
        "--environment.target_velocity_x=0.6",
        "--algorithm.phase=finetune",
        "--algorithm.qsafe.enabled=false",
        "--algorithm.eval_policy=task",
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
