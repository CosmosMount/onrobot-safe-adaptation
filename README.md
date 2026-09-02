# onrobot-safe-adaptation

Safe Go2 adaptation with PyTorch/Isaac Lab pre-training and Flax/JAX online
fine-tuning through the same SDK2 interface used by unitree_mujoco and a future
robot runtime.

## Go2 SQRL workflow

Project-specific environments live under `src/`; `rl_x/` remains the reusable
algorithm and environment-interface layer. The workflow has one entrypoint:

```bash
python -m src.run sim
python -m src.run pretrain
python -m src.run pretrain-sac
python -m src.run isaac-eval --checkpoint /path/to/models/step_000300032.model
python -m src.run isaac-finetune-sac --checkpoint /path/to/models
python -m src.run zero-shot
python -m src.run finetune
python -m src.run finetune-sac
python -m src.run eval
python tools/run_flat_safe_adaptation.py --steps 200000 --seeds 0,1,2
```

The simulator command locates the existing `unitree_mujoco` checkout under the
adjacent `modules/` directory. Override that location only when necessary with
`UNITREE_MUJOCO_ROOT=/path/to/unitree_mujoco`.

The SDK training reset path is software-controlled and does not synthesize
keyboard input. Run `python tools/patch_unitree_mujoco_software_reset.py` and
rebuild the sibling checkout's `simulate/build/unitree_mujoco` binary. The
training process sends a domain-scoped local reset request; the simulator
services it in the physics thread with `mj_resetDataKeyframe(home)`.

## Environment Setup 

### Installation

For Isaac Lab simulation, use Isaac Lab `v2.3.2`, Isaac Sim `5.1.0`, and
Python `3.11` in a dedicated micromamba environment:

```bash
micromamba create -n onrobot-safe-adaptation python=3.11 pip git -c conda-forge -y
eval "$(micromamba shell hook -s bash)"
micromamba activate onrobot-safe-adaptation

python -m pip install -U pip
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com

git clone --depth 1 --branch v2.3.2 \
  https://github.com/isaac-sim/IsaacLab.git /opt/IsaacLab-v2.3.2
cd /opt/IsaacLab-v2.3.2
./isaaclab.sh -i

cd /path/to/onrobot-safe-adaptation
python -m pip install -e ".[mujoco]"
```

Isaac Sim is installed from NVIDIA’s Python package index. The Python version
must match the Isaac Sim release.

For MuJoCo/JAX-only use, Isaac Sim and Isaac Lab are not required:

```bash
python -m pip install -e ".[mujoco]"
```

### Verification

```bash
python -m pip check
python -c "import rl_x, jax, flax, mujoco; print('RL-X environment is ready')"
python -c "import isaaclab, isaacsim; print('Isaac Lab environment is ready')"
```

The MuJoCo extra pins the JAX/Flax versions validated with the NumPy 1.26 Isaac
environment, preventing the package resolver from assembling an incompatible
major-version stack. DDS transport and simulator reset remain host-side Python
operations; policy inference, safety projection and online gradient updates use
JAX.

## Training budget and checkpoint contract

The default pre-training horizon remains 500,000 **task transitions**, matching
the SQRL reference setting. With 256 task environments this ends at 500,224
because a vector step cannot be split. The partitioned trainer now reports
vector steps and optimizer updates separately and uses a default task UTD of
one update per newly collected transition; changing the number of parallel
environments therefore no longer silently changes the optimization budget.
QSafe updates are credited only when complete safety trajectories are committed.

The Go2 transfer manifest is deliberately strict. New Universal-QSafe actors
use manifest v11. The formal flat Safe-Adaptation regression intentionally
retains one older, deployment-verified manifest-v9/action-v1 actor and rejects
every other legacy combination. Both contracts record the SDK joint order,
shared reset pose, absolute joint-target semantics, PD gains, torque limits,
and the common tilt-or-low-base failure label. The source and target tasks both
use flat ground. Rough terrain remains a later robustness gate.

## Verified MuJoCo flat SAC fine-tuning

There is one supported standard-SAC target recipe. It transfers only the actor
and its observation normalizer from the verified
`isaac_sac_height_dr_v1` manifest-v9 flat policy.
The target task creates new task critics and targets, replay, optimizers, and
entropy temperature. The actor and alpha remain frozen for the first 10,000
transitions while the new critic learns; afterwards they update once per ten
critic updates at the configured learning rate. There is no second learning
rate ramp. Task critics, target critics, and replay all consume the projected
normalized action that the robot actually executed.

Start the flat simulator in one terminal:

```bash
python -m src.run sim \
  --scene assets/robots/go2/mjcf/scene.xml \
  --domain-id 1 \
  --interface lo
```

Then run the retained 30k regression recipe in another terminal:

```bash
python -m src.run finetune-sac \
  --checkpoint runs/go2_sqrl/pretrain/isaac_sac_height_dr_v1/models \
  --seed 0 \
  --domain-id 1 \
  --interface lo \
  --algorithm.total_timesteps=30000 \
  --algorithm.learning_starts=1000 \
  --algorithm.finetune_actor_warmup_steps=10000 \
  --algorithm.finetune_actor_update_interval=10 \
  --algorithm.task_utd_ratio=1.0 \
  --algorithm.alpha_init=0.0002 \
  --environment.target_velocity_x=0.6 \
  --environment.terrain_profile=flat \
  --environment.foot_clearance_target=0.07 \
  --environment.clearance_reward_mode=legacy_mean \
  --environment.phase_reward_scale=0.0 \
  --environment.stable_progress_scale=0.0 \
  --environment.terminal_failure_penalty=0.0 \
  --runner.track_tb=true \
  --runner.run_name=mujoco_sac_old_applied_flat_repro_s0_30k
```

The regression run finished at `0.534 m/s` forward velocity. A separate
five-episode evaluation completed all five 500-step episodes without a fall,
with `0.537 m/s` mean forward velocity, `0.548 m/s` mean last-100-step
velocity, `0.064 m/s` mean command error, and `0.073 m/s` mean velocity
estimation error. Training itself accumulated 54 falls, including a
concentrated unstable period around 20k--21k transitions. This validates that
the SAC fine-tune continues moving instead of collapsing to a standing policy;
it is not evidence that the unprotected learner is safe.

Evaluate the saved task policy with the same flat reward contract:

```bash
python -m src.run eval \
  --checkpoint runs/go2_sqrl/finetune/mujoco_sac_old_applied_flat_repro_s0_30k/models/final.model \
  --seed 0 \
  --domain-id 1 \
  --interface lo \
  --runner.nr_test_episodes=5 \
  --algorithm.eval_policy=task \
  --environment.target_velocity_x=0.6 \
  --environment.terrain_profile=flat \
  --environment.foot_clearance_target=0.07 \
  --environment.phase_reward_scale=0.0
```

For the formal QSafe on/off comparison, both arms use that same actor,
normalizer, target-task initialization seed, alpha, and update schedule. The
independent frozen safety critic is
`isaac_sqrl_height_dr_v1/models/qsafe.model`; only
`algorithm.qsafe.enabled` changes between arms. Run three paired 200k seeds
from the repository root with:

```bash
python tools/run_flat_safe_adaptation.py \
  --steps 200000 \
  --seeds 0,1,2 \
  --parallel-seeds 1
```

The launcher starts a fresh flat simulator for each arm, verifies matching
actor/task-critic/target-critic/alpha/normalizer fingerprints, stops on major
runtime or learning failures, evaluates the final deterministic task actors,
and writes falls-per-100k, episode fall probability, velocity, rejection,
fallback, and action-selection diagnostics under
`runs/go2_sqrl/ablation/flat_safe_adaptation_v1`.

## Deterministic gait-capability gate

The actor input remains one 46D frame. The optional diagonal-trot reward is
estimated only from current foot clearance and vertical velocity; it does not
add a clock, phase observation, residual action, or contact sensor. A fixed
four-centimetre upward ledge can be selected with:

```bash
python -m src.run isaac-eval \
  --checkpoint /path/to/models \
  --runner.nr_test_episodes=20 \
  --environment.terrain_mode=rough \
  --environment.terrain_profile=single_step_up \
  --environment.step_height=0.04 \
  --environment.terrain_num_rows=1 \
  --environment.terrain_num_cols=1 \
  --environment.foot_clearance_target=0.07 \
  --environment.clearance_reward_mode=swing_weighted \
  --environment.phase_reward_scale=0.3
```

Training a short reward-screen candidate from the same flat SAC checkpoint
transfers only its actor and actor normalizer. Task critics/targets, entropy
temperature, replay, and every optimizer are new. This matches the formal
target-stage contract instead of attaching source-reward values to a modified
reward. Keep 20 environments for a 10k screen so every environment sees one
complete 500-step ledge episode (`10000 / 20 = 500`):

```bash
python -m src.run isaac-finetune-sac \
  --checkpoint /path/to/flat-sac/models \
  --algorithm.total_timesteps=10000 \
  --algorithm.finetune_actor_warmup_steps=2500 \
  --environment.nr_envs=20 \
  --environment.nr_task_envs=20 \
  --environment.nr_safety_envs=0 \
  --environment.step_height=0.04 \
  --environment.foot_clearance_target=0.07 \
  --environment.clearance_reward_mode=swing_weighted \
  --environment.phase_reward_scale=0.3
```

Data-producing Isaac SAC actors use the frozen optimization recipe recovered
from the successful `gait_h07_p03_50k_s2` checkpoint. The first target stage
transfers only actor and actor normalizer; task critics/targets, entropy
temperature, replay, and optimizers restart. Explicit full-task restoration is
reserved for continuation between stages of the same target-task curriculum.
Learning starts and actor warm-up are 2,500 transitions; actor/critic update
interval and task UTD are both 1; and training uses 20 environments. Terrain
and reward may vary by actor role, but these SAC settings must not be changed
between roles or seeds.

Evaluation reports fall, true crossing success, stuck episodes, foot
clearance, per-leg swing activity, and action/torque saturation. A crossing is
successful only after climbing the ledge and making two metres of total
forward progress; an episode with last-100-step velocity below 0.1 m/s is
reported as stuck rather than successful.

## Universal QSafe v2 gated workflow

Universal QSafe does not change the actor checkpoint contract: the actor still
reads one 46D frame.  QSafe v2 alone reads five raw frames (230D, 100 ms), owns
its normalizer, consumes the projected normalized 12D action, and emits a
sigmoid risk in `[0, 1]`.  Its checkpoint records the observation/action/failure
contract, history length/control period, Bellman gamma, rejection epsilon,
normalizer, and held-out calibration report.

Collect one complete-trajectory shard for an actor/noise/terrain condition:

```bash
python -m src.run isaac-collect-qsafe \
  --checkpoint /path/to/behavior_actor/models \
  --runner.nr_test_episodes=20 \
  --environment.terrain_profile=single_step_up \
  --environment.step_height=0.04 \
  --algorithm.qsafe.dataset.directory=runs/go2_sqrl/qsafe_dataset/v2 \
  --algorithm.qsafe.dataset.actor_id=flat_seed0 \
  --algorithm.qsafe.dataset.map_seed=10000 \
  --algorithm.qsafe.dataset.episode_offset=0 \
  --algorithm.qsafe.dataset.split=train \
  --algorithm.qsafe.dataset.terrain=step_4cm \
  --algorithm.qsafe.dataset.action_noise=0.10
```

Collection transfers only `policy.model` and the actor normalizer.  It does not
load the behavior actor's task critic or QSafe.  Each `.npz` is committed only
after the episode finishes and includes the 100 stochastic policy candidates
needed for the fallback-rate gate.  To execute the full flat/step/boxes/mixed
and `0/0.05/0.10/0.20` matrix, provide an actor inventory JSON to:

```bash
python tools/collect_universal_qsafe_dataset.py \
  --actors /path/to/actors.json \
  --dataset runs/go2_sqrl/qsafe_dataset/v2 \
  --episodes-per-run 20
```

An inventory item has this form; every actor ID must occur in exactly one
train/validation/test split:

```json
[
  {
    "id": "flat_seed0",
    "checkpoint": "runs/go2_sqrl/pretrain/example/models",
    "split": "train",
    "environment_overrides": {}
  }
]
```

Offline training refuses to proceed below 500 fall plus 500 successful-crossing
trajectories, or when actor/map seeds leak between splits.  It trains gamma
`0.7/0.9/0.97`, scans epsilon `0.05/0.10/0.15/0.20`, and writes the protected
name `universal_qsafe_v2.model` only if the held-out Isaac calibration gate
passes.  `--allow-incomplete-data` exists solely for implementation smoke tests
and produces only `best_candidate_qsafe_v2.model`.

```bash
python -m rl_x.algorithms.qsafe.offline_train \
  --dataset runs/go2_sqrl/qsafe_dataset/v2 \
  --output runs/go2_sqrl/universal_qsafe_v2 \
  --updates 100000
```

The frozen, unknown-actor protection test and formal A/B/C/D fine-tune matrix
are then launched with:

```bash
python tools/evaluate_universal_qsafe_protection.py \
  --actor /path/to/held_out_actor/models \
  --qsafe runs/go2_sqrl/universal_qsafe_v2/universal_qsafe_v2.model

python tools/run_universal_qsafe_finetune.py \
  --actor /path/to/held_out_actor/models \
  --qsafe runs/go2_sqrl/universal_qsafe_v2/universal_qsafe_v2.model \
  --steps 10000 --seeds 0 --groups A,B,C,D
```

The experiment launcher enforces the 10k actor freeze and limits vectorization
so each environment receives at least one full 500-step episode.  It refuses
50k/100k/300k experiments with an uncalibrated candidate.  For the deterministic
MuJoCo gate, start `scene_step_4cm.xml` and set
`--environment.terrain_profile=single_step_up`; reward, base-height failure,
foot clearance, crossing and stuck metrics then use the matching local 4 cm
ground truth.  SportModeState remains excluded from actor and QSafe inputs.
