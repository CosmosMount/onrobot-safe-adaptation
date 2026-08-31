# onrobot-safe-adaptation

Safe Go2 adaptation with PyTorch/Isaac Lab pre-training and PyTorch online
fine-tuning through the same SDK2 interface used by unitree_mujoco and a future
robot runtime.

## Go2 SQRL workflow

`train/` owns backend construction, environment adapters, configuration, and
command dispatch. The complete SQRL execution path (pre-training, fine-tuning,
evaluation, and checkpoints) lives under `sqrl/` and depends only on the shared
environment interface. Network definitions remain under `algorithms/`. The
workflow keeps one entrypoint:

```text
sqrl/sac/pytorch/
  workflow.py       public phase, evaluation, and checkpoint API
  pretrainer.py     Algorithm 1 task/safety phase coordinator
  task_trainer.py   task-partition SAC updates
  safety_trainer.py on-policy safety rollouts and QSafe updates
  finetuner.py      target-domain SQRL fine-tuning
  safety_ops.py     shared safe-action selection and QSafe targets
  replay_buffer.py  transition replay and update eligibility

train/core/
  contracts.py      tensor, physics, reset, and state contracts
  actions.py        normalized action-to-joint-target mapping
  observation.py    policy observation construction
  reward.py         reward and episode accounting
  manifest.py       checkpoint environment contract
  estimation_*.py   NumPy and Torch velocity estimators

train/isaac/pytorch/
  environment.py    Go2 Isaac environment
  runtime.py        Isaac manager stepping and physics frames
  contract.py       tensor and joint-order adapter
  terrain.py        terrain configuration
  randomization.py  transfer randomization
  setup.py          Isaac scene configuration

train/mujoco/pytorch/
  environment.py    Go2 MuJoCo environment
  client.py         SDK2 DDS client
  buffers.py        synchronized state buffers
  messages.py       SDK message decoding
  mjcf.py           model-contract validation
  reset.py          simulator reset controller
  sdk.py            stable public bridge API
```

```bash
python -m train.runner sim
python -m train.runner pretrain
python -m train.runner zero-shot --checkpoint runs/sqrl/pretrain/final.model
python -m train.runner finetune --checkpoint runs/sqrl/pretrain/final.model
python -m train.runner eval --checkpoint runs/sqrl/finetune/final.model
```

The simulator command uses the MJCF and meshes in this repository and locates
the existing `unitree_mujoco` checkout under the adjacent `modules/` directory.
Override only the simulator executable location with
`UNITREE_MUJOCO_ROOT=/path/to/unitree_mujoco` or `--unitree-root`.

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

For SDK2/MuJoCo-only use, Isaac Sim and Isaac Lab are not required:

```bash
python -m pip install -e ".[mujoco]"
```

### Verification

```bash
python -m pip check
python -c "import torch, mujoco; from train.core.base import ACTION_SPEC; print(ACTION_SPEC.size)"
python -c "import isaaclab, isaacsim; print('Isaac Lab environment is ready')"
```

DDS transport and simulator reset remain host-side Python operations; policy
inference, safety filtering and online gradient updates use PyTorch.

## Training budget and checkpoint contract

The default pre-training horizon remains 500,000 **task transitions**, matching
the SQRL reference setting. With 256 task environments this ends at 500,224
because a vector step cannot be split. The single pretrainer reports
vector steps and optimizer updates separately and uses a default task UTD of
one update per newly collected transition; changing the number of parallel
environments therefore no longer silently changes the optimization budget.
Each pre-training block follows Algorithm 1 in serial order: `n_off` task
**vector** interactions (with transition-based SAC update credit), `k` complete
trajectories while that policy remains unchanged, then one QSafe gradient update.
Following the practical implementation in section 6, `D_safe` is the small
on-policy buffer containing the latest policy snapshot's `k` complete
trajectories; the next block atomically replaces that batch. Section 5.1 also
describes a "mixture of policies represented in the replay buffer", which is
internally ambiguous with section 6. This reproduction gives precedence to
the explicit practical-implementation wording and records the alternative as
a paper ambiguity rather than silently accumulating off-policy safety data.
An experimental replay-epoch schedule is available only when
`qsafe_epochs_per_block` is explicitly configured. The paper's pseudocode
literally nests `n_off` inside `n_pre`, while Table 1 describes `n_pre` as the
total number of pre-training steps. This implementation adopts the Table 1
interpretation: `n_pre` is a scalar task-transition budget and may overshoot by
at most one vector batch. `learning_starts` is an explicit SAC warm-up setting;
the paper does not specify one.

Pre-training samples the accepted action with the largest QSafe value strictly
below `epsilon_safe`, matching the safety-boundary exploration rule in section
6. Fine-tuning and evaluation use finite-candidate rejection sampling for Eq. 3:
accepted i.i.d. policy samples are selected uniformly. If no candidate is
accepted, the least-risk candidate is used and the fallback is reported. The
default is 100 candidates, so this remains the paper's practical finite-sample
approximation rather than exact analytical conditioning.

Section 6 describes weighting retained candidates by their original policy
density but does not state the proposal distribution. Weighting candidates
that were already sampled from the policy would bias the result toward
`pi(a|s)^2`. This implementation instead uses ordinary rejection sampling from
the policy, which matches the conditional distribution defined by Eq. 3. The
choice is intentional and tested, but is not a claim of reproducing that
ambiguous sentence word for word.

The original QSafe network implementation is retained, including its final
`tanh` activation. Checkpoints are policy/QSafe transfer artifacts, not
optimizer-state training resumes.

The SAC actor uses the exact stable tanh change-of-variables correction. The
actor itself is not wrapped in an environment-specific action projection.
Physical joint limits and target-rate limits remain environment responsibilities;
the environment reports the applied action and replay stores that action rather
than an unapplied proposal.

## Shared Isaac Lab and MuJoCo contract

Both adapters construct the same 46-value policy observation:
`joint_q[0:12]`, `joint_dq[12:24]`, body-frame IMU gyro `[24:27]`, the shared
proprioceptive body-velocity estimate `[27:30]`, continuous WXYZ quaternion
`[30:34]`, and the previous applied absolute joint target `[34:46]`. NumPy and
Torch implementations are cross-tested on the same inputs. Commands and
simulator root velocity do not enter the policy tensor.

Both adapters also share the reward formula, 2 ms physics / 20 ms policy
cadence, 500-step finite horizon, terminal-frame reward and observation,
physical reset on failure or timeout, and the same sparse failure rule: roll or
pitch above 0.8 rad, or local base clearance below 0.18 m, for five consecutive
2 ms frames. The canonical joint order is
`FR, FL, RR, RL × hip, thigh, calf` and is checked as an exact one-to-one map.
For task SAC, the 500-step `TimeLimit` is an artificial truncation and therefore
bootstraps from the preserved pre-reset final observation; only true failure
termination masks the task target. QSafe instead estimates failure within a
complete finite safety rollout, so either termination or horizon truncation
ends its Bellman target. This distinction is intentional and shared across
backends.
The nominal joint pose is `[0, 0.9, -1.8]` per leg and the base origin is 0.289 m
above the flat ground. The bundled MuJoCo model is tested after ordinary
`mj_resetData` for all four feet contacting the floor. Both deterministic
Isaac and live MuJoCo resets validate joint pose within 1 mrad, base height
within 2 mm, identity base orientation within 1 mrad, and all four foot
surfaces within 3 mm of the ground.

The Go2 transfer manifest records the SDK joint order, shared reset pose,
absolute joint-target semantics, PD gains, torque limits, and the common
tilt-or-low-base failure label. Transfer checks keep those semantics strict
while allowing the intended phase differences: Isaac pre-training enables
domain randomization, and MuJoCo fine-tuning/evaluation uses a 0.6 m/s forward
target instead of the 0.5 m/s pre-training target. Both backends retain flat
ground, nominal friction, control timing, actuator contract, reset pose,
failure label, episode horizon, and the same reward formula. MuJoCo also models
small actuator-force sensor noise to better represent the measurement path used
by the torque penalty. The reward is the non-negative Gait in Eight
fixed-forward tracking objective: piecewise velocity tracking with yaw-rate,
upright-orientation, and joint-torque energy penalties.

The manifest also records every policy-visible velocity-estimator covariance,
confidence, robust-loss, and innovation-gate parameter. A pre-training and
fine-tuning configuration that changes any of them is rejected at checkpoint
load rather than changing observation semantics silently.

## MuJoCo bridge requirement and verification scope

Online MuJoCo training requires the external bridge to run one command-driven
transaction at a time. The checked-in patch pauses simulation during policy and
learner work, advances exactly ten 2 ms steps per sequenced command, publishes
post-step command acknowledgements, and timestamps root-state truth. Apply,
rebuild, and launch it as described in
[`assets/robots/go2/UNITREE_MUJOCO_BRIDGE.md`](assets/robots/go2/UNITREE_MUJOCO_BRIDGE.md).
The Python adapter rejects learner gaps, missing first frames, mixed command
sequences, corrupt LowState CRCs, and missing or stale root truth before replay
insertion. The supported `sim` command pins the canonical bundled scene and
validates its compiled actuator, passive-joint, contact-friction, reset-pose,
and sensor-order contract before launch. Custom `--scene` values are rejected
because the learner cannot authenticate an arbitrary external model over the
stock DDS schema. The runner intentionally does not modify an external
checkout.

Run the repository contract suite with:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -q
```

The suite compiles and forwards the real bundled MuJoCo model and exercises the
algorithm, reset, timing, observation, reward, joint-order, checkpoint, and
delegation contracts. The external C++ patch is also compile-tested against the
pinned upstream revision. Isaac tests are dependency-light adapter contract
tests; a live Isaac Sim/PhysX run and an end-to-end live DDS/X11 run still
require their respective external runtimes.
