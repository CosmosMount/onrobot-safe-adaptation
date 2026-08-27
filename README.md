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
python -m src.run isaac-eval --checkpoint /path/to/models/step_000300032.model
python -m src.run zero-shot
python -m src.run finetune
python -m src.run eval
```

The simulator command locates the existing `unitree_mujoco` checkout under the
adjacent `modules/` directory. Override that location only when necessary with
`UNITREE_MUJOCO_ROOT=/path/to/unitree_mujoco`.

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

With `unitree_mujoco` and an evaluation policy already running, compare the
legacy 50 Hz complementary estimator with the robust MuJoCo 500 Hz Kalman
filter against `SportModeState` truth:

```bash
python tools/velocity_estimator_benchmark.py \
  --duration-seconds 30 --warmup-seconds 2 \
  --domain-id 1 --interface lo \
  --save-samples /tmp/go2_velocity_samples.npz
```

The benchmark is read-only: it subscribes to SDK2 state topics, publishes no
joint commands, and reports per-axis and three-dimensional bias, RMSE and P95.
It also reports the same metrics in absolute `SportModeState` forward-speed
bins (`0-0.15`, `0.15-0.3`, `0.3-0.45`, `0.45-0.6`, and `0.6+` m/s). Simulator
tick restarts reset both estimators and are stored in the recording, so reset
and acceleration transients can be replayed without leaking filter state
across episodes. Re-evaluate public KF parameters offline with the exact same
trajectory, for example:

```bash
python tools/velocity_estimator_benchmark.py \
  --load-samples /tmp/go2_velocity_samples.npz \
  --leg-variance 0.002 --height-scale 0.05 --prior-temperature 0.05
```

The common defaults use those three values. They were selected with a frozen
deterministic policy and synchronized MuJoCo truth; process variance `0.03059`,
vertical velocity scale `0.35`, Huber threshold `0.25`, and the 3-DoF NIS gate
`11.34` remain unchanged. Isaac Lab and MuJoCo therefore consume the same
estimator configuration while retaining their respective 20 ms and 2 ms
update rates.

The MuJoCo extra pins the JAX/Flax versions validated with the NumPy 1.26 Isaac
environment, preventing the package resolver from assembling an incompatible
major-version stack. DDS transport and simulator reset remain host-side Python
operations; policy inference, safety projection and online gradient updates use
JAX.

## Training budget and checkpoint contract

The reusable SAC-QSafe default remains 500,000 task transitions, matching the
SQRL reference setting. The Go2 experiment preset uses a 300,000-transition
budget and 300 task plus 100 safety environments to shorten the flat-ground
iteration while retaining two complete 500-step safety horizons. The
partitioned trainer reports
vector steps and optimizer updates separately and uses a default task UTD of
one update per newly collected transition; changing the number of parallel
environments therefore no longer silently changes the optimization budget.
QSafe updates are credited only when complete safety trajectories are committed.

As in the SQRL quadruped velocity-transfer experiment, the task changes across
stages: Isaac pre-training targets 0.5 m/s and MuJoCo fine-tuning targets 0.6
m/s. The target is reward-only and does not change the 46D policy observation.
Foot sliding friction remains 0.4 in both stages so the QSafe ablation changes
one safety mechanism rather than mixing velocity and dynamics shifts.

The MuJoCo preset uses SORL reward shaping during target adaptation. It follows
Equations 10/11, updates the terminal cost and lambda from the empirical reward
range and configured Delta, uses twin safety critics, and duplicates only the
unsafe transition in `D_safe`. The Go2 command reward has one explicit
compatibility safeguard: a negative speed/task reward is never made less
negative merely because the current safety risk is small. Set
`--algorithm.sorl.preserve_negative_task_penalty=false` to run the paper's
strict negative-reward branch. Transfer actor updates start after 50,000 target
transitions and use a low-rate mean-policy anchor; this protects the viable
source gait while the target critic adapts.

The Go2 transfer manifest is deliberately strict. Manifest v8 records the SDK
joint order, shared reset pose, absolute joint-target semantics, PD gains,
torque limits, and the policy-visible proprioceptive body-velocity field. Older
checkpoints are rejected because their observation or Isaac action contract is
different; run `python -m src.run pretrain` again instead of reusing those
policy and safety critic weights. The parity checkpoint and default MuJoCo
transfer gate both use flat ground. Rough terrain and domain randomization
should be enabled only after that deterministic task-policy gate passes.
