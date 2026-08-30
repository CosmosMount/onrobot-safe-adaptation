# onrobot-safe-adaptation

Safe Go2 adaptation with PyTorch/Isaac Lab pre-training and PyTorch online
fine-tuning through the same SDK2 interface used by unitree_mujoco and a future
robot runtime.

## Go2 SQRL workflow

Training is organized backend-first under `train/isaac/` and `train/mujoco/`,
then by framework. PyTorch contains the current implementation; the JAX directories
are reserved and intentionally empty. Shared contracts live under `train/common/`,
while algorithm code remains under `algorithms/` and `sqrl/`. The workflow
keeps one entrypoint:

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
python -c "import torch, mujoco; from train.common.base import ACTION_SPEC; print(ACTION_SPEC.size)"
python -c "import isaaclab, isaacsim; print('Isaac Lab environment is ready')"
```

DDS transport and simulator reset remain host-side Python operations; policy
inference, safety projection and online gradient updates use PyTorch.

## Training budget and checkpoint contract

The default pre-training horizon remains 500,000 **task transitions**, matching
the SQRL reference setting. With 256 task environments this ends at 500,224
because a vector step cannot be split. The partitioned trainer now reports
vector steps and optimizer updates separately and uses a default task UTD of
one update per newly collected transition; changing the number of parallel
environments therefore no longer silently changes the optimization budget.
QSafe updates are credited only when complete safety trajectories are committed.

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
