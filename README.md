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

The Go2 transfer manifest is deliberately strict. Manifest v5 records the SDK
joint order, shared reset pose, absolute joint-target semantics, PD gains and
torque limits. Older checkpoints are rejected because their Isaac action term
used a different joint ordering and articulation-default offset; run
`python -m src.run pretrain` again instead of reusing those policy and safety
critic weights. The parity checkpoint and default MuJoCo transfer gate both use
flat ground. Rough terrain and domain randomization should be enabled only
after that deterministic task-policy gate passes.
