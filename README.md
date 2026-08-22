# onrobot-safe-adaptation

Safe on-robot adaptation with Isaac Lab, MuJoCo, and JAX/Flax inference.

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
python -m pip install -e ".[mujoco,inference]"
```

Isaac Sim is installed from NVIDIA’s Python package index. The Python version
must match the Isaac Sim release.

For MuJoCo/JAX-only use, Isaac Sim and Isaac Lab are not required:

```bash
python -m pip install -e ".[mujoco,inference]"
```

### Verification

```bash
python -m pip check
python -c "import rl_x, jax, flax, mujoco; print('RL-X environment is ready')"
python -c "import isaaclab, isaacsim; print('Isaac Lab environment is ready')"
```
