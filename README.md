# onrobot-safe-adaptation

Aiming to build up a on-robot training system enabling real-time adaptation with pretrained policy.

## Environment Setup

Our algorithm implementation comes from [RL-X](https://github.com/nico-bohlinger/RL-X).

The project targets Python 3.10 or newer. Python 3.11 is recommended. The
dependencies declared in `pyproject.toml` cover the RL-X runner, Gymnasium
MuJoCo environments, PyTorch, JAX/Flax, checkpointing, TensorBoard, and Weights
& Biases.

### Standard installation

Create an isolated environment and install this repository in editable mode:

```bash
conda create -n onrobot-safe-adaptation python=3.11 -y
conda activate onrobot-safe-adaptation

python -m pip install --upgrade pip
python -m pip install -e .
```

The default runner environment is `gym.mujoco.humanoid_v4`, so the
`gymnasium[mujoco]` dependency is installed automatically. If you only need a
specific algorithm/backend, the complete dependency set is still installed so
that the dynamic algorithm and environment imports used by `Runner` remain
available.

### GPU environments

Install the framework wheels that match the machine before installing this
repository. Use the official PyTorch and JAX installation commands for the
CUDA version and driver available on the system. For example, the JAX CUDA
extra can be installed with:

```bash
python -m pip install -U "jax[cuda12]"
python -m pip install -e .
```

For CPU-only or Apple Silicon development, the regular `jax` and `torch`
dependencies selected by `pip install -e .` are sufficient. Select the runtime
device in an experiment with `--algorithm.device=cpu`, `gpu`, or `mps` as
appropriate.

### Isaac Lab environments

Isaac Lab is an external simulator/runtime and is intentionally not listed as a
PyPI dependency. Install Isaac Lab following the instructions for the Isaac
Lab version and CUDA stack used by the robot project, activate that environment,
and then install this repository:

```bash
python -m pip install -e /path/to/onrobot-safe-adaptation
```

Any external environment or algorithm package must be importable in the same
environment. Pass its top-level package name in
`Runner(implementation_package_names=[...])` when it is registered outside
`rl_x`.

### Verify the installation

```bash
python -m pip check
python -c "import gymnasium, jax, torch, wandb; import rl_x; from rl_x.runner.runner import Runner; print('rl_x environment is ready')"
```
