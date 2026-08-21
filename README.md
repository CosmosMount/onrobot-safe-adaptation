# onrobot-safe-adaptation

Aiming to build up a on-robot training system enabling real-time adaptation with pretrained policy.

## Environment Setup

Our algorithm implementation comes from [RL-X](https://github.com/nico-bohlinger/RL-X).

This repository is configured for three runtime targets:

- Isaac Lab simulation;
- native MuJoCo simulation;
- JAX/Flax-based real-robot inference.

The project does not depend on Gymnasium or PyTorch. The JAX/Flax packages are
installed through the `inference` extra because the current model
implementations use Flax modules and checkpoint utilities. Isaac Lab is not a
PyPI dependency: its Python, CUDA, PyTorch, and Omniverse versions must match
the Isaac Lab release used by the project.

### 1. Create the runtime environment

Use the Python version required by your Isaac Lab release. Python 3.10 or 3.11
is typical, but the Isaac Lab release is authoritative. If Isaac Lab provides
an environment creation script, use that script first. Otherwise:

```bash
conda create -n onrobot-safe-adaptation python=3.10 -y
conda activate onrobot-safe-adaptation
```

### 2. Install Isaac Lab (when simulation is needed)

Install Isaac Lab using the instructions bundled with the exact Isaac Lab
release and CUDA driver stack. For example, from an Isaac Lab checkout, use its
provided `isaaclab.sh` setup command rather than installing a random PyTorch or
Isaac Sim version with pip.

After activating the Isaac Lab environment, install this repository:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[mujoco,inference]"
```

Isaac Lab may provide PyTorch internally for simulation. This project does not
declare PyTorch as a dependency, and the JAX policy should use the JAX device
configuration appropriate for the target machine.

For CPU inference, the regular `jax` package installed by the inference extra
is sufficient. For NVIDIA GPU inference, install the JAX wheel matching the
CUDA stack before installing the project, for example:

```bash
python -m pip install -U "jax[cuda12]"
python -m pip install -e ".[mujoco,inference]"
```

Use the JAX installation command for the CUDA version supported by the machine.

### 3. MuJoCo simulation

The `mujoco` extra installs the native Python MuJoCo bindings. Install it with
the JAX inference stack in the same environment:

```bash
python -m pip install -e ".[mujoco,inference]"
```

No system MuJoCo installation is required for the standard pip bindings. A
custom MuJoCo environment should expose observations and actions through the
JAX interface expected by the selected policy.

### 4. JAX inference on the real robot

The robot driver, transport layer, and hardware-specific environment are not
part of this repository and must be installed into the same environment by the
robot project. Typical examples include a vendor SDK, ROS package, or a local
Python package. The package must be importable before creating `Runner`:

```bash
# Example only; replace with the robot project's package/install command.
python -m pip install -e /path/to/robot-environment
python -c "import robot_environment; print('robot environment is ready')"
```

When the environment or algorithm is registered outside `rl_x`, pass its
top-level package in the runner:

```python
from rl_x.runner.runner import Runner

Runner(implementation_package_names=["rl_x", "robot_environment"])
```

Because the repository does not bundle a robot or MuJoCo environment, always
provide the registered environment name, for example:

```bash
python your_inference_entrypoint.py \
  --algorithm.name=ppo.flax \
  --environment.name=robot_environment.mujoco_task
```

### Verify the environment

For the common JAX/MuJoCo path:

```bash
python -m pip check
python -c "import jax, flax, mujoco; import rl_x; print('JAX/MuJoCo environment is ready')"
```

For Isaac Lab, run the release-specific smoke test from its installation guide
and then verify that the package is importable:

```bash
python -c "import isaaclab; print('Isaac Lab environment is ready')"
```

The current RL-X Flax modules import `wandb` even when experiment tracking is
disabled, so it remains a compatibility dependency. No W&B login is needed for
local inference when tracking is not enabled.
