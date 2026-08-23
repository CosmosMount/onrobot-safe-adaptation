import importlib

import numpy as np
import pytest


def test_isaac_environment_registers_without_importing_isaaclab():
    module = importlib.import_module("src.environments.go2_sqrl.isaac_lab")
    assert module.GO2_SQRL_ISAAC_LAB == "go2_sqrl.isaac_lab"


def test_isaac_launcher_string_defaults_are_not_none():
    from src.environments.go2_sqrl.isaac_lab.default_config import get_config

    config = get_config("go2_sqrl.isaac_lab")
    assert config.experience == ""
    assert config.kit_args == ""
    assert isinstance(config.rendering_mode, str)


def test_isaac_joint_gather_uses_sdk_order():
    torch = pytest.importorskip("torch")
    from src.environments.go2_sqrl.common.specs import JOINT_NAMES
    from src.environments.go2_sqrl.isaac_lab.mdp import sdk_joint_indices

    source = [f"{name}_joint" for name in reversed(JOINT_NAMES)]
    indices = sdk_joint_indices(source)
    torch.testing.assert_close(indices, torch.arange(11, -1, -1))
