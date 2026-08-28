from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


try:
    import jax
    import jax.numpy as jnp
    import flax  # noqa: F401
except Exception as exc:  # osa may contain an incompatible JAX/NumPy pair.
    pytest.skip(f"JAX runtime is unavailable: {exc}", allow_module_level=True)

torch = pytest.importorskip("torch")
import torch.nn.functional as torch_functional

from gymnasium.spaces import Box

from rl_x.algorithms.qsafe.flax.safety_critic import SafetyQNetwork
from rl_x.algorithms.sac.flax.policy import get_policy
from rl_x.algorithms.sac_qsafe.flax import sac_qsafe as sac_qsafe_module
from rl_x.algorithms.sac_qsafe.flax.sac_qsafe import SAC_QSafe
from rl_x.algorithms.sac_qsafe.flax.checkpoint import (
    load_policy_artifact,
    load_torch_qsafe_artifact,
)
from rl_x.algorithms.sac_qsafe.flax.distributions import (
    squashed_gaussian_log_probability,
)
from rl_x.algorithms.sac_qsafe.flax.observation_normalizer import (
    ObservationNormalizer,
)
from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


OBSERVATION_SIZE = 46
ACTION_SIZE = 12
HIDDEN_SIZE = 256


class _Properties:
    action_space_type = ActionSpaceType.CONTINUOUS
    observation_space_type = ObservationSpaceType.FLAT_VALUES


class _Environment:
    general_properties = _Properties
    single_observation_space = Box(
        -np.inf, np.inf, shape=(OBSERVATION_SIZE,), dtype=np.float32
    )
    single_action_space = Box(
        -1.0, 1.0, shape=(ACTION_SIZE,), dtype=np.float32
    )
    policy_observation_indices = np.arange(OBSERVATION_SIZE)


def _policy_and_template():
    config = SimpleNamespace(
        algorithm=SimpleNamespace(
            log_std_min=-20.0,
            log_std_max=2.0,
            nr_hidden_units=HIDDEN_SIZE,
        )
    )
    policy, _ = get_policy(config, _Environment())
    params = policy.init(
        jax.random.PRNGKey(0), jnp.zeros((1, OBSERVATION_SIZE), dtype=jnp.float32)
    )
    return policy, params


def _tensor(values):
    return torch.as_tensor(np.asarray(values, dtype=np.float32))


def _linear_state(out_features, in_features, offset):
    weight = (
        np.arange(out_features * in_features, dtype=np.float32).reshape(
            out_features, in_features
        )
        * np.float32(1e-5)
        + np.float32(offset)
    )
    bias = np.linspace(-0.1, 0.1, out_features, dtype=np.float32) + np.float32(
        offset
    )
    return _tensor(weight), _tensor(bias)


def _torch_policy_state():
    first_weight, first_bias = _linear_state(HIDDEN_SIZE, OBSERVATION_SIZE, 0.01)
    second_weight, second_bias = _linear_state(HIDDEN_SIZE, HIDDEN_SIZE, -0.02)
    mean_weight, mean_bias = _linear_state(ACTION_SIZE, HIDDEN_SIZE, 0.03)
    log_std_weight, log_std_bias = _linear_state(
        ACTION_SIZE, HIDDEN_SIZE, -0.04
    )
    return {
        "_orig_mod.torso.0.weight": first_weight,
        "_orig_mod.torso.0.bias": first_bias,
        "_orig_mod.torso.2.weight": second_weight,
        "_orig_mod.torso.2.bias": second_bias,
        "_orig_mod.mean.weight": mean_weight,
        "_orig_mod.mean.bias": mean_bias,
        "_orig_mod.log_std.weight": log_std_weight,
        "_orig_mod.log_std.bias": log_std_bias,
    }


def _manifest(normalizer_metadata):
    return {
        "manifest_version": 1,
        "observation": {"version": "test", "size": OBSERVATION_SIZE},
        "action": {"version": "test", "size": ACTION_SIZE},
        "normalizer": dict(normalizer_metadata),
    }


def test_torch_policy_transfer_matches_fixed_input_and_freezes_normalizer(tmp_path):
    policy, template = _policy_and_template()
    torch_state = _torch_policy_state()
    normalizer_metadata = {
        "observation_size": OBSERVATION_SIZE,
        "enabled": True,
        "epsilon": 1e-8,
        "count": 8,
    }
    running_mean = np.linspace(-0.5, 0.5, OBSERVATION_SIZE, dtype=np.float32)[
        None, :
    ]
    running_var = np.linspace(0.5, 1.5, OBSERVATION_SIZE, dtype=np.float32)[
        None, :
    ]
    checkpoint_path = tmp_path / "policy.model"
    torch.save(
        {
            "policy_state_dict": torch_state,
            "log_alpha": torch.tensor(-8.0),
            "observation_normalizer_state_dict": {
                "running_mean": _tensor(running_mean),
                "running_var": _tensor(running_var),
                "count": torch.tensor(8.0, dtype=torch.float64),
            },
            "observation_normalizer_metadata": normalizer_metadata,
            "environment_manifest": _manifest(normalizer_metadata),
        },
        checkpoint_path,
    )

    artifact = load_policy_artifact(
        checkpoint_path, template, OBSERVATION_SIZE, ACTION_SIZE
    )
    assert artifact["log_alpha"] == pytest.approx(-8.0)
    inputs = np.linspace(
        -1.0, 1.0, 3 * OBSERVATION_SIZE, dtype=np.float32
    ).reshape(3, OBSERVATION_SIZE)
    torch_inputs = _tensor(inputs)
    hidden = torch_functional.relu(
        torch_functional.linear(
            torch_inputs,
            torch_state["_orig_mod.torso.0.weight"],
            torch_state["_orig_mod.torso.0.bias"],
        )
    )
    hidden = torch_functional.relu(
        torch_functional.linear(
            hidden,
            torch_state["_orig_mod.torso.2.weight"],
            torch_state["_orig_mod.torso.2.bias"],
        )
    )
    torch_mean = torch_functional.linear(
        hidden,
        torch_state["_orig_mod.mean.weight"],
        torch_state["_orig_mod.mean.bias"],
    )
    torch_log_std = torch_functional.linear(
        hidden,
        torch_state["_orig_mod.log_std.weight"],
        torch_state["_orig_mod.log_std.bias"],
    ).clamp(-20.0, 2.0)
    flax_mean, flax_log_std = policy.apply(
        artifact["policy_params"], jnp.asarray(inputs)
    )
    np.testing.assert_allclose(np.asarray(flax_mean), torch_mean.numpy(), atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(flax_log_std), torch_log_std.numpy(), atol=1e-5
    )

    normalizer = ObservationNormalizer(OBSERVATION_SIZE)
    normalizer.load_state_dict(
        artifact["normalizer_state"], artifact["normalizer_metadata"]
    )
    normalizer.freeze()
    before = normalizer.state_dict()
    normalizer.normalize(np.full((1, OBSERVATION_SIZE), 100.0), update=True)
    after = normalizer.state_dict()
    np.testing.assert_array_equal(before["running_mean"], after["running_mean"])
    np.testing.assert_array_equal(before["running_var"], after["running_var"])
    assert normalizer.metadata()["count"] == 8


def test_finetune_loader_ignores_pretrain_alpha_and_task_learner(monkeypatch):
    class State:
        def __init__(self, params):
            self.params = params

        def replace(self, **changes):
            return State(changes.get("params", self.params))

    class Normalizer:
        def __init__(self):
            self.loaded = None
            self.frozen = False

        def load_state_dict(self, state, metadata):
            self.loaded = (state, metadata)

        def metadata(self):
            return {"count": 12}

        def freeze(self):
            self.frozen = True

    class TransferEnvironment(_Environment):
        def __init__(self):
            self.validated = None

        def validate_transfer_checkpoint_manifest(self, manifest, metadata):
            self.validated = (manifest, metadata)

    transferred_actor = {"actor": "pretrained"}
    artifact = {
        "policy_params": transferred_actor,
        "normalizer_state": {"mean": "pretrained"},
        "normalizer_metadata": {"count": 12},
        "environment_manifest": {"manifest": "pretrained"},
        # Legacy policy sidecars may contain this; SQRL target-task transfer
        # intentionally ignores it and starts entropy optimization from alpha_init.
        "log_alpha": -8.0,
    }
    monkeypatch.setattr(
        sac_qsafe_module, "load_policy_artifact", lambda *args, **kwargs: artifact
    )

    model = object.__new__(SAC_QSafe)
    model.policy_state = State({"actor": "fresh"})
    model.entropy_coefficient_state = State({"alpha": "fresh"})
    model.critic_state = object()
    fresh_entropy_state = model.entropy_coefficient_state
    fresh_critic_state = model.critic_state
    model.observation_normalizer = Normalizer()
    model.train_env = TransferEnvironment()

    model._load_pretrained_policy("policy.model")

    assert model.policy_state.params is transferred_actor
    assert model.entropy_coefficient_state is fresh_entropy_state
    assert model.critic_state is fresh_critic_state
    assert model.observation_normalizer.loaded == (
        artifact["normalizer_state"],
        artifact["normalizer_metadata"],
    )
    assert model.observation_normalizer.frozen is True
    assert model.train_env.validated == (
        artifact["environment_manifest"],
        {"count": 12},
    )


def _torch_qsafe_state(offset):
    first_weight, first_bias = _linear_state(
        HIDDEN_SIZE, OBSERVATION_SIZE + ACTION_SIZE, offset
    )
    second_weight, second_bias = _linear_state(
        HIDDEN_SIZE, HIDDEN_SIZE, offset + 0.01
    )
    output_weight, output_bias = _linear_state(1, HIDDEN_SIZE, offset - 0.01)
    return {
        "observation_indices": torch.arange(OBSERVATION_SIZE),
        "network.0.weight": first_weight,
        "network.0.bias": first_bias,
        "network.2.weight": second_weight,
        "network.2.bias": second_bias,
        "network.4.weight": output_weight,
        "network.4.bias": output_bias,
    }


def test_torch_qsafe_transfer_matches_fixed_input(tmp_path):
    network = SafetyQNetwork(list(range(OBSERVATION_SIZE)), HIDDEN_SIZE)
    template = network.init(
        jax.random.PRNGKey(1),
        jnp.zeros((1, OBSERVATION_SIZE), dtype=jnp.float32),
        jnp.zeros((1, ACTION_SIZE), dtype=jnp.float32),
    )
    online = _torch_qsafe_state(0.01)
    target = _torch_qsafe_state(-0.02)
    checkpoint_path = tmp_path / "qsafe.model"
    torch.save(
        {
            "metadata": {
                "checkpoint_version": 1,
                "observation_shape": (OBSERVATION_SIZE,),
                "action_shape": (ACTION_SIZE,),
                "observation_indices": list(range(OBSERVATION_SIZE)),
                "nr_hidden_units": HIDDEN_SIZE,
                "gamma": 0.7,
                "epsilon": 0.1,
            },
            "online_state_dict": online,
            "target_state_dict": target,
        },
        checkpoint_path,
    )
    artifact = load_torch_qsafe_artifact(
        checkpoint_path, template, template, np.arange(OBSERVATION_SIZE)
    )

    observations = np.linspace(
        -0.75, 0.75, 2 * OBSERVATION_SIZE, dtype=np.float32
    ).reshape(2, OBSERVATION_SIZE)
    actions = np.linspace(-0.5, 0.5, 2 * ACTION_SIZE, dtype=np.float32).reshape(
        2, ACTION_SIZE
    )
    torch_values = torch.cat((_tensor(observations), _tensor(actions)), dim=-1)
    torch_values = torch_functional.relu(
        torch_functional.linear(
            torch_values, online["network.0.weight"], online["network.0.bias"]
        )
    )
    torch_values = torch_functional.relu(
        torch_functional.linear(
            torch_values, online["network.2.weight"], online["network.2.bias"]
        )
    )
    torch_values = torch.tanh(
        torch_functional.linear(
            torch_values, online["network.4.weight"], online["network.4.bias"]
        )
    )
    flax_values = network.apply(
        artifact["online_params"], jnp.asarray(observations), jnp.asarray(actions)
    )
    np.testing.assert_allclose(
        np.asarray(flax_values), torch_values.numpy(), atol=1e-5
    )


def test_policy_transfer_rejects_manifest_normalizer_mismatch(tmp_path):
    _, template = _policy_and_template()
    metadata = {
        "observation_size": OBSERVATION_SIZE,
        "enabled": True,
        "epsilon": 1e-8,
        "count": 1,
    }
    manifest_metadata = dict(metadata)
    manifest_metadata["count"] = 2
    checkpoint_path = tmp_path / "bad-policy.model"
    torch.save(
        {
            "policy_state_dict": _torch_policy_state(),
            "observation_normalizer_state_dict": {
                "running_mean": torch.zeros(1, OBSERVATION_SIZE),
                "running_var": torch.ones(1, OBSERVATION_SIZE),
                "count": torch.tensor(1.0, dtype=torch.float64),
            },
            "observation_normalizer_metadata": metadata,
            "environment_manifest": _manifest(manifest_metadata),
        },
        checkpoint_path,
    )
    with pytest.raises(ValueError, match="manifest normalizer"):
        load_policy_artifact(
            checkpoint_path, template, OBSERVATION_SIZE, ACTION_SIZE
        )


def test_squashed_gaussian_log_probability_matches_torch_before_projection():
    mean = np.asarray([[0.2, -0.4, 0.1]], dtype=np.float32)
    log_std = np.asarray([[-0.3, 0.25, -0.7]], dtype=np.float32)
    pretanh = np.asarray([[0.9, -1.1, 0.35]], dtype=np.float32)

    torch_mean = torch.from_numpy(mean)
    torch_log_std = torch.from_numpy(log_std)
    torch_pretanh = torch.from_numpy(pretanh)
    torch_squashed = torch.tanh(torch_pretanh)
    expected = torch.distributions.Normal(
        torch_mean, torch_log_std.exp()
    ).log_prob(torch_pretanh)
    expected -= torch.log(1.0 - torch_squashed.pow(2) + 1e-6)
    expected = expected.sum(dim=-1).numpy()

    actual = np.asarray(
        squashed_gaussian_log_probability(pretanh, mean, log_std)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    # A rate limiter may project the Q-evaluated action elsewhere; that value
    # must not replace tanh(pretanh) in the policy Jacobian correction.
    projected_action = np.zeros_like(pretanh)
    gaussian_term = (
        -0.5 * ((pretanh - mean) / np.exp(log_std)) ** 2
        - 0.5 * np.log(2.0 * np.pi)
        - log_std
    )
    wrong_projected_correction = np.sum(
        gaussian_term - np.log(1.0 - projected_action**2 + 1e-6), axis=-1
    )
    assert not np.allclose(actual, wrong_projected_correction)
