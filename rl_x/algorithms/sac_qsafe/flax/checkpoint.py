"""Strict PyTorch-to-Flax transfer helpers for SAC-QSafe artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from flax import serialization


POLICY_ARTIFACT_FORMAT = "sac_qsafe_flax_policy"
POLICY_ARTIFACT_VERSION = 1


def looks_like_torch_checkpoint(file_path: str | Path) -> bool:
    with open(file_path, "rb") as checkpoint_file:
        prefix = checkpoint_file.read(4)
    # Modern torch.save is a zip; older artifacts use the pickle protocol.
    return prefix.startswith(b"PK") or prefix[:1] == b"\x80"


def _load_torch(file_path: str | Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Loading a PyTorch SAC-QSafe artifact requires torch to be installed"
        ) from exc
    checkpoint = torch.load(str(file_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"PyTorch checkpoint {file_path} is not a mapping")
    return checkpoint


def _as_numpy(value, *, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _strip_compilation_prefix(name: str) -> str:
    while name.startswith("_orig_mod."):
        name = name[len("_orig_mod.") :]
    return name


def _normalized_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {_strip_compilation_prefix(str(key)): value for key, value in state.items()}
    if len(normalized) != len(state):
        raise ValueError("Checkpoint contains duplicate keys after removing compile prefixes")
    return normalized


def _require_exact_keys(state: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(state)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        raise ValueError(
            f"{label} keys do not match: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def _set_dense(
    state: dict[str, Any], dense_name: str, torch_state: Mapping[str, Any], prefix: str
) -> None:
    kernel = _as_numpy(torch_state[f"{prefix}.weight"], dtype=np.float32).T
    bias = _as_numpy(torch_state[f"{prefix}.bias"], dtype=np.float32)
    expected_kernel = np.asarray(state["params"][dense_name]["kernel"]).shape
    expected_bias = np.asarray(state["params"][dense_name]["bias"]).shape
    if kernel.shape != expected_kernel or bias.shape != expected_bias:
        raise ValueError(
            f"{prefix} shape mismatch: expected kernel={expected_kernel}, "
            f"bias={expected_bias}; got kernel={kernel.shape}, bias={bias.shape}"
        )
    state["params"][dense_name]["kernel"] = kernel
    state["params"][dense_name]["bias"] = bias


def convert_torch_policy_params(template_params, torch_state: Mapping[str, Any]):
    """Convert the compiled or eager PyTorch SAC policy into a Flax param tree."""

    torch_state = _normalized_state_dict(torch_state)
    expected = {
        "torso.0.weight",
        "torso.0.bias",
        "torso.2.weight",
        "torso.2.bias",
        "mean.weight",
        "mean.bias",
        "log_std.weight",
        "log_std.bias",
    }
    _require_exact_keys(torch_state, expected, "PyTorch policy")
    state = serialization.to_state_dict(template_params)
    expected_dense = {"Dense_0", "Dense_1", "Dense_2", "Dense_3"}
    if set(state.get("params", {})) != expected_dense:
        raise ValueError(
            "Unexpected Flax policy parameter structure: "
            f"{sorted(state.get('params', {}))}"
        )
    _set_dense(state, "Dense_0", torch_state, "torso.0")
    _set_dense(state, "Dense_1", torch_state, "torso.2")
    _set_dense(state, "Dense_2", torch_state, "mean")
    _set_dense(state, "Dense_3", torch_state, "log_std")
    return serialization.from_state_dict(template_params, state)


def convert_torch_qsafe_params(
    template_params,
    torch_state: Mapping[str, Any],
    expected_observation_indices,
):
    """Convert one PyTorch QSafe online/target network into Flax parameters."""

    torch_state = _normalized_state_dict(torch_state)
    expected = {
        "observation_indices",
        "network.0.weight",
        "network.0.bias",
        "network.2.weight",
        "network.2.bias",
        "network.4.weight",
        "network.4.bias",
    }
    _require_exact_keys(torch_state, expected, "PyTorch QSafe")
    actual_indices = _as_numpy(torch_state["observation_indices"], dtype=np.int64)
    expected_indices = np.asarray(expected_observation_indices, dtype=np.int64)
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError(
            "PyTorch QSafe observation_indices do not match the environment: "
            f"expected {expected_indices.tolist()}, got {actual_indices.tolist()}"
        )
    state = serialization.to_state_dict(template_params)
    expected_dense = {"Dense_0", "Dense_1", "Dense_2"}
    if set(state.get("params", {})) != expected_dense:
        raise ValueError(
            "Unexpected Flax QSafe parameter structure: "
            f"{sorted(state.get('params', {}))}"
        )
    _set_dense(state, "Dense_0", torch_state, "network.0")
    _set_dense(state, "Dense_1", torch_state, "network.2")
    _set_dense(state, "Dense_2", torch_state, "network.4")
    return serialization.from_state_dict(template_params, state)


def _normalizer_state_from_torch(state: Mapping[str, Any]) -> dict[str, Any]:
    required = {"running_mean", "running_var", "count"}
    _require_exact_keys(state, required, "PyTorch observation normalizer")
    return {
        "running_mean": _as_numpy(state["running_mean"], dtype=np.float32),
        "running_var": _as_numpy(state["running_var"], dtype=np.float32),
        "count": _as_numpy(state["count"], dtype=np.float64),
    }


def validate_policy_contract(
    normalizer_state: Mapping[str, Any],
    normalizer_metadata: Mapping[str, Any],
    manifest: Mapping[str, Any],
    observation_size: int,
    action_size: int,
) -> None:
    if int(normalizer_metadata.get("observation_size", -1)) != int(observation_size):
        raise ValueError("Policy normalizer observation size does not match the environment")
    count = float(np.asarray(normalizer_state["count"]).reshape(()))
    if int(normalizer_metadata.get("count", -1)) != int(count):
        raise ValueError("Policy normalizer metadata count does not match its state")
    if not isinstance(manifest, Mapping):
        raise ValueError("Policy checkpoint is missing its environment manifest")
    if int(manifest.get("manifest_version", -1)) < 1:
        raise ValueError("Policy checkpoint has an invalid manifest version")
    observation = manifest.get("observation", {})
    action = manifest.get("action", {})
    if int(observation.get("size", -1)) != int(observation_size):
        raise ValueError("Policy manifest observation size does not match the environment")
    if int(action.get("size", -1)) != int(action_size):
        raise ValueError("Policy manifest action size does not match the environment")
    if manifest.get("normalizer") != dict(normalizer_metadata):
        raise ValueError("Policy manifest normalizer metadata does not match the artifact")


def load_policy_artifact(
    file_path: str | Path,
    template_params,
    observation_size: int,
    action_size: int,
) -> dict[str, Any]:
    """Load either an existing Torch ``policy.model`` or a native Flax sidecar."""

    if looks_like_torch_checkpoint(file_path):
        checkpoint = _load_torch(file_path)
        required = {
            "policy_state_dict",
            "observation_normalizer_state_dict",
            "observation_normalizer_metadata",
            "environment_manifest",
        }
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(f"PyTorch policy checkpoint is missing {sorted(missing)}")
        params = convert_torch_policy_params(
            template_params, checkpoint["policy_state_dict"]
        )
        normalizer_state = _normalizer_state_from_torch(
            checkpoint["observation_normalizer_state_dict"]
        )
        normalizer_metadata = dict(checkpoint["observation_normalizer_metadata"])
        manifest = dict(checkpoint["environment_manifest"])
        log_alpha = (
            float(_as_numpy(checkpoint["log_alpha"]).reshape(()))
            if "log_alpha" in checkpoint
            else None
        )
    else:
        with open(file_path, "rb") as policy_file:
            payload = serialization.msgpack_restore(policy_file.read())
        if not isinstance(payload, Mapping):
            raise ValueError("Native Flax policy artifact is not a mapping")
        if payload.get("format") != POLICY_ARTIFACT_FORMAT:
            raise ValueError(
                "Legacy Flax policy artifacts lack the required normalizer and manifest"
            )
        if int(payload.get("version", -1)) != POLICY_ARTIFACT_VERSION:
            raise ValueError(f"Unsupported Flax policy artifact version: {payload.get('version')}")
        params = serialization.from_state_dict(
            template_params, payload["policy_params"]
        )
        normalizer_state = dict(payload["normalizer_state"])
        normalizer_metadata = dict(payload["normalizer_metadata"])
        manifest = dict(payload["environment_manifest"])
        log_alpha = (
            float(np.asarray(payload["log_alpha"]).reshape(()))
            if "log_alpha" in payload
            else None
        )

    validate_policy_contract(
        normalizer_state,
        normalizer_metadata,
        manifest,
        observation_size,
        action_size,
    )
    return {
        "policy_params": params,
        "normalizer_state": normalizer_state,
        "normalizer_metadata": normalizer_metadata,
        "environment_manifest": manifest,
        "log_alpha": log_alpha,
    }


def make_native_policy_artifact(
    policy_params,
    log_alpha,
    normalizer_state: Mapping[str, Any],
    normalizer_metadata: Mapping[str, Any],
    environment_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "format": POLICY_ARTIFACT_FORMAT,
        "version": POLICY_ARTIFACT_VERSION,
        "policy_params": serialization.to_state_dict(policy_params),
        "normalizer_state": dict(normalizer_state),
        "normalizer_metadata": dict(normalizer_metadata),
        "environment_manifest": dict(environment_manifest),
    }
    if log_alpha is not None:
        payload["log_alpha"] = np.asarray(log_alpha, dtype=np.float32)
    return payload


def load_torch_qsafe_artifact(
    file_path: str | Path,
    online_template,
    target_template,
    expected_observation_indices,
) -> dict[str, Any]:
    checkpoint = _load_torch(file_path)
    required = {"metadata", "online_state_dict", "target_state_dict"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"PyTorch QSafe checkpoint is missing {sorted(missing)}")
    return {
        "metadata": dict(checkpoint["metadata"]),
        "online_params": convert_torch_qsafe_params(
            online_template,
            checkpoint["online_state_dict"],
            expected_observation_indices,
        ),
        "target_params": convert_torch_qsafe_params(
            target_template,
            checkpoint["target_state_dict"],
            expected_observation_indices,
        ),
    }
