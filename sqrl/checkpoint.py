"""Framework-neutral SQRL transfer checkpoints.

The on-disk representation is a standard NumPy ``.npz`` archive.  Dense
kernels use the JAX/Flax ``[input, output]`` layout; PyTorch weights are
transposed while exporting and importing.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


METADATA_KEY = "__metadata__"
PORTABLE_FORMAT = "orsa-portable-npz"


_COMPONENT_LAYER_NAMES = {
    "policy": {
        "torso.0": "Dense_0",
        "torso.2": "Dense_1",
        "mean": "Mean",
        "log_std": "LogStd",
    },
    "qsafe": {
        "critic.0": "Dense_0",
        "critic.2": "Dense_1",
        "critic.4": "Dense_2",
    },
}


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _normalized_state_key(state_key: str) -> str:
    parts = [part for part in state_key.split(".") if part != "_orig_mod"]
    return ".".join(parts)


def _portable_parameter_key(component: str, state_key: str) -> str:
    normalized = _normalized_state_key(state_key)
    module_name, separator, parameter_name = normalized.rpartition(".")
    if not separator:
        module_name, parameter_name = "Root", normalized
    layer_name = _COMPONENT_LAYER_NAMES.get(component, {}).get(
        module_name, module_name.replace(".", "_")
    )
    if parameter_name == "weight":
        parameter_name = "kernel"
    return f"{component}/{layer_name}/{parameter_name}"


def torch_state_to_arrays(component: str, state_dict: Mapping[str, Any]):
    """Convert a PyTorch state dict to portable NumPy arrays.

    This function imports torch only through the tensor objects supplied by the
    caller, so reading the resulting archive never requires PyTorch.
    """

    arrays = {}
    for state_key, tensor in state_dict.items():
        value = tensor.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
        if _normalized_state_key(state_key).endswith(".weight") or (
            _normalized_state_key(state_key) == "weight"
        ):
            if value.ndim == 2:
                value = value.T
        portable_key = _portable_parameter_key(component, state_key)
        if portable_key in arrays:
            raise ValueError(f"duplicate portable parameter key {portable_key!r}")
        arrays[portable_key] = np.ascontiguousarray(value)
    return arrays


def load_torch_component(module, arrays: Mapping[str, np.ndarray], component: str):
    """Load one portable component into a PyTorch module strictly."""

    import torch

    converted = {}
    missing = []
    for state_key, reference in module.state_dict().items():
        portable_key = _portable_parameter_key(component, state_key)
        if portable_key not in arrays:
            missing.append(portable_key)
            continue
        value = np.asarray(arrays[portable_key])
        if _normalized_state_key(state_key).endswith(".weight") or (
            _normalized_state_key(state_key) == "weight"
        ):
            if value.ndim == 2:
                value = value.T
        converted[state_key] = torch.as_tensor(
            np.ascontiguousarray(value), dtype=reference.dtype
        )
    if missing:
        raise ValueError(
            f"portable checkpoint is missing {component} parameters: "
            + ", ".join(sorted(missing))
        )
    module.load_state_dict(converted, strict=True)


def flax_params(arrays: Mapping[str, np.ndarray], component: str):
    """Return a Flax-compatible ``{'params': ...}`` tree of NumPy arrays.

    JAX callers may pass the result directly to ``Module.apply``; JAX will
    device-transfer the NumPy arrays as needed.
    """

    prefix = f"{component}/"
    params = {}
    for key, value in arrays.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix) :].split("/")
        if len(path) != 2:
            raise ValueError(f"invalid portable parameter key {key!r}")
        layer_name, parameter_name = path
        params.setdefault(layer_name, {})[parameter_name] = np.asarray(value)
    if not params:
        raise ValueError(f"portable checkpoint has no {component!r} parameters")
    return {"params": params}


def save_portable_checkpoint(
    path: str | os.PathLike[str],
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
):
    """Atomically save metadata and arrays as a portable NPZ checkpoint."""

    target = Path(path)
    if target.suffix.lower() != ".npz":
        raise ValueError("portable checkpoints must use the .npz extension")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded_metadata = json.dumps(
        dict(metadata),
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    payload = {key: np.asarray(value) for key, value in arrays.items()}
    if METADATA_KEY in payload:
        raise ValueError(f"{METADATA_KEY!r} is reserved for checkpoint metadata")
    payload[METADATA_KEY] = np.frombuffer(encoded_metadata, dtype=np.uint8)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as checkpoint_file:
            temporary_path = Path(checkpoint_file.name)
            np.savez_compressed(checkpoint_file, **payload)
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_portable_checkpoint(path: str | os.PathLike[str]):
    """Read a portable checkpoint without pickle or framework dependencies."""

    with np.load(path, allow_pickle=False) as archive:
        if METADATA_KEY not in archive.files:
            raise ValueError("portable checkpoint is missing metadata")
        try:
            metadata = json.loads(
                np.asarray(archive[METADATA_KEY], dtype=np.uint8).tobytes()
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("portable checkpoint metadata is invalid") from error
        arrays = {
            key: np.array(archive[key], copy=True)
            for key in archive.files
            if key != METADATA_KEY
        }
    if not isinstance(metadata, dict):
        raise ValueError("portable checkpoint metadata must be a mapping")
    return metadata, arrays
