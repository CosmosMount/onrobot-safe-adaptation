"""Generic atomic directory checkpoints for PyTorch safe-RL algorithms."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch


MANIFEST_FILE = "manifest.json"


def restore_parameter_(parameter: torch.nn.Parameter, value: torch.Tensor) -> None:
    """Restore a parameter value without breaking optimizer object identity."""

    with torch.no_grad():
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def save_checkpoint_bundle(
    directory: str | os.PathLike[str],
    manifest: dict[str, Any],
    artifacts: dict[str, Any],
) -> Path:
    destination = Path(directory).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        (temporary / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        for file_name, payload in artifacts.items():
            if Path(file_name).name != file_name:
                raise ValueError(f"Checkpoint artifact must be a file name: {file_name}")
            torch.save(payload, temporary / file_name)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_checkpoint_bundle(
    directory: str | os.PathLike[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(directory).resolve()
    manifest = json.loads((source / MANIFEST_FILE).read_text(encoding="utf-8"))
    artifacts = {
        path.name: torch.load(path, map_location="cpu", weights_only=False)
        for path in source.glob("*.pt")
    }
    return manifest, artifacts
