"""Native Flax CrossQ policy artifacts with BatchRenorm state."""

from __future__ import annotations

from pathlib import Path

from flax import serialization

from rl_x.algorithms.sac_qsafe.flax.checkpoint import (
    looks_like_torch_checkpoint,
    validate_policy_contract,
)


POLICY_ARTIFACT_FORMAT = "crossq_qsafe_flax_policy"
POLICY_ARTIFACT_VERSION = 1


def make_policy_artifact(
    params,
    batch_stats,
    normalizer_state,
    normalizer_metadata,
    environment_manifest,
):
    return {
        "format": POLICY_ARTIFACT_FORMAT,
        "version": POLICY_ARTIFACT_VERSION,
        "policy_params": serialization.to_state_dict(params),
        "policy_batch_stats": serialization.to_state_dict(batch_stats),
        "normalizer_state": normalizer_state,
        "normalizer_metadata": normalizer_metadata,
        "environment_manifest": environment_manifest,
    }


def save_policy_artifact(file_path, artifact):
    with open(file_path, "wb") as policy_file:
        policy_file.write(serialization.msgpack_serialize(artifact))


def load_policy_artifact(
    file_path: str | Path,
    params_template,
    batch_stats_template,
    observation_size,
    action_size,
):
    """Load a native policy and explicitly reject unconverted Torch BN state."""

    if looks_like_torch_checkpoint(file_path):
        raise ValueError(
            "PyTorch CrossQ-QSafe policy transfer is not supported: BatchRenorm "
            "parameters and running statistics require an explicit Torch-to-Flax "
            "conversion. Use a native crossq_qsafe.flax policy artifact."
        )
    with open(file_path, "rb") as policy_file:
        artifact = serialization.msgpack_restore(policy_file.read())
    if artifact.get("format") != POLICY_ARTIFACT_FORMAT:
        raise ValueError(
            "Not a native Flax CrossQ-QSafe policy artifact: "
            f"{artifact.get('format')!r}"
        )
    if int(artifact.get("version", -1)) != POLICY_ARTIFACT_VERSION:
        raise ValueError(
            "Unsupported CrossQ-QSafe policy artifact version: "
            f"{artifact.get('version')!r}"
        )
    required = {
        "policy_params",
        "policy_batch_stats",
        "normalizer_state",
        "normalizer_metadata",
        "environment_manifest",
    }
    missing = required.difference(artifact)
    if missing:
        raise ValueError(
            f"CrossQ-QSafe policy artifact is missing {sorted(missing)}"
        )
    validate_policy_contract(
        artifact["normalizer_state"],
        artifact["normalizer_metadata"],
        artifact["environment_manifest"],
        observation_size,
        action_size,
    )
    return {
        "params": serialization.from_state_dict(
            params_template, artifact["policy_params"]
        ),
        "batch_stats": serialization.from_state_dict(
            batch_stats_template, artifact["policy_batch_stats"]
        ),
        "normalizer_state": artifact["normalizer_state"],
        "normalizer_metadata": artifact["normalizer_metadata"],
        "environment_manifest": artifact["environment_manifest"],
    }
