"""Framework-neutral running observation statistics for Flax SAC-QSafe.

The state intentionally mirrors the PyTorch implementation so a policy trained
in another backend can be transferred without changing its input distribution.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np


class ObservationNormalizer:
    def __init__(
        self, observation_size: int, enabled: bool = True, epsilon: float = 1e-8
    ):
        self.observation_size = int(observation_size)
        self.enabled = bool(enabled)
        self.epsilon = float(epsilon)
        self.running_mean = jnp.zeros((1, self.observation_size), dtype=jnp.float32)
        self.running_var = jnp.ones((1, self.observation_size), dtype=jnp.float32)
        self.count = 0.0
        self.frozen = False

    @property
    def running_std(self):
        return jnp.sqrt(jnp.maximum(self.running_var, 0.0))

    def update(self, observations) -> None:
        if not self.enabled or self.frozen:
            return
        values = np.asarray(observations, dtype=np.float32).reshape(
            -1, self.observation_size
        )
        if values.shape[0] == 0:
            return
        batch_count = float(values.shape[0])
        batch_mean = values.mean(axis=0, keepdims=True, dtype=np.float32)
        batch_var = values.var(axis=0, keepdims=True, dtype=np.float32)
        if self.count == 0:
            self.running_mean = jnp.asarray(batch_mean)
            self.running_var = jnp.asarray(batch_var)
            self.count = batch_count
            return

        old_mean = np.asarray(self.running_mean)
        old_var = np.asarray(self.running_var)
        old_count = np.float32(self.count)
        new_count = old_count + np.float32(batch_count)
        delta = batch_mean - old_mean
        new_mean = old_mean + delta * np.float32(batch_count) / new_count
        m_a = old_var * old_count
        m_b = batch_var * np.float32(batch_count)
        m2 = (
            m_a
            + m_b
            + np.square(delta) * old_count * np.float32(batch_count) / new_count
        )
        self.running_mean = jnp.asarray(new_mean, dtype=jnp.float32)
        self.running_var = jnp.asarray(m2 / new_count, dtype=jnp.float32)
        self.count = float(new_count)

    def normalize(self, observations, update: bool = False):
        if update:
            self.update(observations)
        values = jnp.asarray(observations, dtype=jnp.float32)
        if not self.enabled:
            return values
        return (values - self.running_mean) / (self.running_std + self.epsilon)

    def freeze(self) -> None:
        self.frozen = True

    def parameters(self):
        """Return array arguments suitable for passing through a JIT boundary."""

        if not self.enabled:
            return (
                jnp.zeros_like(self.running_mean),
                jnp.ones_like(self.running_var),
                jnp.asarray(0.0, dtype=jnp.float32),
            )
        return (
            self.running_mean,
            self.running_std,
            jnp.asarray(self.epsilon, dtype=jnp.float32),
        )

    def metadata(self) -> dict[str, object]:
        return {
            "observation_size": self.observation_size,
            "enabled": self.enabled,
            "epsilon": self.epsilon,
            "count": int(self.count),
        }

    def validate_metadata(self, metadata: Mapping[str, Any]) -> None:
        expected = self.metadata()
        for key in ("observation_size", "enabled", "epsilon"):
            if metadata.get(key) != expected[key]:
                raise ValueError(
                    f"Incompatible observation normalizer {key}: "
                    f"expected {expected[key]}, got {metadata.get(key)}"
                )
        metadata_count = int(metadata.get("count", -1))
        if self.count and metadata_count != int(self.count):
            raise ValueError(
                "Incompatible observation normalizer count: "
                f"expected {int(self.count)}, got {metadata_count}"
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "running_mean": np.asarray(self.running_mean, dtype=np.float32),
            "running_var": np.asarray(self.running_var, dtype=np.float32),
            "count": np.asarray(self.count, dtype=np.float64),
        }

    def load_state_dict(
        self, state: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
    ) -> None:
        required = {"running_mean", "running_var", "count"}
        missing = required.difference(state)
        if missing:
            raise ValueError(f"Observation normalizer state is missing {sorted(missing)}")
        mean = np.asarray(state["running_mean"], dtype=np.float32)
        variance = np.asarray(state["running_var"], dtype=np.float32)
        expected_shape = (1, self.observation_size)
        if mean.shape != expected_shape or variance.shape != expected_shape:
            raise ValueError(
                "Observation normalizer shape mismatch: "
                f"expected {expected_shape}, got mean={mean.shape}, var={variance.shape}"
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
            raise ValueError("Observation normalizer contains NaN or infinity")
        if np.any(variance < 0):
            raise ValueError("Observation normalizer variance must be non-negative")
        count = float(np.asarray(state["count"]).reshape(()))
        if not np.isfinite(count) or count < 0 or not count.is_integer():
            raise ValueError(f"Invalid observation normalizer count: {count}")
        self.running_mean = jnp.asarray(mean)
        self.running_var = jnp.asarray(variance)
        self.count = count
        if metadata is not None:
            self.validate_metadata(metadata)
