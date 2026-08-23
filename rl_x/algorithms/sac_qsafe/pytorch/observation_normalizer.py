"""Running observation statistics shared by policy and both critics."""

from __future__ import annotations

import torch
import torch.nn as nn


class ObservationNormalizer(nn.Module):
    def __init__(self, observation_size: int, enabled: bool = True, epsilon: float = 1e-8):
        super().__init__()
        self.observation_size = int(observation_size)
        self.enabled = bool(enabled)
        self.epsilon = float(epsilon)
        self.register_buffer("running_mean", torch.zeros(1, observation_size))
        self.register_buffer("running_var", torch.ones(1, observation_size))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.frozen = False

    @property
    def running_std(self):
        return torch.sqrt(torch.clamp(self.running_var, min=0.0))

    @torch.no_grad()
    def update(self, observations: torch.Tensor) -> None:
        if not self.enabled or self.frozen:
            return
        observations = observations.detach().reshape(-1, self.observation_size).float()
        if observations.shape[0] == 0:
            return
        batch_count = float(observations.shape[0])
        batch_mean = observations.mean(dim=0, keepdim=True)
        batch_var = observations.var(dim=0, unbiased=False, keepdim=True)
        if self.count.item() == 0:
            self.running_mean.copy_(batch_mean)
            self.running_var.copy_(batch_var)
            self.count.fill_(batch_count)
            return
        old_count = self.count.to(dtype=observations.dtype)
        new_count = old_count + batch_count
        delta = batch_mean - self.running_mean
        new_mean = self.running_mean + delta * batch_count / new_count
        m_a = self.running_var * old_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * old_count * batch_count / new_count
        self.running_mean.copy_(new_mean)
        self.running_var.copy_(m2 / new_count)
        self.count.fill_(float(new_count.item()))

    def normalize(self, observations: torch.Tensor, update: bool = False) -> torch.Tensor:
        observations = observations.float()
        if update:
            self.update(observations)
        if not self.enabled:
            return observations
        return (observations - self.running_mean) / (self.running_std + self.epsilon)

    def freeze(self) -> None:
        self.frozen = True
        self.eval()

    def metadata(self) -> dict[str, object]:
        return {
            "observation_size": self.observation_size,
            "enabled": self.enabled,
            "epsilon": self.epsilon,
            "count": int(self.count.item()),
        }

    def validate_metadata(self, metadata: dict[str, object]) -> None:
        expected = self.metadata()
        for key in ("observation_size", "enabled", "epsilon"):
            if metadata.get(key) != expected[key]:
                raise ValueError(
                    f"Incompatible observation normalizer {key}: "
                    f"expected {expected[key]}, got {metadata.get(key)}"
                )

