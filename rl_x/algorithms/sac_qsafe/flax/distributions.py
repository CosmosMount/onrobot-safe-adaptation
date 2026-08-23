"""Distribution helpers shared by Flax SAC-QSafe kernels and parity tests."""

from __future__ import annotations

import jax.numpy as jnp


def squashed_gaussian_log_probability(
    pretanh,
    mean,
    log_std,
    *,
    epsilon: float = 1e-6,
):
    """Return ``log pi(tanh(z) | state)`` over action dimensions.

    The tanh Jacobian is defined by the policy sample, before any environment
    action projection. A rate/limit projector changes the action evaluated by
    Q, but is not part of the policy's invertible tanh transform.
    """

    std = jnp.exp(log_std)
    log_probability = (
        -0.5 * ((pretanh - mean) / std) ** 2
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - log_std
    )
    squashed_action = jnp.tanh(pretanh)
    log_probability -= jnp.log(1.0 - squashed_action**2 + epsilon)
    return jnp.sum(log_probability, axis=-1)
