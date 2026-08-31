"""Distribution helpers shared by Flax SAC-QSafe kernels and parity tests."""

from __future__ import annotations

import jax.numpy as jnp
import jax.nn as jnn


def squashed_gaussian_log_probability(
    pretanh,
    mean,
    log_std,
):
    """Return ``log pi(tanh(z) | state)`` over action dimensions.

    The tanh Jacobian is defined by the policy sample, before any environment
    action projection. A rate/limit projector changes the action evaluated by
    Q, but is not part of the policy's invertible tanh transform.

    The softplus identity for ``log(1 - tanh(z) ** 2)`` remains finite and
    differentiable after ``tanh(z)`` itself has rounded to ``+/-1``. This is
    important when a temporarily inaccurate critic pushes a policy toward
    saturated actions.
    """

    std = jnp.exp(log_std)
    log_probability = (
        -0.5 * ((pretanh - mean) / std) ** 2
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - log_std
    )
    tanh_log_abs_det_jacobian = 2.0 * (
        jnp.log(2.0) - pretanh - jnn.softplus(-2.0 * pretanh)
    )
    log_probability -= tanh_log_abs_det_jacobian
    return jnp.sum(log_probability, axis=-1)
