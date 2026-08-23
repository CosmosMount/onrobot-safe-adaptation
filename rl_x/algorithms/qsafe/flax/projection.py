"""Resolve optional environment action projection across host/JAX boundaries."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def identity_project_actions(states, actions):
    del states
    return actions


def resolve_action_projectors(env):
    """Return ``(jax_projector, host_projector, is_jax_compatible)``.

    Environments may explicitly set ``project_actions_jax_compatible``. Without
    that marker we use ``jax.eval_shape`` to conservatively probe the documented
    side-effect-free hook. A NumPy-only hook remains available for rollout, but
    cannot be used in differentiable learner kernels.
    """

    projector = getattr(env, "project_actions", None)
    if projector is None:
        return identity_project_actions, None, True

    marker = getattr(env, "project_actions_jax_compatible", None)
    if marker is False:
        return identity_project_actions, projector, False
    if marker is True:
        return projector, projector, True

    observation_shape = tuple(env.single_observation_space.shape)
    action_shape = tuple(env.single_action_space.shape)
    dummy_states = jnp.zeros((1,) + observation_shape, dtype=jnp.float32)
    dummy_actions = jnp.zeros((1,) + action_shape, dtype=jnp.float32)
    try:
        output = jax.eval_shape(projector, dummy_states, dummy_actions)
        if tuple(output.shape) != tuple(dummy_actions.shape):
            raise ValueError(
                "project_actions must preserve action shape: "
                f"expected {dummy_actions.shape}, got {output.shape}"
            )
    except Exception:
        return identity_project_actions, projector, False
    return projector, projector, True

