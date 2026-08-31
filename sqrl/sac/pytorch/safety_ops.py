"""Pure SQRL safety operations shared by every training phase."""

from __future__ import annotations

import math

import torch


def select_safe_candidate_indices(
    safe_q: torch.Tensor,
    epsilon_safe: float,
    *,
    selection: str = "uniform",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select an accepted iid policy sample per batch row.

    ``selection="uniform"`` implements the fine-tuning policy in Eq. 3.
    Candidate actions are already iid draws from the task policy, so uniform
    selection among accepted candidates is finite-sample rejection sampling.
    Conditional on at least one acceptance, the selected action has the task
    policy restricted to the safe set; policy-density weighting here would
    incorrectly produce a pi-squared bias.

    ``selection="boundary"`` implements SQRL's pre-training exploration rule:
    select the accepted candidate immediately below epsilon with the largest
    predicted QSafe value.

    If a row contains no accepted candidate, SQRL's practical fallback selects
    the candidate with the smallest predicted failure probability.
    """

    if safe_q.ndim != 2 or safe_q.shape[1] < 1:
        raise ValueError("safe_q must have shape [batch, candidates] with candidates >= 1")
    if selection not in {"uniform", "boundary"}:
        raise ValueError("selection must be 'uniform' or 'boundary'")
    epsilon_safe = float(epsilon_safe)
    if not math.isfinite(epsilon_safe) or not 0.0 < epsilon_safe <= 1.0:
        raise ValueError("epsilon_safe must be finite and in (0, 1]")
    # Eq. 3 includes the threshold itself (QSafe <= epsilon).  Section 6's
    # pre-training exploration rule is deliberately stricter: it asks for an
    # action just *below* the boundary.
    safe_mask = (
        safe_q <= epsilon_safe
        if selection == "uniform"
        else safe_q < epsilon_safe
    )
    fallback = ~safe_mask.any(dim=1)
    if selection == "uniform":
        weights = safe_mask.to(dtype=torch.float32)
        weights = torch.where(fallback[:, None], torch.ones_like(weights), weights)
        selected = torch.multinomial(
            weights, 1, replacement=True, generator=generator
        ).squeeze(1)
    else:
        boundary_q = torch.where(
            safe_mask, safe_q, torch.full_like(safe_q, -torch.inf)
        )
        selected = boundary_q.argmax(dim=1)
    selected = torch.where(fallback, safe_q.argmin(dim=1), selected)
    return selected, fallback


def sample_safe_actions(
    states: torch.Tensor,
    policy,
    qsafe,
    nr_action_candidates: int,
    epsilon_safe: float,
    *,
    selection: str = "uniform",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample the finite-candidate SQRL policy used for rollout and evaluation.

    Returns normalized actions, environment-scaled actions, selected QSafe
    values, and a boolean flag indicating the minimum-Q fallback.
    """

    if not torch.is_tensor(states) or states.ndim < 2:
        raise ValueError("states must be a batched torch.Tensor")
    nr_action_candidates = int(nr_action_candidates)
    if nr_action_candidates < 1:
        raise ValueError("nr_action_candidates must be at least 1")

    nr_envs = states.shape[0]
    candidate_states = states[:, None, :].expand(-1, nr_action_candidates, -1)
    flat_states = candidate_states.reshape(nr_envs * nr_action_candidates, -1)
    actions, processed_actions, _ = policy.get_action(flat_states)
    action_shape = actions.shape[1:]
    actions = actions.reshape(nr_envs, nr_action_candidates, *action_shape)
    processed_actions = processed_actions.reshape_as(actions)
    safe_q = qsafe(
        flat_states, actions.reshape(nr_envs * nr_action_candidates, -1)
    ).reshape(nr_envs, nr_action_candidates)
    selected, fallback = select_safe_candidate_indices(
        safe_q, epsilon_safe, selection=selection, generator=generator
    )
    indices = torch.arange(nr_envs, device=states.device)
    return (
        actions[indices, selected],
        processed_actions[indices, selected],
        safe_q[indices, selected],
        fallback,
    )


def qsafe_bellman_target(
    next_failures: torch.Tensor,
    episode_ends: torch.Tensor,
    next_safe_q: torch.Tensor,
    safe_gamma: float,
) -> torch.Tensor:
    """Compute SQRL's one-step target for transitions from safe states.

    ``episode_ends`` is ``terminated OR truncated``.  SQRL defines QSafe over
    the finite rollout horizon ``T`` and Algorithm 1 stores complete length-T
    trajectories, so the horizon transition keeps its immediate
    ``I(s_{t+1})`` label but does not bootstrap beyond T.  ``next_failures`` is
    explicitly ``I(s_{t+1})``; D_safe stores transitions whose current state
    ``s_t`` is safe, so the paper's current-state Bellman equation reduces to
    this data form without an off-by-one error.
    """

    safe_gamma = float(safe_gamma)
    if not math.isfinite(safe_gamma) or not 0.0 <= safe_gamma <= 1.0:
        raise ValueError("safe_gamma must be finite and in [0, 1]")
    next_failures = next_failures.reshape(-1, 1).to(dtype=next_safe_q.dtype)
    episode_ends = episode_ends.reshape(-1, 1).to(dtype=next_safe_q.dtype)
    return safe_gamma * (
        next_failures
        + (1.0 - next_failures) * (1.0 - episode_ends) * next_safe_q
    )


def qsafe_optimizer_steps_for_block(
    replay_size: int,
    batch_size: int,
    epochs_per_block: float = 1.0,
) -> int:
    """Return optimizer steps for at least one expected pass over D_safe."""

    replay_size = int(replay_size)
    batch_size = int(batch_size)
    epochs_per_block = float(epochs_per_block)
    if replay_size < 1:
        raise ValueError("replay_size must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not math.isfinite(epochs_per_block) or epochs_per_block <= 0.0:
        raise ValueError("epochs_per_block must be finite and positive")
    return max(1, math.ceil(replay_size * epochs_per_block / batch_size))
