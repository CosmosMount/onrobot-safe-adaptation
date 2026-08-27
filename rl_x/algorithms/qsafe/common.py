import numpy as np


class SorlRewardShaper:
    """Stateful implementation of SORL equations (10) and (11).

    The empirical reward range updates the terminal cost and, when requested,
    the safety significance ``lambda``.  Equation (19) is the lower envelope
    of ``H*`` affine functions of lambda when the unsafe-trajectory length is
    evaluated at the integer horizons 1..H*.  Solving those line intersections
    is deterministic and avoids a SciPy dependency while implementing the
    paper's local solution around the previous lambda.
    """

    def __init__(
        self,
        gamma,
        gamma_safe,
        horizon,
        significance,
        cost_margin,
        target_delta=0.0,
        solve_significance=False,
        preserve_negative_task_penalty=False,
    ):
        self.gamma = float(gamma)
        self.gamma_safe = float(gamma_safe)
        self.horizon = int(horizon)
        self.significance = float(significance)
        self.cost_margin = float(cost_margin)
        self.target_delta = float(target_delta)
        self.solve_significance = bool(solve_significance)
        self.preserve_negative_task_penalty = bool(
            preserve_negative_task_penalty
        )
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("SORL gamma must be in (0, 1)")
        if not 0.0 < self.gamma_safe < 1.0:
            raise ValueError("SORL gamma_safe must be in (0, 1)")
        if self.horizon < 1:
            raise ValueError("SORL horizon must be at least one")
        if self.significance < 0.0:
            raise ValueError("SORL significance must be non-negative")
        if self.cost_margin <= 0.0:
            raise ValueError("SORL cost_margin must be positive")
        if not np.isfinite(self.target_delta):
            raise ValueError("SORL target_delta must be finite")
        self.reward_min = np.inf
        self.reward_max = -np.inf
        self.failure_cost = self.cost_margin
        self.achieved_delta = np.nan
        self.negative_penalty_floor_fraction = 0.0


    def _delta_lines(self):
        """Return Eq. (19) as ``min_x(intercept_x + slope_x*lambda)``."""
        horizons = np.arange(1, self.horizon + 1, dtype=np.float64)
        ratio = self.gamma / self.gamma_safe
        unsafe_scale = (
            self.gamma_safe**self.horizon
            * self.reward_max
            / (1.0 - ratio)
        )
        unsafe_intercepts = (
            self.reward_max / (1.0 - self.gamma)
            - (self.reward_max + self.failure_cost)
            / (1.0 - self.gamma)
            * self.gamma**horizons
        )
        unsafe_slopes = unsafe_scale * (ratio**horizons - 1.0)
        safe_slope = (
            self.gamma_safe * self.reward_min / (1.0 - self.gamma)
        )
        return -unsafe_intercepts, safe_slope - unsafe_slopes


    def _delta(self, significance):
        intercepts, slopes = self._delta_lines()
        return float(np.min(intercepts + slopes * float(significance)))


    def _solve_significance(self):
        # The theorem assumes r_min < 0 < r_max.  Before the online samples
        # establish that range, retaining the configured initial lambda is the
        # only well-defined behavior.
        if not self.reward_min < 0.0 < self.reward_max:
            self.achieved_delta = np.nan
            return
        intercepts, slopes = self._delta_lines()
        candidates = [max(self.significance, np.finfo(np.float64).tiny)]
        for intercept, slope in zip(intercepts, slopes):
            if abs(slope) <= np.finfo(np.float64).eps:
                continue
            candidate = (self.target_delta - intercept) / slope
            if np.isfinite(candidate) and candidate > 0.0:
                candidates.append(float(candidate))

        previous = max(self.significance, np.finfo(np.float64).tiny)
        scored = []
        for candidate in candidates:
            achieved = self._delta(candidate)
            error = abs(achieved - self.target_delta)
            locality = abs(np.log(candidate / previous))
            scored.append((error, locality, candidate, achieved))
        _, _, self.significance, self.achieved_delta = min(scored)


    def shape(self, rewards, risks, failures):
        rewards = np.asarray(rewards, dtype=np.float32)
        risks = np.asarray(risks, dtype=np.float32).reshape(rewards.shape)
        failures = np.asarray(failures, dtype=np.float32).reshape(rewards.shape)
        if not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(risks)):
            raise ValueError("SORL rewards and risks must be finite")
        if not np.all((failures == 0.0) | (failures == 1.0)):
            raise ValueError("SORL failures must be binary")

        self.reward_min = min(self.reward_min, float(np.min(rewards)))
        self.reward_max = max(self.reward_max, float(np.max(rewards)))
        reward_range = max(self.reward_max - self.reward_min, 0.0)
        theoretical_cost = (
            reward_range / (self.gamma**self.horizon) - self.reward_max
        )
        self.failure_cost = max(
            self.cost_margin, theoretical_cost + self.cost_margin
        )
        if self.solve_significance:
            self._solve_significance()
        elif self.reward_min < 0.0 < self.reward_max:
            self.achieved_delta = self._delta(self.significance)

        risks = np.clip(risks, 0.0, 1.0)
        positive = (1.0 - self.significance * risks) * rewards
        negative = self.significance * risks * rewards
        shaped = np.where(rewards >= 0.0, positive, negative)
        if self.preserve_negative_task_penalty:
            # Eq. (10) assumes that suppressing a negative reward in states
            # judged safe is desirable.  For command tracking this reverses
            # the task ordering: a slow but safe gait can turn a negative
            # speed reward into a larger value than walking at the command.
            # Retain the paper value whenever it is more conservative, but
            # never allow safety shaping to erase a negative task signal.
            floor_applied = (
                (failures == 0.0)
                & (rewards < 0.0)
                & (shaped > rewards)
            )
            self.negative_penalty_floor_fraction = float(
                np.mean(floor_applied)
            )
            shaped = np.where(rewards < 0.0, np.minimum(shaped, rewards), shaped)
        else:
            self.negative_penalty_floor_fraction = 0.0
        return np.where(failures > 0.0, -self.failure_cost, shaped).astype(
            np.float32
        )


    def metrics(self):
        return {
            "sorl/reward_min": self.reward_min,
            "sorl/reward_max": self.reward_max,
            "sorl/failure_cost": self.failure_cost,
            "sorl/significance": self.significance,
            "sorl/target_delta": self.target_delta,
            "sorl/achieved_delta": self.achieved_delta,
            "sorl/negative_penalty_floor_fraction": (
                self.negative_penalty_floor_fraction
            ),
        }


class VectorTrajectoryAccumulator:
    """Stage vector-env transitions and emit complete per-env trajectories."""

    def __init__(self, nr_envs):
        self.nr_envs = int(nr_envs)
        self.pending = [[] for _ in range(self.nr_envs)]

    def add_step(
        self,
        states,
        next_states,
        actions,
        failures,
        terminations,
        truncations,
    ):
        states = np.asarray(states)
        next_states = np.asarray(next_states)
        actions = np.asarray(actions)
        failures = np.asarray(failures).reshape(-1)
        terminations = np.asarray(terminations, dtype=bool).reshape(-1)
        truncations = np.asarray(truncations, dtype=bool).reshape(-1)
        completed = []
        for env_index in range(self.nr_envs):
            self.pending[env_index].append(
                (
                    np.array(states[env_index], copy=True),
                    np.array(next_states[env_index], copy=True),
                    np.array(actions[env_index], copy=True),
                    float(failures[env_index]),
                    float(terminations[env_index]),
                    float(truncations[env_index]),
                )
            )
            if terminations[env_index] or truncations[env_index]:
                completed.append(self.pending[env_index])
                self.pending[env_index] = []
        return completed


class CompletedTrajectoryCollector(VectorTrajectoryAccumulator):
    """Collect an exact number of complete vector-env trajectories."""

    def __init__(self, nr_envs, target_trajectories):
        if target_trajectories < 1:
            raise ValueError("algorithm.n_safe must be at least 1 during pretraining.")
        super().__init__(nr_envs)
        self.target_trajectories = int(target_trajectories)
        self.nr_completed = 0

    @property
    def complete(self):
        return self.nr_completed >= self.target_trajectories

    def add_step(
        self,
        states,
        next_states,
        actions,
        failures,
        terminations,
        truncations,
    ):
        """Return trajectories completed by this vector step, capped at target."""
        completed = super().add_step(
            states,
            next_states,
            actions,
            failures,
            terminations,
            truncations,
        )
        remaining = self.target_trajectories - self.nr_completed
        accepted = completed[:remaining]
        self.nr_completed += len(accepted)
        return accepted


def extract_failure_signal(info, terminations, nr_envs):
    """Return the post-transition SQRL indicator and validate its contract.

    RL-X vector environments auto-reset terminal environments.  Consequently a
    failure must already be represented as an environment termination; changing
    the termination only inside an algorithm would be too late to trigger reset.
    """
    if "failure" not in info:
        raise KeyError(
            "SQRL environments must expose the post-transition binary safety "
            "indicator as info['failure']."
        )

    failures = np.asarray(info["failure"])
    if failures.shape == () and nr_envs == 1:
        failures = failures.reshape(1)
    failures = failures.reshape(-1)
    if failures.shape != (nr_envs,):
        raise ValueError(
            f"info['failure'] must have shape ({nr_envs},), got {failures.shape}."
        )
    if not np.all((failures == 0) | (failures == 1)):
        raise ValueError("info['failure'] must contain only binary values (0/1 or bool).")

    terminations = np.asarray(terminations, dtype=bool).reshape(-1)
    if terminations.shape != (nr_envs,):
        raise ValueError(
            f"terminations must have shape ({nr_envs},), got {terminations.shape}."
        )
    failure_mask = failures.astype(bool)
    if np.any(failure_mask & ~terminations):
        raise ValueError(
            "SQRL contract violation: every info['failure'] must also set "
            "terminated=True so the vector environment resets the failed episode."
        )
    return failures.astype(np.float32)


def safety_bootstrap_mask(terminations, truncations):
    """Bootstrap through pure time limits, never through true terminals.

    Gymnasium permits ``terminated`` and ``truncated`` to both be true.  A real
    terminal takes precedence in that case; only ``terminated=False`` permits
    bootstrapping from the final observation.
    """
    del truncations
    return 1.0 - terminations


def safety_bellman_target(failures, terminations, truncations, next_values, gamma):
    """SQRL TD target using the post-transition indicator ``failure=I(s')``."""
    bootstrap = safety_bootstrap_mask(terminations, truncations)
    return failures + (1.0 - failures) * gamma * bootstrap * next_values


def restore_algorithm_config(
    target_config,
    loaded_config,
    explicitly_set_algorithm_params,
    prefix="algorithm",
):
    """Recursively restore checkpoint config without clobbering CLI overrides."""
    explicit = set(explicitly_set_algorithm_params)
    for key, value in loaded_config.items():
        path = f"{prefix}.{key}"
        if path in explicit or key not in target_config:
            continue
        target_value = target_config[key]
        if hasattr(value, "items") and hasattr(target_value, "items"):
            restore_algorithm_config(target_value, value, explicit, path)
        else:
            target_config[key] = value


def validate_safety_rollout_environment(train_env, safety_env, nr_envs):
    """Validate the observable contract for the independent SQRL rollout env."""
    if safety_env is train_env:
        raise ValueError(
            "SQRL requires an independent safety rollout/evaluation environment "
            "so D_safe collection cannot reset the D_offline task trajectory."
        )
    train_observation_shape = tuple(train_env.single_observation_space.shape)
    safety_observation_shape = tuple(safety_env.single_observation_space.shape)
    if safety_observation_shape != train_observation_shape:
        raise ValueError(
            "SQRL safety rollout environment observation shape must match the "
            f"task environment: {safety_observation_shape} != "
            f"{train_observation_shape}."
        )
    train_action_shape = tuple(train_env.single_action_space.shape)
    safety_action_shape = tuple(safety_env.single_action_space.shape)
    if safety_action_shape != train_action_shape:
        raise ValueError(
            "SQRL safety rollout environment action shape must match the task "
            f"environment: {safety_action_shape} != {train_action_shape}."
        )
    for attribute in ("low", "high"):
        if not np.allclose(
            getattr(safety_env.single_action_space, attribute),
            getattr(train_env.single_action_space, attribute),
        ):
            raise ValueError(
                "SQRL safety rollout environment action bounds must match the "
                f"task environment ({attribute} differs)."
            )
    safety_nr_envs = getattr(
        safety_env, "nr_envs", getattr(safety_env, "num_envs", None)
    )
    if safety_nr_envs is not None and int(safety_nr_envs) != int(nr_envs):
        raise ValueError(
            "SQRL safety rollout environment must use algorithm nr_envs: "
            f"{safety_nr_envs} != {nr_envs}."
        )
