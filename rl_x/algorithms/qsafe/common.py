import numpy as np


# The task MDP is defined by the action that actually reaches the actuator
# after joint/rate projection.  Actor log-probabilities are still computed in
# the original pre-tanh distribution; only critic/replay action semantics use
# this marker.
TASK_ACTION_CONTRACT = "applied_rate_limited_v1"


class SafetyObservationHistory:
    """Per-environment fixed-length history of raw policy observations.

    A newly reset environment is initialized by repeating its first frame, so
    neither the actor observation nor the environment API needs to change.
    """

    def __init__(self, nr_envs, observation_size, history_length=5):
        self.nr_envs = int(nr_envs)
        self.observation_size = int(observation_size)
        self.history_length = int(history_length)
        if self.nr_envs < 1 or self.observation_size < 1:
            raise ValueError("Safety history dimensions must be positive.")
        if self.history_length < 1:
            raise ValueError("QSafe history_length must be at least 1.")
        self.frames = np.zeros(
            (self.nr_envs, self.history_length, self.observation_size),
            dtype=np.float32,
        )
        self.initialized = np.zeros(self.nr_envs, dtype=bool)

    @property
    def flattened_size(self):
        return self.history_length * self.observation_size

    def clear(self):
        self.frames.fill(0.0)
        self.initialized.fill(False)

    def append(self, observations, reset_mask=None):
        observations = np.asarray(observations, dtype=np.float32)
        expected = (self.nr_envs, self.observation_size)
        if observations.shape != expected:
            raise ValueError(
                f"Safety observations must have shape {expected}, got "
                f"{observations.shape}."
            )
        if not np.all(np.isfinite(observations)):
            raise ValueError("Safety observations contain NaN or infinity.")
        reset = np.zeros(self.nr_envs, dtype=bool)
        if reset_mask is not None:
            reset = np.asarray(reset_mask, dtype=bool).reshape(-1)
            if reset.shape != (self.nr_envs,):
                raise ValueError(
                    f"reset_mask must have shape ({self.nr_envs},), got "
                    f"{reset.shape}."
                )
        initialize = reset | ~self.initialized
        continuing = ~initialize
        if np.any(continuing):
            self.frames[continuing, :-1] = self.frames[continuing, 1:]
            self.frames[continuing, -1] = observations[continuing]
        if np.any(initialize):
            self.frames[initialize] = observations[initialize, None, :]
        self.initialized[:] = True
        return self.frames.reshape(self.nr_envs, self.flattened_size).copy()


def trajectory_with_observation_history(
    trajectory, observation_size, history_length=5
):
    """Replace raw states in one complete trajectory with flattened history."""

    if not trajectory:
        raise ValueError("Cannot build history for an empty trajectory.")
    history = SafetyObservationHistory(1, observation_size, history_length)
    converted = []
    for transition_index, transition in enumerate(trajectory):
        if len(transition) != 6:
            raise ValueError("A safety transition must contain exactly six fields.")
        state, next_state, action, failure, termination, truncation = transition
        state = np.asarray(state, dtype=np.float32).reshape(observation_size)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(observation_size)
        state_history = history.append(
            state[None, :], reset_mask=np.asarray([transition_index == 0])
        )[0]
        # Derive s' from the same pre-transition history without advancing the
        # persistent sequence twice.  The next loop appends the actual s frame.
        next_frames = history.frames.copy()
        next_frames[:, :-1] = next_frames[:, 1:]
        next_frames[:, -1] = next_state
        next_history = next_frames.reshape(-1).copy()
        converted.append(
            (
                state_history,
                next_history,
                np.asarray(action, dtype=np.float32),
                failure,
                termination,
                truncation,
            )
        )
    return converted


class GaitEvaluationMetrics:
    """Aggregate environment-independent locomotion benchmark diagnostics."""

    _MEAN_KEYS = (
        "mean_foot_clearance",
        "swing_weighted_foot_clearance",
        "swing_clearance/fr",
        "swing_clearance/fl",
        "swing_clearance/rr",
        "swing_clearance/rl",
        "action_saturation_ratio",
        "torque_saturation_ratio",
        "swing_ratio/fr",
        "swing_ratio/fl",
        "swing_ratio/rr",
        "swing_ratio/rl",
    )

    def __init__(self):
        self.values = {key: [] for key in self._MEAN_KEYS}
        self.max_clearance = []
        self.success = False
        self.stuck = False

    def update(self, info):
        for key in self._MEAN_KEYS:
            if key in info:
                values = np.asarray(info[key], dtype=np.float64)
                finite_values = values[np.isfinite(values)]
                if finite_values.size:
                    self.values[key].append(float(np.mean(finite_values)))
        if "max_foot_clearance" in info:
            value = float(np.nanmax(np.asarray(info["max_foot_clearance"])))
            if np.isfinite(value):
                self.max_clearance.append(value)
        if "terrain/success" in info:
            self.success = self.success or bool(np.any(info["terrain/success"]))
        if "terrain/stuck" in info:
            self.stuck = bool(np.any(info["terrain/stuck"]))

    def result(self, falls):
        result = {
            "fall": bool(falls),
            "success": bool(self.success),
            "stable_success": bool(
                self.success and not falls and not self.stuck
            ),
            "stuck": bool(self.stuck and not falls),
            "max_foot_clearance": (
                float(max(self.max_clearance)) if self.max_clearance else float("nan")
            ),
        }
        result.update(
            {
                key: float(np.mean(values)) if values else float("nan")
                for key, values in self.values.items()
            }
        )
        swing_clearance = np.asarray(
            self.values["swing_weighted_foot_clearance"], dtype=np.float64
        )
        swing_clearance = swing_clearance[np.isfinite(swing_clearance)]
        result["p95_swing_clearance"] = (
            float(np.percentile(swing_clearance, 95.0))
            if swing_clearance.size
            else float("nan")
        )
        return result

    @staticmethod
    def format_suite(results):
        count = len(results)
        if not count:
            return "Gait benchmark: no episodes"
        falls = sum(item["fall"] for item in results)
        successes = sum(item["success"] for item in results)
        stable_successes = sum(item["stable_success"] for item in results)
        stuck = sum(item["stuck"] for item in results)

        def finite_mean(key):
            values = np.asarray([item[key] for item in results], dtype=np.float64)
            values = values[np.isfinite(values)]
            return float(np.mean(values)) if values.size else float("nan")

        swing = "/".join(
            f"{finite_mean(f'swing_ratio/{leg}'):.3f}"
            for leg in ("fr", "fl", "rr", "rl")
        )
        swing_clearance = "/".join(
            f"{finite_mean(f'swing_clearance/{leg}'):.4f}"
            for leg in ("fr", "fl", "rr", "rl")
        )
        return (
            f"Gait benchmark ({count} episodes) - Falls: {falls} ({falls/count:.1%}), "
            f"Successes: {successes} ({successes/count:.1%}), "
            f"Stable successes: {stable_successes} "
            f"({stable_successes/count:.1%}), "
            f"Stuck: {stuck} ({stuck/count:.1%}), Mean/max foot clearance: "
            f"{finite_mean('mean_foot_clearance'):.4f}/"
            f"{finite_mean('max_foot_clearance'):.4f} m, Swing clearance "
            f"mean/P95: {finite_mean('swing_weighted_foot_clearance'):.4f}/"
            f"{finite_mean('p95_swing_clearance'):.4f} m, Swing clearance "
            f"FR/FL/RR/RL: "
            f"{swing_clearance}, "
            f"Swing ratio FR/FL/RR/RL: "
            f"{swing}, Action/torque saturation: "
            f"{finite_mean('action_saturation_ratio'):.3f}/"
            f"{finite_mean('torque_saturation_ratio'):.3f}"
        )


def finetune_constraints_enabled(phase, qsafe_enabled):
    """Use one gate for both target-task action masking and Eq. 4."""

    return str(phase) == "finetune" and bool(qsafe_enabled)


def actor_updates_enabled(
    phase, global_step, warmup_steps, update_interval=1
):
    """Keep the transferred actor fixed while a fresh target-task critic warms up."""

    warmup_steps = int(warmup_steps)
    if warmup_steps < 0:
        raise ValueError("algorithm.finetune_actor_warmup_steps must be non-negative.")
    update_interval = int(update_interval)
    if update_interval < 1:
        raise ValueError(
            "algorithm.finetune_actor_update_interval must be at least 1."
        )
    if str(phase) != "finetune":
        return True
    global_step = int(global_step)
    return global_step >= warmup_steps and (
        global_step - warmup_steps
    ) % update_interval == 0


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
