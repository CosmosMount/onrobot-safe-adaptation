import numpy as np


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
