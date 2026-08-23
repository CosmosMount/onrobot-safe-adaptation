from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class QSafeComponent(Protocol):
    """Structural interface consumed by task-algorithm SQRL wrappers."""

    def select_safe_action(
        self,
        states: Any,
        candidate_actions: Any,
        candidate_log_probs: Any,
        phase: str | None = None,
    ) -> tuple[Any, Any, dict[str, float]]: ...

    def add_transition(
        self,
        states: Any,
        actions: Any,
        next_states: Any,
        failures: Any,
        terminations: Any,
        truncations: Any,
    ) -> None: ...

    def add_trajectory(self, trajectory: Any) -> None: ...

    def update(
        self, policy_sampler: Any, state_transform: Any | None = None
    ) -> dict[str, float]: ...

    def save(self, file_path: str, include_optimizer: bool = True) -> None: ...

    def load(self, file_path: str, load_optimizer: bool = True) -> None: ...

    def freeze(self) -> None: ...
