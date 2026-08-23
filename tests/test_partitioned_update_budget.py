import pytest

torch = pytest.importorskip("torch")

from rl_x.algorithms.sac_qsafe.pytorch.checkpoint import restore_parameter_
from rl_x.algorithms.sac_qsafe.pytorch.rollout import (
    AtomicTrajectoryUpdateBudget,
    TransitionUpdateBudget,
)


def test_task_update_budget_is_invariant_to_vector_pool_size():
    scalar = TransitionUpdateBudget(update_ratio=0.25)
    vector = TransitionUpdateBudget(update_ratio=0.25)

    for _ in range(8):
        scalar.add_transitions(1)
    vector.add_transitions(8)

    assert scalar.consume_ready_updates() == 2
    assert vector.consume_ready_updates() == 2
    assert scalar.effective_ratio == pytest.approx(0.25)
    assert vector.effective_ratio == pytest.approx(0.25)


def test_fractional_update_credit_is_carried_between_steps():
    budget = TransitionUpdateBudget(update_ratio=0.1)

    budget.add_transitions(6)
    assert budget.consume_ready_updates() == 0
    assert budget.credit == pytest.approx(0.6)

    budget.add_transitions(4)
    assert budget.consume_ready_updates() == 1
    assert budget.credit == pytest.approx(0.0)
    assert budget.transitions == 10
    assert budget.updates == 1


def test_default_ratio_schedules_one_update_per_new_transition():
    budget = TransitionUpdateBudget(update_ratio=1.0)
    budget.add_transitions(256)

    assert budget.consume_ready_updates() == 256
    assert budget.effective_ratio == pytest.approx(1.0)


def test_qsafe_credit_counts_only_atomically_completed_trajectories():
    completed = [list(range(3)), list(range(5))]
    pending_trajectory_prefix = list(range(100))
    budget = AtomicTrajectoryUpdateBudget(updates_per_trajectory=1.0)

    committed = budget.add_completed(completed)

    assert committed == 8
    assert budget.consume_ready_updates() == 2
    assert budget.transitions == 8
    assert budget.trajectories == 2
    assert budget.effective_ratio == pytest.approx(1.0)
    assert pending_trajectory_prefix  # pending prefixes never enter the budget


def test_qsafe_synchronized_timeouts_do_not_schedule_per_transition_burst():
    budget = AtomicTrajectoryUpdateBudget(updates_per_trajectory=1.0)
    completed = [list(range(500)) for _ in range(64)]

    assert budget.add_completed(completed) == 32000
    assert budget.consume_ready_updates() == 64


def test_parameter_restore_preserves_optimizer_reference():
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    original_identity = id(parameter)

    restore_parameter_(parameter, torch.tensor([2.5], dtype=torch.float64))

    assert id(parameter) == original_identity
    assert optimizer.param_groups[0]["params"][0] is parameter
    torch.testing.assert_close(parameter, torch.tensor([2.5]))


@pytest.mark.parametrize("ratio", [-1.0, float("nan"), float("inf")])
def test_update_budget_rejects_invalid_ratios(ratio):
    with pytest.raises(ValueError, match="finite non-negative"):
        TransitionUpdateBudget(ratio)
