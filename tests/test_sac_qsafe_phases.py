from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl_x.algorithms.qsafe.common import TASK_ACTION_CONTRACT
from rl_x.algorithms.sac_qsafe.pytorch.checkpoint import (
    load_checkpoint_bundle,
    save_checkpoint_bundle,
)
from rl_x.algorithms.sac_qsafe.pytorch.observation_normalizer import (
    ObservationNormalizer,
)
from rl_x.algorithms.sac_qsafe.pytorch.rollout import (
    PartitionedRolloutCounter,
    preserve_policy_outputs,
)
from rl_x.algorithms.sac.pytorch.entropy_coefficient import EntropyCoefficient
from rl_x.algorithms.sac_qsafe.pytorch.sac_qsafe import SAC_QSafe


class _IdentityNormalizer:
    def normalize(self, states, update=False):
        del update
        return states


class _ProjectedActionEnvironment:
    single_action_space = SimpleNamespace(shape=(2,))

    def project_actions(self, raw_states, actions):
        del raw_states
        return actions + 0.25


class _FixedPolicy(torch.nn.Module):
    def __init__(self, action_size=2):
        super().__init__()
        self.action_size = action_size

    def get_action(self, states):
        values = (
            (torch.arange(states.shape[0], device=states.device) % 3 + 1)
            .to(dtype=states.dtype)
            .mul(0.1)
            .unsqueeze(1)
        )
        actions = values.expand(-1, self.action_size)
        return actions, actions, torch.zeros_like(values)


class _SelectingQSafe:
    candidate_actions = 3
    version = 1

    def __init__(self):
        self.seen_candidates = None

    def select_safe_action(
        self, states, candidate_actions, candidate_log_probs, phase
    ):
        del states, candidate_log_probs, phase
        self.seen_candidates = candidate_actions.detach().clone()
        selected_indices = torch.ones(
            candidate_actions.shape[0], dtype=torch.long
        )
        batch_indices = torch.arange(candidate_actions.shape[0])
        return (
            candidate_actions[batch_indices, selected_indices],
            selected_indices,
            {},
        )

    def normalize_observations(self, states):
        return states


def _action_sampling_model():
    model = object.__new__(SAC_QSafe)
    model.device = torch.device("cpu")
    model.train_env = _ProjectedActionEnvironment()
    model.observation_normalizer = _IdentityNormalizer()
    model.policy = _FixedPolicy()
    model._env_as_low_tensor = torch.full((2,), -1.0)
    model._env_as_high_tensor = torch.full((2,), 1.0)
    model.phase = "finetune"
    model.qsafe = _SelectingQSafe()
    model._qsafe_reset_masks = {}
    model._last_qsafe_observations = {}
    return model


def test_unconstrained_sampling_returns_raw_task_action_but_applies_projection():
    model = _action_sampling_model()

    raw_action, processed_action = model._sample_unconstrained_action(
        np.zeros((1, 4), dtype=np.float32)
    )

    torch.testing.assert_close(raw_action, torch.full((1, 2), 0.1))
    torch.testing.assert_close(processed_action, torch.full((1, 2), 0.35))


def test_qsafe_selection_keeps_raw_and_applied_candidate_indices_paired():
    model = _action_sampling_model()

    raw_action, processed_action, _ = model._sample_policy_candidates(
        np.zeros((1, 4), dtype=np.float32)
    )

    torch.testing.assert_close(
        model.qsafe.seen_candidates[0, :, 0],
        torch.tensor([0.35, 0.45, 0.55]),
    )
    # QSafe selected candidate one in applied-action space.  Task replay must
    # receive raw candidate one, while the environment receives its paired
    # projected value.
    torch.testing.assert_close(raw_action, torch.full((1, 2), 0.2))
    torch.testing.assert_close(processed_action, torch.full((1, 2), 0.45))


class _TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_action_parameter = torch.nn.Parameter(
            torch.tensor([0.2, -0.1], dtype=torch.float32)
        )

    def get_action(self, states):
        actions = torch.tanh(self.raw_action_parameter).expand(
            states.shape[0], -1
        )
        log_probs = (
            -0.5 * self.raw_action_parameter.square().sum()
        ).expand(states.shape[0], 1)
        return actions, actions, log_probs


class _RecordingQ(torch.nn.Module):
    def __init__(self, state_size=3, action_size=2):
        super().__init__()
        self.linear = torch.nn.Linear(state_size + action_size, 1)
        self.seen_actions = []

    def forward(self, states, actions):
        self.seen_actions.append(actions.detach().clone())
        return self.linear(torch.cat((states, actions), dim=-1))


class _FixedReplay:
    def __init__(self):
        self.states = torch.zeros((2, 3), dtype=torch.float32)
        self.next_states = torch.ones((2, 3), dtype=torch.float32) * 0.1
        self.actions = torch.tensor(
            [[0.05, -0.05], [0.1, -0.1]], dtype=torch.float32
        )

    def sample(self, batch_size):
        assert batch_size == 2
        return (
            self.states,
            self.next_states,
            self.actions,
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(2, dtype=torch.float32),
        )


def _partitioned_update_model():
    model = object.__new__(SAC_QSafe)
    model.device = torch.device("cpu")
    model.batch_size = 2
    model.bf16_mixed_precision_training = False
    model.policy = _TinyPolicy()
    model.critic = SimpleNamespace(
        q1=_RecordingQ(),
        q2=_RecordingQ(),
        q1_target=_RecordingQ(),
        q2_target=_RecordingQ(),
    )
    model.entropy_coefficient = EntropyCoefficient(
        SimpleNamespace(
            algorithm=SimpleNamespace(target_entropy=-2.0, alpha_init=0.1)
        ),
        SimpleNamespace(single_action_space=SimpleNamespace(shape=(2,))),
        model.device,
    )
    model.q_optimizer = torch.optim.Adam(
        [
            *model.critic.q1.parameters(),
            *model.critic.q2.parameters(),
        ],
        lr=1e-3,
    )
    model.policy_optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    model.entropy_optimizer = torch.optim.Adam(
        model.entropy_coefficient.parameters(), lr=1e-2
    )
    model.phase = "finetune"
    model.finetune_actor_warmup_steps = 10_000
    model.finetune_actor_handoff_steps = 10_000
    model.finetune_actor_update_interval = 10
    model._actor_handoff_enabled = True
    model._policy_nominal_learning_rates = [1e-3]
    model.finetune_constraints_enabled = False
    model.gamma = 0.99
    model.tau = 0.005
    model.nu = torch.tensor(0.0)
    model.anneal_learning_rate = False
    model.learning_rate = 1e-3
    model._normalize_states = lambda states, update=False: states
    model._project_actions = lambda raw_states, actions: actions + 0.25
    return model


def _task_checkpoint(model, marker=TASK_ACTION_CONTRACT):
    checkpoint = {
        "q1_state_dict": model.critic.q1.state_dict(),
        "q2_state_dict": model.critic.q2.state_dict(),
        "q1_target_state_dict": model.critic.q1_target.state_dict(),
        "q2_target_state_dict": model.critic.q2_target.state_dict(),
        "log_alpha": torch.tensor([-4.0]),
    }
    if marker is not None:
        checkpoint["task_action_contract"] = marker
    return checkpoint


def test_pytorch_task_warm_start_accepts_matching_raw_action_contract(tmp_path):
    model = _partitioned_update_model()
    with torch.no_grad():
        model.critic.q1.linear.weight.fill_(0.17)
        model.critic.q1.linear.bias.fill_(-0.23)
    checkpoint_path = tmp_path / "task.model"
    torch.save(_task_checkpoint(model), checkpoint_path)
    with torch.no_grad():
        model.critic.q1.linear.weight.zero_()
        model.critic.q1.linear.bias.zero_()
        model.entropy_coefficient.log_alpha.zero_()

    model._load_pretrained_task_state(checkpoint_path)

    torch.testing.assert_close(
        model.critic.q1.linear.weight,
        torch.full_like(model.critic.q1.linear.weight, 0.17),
    )
    torch.testing.assert_close(
        model.critic.q1.linear.bias,
        torch.full_like(model.critic.q1.linear.bias, -0.23),
    )
    torch.testing.assert_close(
        model.entropy_coefficient.log_alpha,
        torch.full_like(model.entropy_coefficient.log_alpha, -4.0),
    )


@pytest.mark.parametrize(
    ("marker", "message"),
    [
        (None, "missing: task_action_contract"),
        ("applied_rate_limited_v0", "action contract mismatch"),
    ],
)
def test_pytorch_task_warm_start_rejects_legacy_action_contract(
    tmp_path, marker, message
):
    model = _partitioned_update_model()
    checkpoint_path = tmp_path / "legacy-task.model"
    torch.save(_task_checkpoint(model, marker=marker), checkpoint_path)

    with pytest.raises(ValueError, match=message):
        model._load_pretrained_task_state(checkpoint_path)


def test_fresh_alpha_is_held_with_transferred_actor_during_warmup():
    model = _partitioned_update_model()
    replay = _FixedReplay()
    actor_before = model.policy.raw_action_parameter.detach().clone()
    log_alpha_before = model.entropy_coefficient.log_alpha.detach().clone()

    metrics = model._partitioned_task_update(replay, global_step=5_000)

    torch.testing.assert_close(model.policy.raw_action_parameter, actor_before)
    torch.testing.assert_close(
        model.entropy_coefficient.log_alpha.detach(), log_alpha_before
    )
    assert metrics["updates/actor_enabled"] == 0.0
    assert metrics["updates/alpha_enabled"] == 0.0
    assert metrics["finetune/alpha_anti_windup"] == 1.0
    assert metrics["lr/policy_learning_rate"] == pytest.approx(0.0)


def test_actor_and_alpha_share_interval_and_optimizer_lr_handoff():
    model = _partitioned_update_model()
    replay = _FixedReplay()

    alpha_before = model.entropy_coefficient.log_alpha.detach().clone()
    skipped = model._partitioned_task_update(replay, global_step=15_001)
    torch.testing.assert_close(
        model.entropy_coefficient.log_alpha.detach(), alpha_before
    )
    assert skipped["updates/actor_enabled"] == 0.0
    assert skipped["updates/alpha_enabled"] == 0.0

    updated = model._partitioned_task_update(replay, global_step=15_010)
    assert updated["updates/actor_enabled"] == 1.0
    assert updated["updates/alpha_enabled"] == 1.0
    assert updated["lr/policy_learning_rate"] == pytest.approx(0.000501)
    assert not torch.equal(
        model.entropy_coefficient.log_alpha.detach(), alpha_before
    )


def test_actor_optimizer_takes_a_real_zero_lr_step_at_handoff_boundary():
    model = _partitioned_update_model()
    replay = _FixedReplay()
    actor_before = model.policy.raw_action_parameter.detach().clone()

    boundary = model._partitioned_task_update(replay, global_step=10_000)

    assert boundary["updates/actor_enabled"] == 1.0
    assert boundary["lr/policy_learning_rate"] == pytest.approx(0.0)
    torch.testing.assert_close(model.policy.raw_action_parameter, actor_before)
    optimizer_state = model.policy_optimizer.state[
        model.policy.raw_action_parameter
    ]
    assert int(optimizer_state["step"]) == 1


def test_task_critics_consume_raw_actions_after_actor_unfreezes():
    model = _partitioned_update_model()
    replay = _FixedReplay()
    expected_policy_action = torch.tanh(
        model.policy.raw_action_parameter.detach()
    ).expand(2, -1)

    metrics = model._partitioned_task_update(replay, global_step=10_000)

    # Target critics consume the raw next-policy action.  Online critics see
    # replay raw actions first, then the raw current-policy action.  The +0.25
    # projection is reserved for QSafe/environment execution and diagnostics.
    torch.testing.assert_close(
        model.critic.q1_target.seen_actions[-1], expected_policy_action
    )
    torch.testing.assert_close(model.critic.q1.seen_actions[0], replay.actions)
    torch.testing.assert_close(
        model.critic.q1.seen_actions[-1], expected_policy_action
    )
    assert metrics["updates/actor_enabled"] == 1.0


def test_normalizer_updates_then_freezes_exactly():
    normalizer = ObservationNormalizer(2)
    normalizer.normalize(torch.tensor([[1.0, 3.0], [3.0, 5.0]]), update=True)
    before = {key: value.clone() for key, value in normalizer.state_dict().items()}
    normalizer.freeze()
    normalizer.normalize(torch.tensor([[100.0, 100.0]]), update=True)
    for key, value in before.items():
        torch.testing.assert_close(normalizer.state_dict()[key], value)


def test_checkpoint_bundle_and_partition_counters(tmp_path):
    path = save_checkpoint_bundle(
        tmp_path / "step_0001",
        {"contract": "v1"},
        {"policy.pt": {"weight": torch.tensor([1.0])}},
    )
    manifest, artifacts = load_checkpoint_bundle(path)
    assert manifest == {"contract": "v1"}
    torch.testing.assert_close(artifacts["policy.pt"]["weight"], torch.tensor([1.0]))
    counter = PartitionedRolloutCounter(256, 64)
    assert counter.advance() == (256, 64)
    assert counter.advance() == (512, 128)


def test_preserved_policy_outputs_do_not_alias_reused_graph_storage():
    graph_storage = torch.tensor([[1.0, 2.0]])
    action, processed_action = preserve_policy_outputs(
        graph_storage, graph_storage + 1.0
    )

    graph_storage.fill_(99.0)

    torch.testing.assert_close(action, torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(processed_action, torch.tensor([[2.0, 3.0]]))
