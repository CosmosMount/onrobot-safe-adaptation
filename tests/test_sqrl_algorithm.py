import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from algorithms.qsafe.pytorch.qnetwork import (
    QNetwork as QSafeNetwork,
    get_q_network as get_qsafe_network,
)
from algorithms.sac.pytorch.entropy_coefficient import EntropyCoefficient
from algorithms.types import ObservationSpaceType
from sqrl.sac.pytorch.pretrainer import (
    SQRLPretrainer,
    qsafe_update_schedule,
)
from sqrl.sac.environment import resolve_executed_actions
from sqrl.sac.pytorch.replay_buffer import newly_eligible_transitions
from sqrl.sac.pytorch.rollout_buffer import RolloutBuffer
from sqrl.sac.pytorch.safety_ops import (
    qsafe_bellman_target,
    qsafe_optimizer_steps_for_block,
    sample_safe_actions,
    select_safe_candidate_indices,
)
from sqrl.sac.pytorch.workflow import (
    CHECKPOINT_ALGORITHM_ID,
    CHECKPOINT_ARTIFACT_TYPE,
    CHECKPOINT_SCHEMA_VERSION,
    SQRLWorkflow,
)


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def to_dict(self):
        return dict(self)


class BoxStub:
    def __init__(self, shape, low=-1.0, high=1.0):
        self.shape = tuple(shape)
        self.low = np.full(self.shape, low, dtype=np.float32)
        self.high = np.full(self.shape, high, dtype=np.float32)

    def sample(self):
        return np.zeros(self.shape, dtype=np.float32)


class EnvStub:
    def __init__(self, nr_envs=1, env_low=-1.0, env_high=1.0):
        self.num_envs = nr_envs
        self.single_action_space = BoxStub((12,), env_low, env_high)
        self.single_observation_space = BoxStub((46,))


class SafetyComponentsTests(unittest.TestCase):
    def test_dsafe_keeps_only_latest_k_complete_trajectories(self):
        replay = RolloutBuffer(
            10,
            1,
            (1,),
            (1,),
            np.random.default_rng(4),
            max_trajectories=2,
        )
        for value in range(3):
            vector = np.asarray([value], dtype=np.float32)
            replay.add_trajectory(
                [
                    (
                        vector,
                        vector,
                        vector,
                        np.float32(0.0),
                        np.float32(1.0),
                        np.float32(0.0),
                    )
                ]
            )

        self.assertEqual(replay.nr_trajectories, 2)
        retained = [
            float(trajectory[0][0, 0]) for trajectory in replay.trajectories
        ]
        self.assertEqual(retained, [1.0, 2.0])

    def test_uniform_accepted_selection_is_not_density_reweighted(self):
        rows = 10_000
        safe_q = torch.tensor([[0.01, 0.02, 0.9]], dtype=torch.float32).expand(rows, -1)
        generator = torch.Generator().manual_seed(7)
        selected, fallback = select_safe_candidate_indices(
            safe_q, 0.1, generator=generator
        )
        self.assertFalse(fallback.any())
        self.assertFalse((selected == 2).any())
        first_fraction = (selected == 0).float().mean().item()
        self.assertLess(abs(first_fraction - 0.5), 0.03)

    def test_no_safe_candidate_uses_minimum_q_fallback(self):
        safe_q = torch.tensor([[0.8, 0.2, 0.5], [0.01, 0.9, 0.8]])
        selected, fallback = select_safe_candidate_indices(
            safe_q, 0.1, generator=torch.Generator().manual_seed(1)
        )
        self.assertEqual(selected.tolist(), [1, 0])
        self.assertEqual(fallback.tolist(), [True, False])

    def test_boundary_selection_picks_largest_q_strictly_below_epsilon(self):
        safe_q = torch.tensor(
            [[0.01, 0.099, 0.1, 0.8], [0.8, 0.2, 0.5, 0.3]]
        )
        selected, fallback = select_safe_candidate_indices(
            safe_q, 0.1, selection="boundary"
        )
        self.assertEqual(selected.tolist(), [1, 1])
        self.assertEqual(fallback.tolist(), [False, True])

    def test_eq3_accepts_a_candidate_exactly_at_epsilon(self):
        selected, fallback = select_safe_candidate_indices(
            torch.tensor([[0.1, 0.2]]),
            0.1,
            selection="uniform",
            generator=torch.Generator().manual_seed(1),
        )
        self.assertEqual(selected.tolist(), [0])
        self.assertEqual(fallback.tolist(), [False])

    def test_shared_sampler_returns_only_safe_policy_candidates(self):
        class CandidatePolicy:
            def get_action(self, states):
                candidate = torch.arange(states.shape[0], device=states.device) % 3
                actions = torch.zeros(states.shape[0], 2, device=states.device)
                actions[:, 0] = candidate.to(torch.float32)
                # These deliberately unequal values must not affect selection.
                log_probs = (100.0 * candidate).to(torch.float32).reshape(-1, 1)
                return actions, actions + 10.0, log_probs

        class QSafe:
            def __call__(self, states, actions):
                return torch.where(
                    actions[:, :1] < 2.0,
                    torch.full_like(actions[:, :1], 0.01),
                    torch.full_like(actions[:, :1], 0.9),
                )

        actions, processed, selected_q, fallback = sample_safe_actions(
            torch.zeros(128, 5),
            CandidatePolicy(),
            QSafe(),
            3,
            0.1,
            generator=torch.Generator().manual_seed(3),
        )
        self.assertTrue((actions[:, 0] < 2.0).all())
        torch.testing.assert_close(processed, actions + 10.0)
        torch.testing.assert_close(selected_q, torch.full((128,), 0.01))
        self.assertFalse(fallback.any())

    def test_qsafe_target_keeps_failure_and_terminal_semantics(self):
        target = qsafe_bellman_target(
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            # The second entry is a non-failure terminal and the fourth is a
            # finite-horizon truncation; neither may bootstrap beyond T.
            torch.tensor([1.0, 1.0, 0.0, 1.0]),
            torch.tensor([[0.8], [0.8], [0.8], [0.8]]),
            0.7,
        )
        torch.testing.assert_close(
            target, torch.tensor([[0.7], [0.0], [0.56], [0.0]])
        )

    def test_qsafe_block_steps_cover_the_current_buffer(self):
        self.assertEqual(qsafe_optimizer_steps_for_block(1, 256), 1)
        self.assertEqual(qsafe_optimizer_steps_for_block(257, 256), 2)
        self.assertEqual(qsafe_optimizer_steps_for_block(1_000, 128, 1.5), 12)


class TransitionContractTests(unittest.TestCase):
    def test_learning_starts_counts_only_the_crossing_remainder(self):
        self.assertEqual(newly_eligible_transitions(0, 256, 500), 0)
        self.assertEqual(newly_eligible_transitions(256, 512, 500), 12)
        self.assertEqual(newly_eligible_transitions(512, 768, 500), 256)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            newly_eligible_transitions(0, 1, -1)

    def test_environment_applied_action_takes_precedence(self):
        proposed = np.zeros((2, 3), dtype=np.float32)
        applied = np.full((2, 3), 0.25, dtype=np.float32)
        resolved = resolve_executed_actions(
            {"applied_action": applied}, proposed, 2, (3,)
        )
        np.testing.assert_array_equal(resolved, applied)

    def test_executed_action_shape_and_finiteness_are_strict(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            resolve_executed_actions(
                {"applied_action": np.zeros((3,), dtype=np.float32)},
                np.zeros((2, 3), dtype=np.float32),
                2,
                (3,),
            )
        invalid = np.zeros((2, 3), dtype=np.float32)
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            resolve_executed_actions(
                {"applied_action": invalid}, invalid, 2, (3,)
            )


class EntropyCoefficientTests(unittest.TestCase):
    def test_auto_target_and_gradient_direction_match_sac(self):
        config = SimpleNamespace(algorithm=SimpleNamespace(target_entropy="auto"))
        coefficient = EntropyCoefficient(config, EnvStub(), "cpu")
        self.assertEqual(coefficient.target_entropy, -12.0)

        coefficient.loss(torch.tensor([0.0])).backward()
        self.assertGreater(coefficient.log_alpha.grad.item(), 0.0)
        coefficient.log_alpha.grad = None
        coefficient.loss(torch.tensor([-20.0])).backward()
        self.assertLess(coefficient.log_alpha.grad.item(), 0.0)


class CheckpointTests(unittest.TestCase):
    @staticmethod
    def make_training():
        class CheckpointEnv:
            def __init__(self):
                self.validation_modes = []

            def checkpoint_manifest(self, normalizer):
                return {"contract": "go2-v1"}

            def validate_checkpoint_manifest(self, manifest, normalizer):
                if manifest != self.checkpoint_manifest(normalizer):
                    raise ValueError("evaluation manifest mismatch")
                self.validation_modes.append("evaluation")

            def validate_transfer_checkpoint_manifest(self, manifest, normalizer):
                if manifest.get("contract") != "go2-v1":
                    raise ValueError("transfer manifest mismatch")
                self.validation_modes.append("transfer")

        training = SQRLWorkflow.__new__(SQRLWorkflow)
        training.config = AttrDict(example=True)
        training.env = CheckpointEnv()
        training.device = torch.device("cpu")
        training.policy = nn.Linear(3, 2)
        training.safe_critic = SimpleNamespace(q=nn.Linear(5, 1), q_target=nn.Linear(5, 1))
        return training

    def test_checkpoint_is_versioned_and_validated_before_loading(self):
        training = self.make_training()
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/artifact.model"
            training.save(path, "pretrain")
            artifact = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(artifact["schema_version"], CHECKPOINT_SCHEMA_VERSION)
            self.assertEqual(artifact["algorithm_id"], CHECKPOINT_ALGORITHM_ID)
            self.assertEqual(artifact["artifact_type"], CHECKPOINT_ARTIFACT_TYPE)
            self.assertEqual(artifact["phase"], "pretrain")
            self.assertTrue(artifact["manifest"])

            phase = training.load(path, transfer=False)
            self.assertEqual(phase, "pretrain")
            self.assertEqual(training.env.validation_modes, ["evaluation"])
            training.load(path, transfer=True)
            self.assertEqual(training.env.validation_modes[-1], "transfer")

    def test_legacy_checkpoint_is_rejected_instead_of_silently_loaded(self):
        training = self.make_training()
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/legacy.model"
            torch.save(
                {
                    "phase": "pretrain",
                    "policy": training.policy.state_dict(),
                    "qsafe": training.safe_critic.q.state_dict(),
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "Legacy or incomplete"):
                training.load(path, transfer=True)

class PretrainerConfigurationTests(unittest.TestCase):
    @staticmethod
    def config():
        return SimpleNamespace(
            environment=AttrDict(seed=5, nr_envs=99),
            algorithm=AttrDict(
                compile_mode="default",
                bf16_mixed_precision_training=False,
                gamma=0.99,
                safe_gamma=0.7,
                tau=0.005,
                batch_size=8,
                n_pre=37,
                n_off=2,
                k=2,
                target_entropy="auto",
                log_std_min=-5,
                log_std_max=1,
                nr_hidden_units=8,
            ),
        )

    def test_n_pre_is_transition_budget_and_env_widths_are_explicit(self):
        from sqrl.sac.pytorch import safety_trainer, task_trainer

        env = EnvStub(5)
        env.nr_task_envs = 3
        env.nr_safety_envs = 2
        dummy_policy = nn.Linear(1, 1)
        task_critic = SimpleNamespace(
            q1=nn.Linear(1, 1), q2=nn.Linear(1, 1),
            q1_target=nn.Linear(1, 1), q2_target=nn.Linear(1, 1),
        )
        safe_critic = SimpleNamespace(q=nn.Linear(1, 1), q_target=nn.Linear(1, 1))

        class DummyEntropy(nn.Module):
            def __init__(self):
                super().__init__()
                self.log_alpha = nn.Parameter(torch.zeros(1))

            def forward(self):
                return self.log_alpha.exp()

            def loss(self, entropy):
                return self.log_alpha.exp() * entropy

        with (
            mock.patch.object(
                task_trainer, "get_policy", return_value=dummy_policy
            ),
            mock.patch.object(
                task_trainer, "get_task_critic", return_value=task_critic
            ),
            mock.patch.object(
                task_trainer,
                "get_entropy_coefficient",
                return_value=DummyEntropy(),
            ),
            mock.patch.object(
                safety_trainer, "get_safe_critic", return_value=safe_critic
            ),
        ):
            trainer = SQRLPretrainer(self.config(), env, "cpu")

        self.assertEqual(trainer.nr_pretrain_steps, 37)
        self.assertEqual(trainer.task.nr_envs, 3)
        self.assertEqual(trainer.safety.nr_envs, 2)
        self.assertEqual(trainer.safety.optimizer_steps, 1)
        self.assertIsNone(trainer.safety.epochs_per_block)
        self.assertIs(trainer.safety.policy, trainer.task.policy)
        self.assertFalse(hasattr(trainer.safety, "policy_optimizer"))

    def test_qsafe_schedule_defaults_to_algorithm_1_single_update(self):
        self.assertEqual(qsafe_update_schedule(AttrDict()), (1, None))
        self.assertEqual(
            qsafe_update_schedule(
                AttrDict(qsafe_optimizer_steps_per_block=3)
            ),
            (3, None),
        )
        self.assertEqual(
            qsafe_update_schedule(AttrDict(qsafe_epochs_per_block=1.5)),
            (None, 1.5),
        )
        with self.assertRaisesRegex(ValueError, "only one"):
            qsafe_update_schedule(
                AttrDict(
                    qsafe_optimizer_steps_per_block=1,
                    qsafe_epochs_per_block=1.0,
                )
            )


class PretrainerTests(unittest.TestCase):
    def test_partition_pretrainer_schedules_n_off_then_safety(self):
        events = []

        class PartitionEnv(EnvStub):
            nr_task_envs = 1
            nr_safety_envs = 1

            def reset_partitions(self):
                events.append("reset_partitions")
                state = np.zeros((1, 46), dtype=np.float32)
                return state, state.copy()

            def reset_task_partition(self):
                events.append("reset_task")
                return np.zeros((1, 46), dtype=np.float32)

        task = mock.Mock(
            policy=object(),
            steps=0,
            updates=0,
        )
        safety = mock.Mock(
            critic=SimpleNamespace(q=object()),
            rollouts=0,
            blocks=0,
            update_steps=0,
        )

        def train_task():
            events.append("task")
            task.steps += 1

        def train_safety():
            events.append("safety")
            return {"qsafe_loss": 0.0}

        task.train_step.side_effect = train_task
        task.reset.side_effect = lambda state: events.append("reset_task_trainer")
        safety.train_block.side_effect = train_safety
        config = SimpleNamespace(
            environment=AttrDict(seed=1),
            algorithm=AttrDict(n_pre=3, n_off=2),
        )
        env = PartitionEnv()
        with (
            mock.patch(
                "sqrl.sac.pytorch.pretrainer.TaskTrainer", return_value=task
            ) as task_constructor,
            mock.patch(
                "sqrl.sac.pytorch.pretrainer.SafetyTrainer", return_value=safety
            ) as safety_constructor,
        ):
            trainer = SQRLPretrainer(config, env, "cpu")
            trainer.train()

        task_constructor.assert_called_once()
        safety_constructor.assert_called_once()
        self.assertIs(safety_constructor.call_args.args[2], task.policy)

        self.assertEqual(
            events,
            [
                "reset_partitions",
                "reset_task_trainer",
                "task",
                "task",
                "safety",
                "reset_task",
                "reset_task_trainer",
                "task",
                "safety",
                "reset_task",
                "reset_task_trainer",
            ],
        )


class QSafeNetworkTests(unittest.TestCase):
    def test_original_qsafe_tanh_output_is_preserved(self):
        env = SimpleNamespace(single_action_space=BoxStub((2,)))
        network = QSafeNetwork(env, 8, "cpu", [0, 1, 2])
        self.assertIsInstance(network.critic[-1], nn.Tanh)
        states = torch.randn(5, 3)
        actions = torch.randn(5, 2)
        output = network(states, actions)
        self.assertTrue(((output >= -1.0) & (output <= 1.0)).all())

        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network.critic[-2].bias.fill_(torch.atanh(torch.tensor(0.05)))
        self.assertTrue((network(states, actions) < 0.1).all())
        with torch.no_grad():
            network.critic[-2].bias.fill_(torch.atanh(torch.tensor(0.2)))
        self.assertTrue((network(states, actions) >= 0.1).all())

    def test_factory_preserves_original_critic_observation_indices(self):
        env = SimpleNamespace(
            general_properties=SimpleNamespace(
                observation_space_type=ObservationSpaceType.FLAT_VALUES
            ),
            single_observation_space=BoxStub((4,)),
            single_action_space=BoxStub((2,)),
            critic_observation_indices=np.asarray([0]),
            safety_critic_observation_indices=np.asarray([1, 3]),
        )
        config = SimpleNamespace(
            algorithm=SimpleNamespace(compile_mode="default", nr_hidden_units=8)
        )
        with mock.patch("torch.compile", side_effect=lambda target, **kwargs: target):
            network = get_qsafe_network(config, env, "cpu")
        self.assertEqual(network.critic_observation_indices.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
