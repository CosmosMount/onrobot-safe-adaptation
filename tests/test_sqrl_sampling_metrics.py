import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from sqrl.sac.pytorch.safety_trainer import SafetyTrainer
from sqrl.sac.pytorch.workflow import SQRLWorkflow


class SafetySamplingMetricsTests(unittest.TestCase):
    def test_pretrain_block_aggregates_candidate_metrics(self):
        next_state = np.zeros((2, 1), dtype=np.float32)
        step_result = (
            next_state,
            np.zeros(2, dtype=np.float32),
            np.ones(2, dtype=bool),
            np.zeros(2, dtype=bool),
            {"applied_action": np.full((2, 1), -0.5, dtype=np.float32)},
        )
        env = SimpleNamespace(
            single_action_space=SimpleNamespace(shape=(1,)),
            reset=mock.Mock(
                return_value=(np.zeros((2, 1), dtype=np.float32), {})
            ),
            step=mock.Mock(side_effect=[step_result, step_result]),
        )
        trainer = SafetyTrainer.__new__(SafetyTrainer)
        trainer.env = env
        trainer.policy = mock.Mock()
        trainer.critic = SimpleNamespace(q=mock.Mock())
        trainer.nr_envs = 2
        trainer.rollouts_per_block = 2
        trainer.epsilon = 0.1
        trainer.optimizer_steps = 1
        trainer.epochs_per_block = None
        trainer.replay_buffer = mock.Mock(size=2)
        trainer.replay_buffer.add.side_effect = [
            np.asarray([False, False]),
            np.asarray([True, True]),
        ]
        trainer.rollouts = 0
        trainer.blocks = 0
        trainer.update_steps = 0
        trainer._sample_actions = mock.Mock(
            return_value=(
                np.full((2, 1), 0.25, dtype=np.float32),
                np.full((2, 1), 0.75, dtype=np.float32),
                np.asarray([0.05, 0.15], dtype=np.float32),
                np.asarray([False, True]),
            )
        )
        trainer._process_step = mock.Mock(
            return_value=(
                np.zeros((2, 1), dtype=np.float32),
                np.ones(2, dtype=bool),
                np.zeros(2, dtype=bool),
                np.zeros(2, dtype=np.float32),
            )
        )
        trainer._update_critic = mock.Mock(
            return_value={"qsafe_loss": 0.2, "safe_q": 0.1, "safe_target": 0.1}
        )

        metrics = trainer.train_block()

        self.assertAlmostEqual(metrics["candidate_fallback_rate"], 0.5)
        self.assertAlmostEqual(metrics["selected_safe_q"], 0.1)
        self.assertAlmostEqual(metrics["epsilon_minus_selected_q"], 0.0)
        env.reset.assert_called_once_with()
        self.assertEqual(env.step.call_count, 2)
        np.testing.assert_array_equal(
            env.step.call_args.args[0],
            np.full((2, 1), 0.75, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            trainer.replay_buffer.add.call_args.args[2],
            np.full((2, 1), 0.25, dtype=np.float32),
        )

    def test_evaluation_aggregates_candidate_metrics(self):
        class EvaluationEnv:
            num_envs = 1

            def reset(self):
                return np.zeros((1, 1), dtype=np.float32), {}

            def step(self, actions):
                return (
                    np.zeros((1, 1), dtype=np.float32),
                    np.ones(1, dtype=np.float32),
                    np.ones(1, dtype=bool),
                    np.zeros(1, dtype=bool),
                    {"failure": np.zeros(1, dtype=np.float32)},
                )

        workflow = SQRLWorkflow.__new__(SQRLWorkflow)
        workflow.config = SimpleNamespace(
            algorithm={"max_safe_action_samples": 100, "epsilon_safe": 0.1}
        )
        workflow.env = EvaluationEnv()
        workflow.device = torch.device("cpu")
        workflow.policy = mock.Mock()
        workflow.safe_critic = SimpleNamespace(q=mock.Mock())
        samples = [
            (
                torch.zeros((1, 1)),
                torch.zeros((1, 1)),
                torch.tensor([0.05]),
                torch.tensor([False]),
            ),
            (
                torch.zeros((1, 1)),
                torch.zeros((1, 1)),
                torch.tensor([0.15]),
                torch.tensor([True]),
            ),
        ]

        with mock.patch(
            "sqrl.sac.pytorch.workflow.sample_safe_actions", side_effect=samples
        ):
            metrics = workflow.evaluate(2)

        self.assertAlmostEqual(metrics["candidate_fallback_rate"], 0.5)
        self.assertAlmostEqual(metrics["selected_safe_q"], 0.1)
        self.assertAlmostEqual(metrics["epsilon_minus_selected_q"], 0.0)


if __name__ == "__main__":
    unittest.main()
