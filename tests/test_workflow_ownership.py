import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sqrl.sac.pytorch import workflow as sqrl_workflow


class WorkflowOwnershipTest(unittest.TestCase):
    def test_sqrl_owns_pretraining(self):
        owner = sqrl_workflow.SQRLWorkflow.__new__(sqrl_workflow.SQRLWorkflow)
        owner.config = object()
        owner.device = object()
        owner.env = object()
        safety_env = object()
        trained_policy = object()
        trained_critic = SimpleNamespace(q=object())
        trainer = mock.Mock(policy=trained_policy, safe_critic=trained_critic)

        with mock.patch.object(
            sqrl_workflow, "SQRLPretrainer", return_value=trainer
        ) as constructor:
            result = owner.pretrain(safety_env)

        constructor.assert_called_once_with(
            owner.config, owner.env, safety_env, owner.device
        )
        trainer.train.assert_called_once_with()
        self.assertEqual(result, (trained_policy, trained_critic.q))
        self.assertIs(owner.policy, trained_policy)
        self.assertIs(owner.safe_critic, trained_critic)

    def test_sqrl_owns_finetuning(self):
        owner = sqrl_workflow.SQRLWorkflow.__new__(sqrl_workflow.SQRLWorkflow)
        owner.config = object()
        owner.env = object()
        owner.policy = object()
        owner.safe_critic = SimpleNamespace(q=object())
        owner.device = object()
        original_policy = owner.policy
        trained_policy = object()
        trainer = mock.Mock()
        trainer.train.return_value = trained_policy

        with mock.patch.object(
            sqrl_workflow, "SQRLFinetuner", return_value=trainer
        ) as constructor:
            result = owner.finetune()

        constructor.assert_called_once_with(
            owner.config,
            owner.env,
            original_policy,
            owner.safe_critic.q,
            owner.device,
        )
        self.assertIs(result, trained_policy)
        self.assertIs(owner.policy, trained_policy)

    def test_sqrl_has_no_dependency_on_train_or_process_packages(self):
        root = Path(__file__).resolve().parents[1] / "sqrl"
        violations = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    if any(
                        alias.name == "multiprocessing"
                        or alias.name.startswith("multiprocessing.")
                        or alias.name == "train"
                        or alias.name.startswith("train.")
                        for alias in node.names
                    ):
                        violations.append(str(path))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module == "multiprocessing" or module.startswith(
                        "multiprocessing."
                    ) or module == "train" or module.startswith("train."):
                        violations.append(str(path))
        self.assertEqual(violations, [])

    def test_sqrl_has_no_legacy_partition_transport_api(self):
        root = Path(__file__).resolve().parents[1] / "sqrl"
        forbidden = (
            "EnvironmentProcess",
            "reset_partitions",
            "step_partitions",
            "sync_state",
            "sync_steps",
        )
        violations = []
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for name in forbidden:
                if name in source:
                    violations.append(f"{path}:{name}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
