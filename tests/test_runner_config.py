import unittest

from train.runner import _validate_experiment_config


class RunnerConfigTest(unittest.TestCase):
    def test_command_is_the_single_source_of_runner_mode(self):
        for command, expected_mode in (
            ("pretrain", "train"),
            ("finetune", "train"),
            ("zero-shot", "test"),
            ("eval", "test"),
        ):
            with self.subTest(command=command):
                algorithm, mode = _validate_experiment_config({}, command)
                self.assertEqual(algorithm, "sqrl_sac")
                self.assertEqual(mode, expected_mode)

    def test_unsupported_algorithm_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "unsupported experiment.algorithm"):
            _validate_experiment_config(
                {"experiment": {"algorithm": "droq"}}, "pretrain"
            )

    def test_legacy_experiment_mode_is_rejected_instead_of_ignored(self):
        with self.assertRaisesRegex(ValueError, "unsupported experiment keys: mode"):
            _validate_experiment_config(
                {"experiment": {"mode": "train"}}, "pretrain"
            )

    def test_unsupported_environment_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "unsupported experiment.environment"):
            _validate_experiment_config(
                {"experiment": {"environment": "minitaur"}}, "pretrain"
            )


if __name__ == "__main__":
    unittest.main()
