import multiprocessing
import os
import time
import unittest

import numpy as np
from ml_collections import config_dict

from train.core.process_environment import (
    EnvironmentProcess,
    ProcessEnvironmentError,
)
from train.isaac.pytorch.config import get_config
from train.isaac.pytorch.pretrain_environment import IsaacPretrainEnvironments


class CountingEnvironment:
    """Pickle-safe stand-in for one child-local Isaac environment."""

    def __init__(self, config_data, role, nr_envs):
        self.role = role
        self.seed = int(config_data["environment"]["seed"])
        self.nr_envs = int(nr_envs)
        self.state = 0

    def reset(self):
        self.state = 0
        return self._observations(), {
            "role": self.role,
            "pid": os.getpid(),
            "seed": self.seed,
        }

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.nr_envs, 12):
            raise ValueError(f"invalid action shape {actions.shape}")
        self.state += 1
        return (
            self._observations(),
            np.zeros(self.nr_envs, dtype=np.float32),
            np.zeros(self.nr_envs, dtype=bool),
            np.zeros(self.nr_envs, dtype=bool),
            {"role": self.role, "pid": os.getpid(), "state": self.state},
        )

    def _observations(self):
        return np.full((self.nr_envs, 46), self.state, dtype=np.float32)

    def close(self):
        pass


def create_counting_environment(config_data, role, nr_envs):
    return CountingEnvironment(config_data, role, nr_envs)


class FailingEnvironment:
    def reset(self):
        return None

    def step(self, actions):
        del actions
        raise RuntimeError("deliberate worker failure")

    def close(self):
        pass


def create_failing_environment():
    return FailingEnvironment()


class SlowEnvironment(FailingEnvironment):
    def step(self, actions):
        del actions
        time.sleep(10.0)


def create_slow_environment():
    return SlowEnvironment()


def create_invalid_environment():
    raise ValueError("deliberate startup failure")


class ExitingEnvironment(FailingEnvironment):
    def step(self, actions):
        del actions
        os._exit(7)


def create_exiting_environment():
    return ExitingEnvironment()


class CloseNotifyingEnvironment(FailingEnvironment):
    def __init__(self, closed_event):
        self.closed_event = closed_event

    def close(self):
        self.closed_event.set()


def create_close_notifying_environment(closed_event):
    return CloseNotifyingEnvironment(closed_event)


class CloseFailingEnvironment(FailingEnvironment):
    def close(self):
        raise RuntimeError("deliberate close failure")


def create_close_failing_environment():
    return CloseFailingEnvironment()


def pretrain_config():
    environment = get_config("go2_sqrl.isaac_lab")
    environment.nr_envs = 3
    environment.nr_task_envs = 2
    environment.nr_safety_envs = 1
    return config_dict.ConfigDict({"environment": environment})


class IsaacPretrainProcessTests(unittest.TestCase):
    def test_task_state_is_strictly_paused_during_safety_steps(self):
        environments = IsaacPretrainEnvironments(
            pretrain_config(),
            environment_factory=create_counting_environment,
        )
        try:
            self.assertNotEqual(environments.task.pid, environments.safety.pid)
            self.assertNotEqual(environments.task.pid, os.getpid())
            self.assertNotEqual(environments.safety.pid, os.getpid())
            self.assertEqual(environments.task.num_envs, 2)
            self.assertEqual(environments.safety.num_envs, 1)
            self.assertEqual(environments.task.config.nr_envs, 2)
            self.assertEqual(environments.safety.config.nr_envs, 1)
            self.assertEqual(environments.task.config.seed, 0)
            self.assertEqual(environments.safety.config.seed, 1)

            task_state, task_info = environments.task.reset()
            safety_state, safety_info = environments.safety.reset()
            self.assertEqual(task_state.shape, (2, 46))
            self.assertEqual(safety_state.shape, (1, 46))
            self.assertEqual(task_info["role"], "task")
            self.assertEqual(safety_info["role"], "safety")
            self.assertEqual(task_info["seed"], 0)
            self.assertEqual(safety_info["seed"], 1)

            task_actions = np.zeros((2, 12), dtype=np.float32)
            safety_actions = np.zeros((1, 12), dtype=np.float32)
            first_task_state = environments.task.step(task_actions)[0]
            for expected_state in range(1, 5):
                current_safety_state = environments.safety.step(safety_actions)[0]
                np.testing.assert_array_equal(
                    current_safety_state,
                    np.full((1, 46), expected_state, dtype=np.float32),
                )

            second_task_state = environments.task.step(task_actions)[0]
            np.testing.assert_array_equal(
                first_task_state, np.ones((2, 46), dtype=np.float32)
            )
            np.testing.assert_array_equal(
                second_task_state, np.full((2, 46), 2.0, dtype=np.float32)
            )
        finally:
            environments.close()
            environments.close()

    def test_remote_failure_is_reported_and_worker_becomes_unavailable(self):
        process = EnvironmentProcess(
            create_failing_environment,
            name="failing-test-environment",
        )
        try:
            process.start()
            process.wait_until_ready()
            with self.assertRaises(ProcessEnvironmentError) as raised:
                process.step(None)
            message = str(raised.exception)
            self.assertIn("step failed", message)
            self.assertIn("RuntimeError", message)
            self.assertIn("deliberate worker failure", message)
            with self.assertRaisesRegex(ProcessEnvironmentError, "unavailable"):
                process.reset()
        finally:
            process.close()
        self.assertFalse(process.is_alive)

    def test_timeout_terminates_worker_instead_of_losing_state_alignment(self):
        process = EnvironmentProcess(
            create_slow_environment,
            name="slow-test-environment",
            request_timeout=0.05,
        )
        try:
            process.start()
            process.wait_until_ready()
            with self.assertRaisesRegex(
                ProcessEnvironmentError, "did not respond"
            ):
                process.step(None)
            self.assertFalse(process.is_alive)
        finally:
            process.close()

    def test_worker_exit_code_is_reported_without_waiting_for_timeout(self):
        process = EnvironmentProcess(
            create_exiting_environment,
            name="exiting-test-environment",
            request_timeout=5.0,
        )
        try:
            process.start()
            process.wait_until_ready()
            with self.assertRaisesRegex(ProcessEnvironmentError, "code 7"):
                process.step(None)
        finally:
            process.close()

    def test_normal_close_runs_child_environment_cleanup(self):
        context = multiprocessing.get_context("spawn")
        closed_event = context.Event()
        process = EnvironmentProcess(
            create_close_notifying_environment,
            (closed_event,),
            name="close-test-environment",
        )
        process.start()
        process.wait_until_ready()
        process.close()

        self.assertTrue(closed_event.wait(timeout=1.0))
        self.assertFalse(process.is_alive)
        self.assertIsNone(process.close_error)
        process.close()

    def test_close_failure_keeps_remote_traceback_for_diagnostics(self):
        process = EnvironmentProcess(
            create_close_failing_environment,
            name="close-failing-test-environment",
        )
        process.start()
        process.wait_until_ready()
        with self.assertLogs(
            "train.core.process_environment", level="WARNING"
        ):
            process.close()

        self.assertIsNotNone(process.close_error)
        message = str(process.close_error)
        self.assertIn("deliberate close failure", message)
        self.assertIn("Traceback", message)
        self.assertNotIn("NoneType: None", message)

    def test_startup_failure_is_propagated(self):
        process = EnvironmentProcess(
            create_invalid_environment,
            name="invalid-test-environment",
        )
        try:
            process.start()
            with self.assertRaises(ProcessEnvironmentError) as raised:
                process.wait_until_ready()
            self.assertIn("startup failed", str(raised.exception))
            self.assertIn("deliberate startup failure", str(raised.exception))
        finally:
            process.close()
        self.assertFalse(process.is_alive)


if __name__ == "__main__":
    unittest.main()
