"""SQRL pre-training phase coordinator."""

import logging

import numpy as np
import torch

from sqrl.sac.pytorch.safety_trainer import SafetyTrainer, qsafe_update_schedule
from sqrl.sac.pytorch.task_trainer import TaskTrainer


sqrl_pretrain_logger = logging.getLogger("sqrl_pretrain")


def validate_pretrain_environments(task_env, safety_env):
    """Require independent vector environments with identical model schemas."""

    if task_env is safety_env:
        raise ValueError("Task and safety environments must be independent")
    if int(task_env.num_envs) < 1 or int(safety_env.num_envs) < 1:
        raise ValueError("Task and safety environments must both be non-empty")

    task_observation_shape = tuple(task_env.single_observation_space.shape)
    safety_observation_shape = tuple(safety_env.single_observation_space.shape)
    if task_observation_shape != safety_observation_shape:
        raise ValueError("Task and safety observation shapes must match")

    task_action_space = task_env.single_action_space
    safety_action_space = safety_env.single_action_space
    if tuple(task_action_space.shape) != tuple(safety_action_space.shape):
        raise ValueError("Task and safety action shapes must match")
    if not np.array_equal(
        task_action_space.low, safety_action_space.low
    ) or not np.array_equal(
        task_action_space.high, safety_action_space.high
    ):
        raise ValueError("Task and safety action bounds must match")

    task_properties = getattr(task_env, "general_properties", None)
    safety_properties = getattr(safety_env, "general_properties", None)
    if (task_properties is None) != (safety_properties is None):
        raise ValueError("Task and safety model properties must match")
    if task_properties is not None:
        for name in ("action_space_type", "observation_space_type"):
            if getattr(task_properties, name) != getattr(safety_properties, name):
                raise ValueError(f"Task and safety {name} must match")
        task_policy_indices = getattr(
            task_properties,
            "policy_observation_indices",
            np.arange(task_observation_shape[0]),
        )
        safety_policy_indices = getattr(
            safety_properties,
            "policy_observation_indices",
            np.arange(safety_observation_shape[0]),
        )
        if not np.array_equal(task_policy_indices, safety_policy_indices):
            raise ValueError(
                "Task and safety policy_observation_indices must match"
            )

    task_critic_indices = getattr(
        task_env, "critic_observation_indices", np.arange(task_observation_shape[0])
    )
    safety_critic_indices = getattr(
        safety_env,
        "critic_observation_indices",
        np.arange(safety_observation_shape[0]),
    )
    if not np.array_equal(task_critic_indices, safety_critic_indices):
        raise ValueError("Task and safety critic_observation_indices must match")


class SQRLPretrainer:
    """Run Algorithm 1 by alternating task updates and safety rollouts."""

    def __init__(self, config, task_env, safety_env, device="cpu"):
        validate_pretrain_environments(task_env, safety_env)
        self.task_env = task_env
        self.safety_env = safety_env
        self.device = torch.device(device)
        algorithm = config.algorithm

        self.nr_pretrain_steps = int(algorithm.n_pre)
        self.nr_offline_steps = int(algorithm.n_off)
        if self.nr_pretrain_steps < 1:
            raise ValueError("n_pre must be at least 1")
        if self.nr_offline_steps < 1:
            raise ValueError("n_off must be at least 1")

        seed = int(config.environment.seed)
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.task = TaskTrainer(config, task_env, self.device, rng)
        self.safety = SafetyTrainer(
            config,
            safety_env,
            self.task.policy,
            self.device,
            rng,
        )

    @property
    def policy(self):
        return self.task.policy

    @property
    def safe_critic(self):
        return self.safety.critic

    def train(self, checkpoint_frequency=0, checkpoint_callback=None):
        checkpoint_frequency = int(checkpoint_frequency)
        if checkpoint_frequency < 0:
            raise ValueError("checkpoint_frequency must be non-negative")
        if checkpoint_frequency and checkpoint_callback is None:
            raise ValueError(
                "checkpoint_callback is required when checkpoint_frequency is positive"
            )
        next_checkpoint = checkpoint_frequency

        reset_result = self.task_env.reset()
        task_state = (
            reset_result[0] if isinstance(reset_result, tuple) else reset_result
        )
        self.task.set_state(task_state)

        iteration = 0
        while self.task.steps < self.nr_pretrain_steps:
            for _ in range(self.nr_offline_steps):
                if self.task.steps >= self.nr_pretrain_steps:
                    break
                self.task.train_step()

            # Algorithm 1: collect k complete on-policy trajectories, then
            # update QSafe once by default.
            metrics = self.safety.train_block()
            iteration += 1
            sqrl_pretrain_logger.info(
                "pretrain_iteration=%d task_steps=%d task_updates=%d "
                "safety_rollouts=%d safety_blocks=%d qsafe_optimizer_steps=%d "
                "qsafe_loss=%.6f candidate_fallback_rate=%.6f "
                "selected_safe_q=%.6f epsilon_minus_selected_q=%.6f",
                iteration,
                self.task.steps,
                self.task.updates,
                self.safety.rollouts,
                self.safety.blocks,
                self.safety.update_steps,
                metrics["qsafe_loss"],
                metrics["candidate_fallback_rate"],
                metrics["selected_safe_q"],
                metrics["epsilon_minus_selected_q"],
            )
            if checkpoint_frequency and self.task.steps >= next_checkpoint:
                checkpoint_callback(self.task.steps)
                while next_checkpoint <= self.task.steps:
                    next_checkpoint += checkpoint_frequency

        return self.policy, self.safe_critic.q


__all__ = ["SQRLPretrainer", "validate_pretrain_environments"]
