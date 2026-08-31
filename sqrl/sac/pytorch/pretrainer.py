"""SQRL pre-training phase coordinator."""

import logging

import numpy as np
import torch

from sqrl.sac.pytorch.safety_trainer import SafetyTrainer, qsafe_update_schedule
from sqrl.sac.pytorch.task_trainer import TaskTrainer


sqrl_pretrain_logger = logging.getLogger("sqrl_pretrain")


class SQRLPretrainer:
    """Run Algorithm 1 by alternating task updates and safety rollouts."""

    def __init__(self, config, env, device="cpu"):
        self.env = env
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

        self.task = TaskTrainer(config, env, self.device, rng)
        self.safety = SafetyTrainer(
            config,
            env,
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

    def train(self):
        task_state, _ = self.env.reset_partitions()
        self.task.reset(task_state)

        iteration = 0
        while self.task.steps < self.nr_pretrain_steps:
            for _ in range(self.nr_offline_steps):
                if self.task.steps >= self.nr_pretrain_steps:
                    break
                self.task.train_step()

            # Algorithm 1: collect k complete on-policy trajectories, then
            # update QSafe once by default.
            metrics = self.safety.train_block()
            self.task.reset(self.env.reset_task_partition())
            iteration += 1
            sqrl_pretrain_logger.info(
                "pretrain_iteration=%d task_steps=%d task_updates=%d "
                "safety_rollouts=%d safety_blocks=%d qsafe_optimizer_steps=%d "
                "qsafe_loss=%.6f",
                iteration,
                self.task.steps,
                self.task.updates,
                self.safety.rollouts,
                self.safety.blocks,
                self.safety.update_steps,
                metrics["qsafe_loss"],
            )

        return self.policy, self.safe_critic.q


__all__ = ["SQRLPretrainer"]
