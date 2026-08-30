"""PyTorch SQRL transfer, fine-tuning, and evaluation on MuJoCo."""
import logging

import numpy as np
import torch

from sqrl.sac.pytorch.finetune import SQRLFinetuner
from train.common.pytorch import SQRLTrainingBase


sqrl_training_logger = logging.getLogger("sqrl_training")


class MujocoTraining(SQRLTrainingBase):
    def finetune(self):
        trainer = SQRLFinetuner(self.config, self.env, self.policy, self.safe_critic.q, self.device)
        self.policy = trainer.train()
        return self.policy

    def evaluate(self, episodes):
        episodes = int(episodes)
        if episodes < 1:
            raise ValueError("evaluation episodes must be positive")
        nr_envs = int(self.config.environment.nr_envs)
        nr_candidates = int(self.config.algorithm.get("max_safe_action_samples", 100))
        epsilon_safe = float(self.config.algorithm.get("epsilon_safe", 0.1))
        reset_result = self.env.reset()
        states = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        returns, failures, running_returns = [], 0, np.zeros(nr_envs, dtype=np.float64)
        self.policy.eval()
        self.safe_critic.q.eval()
        while len(returns) < episodes:
            with torch.no_grad():
                states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
                repeated = states_tensor[:, None, :].expand(-1, nr_candidates, -1)
                flat_states = repeated.reshape(nr_envs * nr_candidates, -1)
                actions, processed_actions, log_probs = self.policy.get_action(flat_states)
                safe_q = self.safe_critic.q(flat_states, actions).reshape(nr_envs, nr_candidates)
                actions = actions.reshape((nr_envs, nr_candidates) + self.env.single_action_space.shape)
                processed_actions = processed_actions.reshape_as(actions)
                log_probs = log_probs.reshape(nr_envs, nr_candidates)
                safe_mask = safe_q < epsilon_safe
                fallback = ~safe_mask.any(dim=1)
                logits = torch.where(safe_mask, log_probs, torch.full_like(log_probs, -torch.inf))
                logits = torch.where(fallback[:, None], torch.zeros_like(logits), logits)
                selected = torch.distributions.Categorical(logits=logits).sample()
                selected = torch.where(fallback, safe_q.argmin(dim=1), selected)
                indices = torch.arange(nr_envs, device=self.device)
                processed_actions = processed_actions[indices, selected].float().cpu().numpy()
            states, rewards, terminations, truncations, info = self.env.step(processed_actions)
            running_returns += np.asarray(rewards, dtype=np.float64)
            failures += int(np.sum(np.asarray(info["failure"], dtype=np.float32)))
            for index in np.flatnonzero(np.asarray(terminations) | np.asarray(truncations)):
                if len(returns) == episodes:
                    break
                if hasattr(self.env, "get_final_info_value_at_index"):
                    returns.append(float(self.env.get_final_info_value_at_index(info, "episode_return", index)))
                else:
                    returns.append(float(running_returns[index]))
                running_returns[index] = 0.0
        metrics = {"episodes": episodes, "mean_return": float(np.mean(returns)), "failures": failures}
        sqrl_training_logger.info("evaluation episodes=%d mean_return=%.6f failures=%d", episodes, metrics["mean_return"], failures)
        return metrics
