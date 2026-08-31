"""Public SQRL workflow: phases, evaluation, and checkpoints."""

import logging
import os

import numpy as np
import torch

from algorithms.qsafe.pytorch.critic import get_critic as get_safe_critic
from algorithms.sac.pytorch.policy import get_policy
from sqrl.sac.pytorch.finetuner import SQRLFinetuner
from sqrl.sac.pytorch.pretrainer import SQRLPretrainer
from sqrl.sac.pytorch.safety_ops import sample_safe_actions


CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_ALGORITHM_ID = "sqrl_sac"
CHECKPOINT_ARTIFACT_TYPE = "policy_qsafe_transfer"
CHECKPOINT_PHASES = frozenset({"pretrain", "finetune"})

sqrl_workflow_logger = logging.getLogger("sqrl_workflow")


class SQRLWorkflow:
    """Own all algorithm execution while depending only on environment APIs."""

    def __init__(self, config, env, device):
        self.config = config
        self.env = env
        self.device = torch.device(device)
        self.policy = None
        self.safe_critic = None

        seed = int(config.environment.seed)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def _ensure_models(self):
        if self.policy is None:
            self.policy = get_policy(self.config, self.env, self.device)
        if self.safe_critic is None:
            self.safe_critic = get_safe_critic(self.config, self.env, self.device)

    def pretrain(self):
        """Run source pre-training on the environment partitions."""

        trainer = SQRLPretrainer(self.config, self.env, self.device)
        trainer.train()
        self.policy = trainer.policy
        self.safe_critic = trainer.safe_critic
        return self.policy, self.safe_critic.q

    def finetune(self):
        """Run target-domain SQRL fine-tuning."""

        self._ensure_models()
        trainer = SQRLFinetuner(
            self.config,
            self.env,
            self.policy,
            self.safe_critic.q,
            self.device,
        )
        self.policy = trainer.train()
        return self.policy

    def evaluate(self, episodes):
        """Evaluate the loaded SQRL policy with QSafe action selection."""

        self._ensure_models()
        episodes = int(episodes)
        if episodes < 1:
            raise ValueError("evaluation episodes must be positive")

        nr_envs = int(
            getattr(self.env, "num_envs", getattr(self.env, "nr_envs", 1))
        )
        nr_candidates = int(
            self.config.algorithm.get("max_safe_action_samples", 100)
        )
        epsilon_safe = float(self.config.algorithm.get("epsilon_safe", 0.1))
        reset_result = self.env.reset()
        states = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        returns = []
        failures = 0
        running_returns = np.zeros(nr_envs, dtype=np.float64)

        self.policy.eval()
        self.safe_critic.q.eval()
        while len(returns) < episodes:
            with torch.no_grad():
                states_tensor = torch.as_tensor(
                    states, dtype=torch.float32, device=self.device
                )
                _, processed_actions, _, _ = sample_safe_actions(
                    states_tensor,
                    self.policy,
                    self.safe_critic.q,
                    nr_candidates,
                    epsilon_safe,
                )
            states, rewards, terminations, truncations, info = self.env.step(
                processed_actions.float().cpu().numpy()
            )
            running_returns += np.asarray(rewards, dtype=np.float64)
            failure_values = info.get("failure", info.get("failures"))
            if failure_values is None:
                raise ValueError("evaluation environment must provide failure labels")
            failures += int(np.sum(np.asarray(failure_values, dtype=np.float32)))
            done = np.asarray(terminations) | np.asarray(truncations)
            for index in np.flatnonzero(done):
                if len(returns) == episodes:
                    break
                if hasattr(self.env, "get_final_info_value_at_index"):
                    value = self.env.get_final_info_value_at_index(
                        info, "episode_return", index
                    )
                    returns.append(float(value))
                else:
                    returns.append(float(running_returns[index]))
                running_returns[index] = 0.0

        metrics = {
            "episodes": episodes,
            "mean_return": float(np.mean(returns)),
            "failures": failures,
        }
        sqrl_workflow_logger.info(
            "evaluation episodes=%d mean_return=%.6f failures=%d",
            episodes,
            metrics["mean_return"],
            failures,
        )
        return metrics

    def save(self, path, phase):
        """Save the cross-backend policy/QSafe transfer artifact."""

        self._ensure_models()
        phase = str(phase)
        if phase not in CHECKPOINT_PHASES:
            raise ValueError(
                f"Checkpoint phase must be one of {sorted(CHECKPOINT_PHASES)}, "
                f"got {phase!r}"
            )
        manifest_builder = getattr(self.env, "checkpoint_manifest", None)
        if manifest_builder is None:
            raise ValueError("Environment must provide checkpoint_manifest()")
        manifest = manifest_builder(None)
        if not isinstance(manifest, dict) or not manifest:
            raise ValueError("checkpoint_manifest() must return a non-empty mapping")

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "algorithm_id": CHECKPOINT_ALGORITHM_ID,
                "artifact_type": CHECKPOINT_ARTIFACT_TYPE,
                "phase": phase,
                "policy": self.policy.state_dict(),
                "qsafe": self.safe_critic.q.state_dict(),
                "manifest": manifest,
                "config": self.config.to_dict()
                if hasattr(self.config, "to_dict")
                else dict(self.config),
            },
            path,
        )

    def load(self, path, transfer=False):
        """Load a versioned artifact after validating its environment contract."""

        self._ensure_models()
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a versioned mapping")
        required = {
            "schema_version",
            "algorithm_id",
            "artifact_type",
            "phase",
            "policy",
            "qsafe",
            "manifest",
        }
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise ValueError(
                "Legacy or incomplete checkpoint is not a supported transfer "
                f"artifact; missing fields: {', '.join(missing)}"
            )
        if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported checkpoint schema_version "
                f"{checkpoint['schema_version']!r}; expected "
                f"{CHECKPOINT_SCHEMA_VERSION}"
            )
        if checkpoint["algorithm_id"] != CHECKPOINT_ALGORITHM_ID:
            raise ValueError(
                f"Checkpoint algorithm_id must be {CHECKPOINT_ALGORITHM_ID!r}, "
                f"got {checkpoint['algorithm_id']!r}"
            )
        if checkpoint["artifact_type"] != CHECKPOINT_ARTIFACT_TYPE:
            raise ValueError(
                "Checkpoint artifact_type must be "
                f"{CHECKPOINT_ARTIFACT_TYPE!r}, got "
                f"{checkpoint['artifact_type']!r}"
            )
        if checkpoint["phase"] not in CHECKPOINT_PHASES:
            raise ValueError(f"Invalid checkpoint phase {checkpoint['phase']!r}")

        manifest = checkpoint["manifest"]
        if not isinstance(manifest, dict) or not manifest:
            raise ValueError("Checkpoint manifest must be a non-empty mapping")
        validator_name = (
            "validate_transfer_checkpoint_manifest"
            if transfer
            else "validate_checkpoint_manifest"
        )
        validator = getattr(self.env, validator_name, None)
        if validator is None:
            mode = "transfer" if transfer else "evaluation"
            raise ValueError(
                f"Environment must provide {validator_name}() for {mode} loading"
            )
        validator(manifest, None)

        self.policy.load_state_dict(checkpoint["policy"], strict=True)
        self.safe_critic.q.load_state_dict(checkpoint["qsafe"], strict=True)
        self.safe_critic.q_target.load_state_dict(checkpoint["qsafe"], strict=True)
        return checkpoint["phase"]
