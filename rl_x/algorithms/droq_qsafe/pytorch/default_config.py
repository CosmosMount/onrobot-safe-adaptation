from rl_x.algorithms.droq.pytorch.default_config import get_config as get_droq_config
from rl_x.algorithms.sac_qsafe.pytorch.default_config import (
    get_config as get_sac_qsafe_config,
)


_QSAFE_FIELDS = (
    "enable_observation_normalization",
    "normalizer_epsilon",
    "phase",
    "pretrained_policy_path",
    "dual_learning_rate",
    "initial_nu",
    "n_off",
    "n_safe",
    "rollout_mode",
    "checkpoint_frequency",
    "task_utd_ratio",
    "qsafe",
)


def add_qsafe_defaults(config, algorithm_name):
    qsafe_config = get_sac_qsafe_config(algorithm_name)
    for field in _QSAFE_FIELDS:
        config[field] = qsafe_config[field]
    return config


def get_config(algorithm_name):
    return add_qsafe_defaults(get_droq_config(algorithm_name), algorithm_name)
