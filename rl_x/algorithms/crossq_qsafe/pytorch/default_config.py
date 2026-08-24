from rl_x.algorithms.crossq.pytorch.default_config import get_config as get_crossq_config
from rl_x.algorithms.sac_qsafe.pytorch.default_config import get_config as get_sqrl_config


def get_config(algorithm_name):
    """Return CrossQ defaults extended with the shared SQRL controls.

    Task-learner hyperparameters intentionally remain CrossQ's paper defaults;
    only the phase, rollout, dual and QSafe configuration is borrowed from the
    reference SAC-QSafe implementation.
    """

    config = get_crossq_config(algorithm_name)
    sqrl = get_sqrl_config("sac_qsafe.pytorch")
    for key in (
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
    ):
        config[key] = sqrl[key]
    # CrossQ already normalizes inputs with Batch Renormalization.  Keep the
    # shared transfer-contract object, but make it an identity transform by
    # default so task learning is not normalized twice.
    config.enable_observation_normalization = False
    config.qsafe = sqrl.qsafe.copy_and_resolve_references()
    return config
