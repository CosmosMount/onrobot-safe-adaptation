from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()
    config.name = algorithm_name
    config.device = "gpu"
    config.compile_mode = "reduce-overhead"
    config.bf16_mixed_precision_training = True
    # SQRL Appendix B uses 5e5 steps for both pre-training and fine-tuning.
    config.total_timesteps = 5e5
    config.learning_rate = 3e-4
    config.anneal_learning_rate = False
    config.buffer_size = 1e6
    # Algorithm 1 applies a SAC update in every task step. A non-zero value is
    # still accepted as an explicit engineering override for replay warm-up.
    config.learning_starts = 0
    config.batch_size = 256
    config.tau = 0.005
    config.gamma = 0.99
    config.target_entropy = "auto"
    config.log_std_min = -20
    config.log_std_max = 2
    config.nr_hidden_units = 256
    config.enable_observation_normalization = True
    config.normalizer_epsilon = 1e-8
    # Frequencies are measured in task transitions. With hundreds of parallel
    # environments, logging every few hundred transitions would mean printing
    # almost every vector step.
    config.logging_frequency = 50000
    config.evaluation_frequency = -1
    config.evaluation_episodes = 10

    config.phase = "pretrain"  # pretrain, finetune
    config.pretrained_policy_path = ""
    config.dual_learning_rate = 3e-4
    config.initial_nu = 0.0
    # Algorithm 1: n_off unconstrained task-policy vector steps followed by
    # n_safe complete safety-constrained trajectories per pretrain iteration.
    config.n_off = 1
    config.n_safe = 1
    config.rollout_mode = "serial_reference"  # serial_reference, partitioned
    config.checkpoint_frequency = 50000
    # Task gradient updates per newly committed task transition. Fractional
    # values are accumulated as credit, making the budget independent of pool
    # size. ``qsafe.updates_per_iteration`` remains the single QSafe control and
    # means updates per atomically completed trajectory in partitioned mode.
    config.task_utd_ratio = 1.0

    config.qsafe = config_dict.ConfigDict()
    config.qsafe.checkpoint_path = ""
    config.qsafe.epsilon = 0.1
    config.qsafe.gamma = 0.7
    config.qsafe.learning_rate = 3e-4
    config.qsafe.tau = 0.005
    config.qsafe.buffer_size = 100000
    config.qsafe.batch_size = 256
    config.qsafe.candidate_actions = 10
    config.qsafe.nr_hidden_units = 256
    config.qsafe.updates_per_iteration = 1
    return config
