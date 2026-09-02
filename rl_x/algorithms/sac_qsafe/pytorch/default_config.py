from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()
    config.name = algorithm_name
    config.device = "gpu"
    config.compile_mode = "reduce-overhead"
    # Compilation is valuable for long training runs but its per-process cost
    # dominates short evaluation/data-collection jobs.
    config.compile_policy = True
    # The SQRL reproduction uses full precision. In the Go2 flat baseline,
    # bf16 caused the SAC critic loss to diverge by 10k transitions.
    config.bf16_mixed_precision_training = False
    # SQRL Appendix B uses 5e5 steps for both pre-training and fine-tuning.
    config.total_timesteps = 5e5
    config.learning_rate = 3e-4
    config.anneal_learning_rate = False
    config.buffer_size = 1e6
    # Match the independently validated Go2 reproduction: collect one thousand
    # task transitions before sampling replay. Starting from the first 256-way
    # vector step repeatedly overfits a nearly empty buffer.
    config.learning_starts = 1000
    config.batch_size = 256
    config.tau = 0.005
    config.gamma = 0.99
    config.target_entropy = "auto"
    # Fine-tuning initializes the target-task entropy temperature from this
    # value; alpha is deliberately not part of the SQRL transfer bundle.
    config.alpha_init = 1.0
    # Fine-tuning creates a new task critic. Hold the transferred actor and
    # fresh entropy temperature fixed until that critic has target-task
    # experience; they subsequently update together.
    config.finetune_actor_warmup_steps = 10000
    # A fresh target-task critic can be exploited by the transferred actor if
    # both are updated 1:1 immediately after warm-up.  This target-stage-only
    # interval lets the critic track policy distribution shift; pretraining is
    # unchanged.  Experiments select the interval explicitly.
    config.finetune_actor_update_interval = 1
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
    config.eval_policy = "safe"  # safe, task
    # Optional machine-readable episode results for paired protection and
    # milestone evaluations. Empty keeps ordinary evaluation side-effect free.
    config.evaluation_results_path = ""

    config.phase = "pretrain"  # pretrain, finetune
    config.pretrained_policy_path = ""
    config.dual_learning_rate = 3e-4
    config.initial_nu = 0.0
    # Algorithm 1: n_off unconstrained task-policy vector steps followed by
    # n_safe complete safety-constrained trajectories per pretrain iteration.
    # Not reported by the paper.  Use the independently validated Go2
    # reproduction's explicit choice for the serial Algorithm-1 reference.
    config.n_off = 1000
    config.n_safe = 1
    config.rollout_mode = "serial_reference"  # serial_reference, partitioned
    config.checkpoint_frequency = 50000
    # Task gradient updates per newly committed task transition. Fractional
    # values are accumulated as credit, making the budget independent of pool
    # size. ``qsafe.updates_per_iteration`` remains the single QSafe control and
    # means updates per atomically completed trajectory in partitioned mode.
    config.task_utd_ratio = 1.0

    config.qsafe = config_dict.ConfigDict()
    # ``False`` is the paper's standard-SAC baseline: no safety rollouts,
    # safety critic, action projection, Eq. 4 term, or dual update.
    config.qsafe.enabled = True
    # v2 is actor-independent: five raw 46D proprioceptive frames (100 ms)
    # have their own normalizer and feed a sigmoid failure-risk critic.
    config.qsafe.version = 2
    config.qsafe.history_length = 5
    config.qsafe.control_dt = 0.02
    config.qsafe.enable_observation_normalization = True
    config.qsafe.normalizer_epsilon = 1e-8
    config.qsafe.checkpoint_path = ""
    config.qsafe.epsilon = 0.1
    config.qsafe.gamma = 0.7
    config.qsafe.learning_rate = 3e-4
    config.qsafe.tau = 0.005
    config.qsafe.buffer_size = 100000
    config.qsafe.batch_size = 256
    # ``auto`` preserves historical checkpoints: legacy-v1 uses the old
    # density-resampling implementation while v2 uses exact rejection
    # sampling.  Experiments may explicitly select ``rejection_sampling`` for
    # a legacy critic without changing or retraining that critic.
    config.qsafe.selection_mode = "auto"
    # The paper does not publish the finite rejection pool size.  The verified
    # Go2 reproduction uses 100 candidates and ten recent complete rollouts.
    config.qsafe.candidate_actions = 100
    config.qsafe.max_trajectories = 10
    config.qsafe.nr_hidden_units = 256
    config.qsafe.updates_per_iteration = 1
    # Optional trajectory-atomic collection used to build universal QSafe v2.
    # It is deliberately off for every normal train/eval preset.
    config.qsafe.dataset = config_dict.ConfigDict()
    config.qsafe.dataset.enabled = False
    config.qsafe.dataset.directory = ""
    config.qsafe.dataset.actor_id = ""
    config.qsafe.dataset.map_seed = 0
    config.qsafe.dataset.episode_offset = 0
    config.qsafe.dataset.split = "train"
    config.qsafe.dataset.terrain = ""
    config.qsafe.dataset.action_noise = 0.0
    config.qsafe.dataset.store_candidates = True
    # In paired protection evaluation both arms draw the same ordered stochastic
    # candidate pool. The baseline executes candidate zero; the protected arm
    # applies Eq. 3 to that pool. A step-indexed RNG stream isolates candidate
    # sampling from simulator and QSafe construction randomness.
    config.qsafe.paired_candidate_evaluation = False
    config.qsafe.eval_candidate_seed = 0
    # Target-domain diagnostic only: load the frozen QSafe and score the exact
    # standard-SAC action plus an independent candidate pool, but never let it
    # select or modify the action sent to the environment. This is deliberately
    # separate from qsafe.enabled so the SQRL treatment remains unchanged.
    config.qsafe.shadow_enabled = False
    config.qsafe.shadow_output_path = ""
    return config
