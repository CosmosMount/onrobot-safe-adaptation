from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()
    config.name = algorithm_name
    config.device = "gpu"
    config.compile_mode = "reduce-overhead"
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
    # Keep the historical default while allowing reward-scale-matched target
    # experiments. The Go2 reward is multiplied by control_dt, so resetting a
    # transferred actor to alpha=1 can overwhelm its task objective.
    config.alpha_init = 1.0
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

    config.phase = "pretrain"  # pretrain, finetune
    config.pretrained_policy_path = ""
    # A complete pre-training checkpoint is used during cross-backend
    # fine-tuning to restore the task critics and entropy coefficient alongside
    # the policy. Starting those components from scratch immediately corrupts
    # an otherwise stable transferred actor.
    config.pretrained_task_checkpoint_path = ""
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

    # Safety objective used by the target-task learner. ``sqrl`` preserves the
    # original rejection/dual formulation; ``sorl`` follows arXiv:2402.15197
    # by shaping rewards with a safety critic instead of constraining actions.
    config.safety_objective = "sqrl"  # sqrl, sorl
    config.sorl = config_dict.ConfigDict()
    config.sorl.significance = 1.0
    config.sorl.solve_significance = True
    # Delta=0 is the paper's performance-oriented boundary.  Positive values
    # make the learned policy more conservative and remain command-line tunable.
    config.sorl.target_delta = 0.0
    # Go2's dense command reward is deliberately negative when locomotion is
    # too slow.  Strict Eq. (10) nearly deletes that signal when Q_safe is
    # small, creating a safe stationary optimum.  Keep the strict paper mode
    # available by setting this flag to false.
    config.sorl.preserve_negative_task_penalty = True
    config.sorl.horizon = 10
    config.sorl.cost_margin = 1e-3
    config.sorl.unsafe_replay_fraction = 0.5
    config.sorl.safety_learning_starts = 1000
    config.sorl.safety_updates_per_step = 1

    # Cross-simulator transfer stabilization.  These settings do not alter
    # scratch pre-training: MuJoCo first collects deterministic target data,
    # then adapts the task critic before allowing conservative actor updates.
    config.transfer_deterministic_steps = 5000
    # Contract the source Gaussian during physical target collection.  The
    # unscaled Isaac policy noise is large enough to break a viable gait before
    # any actor gradient is applied; SAC remains stochastic and off-policy.
    config.transfer_exploration_scale = 0.25
    # The target critic is initially a source-task value function.  A 10k gate
    # still sent the transferred gait in the wrong direction within 500 actor
    # updates, so collect enough target transitions for critic calibration
    # before changing the physical policy.
    config.transfer_actor_learning_starts = 50000
    config.transfer_actor_learning_rate = 3e-6
    config.transfer_actor_update_interval = 2
    config.transfer_actor_anchor_coefficient = 1.0

    config.qsafe = config_dict.ConfigDict()
    # Fine-tuning ablation switch. Pre-training always requires QSafe; when
    # disabled in the target phase, both action masking and the actor/dual
    # safety constraint are removed, yielding the matched SAC baseline.
    config.qsafe.enabled = True
    config.qsafe.checkpoint_path = ""
    config.qsafe.epsilon = 0.1
    config.qsafe.gamma = 0.7
    config.qsafe.learning_rate = 3e-4
    config.qsafe.tau = 0.005
    config.qsafe.buffer_size = 100000
    config.qsafe.batch_size = 256
    # The paper does not publish the finite rejection pool size.  The verified
    # Go2 reproduction uses 100 candidates and ten recent complete rollouts.
    config.qsafe.candidate_actions = 100
    # ``importance`` follows the paper's practical finite-candidate selector.
    # ``first_safe`` is the neutral rejection-sampling ablation used by the
    # independent Go2 reproduction.
    config.qsafe.finetune_selector = "importance"
    config.qsafe.max_trajectories = 10
    config.qsafe.nr_hidden_units = 256
    config.qsafe.updates_per_iteration = 1
    return config
