# DroQ-QSafe (SQRL)

DroQ-QSafe combines DroQ's dropout critic ensemble and high update-to-data
ratio with the independent SQRL safety critic used by `sac_qsafe`. It is
registered as `droq_qsafe.pytorch` and `droq_qsafe.flax`.

The safety contract is unchanged: pre-training keeps task and complete safety
trajectories in separate replay buffers; `info["failure"]` is the binary
post-transition failure indicator and must imply `terminated=True`;
fine-tuning freezes QSafe, projects rollout actions, adds the QSafe constraint
to the actor objective, and projects the learned dual variable to be
non-negative.

DroQ semantics are retained in both backends. Every task learner update runs
`q_update_steps` critic updates, samples `in_target_minimization_size` members
from the target ensemble, applies dropout independently to online and target
passes, performs Polyak target updates, and then updates the actor and entropy
coefficient once. The partitioned PyTorch trainer treats this complete DroQ
learner cycle as one credited task update.

Task hyperparameters come from the corresponding native DroQ backend; phase,
normalization, rollout, checkpoint, dual, and nested `qsafe` controls come from
SAC-QSafe. DroQ's policy architecture matches SAC's dense policy, so the
existing strict PyTorch-to-Flax policy transfer format remains supported.
