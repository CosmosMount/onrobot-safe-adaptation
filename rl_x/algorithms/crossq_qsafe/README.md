# CrossQ-QSafe (SQRL)

CrossQ-QSafe combines the task learner from
[CrossQ](https://openreview.net/pdf?id=PczQtTsTIX) with the reusable safety
critic and action projection used by `sac_qsafe`.  Both the PyTorch and Flax
implementations expose the normal RL-X algorithm interface as
`crossq_qsafe.pytorch` and `crossq_qsafe.flax`.

The SQRL environment and replay contracts are identical to SAC-QSafe:

- `info["failure"]` is the post-transition binary indicator `I(s')` and every
  failure must also set `terminated=True`.
- Pre-training alternates `n_off` unconstrained task-policy vector steps with
  `n_safe` complete trajectories from the QSafe boundary policy.  Task and
  safety transitions enter separate replay buffers.
- Fine-tuning freezes QSafe, projects every rollout action, and augments the
  actor objective with the projected non-negative dual variable.
- Pure time-limit truncations bootstrap, true terminals do not, and the action
  recorded in replay is `info["applied_action"]` when the environment supplies
  it.

CrossQ semantics remain intact: the task critic has no target network.  Each
critic update concatenates current and next state-action batches into one
forward pass so they share Batch Renormalization statistics; the bootstrapped
next value is stop-gradient.  Actor and entropy updates retain CrossQ's
`policy_delay`.

Task-learner defaults come from CrossQ.  The phase, rollout, dual and nested
`qsafe` controls match SAC-QSafe.  Fine-tuning requires
`algorithm.pretrained_policy_path` and `algorithm.qsafe.checkpoint_path`.

CrossQ's Batch Renormalization state is part of the policy artifact. The Flax
backend therefore uses its own `crossq_qsafe_flax_policy` sidecar containing
both parameters and batch statistics. PyTorch-to-Flax CrossQ policy transfer is
rejected explicitly until a strict BatchRenorm converter is available; native
Flax resume/fine-tuning and same-backend PyTorch checkpoints are supported.
