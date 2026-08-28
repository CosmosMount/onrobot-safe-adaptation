# SAC-QSafe (SQRL)

SAC-QSafe implements the pre-training and fine-tuning phases from
[Learning to be Safe: Deep RL with a Safety Critic](https://arxiv.org/abs/2010.14603).
The task critic and safety critic use separate networks, replay buffers,
optimizers, target networks, and checkpoints.

The environment must return a binary vector in `info["failure"]`. A failure is
a true episode terminal, so every failed environment must also return
`terminated=True` and expose its final observation through the normal RL-X
environment interface. The signal is the post-transition indicator `I(s')`;
therefore its transition target is exactly 1. Non-failure true terminals have
target 0, while pure time-limit truncations bootstrap from the final
observation. If both terminal flags are true, true termination takes priority.

## Running

Pre-training uses `--algorithm.name=sac_qsafe.pytorch` (or
`sac_qsafe.flax`) with the default `--algorithm.phase=pretrain`. It writes
separate policy, task-critic, and QSafe artifacts under the run's `models/`
directory. `best.model` follows task return as before, while `final.model` and
the three sidecars are written unconditionally after the final safety block so
the last QSafe update is never omitted.

Each pre-training iteration follows Algorithm 1 literally:

1. Run `algorithm.n_off` vector-environment steps with the unconstrained SAC
   policy. These transitions are written only to `D_offline`, and only the SAC
   actor, task critics, targets, and entropy coefficient are updated.
2. Reset to the pre-training initial distribution and collect
   `algorithm.n_safe` complete trajectories with the pre-training QSafe
   boundary policy. A trajectory is committed only after it terminates or
   truncates and is written only to `D_safe`. When the smaller replay reaches
   capacity, complete oldest trajectories are evicted so no partial episode is
   retained. These rollouts use the independent evaluation environment, so the
   task-environment trajectory feeding `D_offline` is paused rather than reset
   between blocks; both environments must use the configured `nr_envs`.
   Their observation/action spaces and action bounds are validated at startup;
   they must also represent the same pre-training dynamics distribution.
3. Run `algorithm.qsafe.updates_per_iteration` Q-learning updates from
   `D_safe`. The target action is sampled from the unconstrained current policy.

`algorithm.total_timesteps` counts task interactions written to `D_offline`;
safety-rollout interactions are reported separately as
`steps/nr_safe_env_steps`. For `nr_envs > 1`, `n_off` counts vector steps while
`n_safe` counts completed individual trajectories.

Fine-tuning uses the same algorithm name with:

```text
--algorithm.phase=finetune
--algorithm.pretrained_policy_path=/path/to/policy.model
--algorithm.pretrained_task_critic_path=/path/to/task_critic.model
--algorithm.qsafe.checkpoint_path=/path/to/qsafe.model
```

The Flax sidecars use the `.msgpack` extension. During fine-tuning QSafe is
frozen; `D_offline` and optimizer moments are initialized from scratch, while
the SAC policy, both task critics and targets, and entropy temperature are
transferred from pre-training. Every
fine-tuning interaction uses the projected policy from Eq. 3. Candidate actions
are sampled from the original policy, unsafe candidates are masked, and the
remaining candidates are importance-sampled using their original policy log
probabilities; if none is safe, the minimum-risk candidate is used. The actor
is updated with Eq. 4 using an unconstrained differentiable policy sample, while
the task critic still uses task reward only. QSafe has no trainable optimizer in
this phase. Both backends optimize the projected non-negative dual variable
with Adam.

The paper defaults are used: `total_timesteps=5e5` in either phase, learning
rate `3e-4`, two 256-unit ReLU hidden layers, tanh QSafe output, Adam,
`epsilon=0.1`, and `gamma_safe=0.7`. Fine-tuning validates both safety
hyperparameters against the QSafe checkpoint, preventing target-task retuning.
