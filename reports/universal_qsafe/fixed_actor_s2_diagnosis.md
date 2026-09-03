# Fixed-actor QSafe diagnosis (seed 2)

Date: 2026-09-03

## Question

For one frozen target-domain SAC actor, does QSafe reduce falls by selecting
different actions, or did the earlier training result come from learner
feedback and seed variance?

## Controlled evaluation

All three arms use the same frozen actor checkpoint and 30 MuJoCo flat-ground
episodes. The two stochastic arms use seed 32002. `stochastic_task` executes
candidate zero directly; `safe` preserves candidate zero unless QSafe rejects
it.

| Arm | Falls | Stable successes | Mean velocity | Action change | Fallback |
|---|---:|---:|---:|---:|---:|
| No QSafe | 11/30 (36.7%) | 19/30 | 0.424 m/s | 0% | 0% |
| Legacy source QSafe, gamma 0.7 | 10/30 (33.3%) | 20/30 | 0.409 m/s | 0.49% | 0.13% |
| Target diagnostic QSafe, gamma 0.9 | 0/30 (0.0%) | 30/30 | 0.399 m/s | 8.81% | 6.88% |

The one-sided Fisher exact p-value is 0.000159 for no-QSafe versus the target
diagnostic QSafe. The 95% Wilson intervals for episode fall probability are
21.9--54.5% and 0--11.4%, respectively.

## Critic evidence

On 15,000 frozen-actor target transitions (29 fall and 22 complete non-fall
episodes), the legacy critic had H25 AUC 0.485 and 4.6% recall at epsilon 0.1.
Threshold tuning alone could not provide high recall without excessive normal
rejection/fallback.

The target diagnostic critic uses five observation frames and gamma 0.9. On
its diagnostic held-out episodes it reached H5/H10/H25 recall of
100%/100%/98.2%, H25 false rejection of 10.7%, and fallback of 13.6%. Its
offline universal gate remains WARN because fallback exceeds 5% and the data
splits contain one behavior actor.

## Conclusion and scope

The action-selection implementation can select actions that materially reduce
falls. The observed three-seed failure is therefore not evidence that QSafe can
only classify danger. It is evidence that the legacy source-trained critic
does not cover the adapted target-policy distribution and that gamma 0.7 gives
an inadequate physical warning horizon at 50 Hz.

This is a target-policy diagnostic success, not policy-generalization evidence.
The next formal artifact must train on multiple target-policy actors and retain
actor-isolated validation/test splits before rerunning the three-seed
fine-tuning A/B.

## Artifacts

- No-QSafe evaluation: `fixed_actor_s2_stochastic_task.json`
- Legacy-QSafe evaluation: `fixed_actor_s2_safe.json`
- Target-QSafe evaluation: `fixed_actor_s2_targetcritic_safe.json`
- Target shadow data: `runs/go2_sqrl/finetune/flat_qsafe_target_collect_s2_fixed_15k/qsafe_shadow.npz`
- Diagnostic QSafe: `runs/go2_sqrl/qsafe_offline/mujoco_target_s2_fixed_diagnostic/qsafe_mujoco_target_s2_fixed_diagnostic.model`
