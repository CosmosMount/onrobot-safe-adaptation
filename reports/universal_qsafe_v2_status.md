# Universal QSafe v2 staged status

Date: 2026-08-30

## 2026-08-30 high-clearance v2 experiment — easy gate failed

This failed run used a now-superseded high-clearance-only SAC recipe.  Future
data-producing actors no longer use actor-only transfer, reset alpha, a 10k
freeze, sparse 10:1/5:1 actor updates, or 64 environments.  The replacement
recipe is locked to the successful `gait_h07_p03_50k_s2` settings: inherit the
actor, normalizer, task critics, target critics, and temperature; restart only
replay and optimizers; use 2,500 learning-start/warm-up transitions, 1:1 actor
updates, UTD 1, and 20 environments.  New runs are written under
`runs/go2_sqrl/universal_high_clearance_v2_stable_sac` so the failed evidence
below remains intact and cannot be resumed accidentally.

The second high-clearance attempt transferred the already successful
`gait_h07_p03_50k_s2` actor rather than restarting from a flat policy.  The
source policy passed all four deterministic 20-episode baselines:

| Terrain | Stable success | Fall | Stuck | Mean speed | Swing mean/P95 |
|---|---:|---:|---:|---:|---:|
| flat | 20/20 | 0/20 | 0/20 | 0.5188 m/s | 0.0268/0.0403 m |
| 2 cm step | 20/20 | 0/20 | 0/20 | 0.5111 m/s | 0.0268/0.0401 m |
| 3 cm step | 20/20 | 0/20 | 0/20 | 0.5127 m/s | 0.0257/0.0387 m |
| 4 cm step | 20/20 | 0/20 | 0/20 | 0.5065 m/s | 0.0267/0.0416 m |

Seeds 6, 7, and 8 were each restarted independently from that source.  Each
received a fresh task critic, target critic, temperature, replay buffer, and
optimizers; the actor was frozen for the first 10k of the 50k easy stage.  The
deterministic 2 cm gates were:

| Seed | Stable success | Fall | Stuck | Mean speed | Swing mean/P95 | Result |
|---:|---:|---:|---:|---:|---:|---|
| 6 | 0/20 | 0/20 | 20/20 | 0.0058 m/s | 0.0217/0.0251 m | standing collapse |
| 7 | 0/20 | 20/20 | 0/20 | 0.0702 m/s | 0.0249/0.0460 m | falling collapse |
| 8 | 0/20 | 0/20 | 20/20 | -0.0024 m/s | 0.0183/0.0463 m | standing collapse |

All three failed the easy requirement of at least 18/20 stable successes,
at most 2/20 falls, and at most 2/20 stuck episodes.  Therefore the staged
runner correctly stopped before medium and hard, did not create a
`high_clearance` role, and did not start universal-QSafe collection.

The repeated collapse despite an initially successful actor indicates a
fine-tuning stability problem rather than a missing source capability.  In
particular, a randomly initialized task critic combined with a freshly high
SAC entropy temperature can drive the actor away from its useful gait as soon
as it is unfrozen.  The current reward also remains negative on average during
easy training, while standing avoids falls and large action/default-pose
penalties.  A follow-up should preserve the actor distribution during critic
bootstrapping (for example with source-action distillation/KL anchoring and a
transferred or explicitly low initial temperature), verify critic action
ranking before actor unfreeze, and only then repeat a small 10k/20k screen.

The full machine-readable record is in
`runs/go2_sqrl/universal_high_clearance_v2/manifest.json` and
`runs/go2_sqrl/universal_high_clearance_v2/summary.csv`.

## 2026-08-30 sequential execution — stopped at the actor gate

The actor-inventory stage was resumed under the strict stop-on-failure rule.
Four missing 300k SAC checkpoints were trained successfully:

- `universal_flat_seed1_v11` and the completely held-out
  `universal_flat_seed2_v11` on flat terrain with the legacy reward;
- `universal_rough_seed0_v11` and `universal_rough_seed1_v11` on the mixed
  rough terrain with 0.07 m swing-weighted clearance and phase scale 0.3.

The high-clearance role was then trained from the same flat seed-0 actor with
the required 2 -> 3 -> 4 cm curriculum.  Seeds 3, 4, and 5 each received 30k
steps per level.  Their independent deterministic 4 cm evaluations were:

| Seed | Fall | Success | Stuck | Mean velocity | Mean/max clearance | Action saturation | Gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 0/20 | 0/20 | 20/20 | -0.0036 m/s | 0.0236/0.0694 m | 25.6% | fail |
| 4 | 0/20 | 0/20 | 20/20 | 0.0400 m/s | 0.0409/0.1697 m | 36.2% | fail |
| 5 | 14/20 | 10/20 | 0/20 | 0.3642 m/s | 0.0313/0.2218 m | 8.9% | fail |

The required gate was success >=16/20, fall <=2/20, and stuck <=2/20.  No
seed passed.  Therefore no checkpoint was assigned the `high_clearance` role,
and the formal actor inventory is incomplete.  In accordance with the plan,
the runner exited with code 2 and did not start the collection, offline QSafe,
protection, fine-tuning, leave-one-actor-out, or MuJoCo stages.

The machine-readable command history, git state, checkpoint chain, timestamps,
metrics, and stop reason are in
`runs/go2_sqrl/universal_qsafe_actor_stage/manifest.json`.  This is a task
capability failure, not evidence for or against QSafe: a safety critic cannot
be evaluated as a safe-learning aid until the actor/reward setup demonstrates
that it can learn the target 4 cm crossing behavior.

## Stage 1: deterministic baseline — complete

Actor: `runs/go2_sqrl/pretrain/isaac_sac_flat_action_v2_legacy_v1/models`

All conditions used a 0.5 m/s command, 500-step episodes, 20 episodes, and the
relative-local-ground fall detector.

| Terrain | Fall | Success | Stuck |
|---|---:|---:|---:|
| flat | 0/20 | 20/20 traversal | 0/20 |
| upward step 2 cm | 0/20 | 20/20 | 0/20 |
| upward step 3 cm | 4/20 | 16/20 | 0/20 |
| upward step 4 cm | 20/20 | 0/20 | 0/20 |
| upward step 5 cm | 17/20 | 1/20 | 2/20 |
| upward step 6 cm | 9/20 | 0/20 | 11/20 |
| random boxes, max adjacent difference 4 cm | 5/20 | 16/20 | 5/20 |

The boxes outcomes are episode diagnostics and are not mutually exclusive: an
episode may pass the progress threshold and subsequently fall.  The boxes run
also measured mean/max foot clearance 0.0248/0.1105 m, swing ratios
FR/FL/RR/RL 0.249/0.239/0.308/0.438, and action/torque saturation 0.044/0.001.

## Stage 2: minimum phase/clearance reward — passed

The valid 10k screen used 20 environments so every environment received all
500 control steps.  The top three combinations were h07/p0.3, h08/p0.1, and
h06/p0.5.  Their 50k, three-seed outcomes were:

| Candidate | Seed 0 | Seed 1 | Seed 2 | Gate |
|---|---|---|---|---|
| h=0.07, phase=0.3 | 18 success, 1 fall, 1 stuck | 0 success, 0 fall, 20 stuck | 20 success, 0 fall, 0 stuck | pass: 2/3 seeds >=80% |
| h=0.08, phase=0.1 | 1 success, 20 fall | 3 success, 5 fall, 11 stuck | 0 success, 20 fall | fail |
| h=0.06, phase=0.5 | 16 success, 4 fall | 0 success, 1 fall, 19 stuck | 0 success, 0 fall, 20 stuck | fail |

The fixed reward is therefore clearance target 0.07 m, swing-weighted
clearance, and phase scale 0.3.  Flat forward speed was 0.45999 m/s for seed 0
and 0.52217 m/s for seed 2 versus the original actor's 0.47733 m/s, retaining
96.4% and 109.4% respectively (both above the 90% gate).

## Stage 3: QSafe v2 implementation — passed

- Actor input remains one 46D observation and the action remains 12D.
- QSafe owns a five-frame/230D raw-history buffer and independent normalizer.
- Reset repeats the first frame; Isaac and MuJoCo histories advance once per
  control step and do not share state between task/safety/eval streams.
- V2 uses sigmoid risk, SQRL Bellman targets, Eq.3 filtering and Eq.4 actor
  constraints.  Target fine-tuning freezes QSafe.
- PyTorch-to-Flax parity is tested at tolerance 1e-5.
- A 100-vector-step Isaac runtime smoke completed a QSafe Bellman update with
  finite value 0.413447 and target 0.357086; the saved independent normalizer
  contained 1,164 samples.
- A separate-checkpoint cross-actor Isaac rollout also completed.  The old,
  deliberately uncalibrated smoke QSafe rejected 100% of candidates and used
  fallback 100% of the time, so it is retained only as an integration check
  and is correctly forbidden from formal stage-6/7 experiments.

## Stage 4: multi-policy dataset — gate not yet met; later experiments stopped

The complete-trajectory collector and matrix launcher are implemented and a
real 500-step Isaac smoke trajectory was verified with shapes `(500,46)`,
`(500,12)`, and `(500,100,12)` for state, executed action, and candidates.

The actor run on 2026-08-30 added the required second training/validation flat
actor, held-out flat actor, and two mixed-rough actors.  Two verified successful
phase-step actors (h07/p0.3 seeds 0 and 2) also remain available.  The formal
inventory is nevertheless insufficient because all three newly trained
high-clearance candidates failed the 4 cm gate.  No persistent v2 dataset with
the required per-split fall/success counts has therefore been collected.

Consequently stages 5–9 have not been claimed or run.  The offline trainer,
calibration gates, frozen-protection launcher, Isaac A/B/C/D launcher, and
deterministic MuJoCo 4 cm scene are implemented, but the tools intentionally
refuse formal downstream experiments until the stage-4 requirements are met.

## Verification

- Repository suite: 116 passed, 1 skipped.
- Isaac trajectory collection runtime smoke: passed.
- Offline gamma/epsilon training and checkpoint smoke: passed; the deliberately
  incomplete synthetic dataset was correctly named only
  `best_candidate_qsafe_v2.model`, never `universal_qsafe_v2.model`.
