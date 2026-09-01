# Strict command-driven MuJoCo bridge

The stock Unitree bridge and MuJoCo physics loop are asynchronous. During a
learner update the simulator continues to advance under the previous command,
and the first `LowState` observed after a new `LowCmd` is not proof that a
physics step used that command. Dropping the resulting backlog would create a
replay tuple whose state, action, reward, and next state do not describe one
transition.

The adjacent patch replaces that behavior, when explicitly enabled, with this
transaction contract:

- MuJoCo remains paused while the policy or learner is running.
- Python writes a monotonic nonzero sequence to `LowCmd.reserve` before CRC.
- One new sequence freezes the complete command and advances exactly ten 2 ms
  physics steps.
- PD torque is recomputed from the current state before every step.
- Each post-step `LowState` echoes the applied sequence in `reserve`.
- `SportModeState.stamp` records the same simulation time as `LowState.tick`,
  and its otherwise-unused `error_code` echoes the same applied sequence.
- Physics, command installation, state reads, and reset share `sim.mtx`.
- Duplicate, skipped, stale, or concurrently published command sequences are
  rejected instead of being silently merged.

The patch is generated and compile-tested against Unitree's
`unitree_mujoco` commit `4134cb5dc7ff1ba7f484deda48b5274b58694519`.
From the root of that checkout:

```sh
git checkout 4134cb5dc7ff1ba7f484deda48b5274b58694519
git apply /absolute/path/to/ora-refactor/assets/robots/go2/unitree_mujoco_bridge_lockstep.patch
git apply /absolute/path/to/ora-refactor/assets/robots/go2/unitree_mujoco_bridge_lossless_publish.patch
cmake -S simulate -B simulate/build -DCMAKE_BUILD_TYPE=Release
cmake --build simulate/build --parallel
```

The second patch makes each of the ten post-step LowState and HighState frames
wait for the SDK real-time publisher slot. Without that backpressure, the
asynchronous publisher can silently skip intermediate frames and the learner
will time out waiting for ten distinct ticks.

Launch through `python -m train.runner sim`; the runner sets
`ORSA_STRICT_LOCKSTEP=1`. If launching the executable directly, set the same
environment variable yourself. Strict mode requires `enable_elastic_band: 0`
and rejects model hot-loading. The repository runner does not mutate or patch
an external checkout, so applying and rebuilding remains an explicit setup
step.

The Python adapter independently validates every `LowState` CRC, synchronizes
its sequence watermark when reconnecting to an already-running bridge, checks
that no tick advanced while it was outside `env.step`, that the first returned
tick is exactly the next 2 ms frame, that all ten ticks are consecutive, that
all ten carry the just-issued sequence, and that each has root truth matched by
both timestamp and sequence. A stock, timestamp-only, or incorrectly patched
bridge therefore fails before a replay sample is written. Sequence wrap is
rejected; after exhausting uint32, restart both simulator and learner.
