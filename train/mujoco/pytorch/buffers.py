"""Thread-safe LowState and simulator-truth buffers."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

from train.core.base import RobotState, TrainingState


class StateBufferError(RuntimeError):
    pass


class StateTimeout(StateBufferError):
    pass


class FrameOrderError(StateBufferError):
    pass


class SimulatorRestarted(StateBufferError):
    pass


class TrainingStateSyncError(StateBufferError):
    """The bridge cannot pair base truth with an exact LowState physics tick."""


@dataclass(slots=True)
class SynchronizedTrainingState(TrainingState):
    """Simulator-only root truth tagged with the matching LowState tick."""

    tick: int = 0
    command_sequence: int = 0


class StateBuffer:
    def __init__(
        self,
        capacity: int = 4096,
        restart_threshold_ticks: int = 100,
        restart_settle_seconds: float = 5.0,
        restart_stable_frames: int = 3,
    ):
        self._frames: deque[RobotState] = deque(maxlen=int(capacity))
        self._condition = threading.Condition()
        self._last_tick: int | None = None
        self._restart_threshold = int(restart_threshold_ticks)
        self._restart_settle_seconds = float(restart_settle_seconds)
        self._restart_stable_frames_required = max(1, int(restart_stable_frames))
        self._restart_stable_frames = 0
        self._restart_stale_tick_floor: int | None = None
        self._restart_armed = False
        self._restart_settle_deadline = 0.0
        self._generation = 0
        self._error: Exception | None = None

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    @property
    def last_tick(self) -> int | None:
        with self._condition:
            return self._last_tick

    @property
    def latest_state(self) -> RobotState | None:
        with self._condition:
            return self._frames[-1] if self._frames else None

    def push(self, state: RobotState) -> bool:
        if state.tick is None:
            raise ValueError("LowState must contain tick")
        tick = int(state.tick)
        with self._condition:
            now = time.monotonic()
            if now >= self._restart_settle_deadline:
                self._restart_stale_tick_floor = None
            if self._last_tick is not None:
                if tick == self._last_tick:
                    return False
                if tick < self._last_tick:
                    difference = self._last_tick - tick
                    if self._restart_armed:
                        previous_tick = self._last_tick
                        self._frames.clear()
                        self._generation += 1
                        self._restart_armed = False
                        self._restart_stable_frames = 0
                        self._restart_stale_tick_floor = previous_tick
                        self._restart_settle_deadline = (
                            now + self._restart_settle_seconds
                        )
                    elif now < self._restart_settle_deadline:
                        # Once an armed reset established a new generation,
                        # any late lower tick belongs to the drained DDS epoch.
                        return False
                    elif difference >= self._restart_threshold:
                        previous_tick = self._last_tick
                        self._frames.clear()
                        self._generation += 1
                        self._restart_stable_frames = 0
                        self._restart_stale_tick_floor = previous_tick
                        self._restart_settle_deadline = (
                            now + self._restart_settle_seconds
                        )
                    else:
                        self._error = FrameOrderError(
                            f"Out-of-order LowState tick: {tick} after {self._last_tick}"
                        )
                        self._condition.notify_all()
                        return False
                elif (
                    now < self._restart_settle_deadline
                    and self._restart_stale_tick_floor is not None
                    and tick >= self._restart_stale_tick_floor
                ):
                    # A reliable DDS reader may finish delivering queued frames
                    # from the previous simulator epoch after reset.  Compare
                    # against the previous epoch watermark instead of imposing
                    # a small maximum jump: the 500 Hz reader may legitimately
                    # skip more than five ticks while Python handles reset.
                    return False
            self._last_tick = tick
            self._frames.append(state)
            if self._restart_stale_tick_floor is not None:
                self._restart_stable_frames += 1
                if self._restart_stable_frames >= self._restart_stable_frames_required:
                    self._restart_stale_tick_floor = None
            self._condition.notify_all()
            return True

    def arm_restart(self) -> None:
        """Accept the next rollback as an explicitly requested simulator reset."""

        with self._condition:
            self._restart_armed = True
            self._error = None

    def cancel_restart(self) -> None:
        with self._condition:
            self._restart_armed = False

    def clear_error(self) -> None:
        with self._condition:
            self._error = None

    def set_error(self, error: Exception) -> None:
        with self._condition:
            self._error = error
            self._condition.notify_all()

    def wait_for_frames(
        self,
        count: int,
        after_tick: int | None,
        timeout: float,
        generation: int | None = None,
    ) -> list[RobotState]:
        deadline = time.monotonic() + float(timeout)
        count = int(count)
        with self._condition:
            initial_generation = self._generation if generation is None else generation
            while True:
                if self._error is not None:
                    error = self._error
                    self._error = None
                    raise error
                if generation is not None and self._generation != initial_generation:
                    raise SimulatorRestarted("Simulator tick reset while waiting for state")
                eligible = [
                    frame
                    for frame in self._frames
                    if after_tick is None or int(frame.tick) > int(after_tick)
                ]
                if len(eligible) >= count:
                    # With an explicit anchor, return the first physics window
                    # after the command. Returning the latest window could hide
                    # a dropped first frame. With no anchor, preserve the
                    # original "latest state" behavior used during startup.
                    return eligible[-count:] if after_tick is None else eligible[:count]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise StateTimeout(
                        f"Timed out waiting for {count} distinct LowState ticks"
                    )
                self._condition.wait(remaining)

    def wait_for_restart(self, generation: int, timeout: float | None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._generation == generation:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise StateTimeout("Timed out waiting for manual simulator reset")
                self._condition.wait(remaining)
            return self._generation


class TrainingStateBuffer:
    """Pair simulator root truth with LowState using the simulator tick.

    ``SportModeState`` callbacks are independent DDS deliveries.  Reading the
    latest callback is therefore not a synchronization contract.  This buffer
    only returns root truth carrying the exact requested simulation tick,
    applied command sequence and generation; missing bridge tags surface as an
    explicit transition error instead of silently mixing frames or reset
    epochs.
    """

    def __init__(self, capacity: int = 4096):
        self._frames: deque[tuple[int, SynchronizedTrainingState]] = deque(
            maxlen=int(capacity)
        )
        self._condition = threading.Condition()
        self._generation = 0
        self._error: Exception | None = None
        self._last_tick: int | None = None
        self._restart_armed = False

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    @property
    def latest_state(self) -> SynchronizedTrainingState | None:
        with self._condition:
            return self._frames[-1][1] if self._frames else None

    def reset_generation(self, generation: int) -> None:
        generation = int(generation)
        with self._condition:
            if generation < self._generation:
                raise ValueError(
                    f"Training-state generation cannot move backward to {generation}"
                )
            if generation > self._generation:
                self._generation = generation
                self._frames.clear()
                self._last_tick = None
                self._error = None
            # If HighState observed the armed rollback first, its exact first
            # sample is already stored in this same generation. Preserve it.
            self._restart_armed = False
            self._condition.notify_all()

    def arm_restart(self) -> None:
        with self._condition:
            self._restart_armed = True
            self._error = None

    def cancel_restart(self) -> None:
        with self._condition:
            self._restart_armed = False

    def set_error(self, error: Exception) -> None:
        with self._condition:
            self._error = error
            self._condition.notify_all()

    def push(
        self,
        state: SynchronizedTrainingState,
        *,
        generation: int,
    ) -> None:
        generation = int(generation)
        tick = int(state.tick)
        with self._condition:
            if generation > self._generation:
                self._generation = generation
                self._frames.clear()
                self._last_tick = None
                self._error = None
                self._restart_armed = False
            elif generation < self._generation:
                # HighState can observe an armed rollback before LowState has
                # advanced StateBuffer.generation. Only that one-generation
                # handoff is valid; all other stale callbacks are discarded.
                if generation + 1 != self._generation:
                    return
            if self._last_tick is not None and tick < self._last_tick:
                if self._restart_armed:
                    self._generation += 1
                    self._frames.clear()
                    self._error = None
                    self._restart_armed = False
                    self._last_tick = tick
                # An unarmed lower timestamp can be a delayed DDS sample. Keep
                # it available for an exact tick join without moving the
                # monotonic watermark backward.
            else:
                self._last_tick = tick
            # Store under the resolved generation, which may have advanced
            # when HighState won the reset callback race.
            self._frames.append((self._generation, state))
            self._condition.notify_all()

    def wait_for_ticks(
        self,
        ticks: list[int] | tuple[int, ...],
        *,
        command_sequences: list[int] | tuple[int, ...],
        generation: int,
        timeout: float,
    ) -> list[SynchronizedTrainingState]:
        requested = tuple(int(tick) for tick in ticks)
        sequences = tuple(int(sequence) for sequence in command_sequences)
        if not requested:
            return []
        if len(sequences) != len(requested):
            raise ValueError("One command sequence is required for every requested tick")
        requested_keys = tuple(zip(requested, sequences))
        deadline = time.monotonic() + float(timeout)
        generation = int(generation)
        with self._condition:
            while True:
                if self._error is not None:
                    error = self._error
                    self._error = None
                    raise error
                if self._generation != generation:
                    raise SimulatorRestarted(
                        "Simulator tick reset while waiting for synchronized root truth"
                    )
                by_key = {
                    (int(state.tick), int(state.command_sequence)): state
                    for frame_generation, state in self._frames
                    if frame_generation == generation
                }
                if all(key in by_key for key in requested_keys):
                    return [by_key[key] for key in requested_keys]
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    missing = [key for key in requested_keys if key not in by_key]
                    raise TrainingStateSyncError(
                        "Timed out waiting for tagged SportModeState at "
                        f"LowState (tick, sequence) keys {missing}. The "
                        "unitree_mujoco bridge must publish mj_data_->time and "
                        "the applied LowCmd sequence; see "
                        "assets/robots/go2/unitree_mujoco_bridge_lockstep.patch."
                    )
                self._condition.wait(remaining)

