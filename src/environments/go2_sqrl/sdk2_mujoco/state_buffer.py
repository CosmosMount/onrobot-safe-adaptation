"""Thread-safe collection of distinct SDK LowState physics ticks."""

from __future__ import annotations

import threading
import time
from collections import deque

from ..common.types import RobotState


class StateBufferError(RuntimeError):
    pass


class StateTimeout(StateBufferError):
    pass


class FrameOrderError(StateBufferError):
    pass


class SimulatorRestarted(StateBufferError):
    pass


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
                    return eligible[-count:]
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
