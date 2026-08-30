"""SDK2 state transport and MuJoCo reset control."""
from __future__ import annotations

from train.common.base import ACTION_SIZE, ACTION_SPEC, RobotState, TrainingState

import threading
import time
from collections import deque


# DDS state transport and restart detection.
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

from typing import Any

import numpy as np


TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_HIGHSTATE = "rt/sportmodestate"


def _read(value: Any) -> Any:
    return value() if callable(value) else value


def decode_low_state(message: Any) -> RobotState:
    motors = _read(message.motor_state)[:ACTION_SIZE]
    imu = _read(message.imu_state)
    return RobotState(
        joint_q=np.asarray([float(_read(motor.q)) for motor in motors], dtype=np.float32),
        joint_dq=np.asarray(
            [float(_read(motor.dq)) for motor in motors], dtype=np.float32
        ),
        imu_gyro=np.asarray(_read(imu.gyroscope), dtype=np.float32)[:3],
        imu_quat=np.asarray(_read(imu.quaternion), dtype=np.float32)[:4],
        imu_accelerometer=np.asarray(_read(imu.accelerometer), dtype=np.float32)[:3],
        tick=int(_read(message.tick)),
    )


def decode_training_state(message: Any, torque: np.ndarray) -> TrainingState:
    return TrainingState(
        world_velocity=np.asarray(_read(message.velocity), dtype=np.float32)[:3],
        base_position=np.asarray(_read(message.position), dtype=np.float32)[:3],
        actuator_torque=np.asarray(torque, dtype=np.float32).copy(),
    )


class SDKClient:
    """Own DDS entities and expose dependency-free records to the environment."""

    _factory_lock = threading.Lock()
    _factory_configuration: tuple[int, str] | None = None

    def __init__(self, domain_id: int = 1, interface: str = "lo"):
        self.domain_id = int(domain_id)
        self.interface = str(interface)
        self.state_buffer = StateBuffer()
        self._latest_torque = np.zeros(ACTION_SIZE, dtype=np.float32)
        self._training_state: TrainingState | None = None
        self._training_lock = threading.Lock()
        self._started = False
        self._publisher = None
        self._command = None
        self._crc = None
        self._entities: list[Any] = []

    def start(self) -> None:
        if self._started:
            return
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
                LowCmd_,
                LowState_,
                SportModeState_,
            )
            from unitree_sdk2py.utils.crc import CRC
        except ImportError as exc:
            raise RuntimeError(
                "unitree_sdk2py is required for go2_sqrl.sdk2_mujoco"
            ) from exc

        with self._factory_lock:
            requested = (self.domain_id, self.interface)
            if self._factory_configuration is None:
                ChannelFactoryInitialize(*requested)
                type(self)._factory_configuration = requested
            elif self._factory_configuration != requested:
                raise RuntimeError(
                    "Unitree ChannelFactory is already initialized for "
                    f"{self._factory_configuration}, cannot switch to {requested}"
                )

        low_state_subscriber = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
        low_state_subscriber.Init(self._on_low_state, 256)
        high_state_subscriber = ChannelSubscriber(TOPIC_HIGHSTATE, SportModeState_)
        high_state_subscriber.Init(self._on_high_state, 32)
        publisher = ChannelPublisher(TOPIC_LOWCMD, LowCmd_)
        publisher.Init()

        command = unitree_go_msg_dds__LowCmd_()
        command.head[0] = 0xFE
        command.head[1] = 0xEF
        command.level_flag = 0xFF
        command.gpio = 0
        for motor in command.motor_cmd:
            motor.mode = 0x01
            motor.q = 0.0
            motor.kp = 0.0
            motor.dq = 0.0
            motor.kd = 0.0
            motor.tau = 0.0

        self._entities = [low_state_subscriber, high_state_subscriber, publisher]
        self._publisher = publisher
        self._command = command
        self._crc = CRC()
        self._started = True

    def _on_low_state(self, message: Any) -> None:
        state = decode_low_state(message)
        motors = _read(message.motor_state)[:ACTION_SIZE]
        self._latest_torque = np.asarray(
            [float(_read(motor.tau_est)) for motor in motors], dtype=np.float32
        )
        self.state_buffer.push(state)

    def _on_high_state(self, message: Any) -> None:
        training_state = decode_training_state(message, self._latest_torque)
        with self._training_lock:
            self._training_state = training_state

    def latest_training_state(self) -> TrainingState | None:
        with self._training_lock:
            state = self._training_state
            if state is None:
                return None
            return TrainingState(
                world_velocity=np.asarray(state.world_velocity).copy(),
                base_position=np.asarray(state.base_position).copy(),
                actuator_torque=np.asarray(state.actuator_torque).copy(),
            )

    def publish_joint_target(
        self,
        q_target: np.ndarray,
        *,
        kp: float | None = None,
        kd: float | None = None,
    ) -> None:
        self.start()
        q_target = np.asarray(q_target, dtype=np.float32)
        if q_target.shape != (ACTION_SIZE,):
            raise ValueError(f"q_target must have shape (12,), got {q_target.shape}")
        kp = ACTION_SPEC.kp if kp is None else float(kp)
        kd = ACTION_SPEC.kd if kd is None else float(kd)
        for index in range(ACTION_SIZE):
            motor = self._command.motor_cmd[index]
            motor.mode = 0x01
            motor.q = float(q_target[index])
            motor.kp = kp
            motor.dq = 0.0
            motor.kd = kd
            motor.tau = 0.0
        self._command.crc = self._crc.Crc(self._command)
        self._publisher.Write(self._command)

    def close(self) -> None:
        # SDK2 Python entities do not expose a stable cross-version close API.
        # Retaining them until process exit also prevents callback use-after-free.
        pass

import ctypes


# Simulator reset and target environment.
class MujocoResetController:
    """Find the MuJoCo window and send its Backspace reset shortcut."""

    def __init__(self, window_title: str = "MuJoCo", search_timeout: float = 5.0):
        self.window_title = str(window_title)
        self.search_timeout = float(search_timeout)

    @staticmethod
    def _libraries():
        try:
            x11 = ctypes.CDLL("libX11.so.6")
            xtst = ctypes.CDLL("libXtst.so.6")
        except OSError as exc:
            raise RuntimeError(
                "Automatic MuJoCo reset requires libX11 and libXtst."
            ) from exc

        window = ctypes.c_ulong
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        x11.XDefaultRootWindow.restype = window
        x11.XQueryTree.argtypes = [
            ctypes.c_void_p,
            window,
            ctypes.POINTER(window),
            ctypes.POINTER(window),
            ctypes.POINTER(ctypes.POINTER(window)),
            ctypes.POINTER(ctypes.c_uint),
        ]
        x11.XQueryTree.restype = ctypes.c_int
        x11.XFetchName.argtypes = [
            ctypes.c_void_p,
            window,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        x11.XFetchName.restype = ctypes.c_int
        x11.XFree.argtypes = [ctypes.c_void_p]
        x11.XSetInputFocus.argtypes = [
            ctypes.c_void_p,
            window,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        x11.XRaiseWindow.argtypes = [ctypes.c_void_p, window]
        x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        x11.XKeysymToKeycode.restype = ctypes.c_uint
        x11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        xtst.XTestFakeKeyEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        xtst.XTestFakeKeyEvent.restype = ctypes.c_int
        return x11, xtst

    @staticmethod
    def _window_name(x11, display, window: int) -> str:
        name = ctypes.c_char_p()
        if not x11.XFetchName(display, window, ctypes.byref(name)) or not name.value:
            return ""
        try:
            return name.value.decode("utf-8", errors="replace")
        finally:
            x11.XFree(name)

    def _find_window(self, x11, display, root: int) -> int | None:
        pending = [int(root)]
        while pending:
            window = pending.pop()
            if self.window_title in self._window_name(x11, display, window):
                return window

            root_return = ctypes.c_ulong()
            parent_return = ctypes.c_ulong()
            children = ctypes.POINTER(ctypes.c_ulong)()
            count = ctypes.c_uint()
            status = x11.XQueryTree(
                display,
                window,
                ctypes.byref(root_return),
                ctypes.byref(parent_return),
                ctypes.byref(children),
                ctypes.byref(count),
            )
            if not status:
                continue
            try:
                pending.extend(int(children[index]) for index in range(count.value))
            finally:
                if children:
                    x11.XFree(children)
        return None

    def reset(self) -> None:
        x11, xtst = self._libraries()
        display = x11.XOpenDisplay(None)
        if not display:
            raise RuntimeError(
                "Cannot open the X11 display for automatic MuJoCo reset. "
                "Set DISPLAY to the display containing the MuJoCo window."
            )
        try:
            deadline = time.monotonic() + self.search_timeout
            window = None
            while window is None and time.monotonic() < deadline:
                root = x11.XDefaultRootWindow(display)
                window = self._find_window(x11, display, root)
                if window is None:
                    time.sleep(0.1)
            if window is None:
                raise RuntimeError(
                    f"MuJoCo window containing {self.window_title!r} was not found."
                )

            # Backspace is MuJoCo's reset shortcut. Focus the simulator before
            # emitting the synthetic key so another terminal cannot consume it.
            x11.XRaiseWindow(display, window)
            x11.XSetInputFocus(display, window, 2, 0)
            # Deliver FocusIn before the key event. If another app was focused,
            # GLFW can otherwise receive press/release before it processes the
            # focus transition and silently discard the reset shortcut.
            x11.XSync(display, 0)
            time.sleep(0.05)
            keycode = x11.XKeysymToKeycode(display, 0xFF08)
            if not keycode:
                raise RuntimeError("X11 could not resolve the Backspace keycode.")
            # Clear a stale synthetic-down state left by an interrupted sender.
            # XTest otherwise emits release/press in an implementation-dependent
            # order and GLFW may never observe a fresh press transition.
            if not xtst.XTestFakeKeyEvent(display, keycode, 0, 0):
                raise RuntimeError("X11 failed to clear the Backspace key state.")
            x11.XSync(display, 0)
            time.sleep(0.02)
            if not xtst.XTestFakeKeyEvent(display, keycode, 1, 0):
                raise RuntimeError("X11 failed to send the Backspace key press.")
            x11.XSync(display, 0)
            time.sleep(0.02)
            if not xtst.XTestFakeKeyEvent(display, keycode, 0, 0):
                raise RuntimeError("X11 failed to send the Backspace key release.")
            x11.XSync(display, 0)
        finally:
            x11.XCloseDisplay(display)
