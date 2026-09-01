"""Unitree SDK2 DDS client used by the MuJoCo environment."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from train.core.base import ACTION_SIZE, ACTION_SPEC

from .state import (
    TOPIC_HIGHSTATE,
    TOPIC_LOWCMD,
    TOPIC_LOWSTATE,
    StateBuffer,
    StateBufferError,
    SynchronizedTrainingState,
    TrainingStateBuffer,
    TrainingStateSyncError,
    decode_low_state,
    decode_training_state,
    validate_low_state_crc,
)


class SDKClient:
    """Own DDS entities and expose dependency-free records to the environment."""

    _factory_lock = threading.Lock()
    _factory_configuration: tuple[int, str] | None = None

    def __init__(self, domain_id: int = 1, interface: str = "lo"):
        self.domain_id = int(domain_id)
        self.interface = str(interface)
        self.state_buffer = StateBuffer()
        self.training_state_buffer = TrainingStateBuffer()
        self._latest_torque = np.zeros(ACTION_SIZE, dtype=np.float32)
        self._started = False
        self._publisher = None
        self._command = None
        self._crc = None
        self._entities: list[Any] = []
        self._command_lock = threading.Lock()
        self._command_sequence = 0
        self._sequence_synchronized = False
        self._commands_published = False

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

        # Construct CRC validation before subscriber Init: DDS implementations
        # may invoke a callback immediately for retained/reliable data.
        self._crc = CRC()
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
        self._started = True

    def _on_low_state(self, message: Any) -> None:
        try:
            validate_low_state_crc(message, self._crc)
            state = decode_low_state(message)
        except (TypeError, ValueError) as exc:
            self.state_buffer.set_error(StateBufferError(str(exc)))
            return
        with self._command_lock:
            if not self._commands_published:
                # Synchronize to an already-running bridge so a restarted
                # learner continues at N+1 rather than sending sequence 1.
                self._command_sequence = max(
                    self._command_sequence, int(state.command_sequence)
                )
                self._sequence_synchronized = True
        self._latest_torque = np.asarray(state.actuator_torque, dtype=np.float32)
        previous_generation = self.state_buffer.generation
        self.state_buffer.push(state)
        generation = self.state_buffer.generation
        if generation != previous_generation:
            self.training_state_buffer.reset_generation(generation)

    def _on_high_state(self, message: Any) -> None:
        try:
            training_state = decode_training_state(message, self._latest_torque)
        except (TrainingStateSyncError, TypeError, ValueError) as exc:
            self.training_state_buffer.set_error(
                exc
                if isinstance(exc, TrainingStateSyncError)
                else TrainingStateSyncError(str(exc))
            )
            return
        self.training_state_buffer.push(
            training_state,
            generation=self.state_buffer.generation,
        )

    def latest_training_state(self) -> SynchronizedTrainingState | None:
        state = self.training_state_buffer.latest_state
        if state is None:
            return None
        return SynchronizedTrainingState(
            world_velocity=np.asarray(state.world_velocity).copy(),
            base_position=np.asarray(state.base_position).copy(),
            actuator_torque=np.asarray(state.actuator_torque).copy(),
            tick=int(state.tick),
            command_sequence=int(state.command_sequence),
        )

    def arm_simulator_restart(self) -> None:
        """Arm LowState and root-truth generation tracking as one operation."""

        self.state_buffer.arm_restart()
        self.training_state_buffer.arm_restart()

    def cancel_simulator_restart(self) -> None:
        self.state_buffer.cancel_restart()
        self.training_state_buffer.cancel_restart()

    def training_states_for_ticks(
        self,
        ticks: list[int] | tuple[int, ...],
        *,
        command_sequences: list[int] | tuple[int, ...],
        generation: int,
        timeout: float,
    ) -> list[SynchronizedTrainingState]:
        return self.training_state_buffer.wait_for_ticks(
            ticks,
            command_sequences=command_sequences,
            generation=generation,
            timeout=timeout,
        )

    def publish_joint_target(
        self,
        q_target: np.ndarray,
        *,
        kp: float | None = None,
        kd: float | None = None,
    ) -> int:
        self.start()
        q_target = np.asarray(q_target, dtype=np.float32)
        if q_target.shape != (ACTION_SIZE,):
            raise ValueError(f"q_target must have shape (12,), got {q_target.shape}")
        kp = ACTION_SPEC.kp if kp is None else float(kp)
        kd = ACTION_SPEC.kd if kd is None else float(kd)
        with self._command_lock:
            if not self._sequence_synchronized:
                raise StateBufferError(
                    "Cannot publish LowCmd before synchronizing the bridge sequence "
                    "from a CRC-valid LowState"
                )
            if self._command_sequence == np.iinfo(np.uint32).max:
                raise StateBufferError(
                    "LowCmd sequence space exhausted; restart both learner and simulator"
                )
            self._command_sequence += 1
            for index in range(ACTION_SIZE):
                motor = self._command.motor_cmd[index]
                motor.mode = 0x01
                motor.q = float(q_target[index])
                motor.kp = kp
                motor.dq = 0.0
                motor.kd = kd
                motor.tau = 0.0
            # reserve is part of the CRC-covered payload. The patched bridge
            # echoes it only on post-mj_step snapshots generated by this exact
            # frozen command transaction.
            self._command.reserve = int(self._command_sequence)
            self._command.crc = self._crc.Crc(self._command)
            self.state_buffer.expect_command_sequence(self._command_sequence)
            self._publisher.Write(self._command)
            self._commands_published = True
            return int(self._command_sequence)

    def close(self) -> None:
        # SDK2 Python entities do not expose a stable cross-version close API.
        # Retaining them until process exit also prevents callback use-after-free.
        pass
