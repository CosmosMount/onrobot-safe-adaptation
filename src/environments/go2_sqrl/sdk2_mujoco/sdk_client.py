"""Minimal Unitree SDK2 DDS transport for Go2 training."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from ..common.specs import ACTION_SPEC, ACTION_SIZE
from ..common.types import RobotState, TrainingState
from .state_buffer import StateBuffer


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

    def publish_joint_target(self, q_target: np.ndarray) -> None:
        self.start()
        q_target = np.asarray(q_target, dtype=np.float32)
        if q_target.shape != (ACTION_SIZE,):
            raise ValueError(f"q_target must have shape (12,), got {q_target.shape}")
        for index in range(ACTION_SIZE):
            motor = self._command.motor_cmd[index]
            motor.mode = 0x01
            motor.q = float(q_target[index])
            motor.kp = ACTION_SPEC.kp
            motor.dq = 0.0
            motor.kd = ACTION_SPEC.kd
            motor.tau = 0.0
        self._command.crc = self._crc.Crc(self._command)
        self._publisher.Write(self._command)

    def close(self) -> None:
        # SDK2 Python entities do not expose a stable cross-version close API.
        # Retaining them until process exit also prevents callback use-after-free.
        pass
