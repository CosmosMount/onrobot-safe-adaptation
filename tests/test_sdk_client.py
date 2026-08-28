from types import SimpleNamespace

import numpy as np
import pytest

from src.environments.go2_sqrl.common.specs import DEFAULT_JOINT_POSITION
from src.environments.go2_sqrl.common.types import RobotState
from src.environments.go2_sqrl.sdk2_mujoco.sdk_client import (
    SDKClient,
    decode_low_state,
)
from src.environments.go2_sqrl.sdk2_mujoco.state_buffer import (
    FrameOrderError,
    StateBuffer,
    StateTimeout,
)


def fake_low_state(tick=2):
    motors = [SimpleNamespace(q=i, dq=-i, tau_est=0.1 * i) for i in range(12)]
    imu = SimpleNamespace(
        quaternion=[1.0, 0.0, 0.0, 0.0],
        gyroscope=[0.1, 0.2, 0.3],
        accelerometer=[0.0, 0.0, 9.81],
    )
    return SimpleNamespace(motor_state=motors, imu_state=imu, tick=tick)


def state(tick):
    result = decode_low_state(fake_low_state(tick))
    assert result.joint_q.shape == (12,)
    return result


def test_decode_low_state_and_distinct_tick_window():
    buffer = StateBuffer()
    assert buffer.push(state(2))
    assert not buffer.push(state(2))
    for tick in (4, 6, 8):
        buffer.push(state(tick))
    frames = buffer.wait_for_frames(3, after_tick=2, timeout=0.01)
    assert [frame.tick for frame in frames] == [4, 6, 8]


def test_buffer_reports_reordering_timeout_and_restart():
    buffer = StateBuffer(restart_threshold_ticks=100)
    buffer.push(state(200))
    assert not buffer.push(state(198))
    with pytest.raises(FrameOrderError):
        buffer.wait_for_frames(1, after_tick=200, timeout=0.01)
    assert buffer.push(state(0))
    assert buffer.generation == 1
    with pytest.raises(StateTimeout):
        buffer.wait_for_frames(2, after_tick=0, timeout=0.001)


def test_buffer_accepts_armed_small_reset_and_ignores_stale_high_tick():
    buffer = StateBuffer(restart_threshold_ticks=100)
    buffer.push(state(50))
    generation = buffer.generation
    buffer.arm_restart()
    assert buffer.push(state(28))
    assert buffer.generation == generation + 1
    assert not buffer.push(state(50))
    assert buffer.push(state(29))
    assert not buffer.push(state(10))
    frames = buffer.wait_for_frames(
        2, after_tick=None, timeout=0.01, generation=buffer.generation
    )
    assert [frame.tick for frame in frames] == [28, 29]


def test_publish_joint_target_populates_pd_and_crc_without_sdk_runtime():
    class CRC:
        def Crc(self, command):
            return 1234

    class Publisher:
        def __init__(self):
            self.written = None

        def Write(self, command):
            self.written = command

    client = SDKClient()
    client._started = True
    client._command = SimpleNamespace(
        motor_cmd=[SimpleNamespace() for _ in range(20)], crc=0
    )
    client._crc = CRC()
    client._publisher = Publisher()
    client.publish_joint_target(DEFAULT_JOINT_POSITION)
    assert client._command.crc == 1234
    assert client._publisher.written is client._command
    assert all(motor.kp == 25.0 and motor.kd == 0.5 for motor in client._command.motor_cmd[:12])
