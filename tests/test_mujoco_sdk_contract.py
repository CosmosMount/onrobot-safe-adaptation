from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from train.core.base import JOINT_NAMES
from train.mujoco.pytorch.sdk import (
    DEFAULT_GO2_SCENE,
    SDKClient,
    SDK_MOTOR_ORDER,
    StateBuffer,
    StateBufferError,
    SynchronizedTrainingState,
    TrainingStateBuffer,
    TrainingStateSyncError,
    decode_low_state,
    decode_training_state,
    decode_simulator_tick,
    validate_low_state_crc,
    validate_go2_mjcf_contract,
)


class _CallableValue:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


class MujocoSdkContractTest(unittest.TestCase):
    @staticmethod
    def _low_state_message(*, reserve=9):
        message = SimpleNamespace(
            motor_state=[
                SimpleNamespace(q=0.0, dq=0.0, tau_est=0.0)
                for _ in range(12)
            ],
            imu_state=SimpleNamespace(
                gyroscope=[0.0, 0.0, 0.0],
                quaternion=[1.0, 0.0, 0.0, 0.0],
                accelerometer=[0.0, 0.0, 9.81],
            ),
            tick=2,
            reserve=reserve,
        )
        return message

    def test_low_state_decodes_post_step_command_sequence(self):
        state = decode_low_state(self._low_state_message(reserve=17))
        self.assertEqual(state.command_sequence, 17)

    def test_low_state_without_sequence_field_fails_fast(self):
        message = self._low_state_message()
        del message.reserve
        with self.assertRaisesRegex(ValueError, "command-sequence reserve"):
            decode_low_state(message)

    def test_low_command_sequence_is_crc_covered_and_monotonic(self):
        client = SDKClient()
        client._started = True
        client._command = SimpleNamespace(
            motor_cmd=[SimpleNamespace() for _ in range(12)],
            reserve=0,
            crc=0,
        )
        client._sequence_synchronized = True
        crc_sequences = []
        published_sequences = []
        client._crc = SimpleNamespace(
            Crc=lambda message: crc_sequences.append(message.reserve) or 123
        )
        client._publisher = SimpleNamespace(
            Write=lambda message: published_sequences.append(message.reserve)
        )

        first = client.publish_joint_target(np.zeros(12, dtype=np.float32))
        second = client.publish_joint_target(np.zeros(12, dtype=np.float32))

        self.assertEqual((first, second), (1, 2))
        self.assertEqual(crc_sequences, [1, 2])
        self.assertEqual(published_sequences, [1, 2])

    def test_low_state_crc_is_verified_before_decode(self):
        message = self._low_state_message()
        message.crc = 0x12345678
        crc = SimpleNamespace(Crc=lambda _: 0x12345678)
        validate_low_state_crc(message, crc)
        message.crc = 0
        with self.assertRaisesRegex(ValueError, "CRC mismatch"):
            validate_low_state_crc(message, crc)

    def test_corrupt_low_state_wakes_waiter_with_error(self):
        client = SDKClient()
        client._crc = SimpleNamespace(Crc=lambda _: 7)
        message = self._low_state_message()
        message.crc = 8
        client._on_low_state(message)
        with self.assertRaisesRegex(StateBufferError, "CRC mismatch"):
            client.state_buffer.wait_for_frames(1, None, 0.0)

    def test_client_restart_continues_bridge_sequence_watermark(self):
        client = SDKClient()
        client._crc = SimpleNamespace(Crc=lambda _: 11)
        message = self._low_state_message(reserve=41)
        message.crc = 11
        client._on_low_state(message)
        self.assertTrue(client._sequence_synchronized)
        self.assertEqual(client._command_sequence, 41)

    def test_publish_requires_sequence_synchronization_and_never_wraps(self):
        client = SDKClient()
        client._started = True
        client._command = SimpleNamespace(
            motor_cmd=[SimpleNamespace() for _ in range(12)], reserve=0, crc=0
        )
        client._crc = SimpleNamespace(Crc=lambda _: 0)
        client._publisher = SimpleNamespace(Write=lambda _: None)
        with self.assertRaisesRegex(StateBufferError, "synchronizing"):
            client.publish_joint_target(np.zeros(12, dtype=np.float32))
        client._sequence_synchronized = True
        client._command_sequence = np.iinfo(np.uint32).max
        with self.assertRaisesRegex(StateBufferError, "exhausted"):
            client.publish_joint_target(np.zeros(12, dtype=np.float32))

    def test_bundled_asset_satisfies_reset_contact_and_joint_contract(self):
        validate_go2_mjcf_contract()
        self.assertEqual(SDK_MOTOR_ORDER, tuple(JOINT_NAMES))

    def test_stock_mj_reset_data_is_home_and_grounded(self):
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(DEFAULT_GO2_SCENE))
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        home_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        np.testing.assert_allclose(data.qpos, model.key_qpos[home_id], atol=1e-12)
        mujoco.mj_forward(model, data)
        floor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        foot_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg)
            for leg in ("FR", "FL", "RR", "RL")
        }
        contacts = {
            geom
            for index in range(data.ncon)
            for geom in (int(data.contact[index].geom1), int(data.contact[index].geom2))
            if floor_id in {
                int(data.contact[index].geom1),
                int(data.contact[index].geom2),
            }
        }
        self.assertTrue(foot_ids.issubset(contacts))

    def test_reset_frame_sensors_report_base_link_not_inertial_frame(self):
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(DEFAULT_GO2_SCENE))
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        expected = {
            "imu_quat": np.asarray([1.0, 0.0, 0.0, 0.0]),
            "frame_pos": np.asarray([0.0, 0.0, 0.289]),
            "frame_vel": np.zeros(3),
        }
        for name, value in expected.items():
            sensor_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SENSOR, name
            )
            address = int(model.sensor_adr[sensor_id])
            np.testing.assert_allclose(
                data.sensordata[address : address + value.size], value, atol=1e-12
            )

    def test_state_buffer_rejects_duplicate_bridge_tick(self):
        buffer = StateBuffer()
        first = SimpleNamespace(tick=2)
        duplicate = SimpleNamespace(tick=2)
        self.assertTrue(buffer.push(first))
        self.assertFalse(buffer.push(duplicate))
        self.assertIs(buffer.latest_state, first)

    def test_state_buffer_rejects_delayed_previous_command_before_tick_dedup(self):
        buffer = StateBuffer()
        anchor = SimpleNamespace(tick=0, command_sequence=653)
        delayed = SimpleNamespace(tick=2, command_sequence=653)
        current = [
            SimpleNamespace(tick=tick, command_sequence=654)
            for tick in range(2, 22, 2)
        ]
        self.assertTrue(buffer.push(anchor))
        buffer.expect_command_sequence(654)
        self.assertFalse(buffer.push(delayed))
        self.assertTrue(all(buffer.push(frame) for frame in current))
        frames = buffer.wait_for_frames(
            10,
            after_tick=0,
            timeout=0.0,
            command_sequence=654,
        )
        self.assertEqual([frame.tick for frame in frames], list(range(2, 22, 2)))

    def test_high_state_stamp_decodes_to_low_state_tick(self):
        message = SimpleNamespace(
            stamp=_CallableValue(
                SimpleNamespace(
                    sec=_CallableValue(3),
                    nanosec=_CallableValue(42_000_000),
                )
            )
        )
        self.assertEqual(decode_simulator_tick(message), 3042)

    def test_explicit_bridge_tick_is_supported(self):
        message = SimpleNamespace(simulation_tick=_CallableValue(86))
        self.assertEqual(decode_simulator_tick(message), 86)

    def test_missing_bridge_timestamp_fails_fast(self):
        with self.assertRaisesRegex(TrainingStateSyncError, "bridge_lockstep"):
            decode_simulator_tick(SimpleNamespace())

    def test_training_truth_join_requires_exact_ticks_and_generation(self):
        buffer = TrainingStateBuffer()
        for tick in (2, 4, 6):
            buffer.push(
                SynchronizedTrainingState(
                    world_velocity=np.asarray([tick, 0.0, 0.0]),
                    base_position=np.asarray([0.0, 0.0, 0.289]),
                    actuator_torque=np.zeros(12),
                    tick=tick,
                    command_sequence=7,
                ),
                generation=0,
            )
        states = buffer.wait_for_ticks(
            (2, 6), command_sequences=(7, 7), generation=0, timeout=0.01
        )
        self.assertEqual([state.tick for state in states], [2, 6])
        with self.assertRaisesRegex(TrainingStateSyncError, r"\(8, 7\)"):
            buffer.wait_for_ticks(
                (8,), command_sequences=(7,), generation=0, timeout=0.0
            )

    def test_training_truth_join_rejects_stale_reset_sequence(self):
        buffer = TrainingStateBuffer()
        for sequence, velocity in ((9, 9.0), (10, 10.0)):
            buffer.push(
                SynchronizedTrainingState(
                    world_velocity=np.asarray([velocity, 0.0, 0.0]),
                    base_position=np.asarray([0.0, 0.0, 0.289]),
                    actuator_torque=np.zeros(12),
                    tick=2,
                    command_sequence=sequence,
                ),
                generation=1,
            )
        state = buffer.wait_for_ticks(
            (2,), command_sequences=(10,), generation=1, timeout=0.01
        )[0]
        self.assertEqual(float(state.world_velocity[0]), 10.0)

    def test_high_state_decodes_applied_command_sequence(self):
        message = SimpleNamespace(
            velocity=[0.0, 0.0, 0.0],
            position=[0.0, 0.0, 0.289],
            simulation_tick=2,
            error_code=17,
        )
        state = decode_training_state(message, np.zeros(12))
        self.assertEqual(state.command_sequence, 17)

    def test_high_state_can_win_the_armed_restart_callback_race(self):
        buffer = TrainingStateBuffer()

        def truth(tick):
            return SynchronizedTrainingState(
                world_velocity=np.zeros(3),
                base_position=np.asarray([0.0, 0.0, 0.289]),
                actuator_torque=np.zeros(12),
                tick=tick,
                command_sequence=5,
            )

        buffer.push(truth(100), generation=0)
        buffer.arm_restart()
        # HighState from the new epoch arrives before LowState advances its
        # generation. The later LowState handoff must preserve this sample.
        buffer.push(truth(0), generation=0)
        self.assertEqual(buffer.generation, 1)
        buffer.reset_generation(1)
        states = buffer.wait_for_ticks(
            (0,), command_sequences=(5,), generation=1, timeout=0.01
        )
        self.assertEqual([state.tick for state in states], [0])


if __name__ == "__main__":
    unittest.main()
