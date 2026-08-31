"""MuJoCo fall detection over individual physics frames."""

import numpy as np

from train.core.task import quaternion_to_rpy_wxyz


class FallDetector:
    def __init__(
        self,
        angle_threshold: float = 0.8,
        consecutive_frames: int = 5,
        min_base_clearance: float = 0.18,
    ):
        self.angle_threshold = float(angle_threshold)
        self.consecutive_frames = int(consecutive_frames)
        self.min_base_clearance = float(min_base_clearance)
        self._count = 0
        self._low_height_count = 0
        self.last_tilt_failure = False
        self.last_height_failure = False

    def reset(self) -> None:
        self._count = 0
        self._low_height_count = 0
        self.last_tilt_failure = False
        self.last_height_failure = False

    @staticmethod
    def _normalized_quaternion(quaternion) -> np.ndarray:
        quaternion = np.asarray(quaternion, dtype=np.float64)
        norm = float(np.linalg.norm(quaternion))
        if quaternion.shape != (4,) or not np.isfinite(norm) or norm < 1.0e-8:
            raise ValueError("Failure detection requires a finite WXYZ quaternion")
        return quaternion / norm

    def update(self, quaternion) -> bool:
        roll, pitch, _ = quaternion_to_rpy_wxyz(
            self._normalized_quaternion(quaternion)
        )
        fallen = abs(roll) > self.angle_threshold or abs(pitch) > self.angle_threshold
        self._count = self._count + 1 if fallen else 0
        self.last_tilt_failure = self._count >= self.consecutive_frames
        return self.last_tilt_failure

    def update_base_clearance(self, clearance: float | None) -> bool:
        if clearance is None or not np.isfinite(clearance):
            raise ValueError("Every physics frame requires finite base clearance")
        if float(clearance) < self.min_base_clearance:
            self._low_height_count += 1
        else:
            self._low_height_count = 0
        self.last_height_failure = (
            self._low_height_count >= self.consecutive_frames
        )
        return self.last_height_failure

    def update_frame(self, quaternion, clearance: float) -> bool:
        """Update both failure channels from one 2 ms physics frame."""

        tilt_failure = self.update(quaternion)
        height_failure = self.update_base_clearance(clearance)
        return tilt_failure or height_failure

    def is_stable(self, quaternion, angle_tolerance: float = 0.25) -> bool:
        roll, pitch, _ = quaternion_to_rpy_wxyz(
            self._normalized_quaternion(quaternion)
        )
        return (
            abs(roll) <= float(angle_tolerance)
            and abs(pitch) <= float(angle_tolerance)
        )

