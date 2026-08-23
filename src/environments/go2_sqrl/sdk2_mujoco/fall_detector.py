"""SDK-observable approximation of the SQRL fall label."""

from __future__ import annotations

from ..common.reward import quaternion_to_rpy_wxyz


class FallDetector:
    def __init__(self, angle_threshold: float = 0.8, consecutive_frames: int = 5):
        self.angle_threshold = float(angle_threshold)
        self.consecutive_frames = int(consecutive_frames)
        self._count = 0

    def reset(self) -> None:
        self._count = 0

    def update(self, quaternion) -> bool:
        roll, pitch, _ = quaternion_to_rpy_wxyz(quaternion)
        fallen = abs(roll) > self.angle_threshold or abs(pitch) > self.angle_threshold
        self._count = self._count + 1 if fallen else 0
        return self._count >= self.consecutive_frames

    def is_stable(self, quaternion) -> bool:
        roll, pitch, _ = quaternion_to_rpy_wxyz(quaternion)
        return abs(roll) < 0.25 and abs(pitch) < 0.25

