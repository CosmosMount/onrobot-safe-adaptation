"""SDK-observable approximation of the SQRL fall label."""

from __future__ import annotations

import numpy as np

from ..common.reward import quaternion_to_rpy_wxyz


class FallDetector:
    def __init__(
        self,
        angle_threshold: float = 0.8,
        consecutive_frames: int = 5,
        min_base_height: float = 0.18,
    ):
        self.angle_threshold = float(angle_threshold)
        self.consecutive_frames = int(consecutive_frames)
        self.min_base_height = float(min_base_height)
        self._count = 0
        self._low_height_count = 0

    def reset(self) -> None:
        self._count = 0
        self._low_height_count = 0

    def update(self, quaternion) -> bool:
        roll, pitch, _ = quaternion_to_rpy_wxyz(quaternion)
        fallen = abs(roll) > self.angle_threshold or abs(pitch) > self.angle_threshold
        self._count = self._count + 1 if fallen else 0
        return self._count >= self.consecutive_frames

    def update_base_height(self, height: float | None) -> bool:
        if height is None or not np.isfinite(height):
            self._low_height_count = 0
            return False
        if float(height) < self.min_base_height:
            self._low_height_count += 1
        else:
            self._low_height_count = 0
        return self._low_height_count >= self.consecutive_frames

    def is_stable(self, quaternion) -> bool:
        roll, pitch, _ = quaternion_to_rpy_wxyz(quaternion)
        return abs(roll) < 0.25 and abs(pitch) < 0.25
