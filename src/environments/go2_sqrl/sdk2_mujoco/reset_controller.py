"""Software reset requests for the external Unitree MuJoCo process."""

from __future__ import annotations

import signal
import socket
import tempfile
from pathlib import Path


# The simulator installs a handler that only raises an atomic request flag;
# the physics thread performs the actual mj_resetDataKeyframe call. SIGWINCH
# is deliberately harmless if an old, unpatched simulator is started.
SOFTWARE_RESET_SIGNAL = signal.SIGWINCH


def reset_socket_path(domain_id: int) -> Path:
    return Path(tempfile.gettempdir()) / f"osa-mujoco-reset-{int(domain_id)}.sock"


class MujocoResetController:
    """Send a reset request to the simulator serving one DDS domain."""

    def __init__(self, domain_id: int):
        self.domain_id = int(domain_id)
        self.socket_path = reset_socket_path(self.domain_id)

    def reset(self) -> None:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
                client.sendto(b"reset", str(self.socket_path))
        except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
            raise RuntimeError(
                "MuJoCo software-reset channel is unavailable on DDS domain "
                f"{self.domain_id}. Start the simulator through `python -m "
                "src.run sim` or the experiment launcher."
            ) from exc
