#!/usr/bin/env python3
"""Install the software-reset hook in the sibling Unitree MuJoCo checkout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import find_unitree_mujoco_root


REPLACEMENTS = (
    (
        "#include <atomic>\n",
        "#include <atomic>\n#include <csignal>\n",
    ),
    (
        "  std::atomic_bool bridge_ready = false;\n\n"
        "  // Unitree robot MJCFs define a standing \"home\" keyframe.",
        "  std::atomic_bool bridge_ready = false;\n"
        "  std::atomic_bool software_reset_requested = false;\n\n"
        "  void RequestSoftwareReset(int)\n"
        "  {\n"
        "    software_reset_requested.store(true);\n"
        "  }\n\n"
        "  // Unitree robot MJCFs define a standing \"home\" keyframe.",
    ),
    (
        "        if (m)\n"
        "        {\n"
        "          // running\n",
        "        if (m)\n"
        "        {\n"
        "          if (software_reset_requested.exchange(false))\n"
        "          {\n"
        "            ResetDataToHome(m, d);\n"
        "            mj_forward(m, d);\n"
        "            syncCPU = {};\n"
        "            syncSim = d->time;\n"
        "            sim.speed_changed = true;\n"
        "            std::printf(\"MuJoCo software reset completed\\n\");\n"
        "            std::fflush(stdout);\n"
        "          }\n\n"
        "          // running\n",
    ),
    (
        "int main(int argc, char **argv)\n"
        "{\n\n"
        "  // display an error if running on macOS under Rosetta 2\n",
        "int main(int argc, char **argv)\n"
        "{\n"
        "#if defined(SIGWINCH)\n"
        "  std::signal(SIGWINCH, RequestSoftwareReset);\n"
        "#endif\n\n"
        "  // display an error if running on macOS under Rosetta 2\n",
    ),
)


def patch_source(source: str) -> tuple[str, bool]:
    if "void ResetDataToHome" not in source:
        raise RuntimeError(
            "unitree_mujoco must contain the Go2 home-keyframe reset patch first"
        )
    changed = False
    for original, replacement in REPLACEMENTS:
        if replacement in source:
            continue
        if source.count(original) != 1:
            raise RuntimeError(
                "unitree_mujoco source does not match the expected reset contract"
            )
        source = source.replace(original, replacement, 1)
        changed = True
    return source, changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source_path = find_unitree_mujoco_root() / "simulate/src/main.cc"
    source, changed = patch_source(source_path.read_text(encoding="utf-8"))
    if args.check:
        if changed:
            raise RuntimeError(f"Software reset patch is missing from {source_path}")
        print(f"Software reset patch is installed: {source_path}")
        return 0
    if changed:
        source_path.write_text(source, encoding="utf-8")
        print(f"Patched {source_path}")
    else:
        print(f"Already patched: {source_path}")
    print(
        "Rebuild with: cmake --build "
        f"{source_path.parents[1] / 'build'} -j2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
