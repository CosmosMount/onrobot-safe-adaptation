"""Audit a frozen Go2 Q_safe with paired MuJoCo snapshot rollouts.

This diagnostic deliberately bypasses DDS and learning.  It first checks that
the deterministic policy walks in the canonical project MuJoCo model, then
restores identical disturbed states to test whether Q_safe ranks candidate
actions in the same order as their short-horizon physical outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environments.go2_sqrl.common.specs import (
    ACTION_SPEC,
    DEFAULT_JOINT_POSITION,
    JOINT_NAMES,
    OBSERVATION_SPEC,
    PHYSICS_STEPS_PER_ACTION,
)
from src.environments.go2_sqrl.common.estimation import VelocityEstimator
from src.environments.go2_sqrl.common.observation import build_observation
from src.environments.go2_sqrl.common.types import RobotState


DEFAULT_MODELS = (
    PROJECT_ROOT
    / "runs/go2_sqrl/pretrain/isaac_flashsac_cmd_reward_v3/models"
)
DEFAULT_SCENE = PROJECT_ROOT / "assets/robots/go2/mjcf/scene.xml"
POSE_1 = np.asarray(
    [
        0.0, 1.36, -2.65,
        0.0, 1.36, -2.65,
        -0.2, 1.36, -2.65,
        0.2, 1.36, -2.65,
    ],
    dtype=np.float64,
)


class Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.torso = nn.Sequential(
            nn.Linear(46, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        )
        self.mean = nn.Linear(256, 12)
        self.log_std = nn.Linear(256, 12)

    def distribution(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.torso(observation)
        return self.mean(latent), self.log_std(latent).clamp(-20.0, 2.0)

    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        mean, _ = self.distribution(observation)
        return torch.tanh(mean)

    def candidates(
        self, observation: torch.Tensor, noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution(observation)
        raw = mean + log_std.exp() * noise
        action = torch.tanh(raw)
        log_probability = -0.5 * (
            noise.square() + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        log_probability -= torch.log(1.0 - action.square() + 1e-6)
        return action, log_probability.sum(dim=-1)


class SafetyCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(58, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((observation, action), dim=-1)).squeeze(-1)


def _strip_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state.items()
    }


@dataclass
class Snapshot:
    state: np.ndarray
    previous_target: np.ndarray
    previous_quaternion: np.ndarray | None
    estimator_world_velocity: np.ndarray


class DirectGo2:
    def __init__(self, scene: Path) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.state_size = mujoco.mj_stateSize(self.model, self.state_spec)
        self.previous_target = DEFAULT_JOINT_POSITION.astype(np.float64).copy()
        self.previous_quaternion: np.ndarray | None = None
        self.velocity_estimator = VelocityEstimator()
        self.joint_qpos = []
        self.joint_dof = []
        for name in JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint"
            )
            self.joint_qpos.append(int(self.model.jnt_qposadr[joint_id]))
            self.joint_dof.append(int(self.model.jnt_dofadr[joint_id]))
        self.joint_qpos = np.asarray(self.joint_qpos)
        self.joint_dof = np.asarray(self.joint_dof)

    def _sensor(self, name: str) -> np.ndarray:
        return np.asarray(self.data.sensor(name).data, dtype=np.float64).copy()

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.qpos[self.joint_qpos].copy(), self.data.qvel[self.joint_dof].copy()

    def _pd_period(self, target: np.ndarray, kp: float, kd: float) -> None:
        for _ in range(PHYSICS_STEPS_PER_ACTION):
            q, dq = self.joint_state()
            self.data.ctrl[:] = np.clip(
                kp * (target - q) - kd * dq,
                -ACTION_SPEC.effort_limit,
                ACTION_SPEC.effort_limit,
            )
            mujoco.mj_step(self.model, self.data)

    def reset(self) -> None:
        home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, home_key)
        mujoco.mj_forward(self.model, self.data)
        home = DEFAULT_JOINT_POSITION.astype(np.float64)
        phases = ((home, POSE_1, 50), (POSE_1, home, 50), (home, home, 100))
        for start, target, steps in phases:
            for index in range(steps):
                alpha = float(index + 1) / float(steps)
                self._pd_period((1.0 - alpha) * start + alpha * target, 60.0, 1.0)
        self.previous_target = home.copy()
        self.previous_quaternion = None
        self.velocity_estimator.reset()

    def snapshot(self) -> Snapshot:
        state = np.empty(self.state_size, dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, state, self.state_spec)
        return Snapshot(
            state,
            self.previous_target.copy(),
            None if self.previous_quaternion is None else self.previous_quaternion.copy(),
            self.velocity_estimator.world_velocity.copy(),
        )

    def restore(self, snapshot: Snapshot) -> None:
        mujoco.mj_setState(self.model, self.data, snapshot.state, self.state_spec)
        mujoco.mj_forward(self.model, self.data)
        self.previous_target = snapshot.previous_target.copy()
        self.previous_quaternion = (
            None
            if snapshot.previous_quaternion is None
            else snapshot.previous_quaternion.copy()
        )
        self.velocity_estimator.reset(snapshot.estimator_world_velocity)

    def disturb(self, disturbance: np.ndarray) -> None:
        # Free-joint qvel begins with world linear velocity followed by angular velocity.
        self.data.qvel[1] += disturbance[0]
        self.data.qvel[2] += disturbance[1]
        self.data.qvel[3] += disturbance[2]
        self.data.qvel[4] += disturbance[3]
        self.data.qvel[5] += disturbance[4]
        self.data.qpos[2] = max(0.19, self.data.qpos[2] - disturbance[5])
        mujoco.mj_forward(self.model, self.data)

    def observation(self) -> np.ndarray:
        q, dq = self.joint_state()
        quaternion = self._sensor("imu_quat")
        state = RobotState(
            joint_q=q,
            joint_dq=dq,
            imu_gyro=self._sensor("imu_gyro"),
            imu_quat=quaternion,
            imu_accelerometer=self._sensor("imu_acc"),
        )
        estimated_body_velocity = self.velocity_estimator.update(state)
        observation, quaternion = build_observation(
            state,
            estimated_body_velocity,
            self.previous_target,
            self.previous_quaternion,
        )
        self.previous_quaternion = quaternion
        return observation

    def project(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        clipped = np.clip(action, -1.0, 1.0)
        target = np.clip(
            DEFAULT_JOINT_POSITION + ACTION_SPEC.scale * clipped,
            self.previous_target - ACTION_SPEC.max_target_rate * ACTION_SPEC.control_dt,
            self.previous_target + ACTION_SPEC.max_target_rate * ACTION_SPEC.control_dt,
        )
        applied = np.clip(
            (target - DEFAULT_JOINT_POSITION) / ACTION_SPEC.scale, -1.0, 1.0
        )
        return applied.astype(np.float32), target.astype(np.float64)

    def step(self, action: np.ndarray) -> tuple[bool, float, float, float]:
        _, target = self.project(action)
        self.previous_target = target
        failure_count = 0
        max_angle = 0.0
        min_height = float("inf")
        for _ in range(10):
            q, dq = self.joint_state()
            self.data.ctrl[:] = np.clip(
                ACTION_SPEC.kp * (target - q) - ACTION_SPEC.kd * dq,
                -ACTION_SPEC.effort_limit,
                ACTION_SPEC.effort_limit,
            )
            mujoco.mj_step(self.model, self.data)
            quaternion = self._sensor("imu_quat")
            roll, pitch = roll_pitch(quaternion)
            height = float(self._sensor("frame_pos")[2])
            max_angle = max(max_angle, abs(roll), abs(pitch))
            min_height = min(min_height, height)
            outside = abs(roll) > 0.8 or abs(pitch) > 0.8 or height < 0.18
            failure_count = failure_count + 1 if outside else 0
            if failure_count >= 5:
                return True, max_angle, min_height, self.forward_velocity()
        return False, max_angle, min_height, self.forward_velocity()

    def forward_velocity(self) -> float:
        quaternion = self._sensor("imu_quat")
        rotation = quaternion_matrix(quaternion)
        return float((rotation.T @ self._sensor("frame_vel"))[0])


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def roll_pitch(quaternion: np.ndarray) -> tuple[float, float]:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, math.asin(sin_pitch)


class Audit:
    def __init__(self, models: Path, scene: Path, seed: int) -> None:
        policy_artifact = torch.load(models / "policy.model", map_location="cpu", weights_only=False)
        qsafe_artifact = torch.load(models / "qsafe.model", map_location="cpu", weights_only=False)
        checkpoint_observation_version = policy_artifact["environment_manifest"][
            "observation"
        ]["version"]
        if checkpoint_observation_version != OBSERVATION_SPEC.version:
            raise ValueError(
                "Snapshot audit observation contract mismatch: expected "
                f"{OBSERVATION_SPEC.version}, got {checkpoint_observation_version}. "
                "Retrain or select a checkpoint produced by the current contract."
            )
        self.policy = Policy().eval()
        self.policy.load_state_dict(
            _strip_prefix(policy_artifact["policy_state_dict"], "_orig_mod.")
        )
        self.qsafe = SafetyCritic().eval()
        qsafe_state = {
            key.removeprefix("network."): value
            for key, value in qsafe_artifact["online_state_dict"].items()
            if key != "observation_indices"
        }
        self.qsafe.network.load_state_dict(qsafe_state)
        normalizer = policy_artifact["observation_normalizer_state_dict"]
        self.mean = normalizer["running_mean"].float()
        self.std = torch.sqrt(torch.clamp(normalizer["running_var"].float(), min=0.0))
        self.sim = DirectGo2(scene)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

    def normalize(self, observation: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        return (tensor - self.mean) / (self.std + 1e-8)

    @torch.no_grad()
    def deterministic_action(self, observation: np.ndarray) -> np.ndarray:
        return self.policy.deterministic(self.normalize(observation))[0].numpy()

    @torch.no_grad()
    def candidates(
        self,
        observation: np.ndarray,
        count: int,
        generator: torch.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        normalized = self.normalize(observation)
        noise = torch.randn(count, 12, generator=generator)
        actions, log_probabilities = self.policy.candidates(
            normalized.expand(count, -1), noise
        )
        previous = observation[34:46]
        projected = []
        for action in actions.numpy():
            old_target = self.sim.previous_target
            self.sim.previous_target = previous.astype(np.float64)
            applied, _ = self.sim.project(action)
            self.sim.previous_target = old_target
            projected.append(applied)
        projected_tensor = torch.as_tensor(np.asarray(projected), dtype=torch.float32)
        q_values = self.qsafe(normalized.expand(count, -1), projected_tensor)
        return projected_tensor.numpy(), log_probabilities.numpy(), q_values.numpy()

    def rollout(
        self,
        snapshot: Snapshot,
        first_action: np.ndarray,
        horizon: int,
    ) -> dict[str, float | bool]:
        self.sim.restore(snapshot)
        failed = False
        max_angle = 0.0
        min_height = float("inf")
        velocities = []
        for step in range(horizon):
            observation = self.sim.observation()
            action = first_action if step == 0 else self.deterministic_action(observation)
            failure, angle, height, velocity = self.sim.step(action)
            failed = failed or failure
            max_angle = max(max_angle, angle)
            min_height = min(min_height, height)
            velocities.append(velocity)
            if failed:
                break
        risk = max(
            max_angle / 0.8,
            max(0.0, (0.18 - min_height) / 0.08 + 1.0),
            2.0 if failed else 0.0,
        )
        return {
            "failure": failed,
            "risk": float(risk),
            "max_angle": float(max_angle),
            "min_height": float(min_height),
            "mean_velocity": float(np.mean(velocities)),
        }

    def selector_rollout(
        self,
        snapshot: Snapshot,
        selector: str,
        horizon: int,
        candidate_count: int,
        seed: int,
    ) -> dict[str, float | bool]:
        self.sim.restore(snapshot)
        generator = torch.Generator().manual_seed(seed)
        selection_rng = np.random.default_rng(seed)
        failed = False
        velocities = []
        selected_q_values = []
        rejected_fractions = []
        for _ in range(horizon):
            observation = self.sim.observation()
            if selector == "task":
                action = self.deterministic_action(observation)
            else:
                actions, log_probabilities, q_values = self.candidates(
                    observation, candidate_count, generator
                )
                safe = q_values < 0.1
                rejected_fractions.append(float(np.mean(~safe)))
                if not safe.any() or selector == "lowest_q":
                    selected = int(np.argmin(q_values))
                elif selector == "first_safe":
                    selected = int(np.flatnonzero(safe)[0])
                elif selector == "importance":
                    indices = np.flatnonzero(safe)
                    logits = log_probabilities[indices]
                    weights = np.exp(logits - np.max(logits))
                    weights /= weights.sum()
                    selected = int(selection_rng.choice(indices, p=weights))
                else:
                    raise ValueError(f"Unknown selector: {selector}")
                action = actions[selected]
                selected_q_values.append(float(q_values[selected]))
            failure, _, _, velocity = self.sim.step(action)
            velocities.append(velocity)
            failed = failed or failure
            if failed:
                break
        return {
            "failure": failed,
            "steps": len(velocities),
            "mean_velocity": float(np.mean(velocities)),
            "last_velocity": float(velocities[-1]),
            "mean_selected_q": (
                float(np.mean(selected_q_values)) if selected_q_values else float("nan")
            ),
            "mean_rejected_fraction": (
                float(np.mean(rejected_fractions)) if rejected_fractions else 0.0
            ),
        }

    def parity_rollout(self, steps: int) -> tuple[dict[str, float | bool], list[Snapshot]]:
        self.sim.reset()
        velocities = []
        snapshots = []
        failed = False
        for step in range(steps):
            observation = self.sim.observation()
            action = self.deterministic_action(observation)
            failure, _, _, velocity = self.sim.step(action)
            velocities.append(velocity)
            failed = failed or failure
            if 40 <= step < min(steps, 340) and step % 20 == 0:
                snapshots.append(self.sim.snapshot())
            if failure:
                break
        window = min(100, len(velocities))
        return (
            {
                "steps": len(velocities),
                "failure": failed,
                "mean_forward_velocity": float(np.mean(velocities)),
                "last_window_forward_velocity": float(np.mean(velocities[-window:])),
            },
            snapshots,
        )


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty(len(values), dtype=np.float64)
        result[order] = np.arange(len(values), dtype=np.float64)
        return result

    xr, yr = ranks(x), ranks(y)
    if np.std(xr) == 0.0 or np.std(yr) == 0.0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def main() -> None:
    # Tiny MLP inference is much faster and more reproducible without BLAS
    # oversubscription, especially when several audit seeds run in parallel.
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parity-steps", type=int, default=500)
    parser.add_argument("--states", type=int, default=12)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument(
        "--disturbance-scale",
        type=float,
        default=1.0,
        help="Multiply all sampled velocity impulses and the base-height drop.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    audit = Audit(args.models.resolve(), args.scene.resolve(), args.seed)
    parity, nominal_snapshots = audit.parity_rollout(args.parity_steps)
    if parity["failure"] or parity["mean_forward_velocity"] < 0.2:
        raise RuntimeError(f"Direct MuJoCo parity gate failed: {parity}")

    records = []
    selector_records = []
    for state_index in range(args.states):
        source = nominal_snapshots[state_index % len(nominal_snapshots)]
        audit.sim.restore(source)
        disturbance = args.disturbance_scale * np.asarray(
            [
                audit.rng.uniform(-1.2, 1.2),
                audit.rng.uniform(-0.5, 0.2),
                audit.rng.uniform(-4.0, 4.0),
                audit.rng.uniform(-4.0, 4.0),
                audit.rng.uniform(-1.5, 1.5),
                audit.rng.uniform(0.0, 0.055),
            ]
        )
        audit.sim.disturb(disturbance)
        disturbed = audit.sim.snapshot()
        observation = audit.sim.observation()
        actions, log_probabilities, q_values = audit.candidates(
            observation, args.candidates
        )
        outcomes = [
            audit.rollout(disturbed, action, args.horizon) for action in actions
        ]
        risks = np.asarray([outcome["risk"] for outcome in outcomes])
        failures = np.asarray([outcome["failure"] for outcome in outcomes])
        records.append(
            {
                "state": state_index,
                "disturbance": disturbance.tolist(),
                "q_values": q_values.tolist(),
                "risks": risks.tolist(),
                "failures": failures.astype(int).tolist(),
                "log_probabilities": log_probabilities.tolist(),
                "spearman": rank_correlation(q_values, risks),
                "lowest_q_failed": bool(failures[np.argmin(q_values)]),
                "first_failed": bool(failures[0]),
                "lowest_q_risk": float(risks[np.argmin(q_values)]),
                "first_risk": float(risks[0]),
                "mixed_failure_outcomes": bool(failures.any() and not failures.all()),
            }
        )
        selector_records.append(
            {
                "state": state_index,
                **{
                    selector: audit.selector_rollout(
                        disturbed,
                        selector,
                        args.horizon,
                        args.candidates,
                        seed=args.seed * 100_000 + state_index,
                    )
                    for selector in (
                        "task",
                        "lowest_q",
                        "first_safe",
                        "importance",
                    )
                },
            }
        )

    correlations = np.asarray([record["spearman"] for record in records])
    result = {
        "parity": parity,
        "settings": {
            "states": args.states,
            "candidates": args.candidates,
            "horizon": args.horizon,
            "seed": args.seed,
            "disturbance_scale": args.disturbance_scale,
        },
        "summary": {
            "mean_spearman_q_vs_risk": float(np.nanmean(correlations)),
            "median_spearman_q_vs_risk": float(np.nanmedian(correlations)),
            "mixed_failure_states": int(sum(r["mixed_failure_outcomes"] for r in records)),
            "lowest_q_failures": int(sum(r["lowest_q_failed"] for r in records)),
            "first_candidate_failures": int(sum(r["first_failed"] for r in records)),
            "mean_lowest_q_risk": float(np.mean([r["lowest_q_risk"] for r in records])),
            "mean_first_candidate_risk": float(np.mean([r["first_risk"] for r in records])),
            "q_min": float(min(min(r["q_values"]) for r in records)),
            "q_max": float(max(max(r["q_values"]) for r in records)),
            "fraction_q_below_epsilon": float(
                np.mean(np.concatenate([np.asarray(r["q_values"]) < 0.1 for r in records]))
            ),
            "selector_failures": {
                selector: int(
                    sum(record[selector]["failure"] for record in selector_records)
                )
                for selector in ("task", "lowest_q", "first_safe", "importance")
            },
            "selector_mean_velocity": {
                selector: float(
                    np.mean(
                        [record[selector]["mean_velocity"] for record in selector_records]
                    )
                )
                for selector in ("task", "lowest_q", "first_safe", "importance")
            },
            "selector_mean_rejected_fraction": {
                selector: float(
                    np.mean(
                        [
                            record[selector]["mean_rejected_fraction"]
                            for record in selector_records
                        ]
                    )
                )
                for selector in ("lowest_q", "first_safe", "importance")
            },
        },
        "records": records,
        "selector_records": selector_records,
    }
    encoded = json.dumps(result, indent=2)
    if not args.quiet:
        print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
