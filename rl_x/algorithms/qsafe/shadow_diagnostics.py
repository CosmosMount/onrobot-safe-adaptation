"""Non-intervening target-domain diagnostics for a frozen QSafe critic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rl_x.algorithms.qsafe.calibration import (
    calibration_report,
    future_failure_labels,
)


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size:
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _roc_auc(positives: np.ndarray, negatives: np.ndarray) -> float | None:
    """Return the probability that a positive has a higher risk score."""

    positives = np.asarray(positives, dtype=np.float64).reshape(-1)
    negatives = np.asarray(negatives, dtype=np.float64).reshape(-1)
    if not positives.size or not negatives.size:
        return None
    ordered = np.sort(negatives)
    lower = np.searchsorted(ordered, positives, side="left")
    upper = np.searchsorted(ordered, positives, side="right")
    wins = lower.astype(np.float64)
    ties = (upper - lower).astype(np.float64)
    return float(np.mean((wins + 0.5 * ties) / ordered.size))


def build_shadow_report(arrays: dict[str, np.ndarray], epsilon: float) -> dict:
    """Compare QSafe scores before target-domain falls and on normal episodes."""

    required = {
        "env_index",
        "episode_id",
        "failure",
        "done",
        "executed_q",
        "candidate_q",
        "candidate_best_action_l2",
        "observation_abs_z_p95",
        "observation_ood_fraction",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"Shadow diagnostic arrays are missing {sorted(missing)}")

    env_index = np.asarray(arrays["env_index"], dtype=np.int64)
    episode_id = np.asarray(arrays["episode_id"], dtype=np.int64)
    failures = np.asarray(arrays["failure"], dtype=bool)
    done = np.asarray(arrays["done"], dtype=bool)
    scores = np.asarray(arrays["executed_q"], dtype=np.float64)
    candidate_q = np.asarray(arrays["candidate_q"], dtype=np.float64)
    if candidate_q.ndim != 2 or candidate_q.shape[0] != scores.size:
        raise ValueError("candidate_q must have shape [steps, candidates]")

    labels_by_horizon = {
        horizon: np.zeros(scores.shape, dtype=bool) for horizon in (5, 10, 25)
    }
    safe_episode = np.zeros(scores.shape, dtype=bool)
    fall_episode_count = 0
    complete_safe_episode_count = 0
    keys = np.stack((env_index, episode_id), axis=1)
    for key in np.unique(keys, axis=0):
        indices = np.flatnonzero(np.all(keys == key, axis=1))
        trajectory_failures = failures[indices]
        trajectory_complete = bool(done[indices[-1]])
        trajectory_fell = bool(np.any(trajectory_failures))
        if trajectory_fell:
            fall_episode_count += 1
        elif trajectory_complete:
            complete_safe_episode_count += 1
            safe_episode[indices] = True
        for horizon in labels_by_horizon:
            labels_by_horizon[horizon][indices] = future_failure_labels(
                trajectory_failures, horizon
            ).astype(bool)

    def pool_summary(mask: np.ndarray) -> dict:
        values = candidate_q[mask]
        if not values.shape[0]:
            return {"steps": 0}
        executed = values[:, 0]
        minimum = np.min(values, axis=1)
        fallback_mask = np.all(values >= float(epsilon), axis=1)
        return {
            "steps": int(values.shape[0]),
            "fallback_fraction": float(np.mean(fallback_mask)),
            "intervention_opportunity_fraction": float(
                np.mean((executed >= float(epsilon)) & ~fallback_mask)
            ),
            "executed_score": _summary(executed),
            "minimum_candidate_score": _summary(minimum),
            "best_score_reduction": _summary(executed - minimum),
            "within_pool_score_range": _summary(np.ptp(values, axis=1)),
        }

    horizons = {}
    for horizon, labels in labels_by_horizon.items():
        normal_mask = safe_episode if np.any(safe_episode) else ~labels
        positive_scores = scores[labels]
        normal_scores = scores[normal_mask]
        report = calibration_report(
            scores,
            labels.astype(np.float32),
            epsilon,
            candidate_probabilities=candidate_q,
        )
        report.update(
            imminent_fall_scores=_summary(positive_scores),
            normal_scores=_summary(normal_scores),
            normal_group=(
                "complete_no_fall_episodes"
                if np.any(safe_episode)
                else "all_non_imminent_transitions"
            ),
            roc_auc=_roc_auc(positive_scores, normal_scores),
            candidate_pool_imminent_fall=pool_summary(labels),
            candidate_pool_normal=pool_summary(normal_mask),
        )
        horizons[str(horizon)] = report

    candidate_range = np.ptp(candidate_q, axis=1)
    candidate_std = np.std(candidate_q, axis=1)
    executed_rank_fraction = np.mean(
        candidate_q <= candidate_q[:, :1], axis=1
    )
    fallback = np.all(candidate_q >= float(epsilon), axis=1)
    report = {
        "status": (
            "enough_target_falls" if fall_episode_count >= 5 else "insufficient_target_falls"
        ),
        "epsilon_unchanged": float(epsilon),
        "transitions": int(scores.size),
        "fall_episodes": int(fall_episode_count),
        "complete_safe_episodes": int(complete_safe_episode_count),
        "horizons": horizons,
        "candidate_pool": {
            "candidates_per_step": int(candidate_q.shape[1]),
            "fallback_fraction": float(np.mean(fallback)),
            "executed_action_rejected_fraction": float(
                np.mean(scores >= float(epsilon))
            ),
            "intervention_opportunity_fraction": float(
                np.mean((scores >= float(epsilon)) & ~fallback)
            ),
            "score_range": _summary(candidate_range),
            "score_std": _summary(candidate_std),
            "executed_rank_fraction": _summary(executed_rank_fraction),
            "lowest_risk_action_l2_from_executed": _summary(
                arrays["candidate_best_action_l2"]
            ),
        },
        "target_observation_shift": {
            "abs_z_p95": _summary(arrays["observation_abs_z_p95"]),
            "ood_fraction": _summary(arrays["observation_ood_fraction"]),
        },
    }
    return report


class ShadowQSafeRecorder:
    """Record frozen-QSafe scores without changing the action sent to the env."""

    def __init__(self, path: str | Path, nr_envs: int, epsilon: float):
        self.path = Path(path)
        self.report_path = self.path.with_suffix(".report.json")
        self.nr_envs = int(nr_envs)
        self.epsilon = float(epsilon)
        self.episode_ids = np.zeros(self.nr_envs, dtype=np.int64)
        self._records: dict[str, list[np.ndarray]] = {}

    def add(
        self,
        *,
        global_step: int,
        states,
        applied_actions,
        failure,
        terminated,
        truncated,
        candidate_q,
        candidate_best_action_l2,
        observation_abs_z_p95,
        observation_ood_fraction,
    ) -> None:
        candidate_q = np.asarray(candidate_q, dtype=np.float32)
        if candidate_q.shape[0] != self.nr_envs:
            raise ValueError("Shadow candidate scores do not match nr_envs")
        done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        values = {
            "global_step": np.arange(
                int(global_step), int(global_step) + self.nr_envs, dtype=np.int64
            ),
            "env_index": np.arange(self.nr_envs, dtype=np.int32),
            "episode_id": self.episode_ids.copy(),
            "state": np.asarray(states, dtype=np.float32),
            "applied_action": np.asarray(applied_actions, dtype=np.float32),
            "failure": np.asarray(failure, dtype=bool),
            "done": done,
            "executed_q": candidate_q[:, 0],
            "candidate_q": candidate_q,
            "candidate_best_action_l2": np.asarray(
                candidate_best_action_l2, dtype=np.float32
            ),
            "observation_abs_z_p95": np.asarray(
                observation_abs_z_p95, dtype=np.float32
            ),
            "observation_ood_fraction": np.asarray(
                observation_ood_fraction, dtype=np.float32
            ),
        }
        for key, value in values.items():
            self._records.setdefault(key, []).append(value)
        self.episode_ids += done.astype(np.int64)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            key: np.concatenate(parts, axis=0)
            for key, parts in self._records.items()
        }

    def flush(self) -> dict | None:
        if not self._records:
            return None
        arrays = self.arrays()
        report = build_shadow_report(arrays, self.epsilon)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(self.path)
        report_temporary = self.report_path.with_suffix(".json.tmp")
        report_temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report_temporary.replace(self.report_path)
        return report
