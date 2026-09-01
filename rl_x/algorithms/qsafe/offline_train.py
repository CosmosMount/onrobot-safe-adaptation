"""Offline universal-QSafe v2 training and held-out calibration.

This module intentionally has no Isaac dependency.  It consumes complete raw
46D trajectories, constructs the 5-frame safety history, trains the SQRL
Bellman risk critic, and selects one gamma/threshold on actor- and map-isolated
validation trajectories before reporting untouched test performance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from rl_x.algorithms.qsafe.calibration import (
    calibration_report,
    future_failure_labels,
)
from rl_x.algorithms.qsafe.common import (
    safety_bellman_target,
    trajectory_with_observation_history,
)
from rl_x.algorithms.qsafe.dataset import SafetyTrajectoryDataset
from rl_x.algorithms.qsafe.pytorch.qsafe import QSafe
from rl_x.algorithms.sac_qsafe.pytorch.default_config import get_config


GAMMAS = (0.7, 0.9, 0.97)
HORIZONS = (5, 10, 25)


class _DatasetEnvironment:
    def __init__(self, contract):
        self.single_observation_space = SimpleNamespace(
            shape=tuple(contract["base_observation_shape"])
        )
        self.single_action_space = SimpleNamespace(
            shape=tuple(contract["action_shape"])
        )
        self._manifest = {
            key: contract[key] for key in ("observation", "action", "failure")
        }

    def checkpoint_manifest(self, _normalizer):
        return self._manifest


def _history_arrays(dataset, entries, history_length):
    result = []
    for entry in entries:
        arrays = dataset.load(entry)
        trajectory = list(
            zip(
                arrays["states"],
                arrays["next_states"],
                arrays["actions"],
                arrays["failures"],
                arrays["terminations"],
                arrays["truncations"],
            )
        )
        converted = trajectory_with_observation_history(
            trajectory, arrays["states"].shape[-1], history_length
        )
        result.append(
            {
                "entry": entry,
                "states": np.stack([item[0] for item in converted]),
                "next_states": np.stack([item[1] for item in converted]),
                "actions": arrays["actions"],
                "next_actions": arrays["next_actions"],
                "failures": arrays["failures"],
                "terminations": arrays["terminations"],
                "truncations": arrays["truncations"],
                "candidate_actions": arrays.get("candidate_actions"),
            }
        )
    return result


def _concatenate(trajectories, key):
    return np.concatenate([item[key] for item in trajectories], axis=0)


def _risk_values(qsafe, states, actions, chunk_size=65536):
    values = []
    with torch.no_grad():
        for start in range(0, len(states), chunk_size):
            state = torch.as_tensor(
                states[start : start + chunk_size],
                dtype=torch.float32,
                device=qsafe.device,
            )
            action = torch.as_tensor(
                actions[start : start + chunk_size],
                dtype=torch.float32,
                device=qsafe.device,
            )
            values.append(qsafe.values(state, action).cpu().numpy().reshape(-1))
    return np.concatenate(values)


def _prediction_arrays(qsafe, trajectories, horizon_steps):
    """Run the critic once; epsilon sweeps only threshold these predictions."""

    probabilities = []
    labels = []
    candidate_probabilities = []
    candidates_complete = True
    for trajectory in trajectories:
        probabilities.append(
            _risk_values(qsafe, trajectory["states"], trajectory["actions"])
        )
        labels.append(
            future_failure_labels(trajectory["failures"], horizon_steps)
        )
        candidates = trajectory["candidate_actions"]
        if candidates is None:
            candidates_complete = False
            continue
        repeated_states = np.repeat(
            trajectory["states"][:, None, :], candidates.shape[1], axis=1
        )
        candidate_probabilities.append(
            _risk_values(
                qsafe,
                repeated_states.reshape(-1, repeated_states.shape[-1]),
                candidates.reshape(-1, candidates.shape[-1]),
            ).reshape(candidates.shape[:2])
        )
    return {
        "probabilities": np.concatenate(probabilities),
        "labels": np.concatenate(labels),
        "candidate_probabilities": (
            np.concatenate(candidate_probabilities)
            if candidates_complete and candidate_probabilities
            else None
        ),
    }


def _evaluate_predictions(predictions, epsilon):
    return calibration_report(
        predictions["probabilities"],
        predictions["labels"],
        epsilon,
        candidate_probabilities=predictions["candidate_probabilities"],
    )


def _evaluate(qsafe, trajectories, epsilon, horizon_steps):
    """Compatibility wrapper for one-threshold callers and focused tests."""

    return _evaluate_predictions(
        _prediction_arrays(qsafe, trajectories, horizon_steps), epsilon
    )


def _build_qsafe(dataset, gamma, device, seed):
    contract = dataset.manifest["contract"]
    config = SimpleNamespace()
    config.algorithm = get_config("sac_qsafe.pytorch")
    config.algorithm.device = "gpu" if device.type == "cuda" else "cpu"
    config.algorithm.phase = "pretrain"
    config.algorithm.qsafe.gamma = float(gamma)
    # This placeholder is replaced by held-out threshold selection after
    # training; it must never be treated as a historical default.
    config.algorithm.qsafe.epsilon = 0.1
    config.environment = SimpleNamespace(nr_envs=1)
    return QSafe(
        config,
        _DatasetEnvironment(contract),
        device,
        np.random.default_rng(seed),
        phase="pretrain",
    )


def _train_one(qsafe, trajectories, updates, batch_size, seed):
    states = _concatenate(trajectories, "states")
    next_states = _concatenate(trajectories, "next_states")
    actions = _concatenate(trajectories, "actions")
    next_actions = _concatenate(trajectories, "next_actions")
    failures = _concatenate(trajectories, "failures")[:, None]
    terminations = _concatenate(trajectories, "terminations")[:, None]
    truncations = _concatenate(trajectories, "truncations")[:, None]
    all_histories = torch.as_tensor(
        np.concatenate((states, next_states)), dtype=torch.float32, device=qsafe.device
    )
    qsafe.observation_normalizer.update(all_histories)
    rng = np.random.default_rng(seed)
    trajectory_offsets = []
    trajectory_lengths = []
    fall_trajectory_indices = []
    safe_trajectory_indices = []
    offset = 0
    for trajectory_index, trajectory in enumerate(trajectories):
        length = len(trajectory["states"])
        trajectory_offsets.append(offset)
        trajectory_lengths.append(length)
        if np.any(trajectory["failures"]):
            fall_trajectory_indices.append(trajectory_index)
        else:
            safe_trajectory_indices.append(trajectory_index)
        offset += length
    trajectory_offsets = np.asarray(trajectory_offsets, dtype=np.int64)
    trajectory_lengths = np.asarray(trajectory_lengths, dtype=np.int64)
    fall_trajectory_indices = np.asarray(fall_trajectory_indices, dtype=np.int64)
    safe_trajectory_indices = np.asarray(safe_trajectory_indices, dtype=np.int64)
    for update in range(int(updates)):
        if fall_trajectory_indices.size and safe_trajectory_indices.size:
            choose_fall = rng.random(int(batch_size)) < 0.5
            chosen_trajectories = np.empty(int(batch_size), dtype=np.int64)
            nr_fall = int(np.sum(choose_fall))
            chosen_trajectories[choose_fall] = fall_trajectory_indices[
                rng.integers(0, fall_trajectory_indices.size, size=nr_fall)
            ]
            chosen_trajectories[~choose_fall] = safe_trajectory_indices[
                rng.integers(
                    0,
                    safe_trajectory_indices.size,
                    size=int(batch_size) - nr_fall,
                )
            ]
            selected_lengths = trajectory_lengths[chosen_trajectories]
            indices = trajectory_offsets[chosen_trajectories] + (
                rng.random(int(batch_size)) * selected_lengths
            ).astype(np.int64)
            selected_group_sizes = np.where(
                choose_fall,
                fall_trajectory_indices.size,
                safe_trajectory_indices.size,
            )
            sampling_probability = (
                0.5 / selected_group_sizes / selected_lengths
            )
            importance_weights = (
                (1.0 / len(states)) / sampling_probability
            ).astype(np.float32)
        else:
            indices = rng.integers(0, len(states), size=int(batch_size))
            importance_weights = np.ones(int(batch_size), dtype=np.float32)

        def tensor(values):
            return torch.as_tensor(
                values[indices], dtype=torch.float32, device=qsafe.device
            )

        state = qsafe.normalize_observations(tensor(states))
        next_state = qsafe.normalize_observations(tensor(next_states))
        action = tensor(actions)
        with torch.no_grad():
            next_q = qsafe.target(next_state, tensor(next_actions))
            target = safety_bellman_target(
                tensor(failures),
                tensor(terminations),
                tensor(truncations),
                next_q,
                qsafe.gamma,
            )
        predicted = qsafe.online(state, action)
        weights = torch.as_tensor(
            importance_weights[:, None], dtype=torch.float32, device=qsafe.device
        )
        loss = torch.mean(weights * F.mse_loss(predicted, target, reduction="none"))
        qsafe.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(qsafe.online.parameters(), 10.0)
        qsafe.optimizer.step()
        with torch.no_grad():
            for source, target_parameter in zip(
                qsafe.online.parameters(), qsafe.target.parameters()
            ):
                target_parameter.mul_(1.0 - qsafe.tau).add_(source, alpha=qsafe.tau)
        if not math.isfinite(float(loss.detach().cpu())):
            raise FloatingPointError(f"Non-finite QSafe loss at update {update}.")
    return float(loss.detach().cpu())


def _prediction_arrays_for_horizons(qsafe, trajectories, horizons):
    """Score states/candidates once and attach each requested future label."""

    horizons = tuple(int(value) for value in horizons)
    first = _prediction_arrays(qsafe, trajectories, horizons[0])
    result = {horizons[0]: first}
    for horizon in horizons[1:]:
        result[horizon] = {
            "probabilities": first["probabilities"],
            "labels": np.concatenate(
                [
                    future_failure_labels(trajectory["failures"], horizon)
                    for trajectory in trajectories
                ]
            ),
            "candidate_probabilities": first["candidate_probabilities"],
        }
    return result


def _threshold_for_minimum_recall(predictions_by_horizon, target_recall=0.80):
    """Choose the highest shared threshold retaining recall on every horizon."""

    horizon_thresholds = {}
    for horizon, predictions in predictions_by_horizon.items():
        probabilities = np.asarray(predictions["probabilities"], dtype=np.float64)
        labels = np.asarray(predictions["labels"], dtype=bool)
        positives = probabilities[labels]
        if not positives.size:
            raise RuntimeError(f"Horizon {horizon} has no future-failure positives.")
        required = max(1, int(math.ceil(float(target_recall) * positives.size)))
        horizon_thresholds[int(horizon)] = float(np.sort(positives)[-required])
    # A lower threshold rejects more actions. Satisfying every horizon requires
    # the most conservative of their individually feasible thresholds.
    epsilon = min(horizon_thresholds.values())
    reports = {
        int(horizon): _evaluate_predictions(predictions, epsilon)
        for horizon, predictions in predictions_by_horizon.items()
    }
    return epsilon, horizon_thresholds, reports


def _passes_action_ranking_gate(report):
    fallback = float(report["fallback_rate"])
    return bool(
        report["recall_future_failure"] >= 0.80
        and report["safe_action_false_rejection_rate"] <= 0.20
        and np.isfinite(fallback)
        and fallback <= 0.05
    )


def _passes_multi_horizon_gate(reports):
    """Require early-warning recall, with rejection quality at max horizon.

    A transition 8 steps before failure is correctly positive at H=10 but is
    labelled negative at H=5. Applying the false-rejection gate independently
    to every nested horizon would therefore penalize the desired early warning.
    """

    primary = reports[max(reports)]
    return bool(
        all(
            report["recall_future_failure"] >= 0.80
            for report in reports.values()
        )
        and _passes_action_ranking_gate(primary)
    )


def _candidate_rank(reports):
    values = tuple(reports.values())
    passed = _passes_multi_horizon_gate(reports)
    minimum_recall = min(report["recall_future_failure"] for report in values)
    primary = reports[max(reports)]
    primary_false_rejection = primary["safe_action_false_rejection_rate"]
    primary_fallback = primary["fallback_rate"]
    mean_brier_improvement = float(
        np.mean([report["brier_improvement"] for report in values])
    )
    return (
        int(passed),
        -primary_false_rejection,
        minimum_recall,
        -primary_fallback,
        mean_brier_improvement,
    )


def train(args):
    dataset = SafetyTrajectoryDataset(args.dataset)
    dataset.validate_isolation()
    statistics = dataset.statistics()
    minimum_outcomes = {
        "train": int(args.min_train_outcomes),
        "validation": int(args.min_validation_outcomes),
        "test": int(args.min_test_outcomes),
    }
    data_gate = all(
        statistics["by_split"][split][outcome] >= minimum
        for split, minimum in minimum_outcomes.items()
        for outcome in ("fall", "success")
    )
    if not data_gate and not args.allow_incomplete_data:
        counts = ", ".join(
            f"{split}=fall:{statistics['by_split'][split]['fall']}/"
            f"success:{statistics['by_split'][split]['success']}"
            for split in ("train", "validation", "test")
        )
        raise RuntimeError(
            "QSafe split data gate failed. Required fall and success trajectories "
            f"per split: {minimum_outcomes}. Found {counts}."
        )
    split_entries = {
        split: dataset.entries(split) for split in ("train", "validation", "test")
    }
    if any(not entries for entries in split_entries.values()):
        raise RuntimeError("Train, validation, and test splits must all be non-empty.")
    for split, entries in split_entries.items():
        outcomes = {bool(entry["fall"]) for entry in entries}
        if outcomes != {False, True}:
            raise RuntimeError(
                f"{split} must contain both fall and non-fall trajectories; "
                f"found fall outcomes {sorted(outcomes)}."
            )
    history_length = int(dataset.manifest["contract"]["history_length"])
    splits = {
        name: _history_arrays(dataset, entries, history_length)
        for name, entries in split_entries.items()
    }
    device = torch.device(
        "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    best = None
    horizons = tuple(int(value) for value in args.horizons.split(",") if value)
    gammas = tuple(float(value) for value in args.gammas.split(",") if value)
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError("--horizons must contain positive integers.")
    if not gammas or any(not 0.0 < value < 1.0 for value in gammas):
        raise ValueError("--gammas must contain values strictly between 0 and 1.")
    for gamma in gammas:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        qsafe = _build_qsafe(dataset, gamma, device, args.seed)
        final_loss = _train_one(
            qsafe, splits["train"], args.updates, args.batch_size, args.seed
        )
        validation_predictions = _prediction_arrays_for_horizons(
            qsafe, splits["validation"], horizons
        )
        epsilon, per_horizon_thresholds, validation_reports = (
            _threshold_for_minimum_recall(validation_predictions)
        )
        candidate = {
            "gamma_safe": gamma,
            "epsilon": epsilon,
            "per_horizon_maximum_threshold_at_80pct_recall": (
                per_horizon_thresholds
            ),
            "final_loss": final_loss,
            "validation_by_horizon": validation_reports,
            "validation_pass": _passes_multi_horizon_gate(validation_reports),
        }
        rank = _candidate_rank(validation_reports)
        candidates.append(candidate)
        if best is None or rank > best[0]:
            best = (rank, qsafe, candidate)
    _, qsafe, selected = best
    qsafe.epsilon = float(selected["epsilon"])
    qsafe.config.epsilon = float(selected["epsilon"])
    test_predictions = _prediction_arrays_for_horizons(
        qsafe, splits["test"], horizons
    )
    test_reports = {
        horizon: _evaluate_predictions(predictions, selected["epsilon"])
        for horizon, predictions in test_predictions.items()
    }
    validation_pass = bool(selected["validation_pass"])
    test_pass = _passes_multi_horizon_gate(test_reports)
    primary_horizon = max(horizons)
    test_report = test_reports[primary_horizon]
    report = {
        "dataset": str(Path(args.dataset).resolve()),
        "statistics": statistics,
        "data_gate_pass": data_gate,
        "horizon_steps": int(primary_horizon),
        "horizons": list(horizons),
        "horizon_seconds": float(
            primary_horizon * dataset.manifest["contract"]["control_dt"]
        ),
        "minimum_outcomes": minimum_outcomes,
        "updates_per_gamma": int(args.updates),
        "gammas": list(gammas),
        "candidates": candidates,
        "selected": {
            **{key: selected[key] for key in ("gamma_safe", "epsilon")},
            "validation_by_horizon": selected["validation_by_horizon"],
            "validation_pass": validation_pass,
            "test": test_report,
            "test_by_horizon": test_reports,
            "test_pass": test_pass,
        },
        "universal_qsafe_v2_pass": bool(
            data_gate and validation_pass and test_pass
        ),
    }
    report["status"] = (
        "PASS" if report["universal_qsafe_v2_pass"] else "WARN"
    )
    report["artifact_status"] = (
        "calibrated_candidate"
        if validation_pass and test_pass
        else "diagnostic_candidate"
    )
    qsafe.calibration_report = report
    checkpoint_name = args.checkpoint_name
    qsafe.save(output / checkpoint_name, include_optimizer=False)
    (output / "calibration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.diagnostic_continue:
        return 0
    return 0 if report["universal_qsafe_v2_pass"] else 2


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--updates", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--horizons",
        default=",".join(map(str, HORIZONS)),
        help="Comma-separated future-failure horizons used by one shared threshold.",
    )
    parser.add_argument(
        "--gammas", default=",".join(map(str, GAMMAS))
    )
    parser.add_argument("--min-train-outcomes", type=int, default=500)
    parser.add_argument("--min-validation-outcomes", type=int, default=100)
    parser.add_argument("--min-test-outcomes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--allow-incomplete-data", action="store_true")
    parser.add_argument("--diagnostic-continue", action="store_true")
    parser.add_argument(
        "--checkpoint-name",
        default="qsafe_flat_rough_step_v1_candidate.model",
    )
    return parser


def main(argv=None):
    return train(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
