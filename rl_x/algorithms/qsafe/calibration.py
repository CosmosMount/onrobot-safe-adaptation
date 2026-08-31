"""Calibration metrics and gates for frozen universal QSafe checkpoints."""

from __future__ import annotations

import numpy as np


def future_failure_labels(failures, horizon_steps):
    failures = np.asarray(failures, dtype=bool).reshape(-1)
    horizon_steps = int(horizon_steps)
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive.")
    labels = np.zeros(failures.shape, dtype=np.float32)
    for index in range(failures.size):
        labels[index] = np.any(failures[index : index + horizon_steps])
    return labels


def expected_calibration_error(probabilities, labels, bins=10):
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    result = 0.0
    for index in range(len(edges) - 1):
        lower, upper = edges[index], edges[index + 1]
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == len(edges) - 2 else probabilities < upper
        )
        if np.any(mask):
            result += np.mean(mask) * abs(
                np.mean(probabilities[mask]) - np.mean(labels[mask])
            )
    return float(result)


def calibration_report(probabilities, labels, epsilon, candidate_probabilities=None):
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if probabilities.shape != labels.shape or not probabilities.size:
        raise ValueError("Probabilities and labels must be non-empty aligned vectors.")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("QSafe probabilities must be finite and in [0, 1].")
    positives = labels.astype(bool)
    rejected = probabilities >= float(epsilon)
    recall = float(np.mean(rejected[positives])) if np.any(positives) else float("nan")
    false_rejection = (
        float(np.mean(rejected[~positives])) if np.any(~positives) else float("nan")
    )
    brier = float(np.mean(np.square(probabilities - labels)))
    prevalence = float(np.mean(labels))
    constant_brier = float(np.mean(np.square(prevalence - labels)))
    improvement = (
        1.0 - brier / constant_brier if constant_brier > 0 else float("nan")
    )
    fallback = float("nan")
    if candidate_probabilities is not None:
        candidates = np.asarray(candidate_probabilities, dtype=np.float64)
        if candidates.ndim != 2:
            raise ValueError("candidate_probabilities must be [samples, candidates].")
        fallback = float(np.mean(np.all(candidates >= float(epsilon), axis=1)))
    return {
        "recall_future_failure": recall,
        "safe_action_false_rejection_rate": false_rejection,
        "ece": expected_calibration_error(probabilities, labels),
        "brier_score": brier,
        "constant_failure_rate_brier": constant_brier,
        "brier_improvement": float(improvement),
        "fallback_rate": fallback,
        "epsilon": float(epsilon),
        "samples": int(labels.size),
        "positive_rate": prevalence,
    }


def passes_isaac_gate(report):
    return bool(
        report["recall_future_failure"] >= 0.80
        and report["safe_action_false_rejection_rate"] <= 0.20
        and report["ece"] <= 0.05
        and report["brier_improvement"] >= 0.20
        and report["fallback_rate"] <= 0.05
    )


def passes_mujoco_gate(report):
    return bool(
        report["recall_future_failure"] >= 0.70
        and report["safe_action_false_rejection_rate"] <= 0.30
        and report["fallback_rate"] <= 0.10
    )
