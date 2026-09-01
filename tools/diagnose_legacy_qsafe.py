#!/usr/bin/env python3
"""Diagnose whether a legacy QSafe ranks imminent falls above normal states.

The legacy (v1) QSafe consumes the actor-normalized 46D policy observation and
the executed 12D normalized action.  This tool intentionally evaluates the
same raw trajectories with one or more actor normalizers so that critic
failure can be separated from a normalizer/distribution-transfer failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_x.algorithms.qsafe.calibration import future_failure_labels
from rl_x.algorithms.qsafe.dataset import SafetyTrajectoryDataset
from rl_x.algorithms.qsafe.pytorch.safety_critic import SafetyQNetwork


DEFAULT_HORIZONS = (5, 10, 25)
DEFAULT_EPSILON = 0.1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _legacy_scale(scale) -> bool:
    values = np.asarray(scale, dtype=np.float64).reshape(-1)
    return values.size in (1, 12) and bool(np.allclose(values, 0.25, atol=1e-7))


def validate_legacy_dataset_contract(contract: dict) -> None:
    observation_shape = tuple(contract.get("base_observation_shape", ()))
    action_shape = tuple(contract.get("action_shape", ()))
    action = dict(contract.get("action") or {})
    problems = []
    if observation_shape != (46,):
        problems.append(f"base_observation_shape={observation_shape}, expected (46,)")
    if action_shape != (12,):
        problems.append(f"action_shape={action_shape}, expected (12,)")
    if action.get("version") != "go2-action-v1":
        problems.append(
            f"action.version={action.get('version')!r}, expected 'go2-action-v1'"
        )
    if action.get("pipeline_version") != "sdk-absolute-position-v2":
        problems.append(
            "action.pipeline_version="
            f"{action.get('pipeline_version')!r}, expected 'sdk-absolute-position-v2'"
        )
    if not _legacy_scale(action.get("scale", [])):
        problems.append(f"action.scale={action.get('scale')!r}, expected 0.25")
    if problems:
        raise ValueError(
            "Dataset cannot be scored by the legacy QSafe because its tensor/action "
            "contract differs:\n- " + "\n- ".join(problems)
        )


def load_legacy_qsafe(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(checkpoint.get("metadata") or {})
    if int(metadata.get("checkpoint_version", -1)) != 1:
        raise ValueError("This diagnostic only accepts a legacy QSafe v1 checkpoint.")
    observation_shape = tuple(metadata.get("observation_shape", ()))
    action_shape = tuple(metadata.get("action_shape", ()))
    if observation_shape != (46,) or action_shape != (12,):
        raise ValueError(
            f"Expected legacy shapes (46,) and (12,), got {observation_shape} and "
            f"{action_shape}."
        )
    model = SafetyQNetwork(
        observation_shape,
        action_shape,
        metadata["observation_indices"],
        int(metadata["nr_hidden_units"]),
        "tanh",
    ).to(device)
    model.load_state_dict(checkpoint["online_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    optimizer_updates = None
    optimizer = checkpoint.get("optimizer_state_dict")
    if optimizer:
        steps = []
        for state in optimizer.get("state", {}).values():
            step = state.get("step")
            if step is not None:
                steps.append(int(torch.as_tensor(step).item()))
        if steps:
            if len(set(steps)) != 1:
                raise ValueError(f"QSafe optimizer parameters disagree on step: {steps}")
            optimizer_updates = steps[0]
    return model, metadata, optimizer_updates


def load_actor_normalizer(name: str, path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("observation_normalizer_state_dict")
    metadata = dict(checkpoint.get("observation_normalizer_metadata") or {})
    if state is None:
        raise ValueError(f"{path} has no observation_normalizer_state_dict.")
    mean = torch.as_tensor(state["running_mean"], dtype=torch.float32).reshape(1, -1)
    variance = torch.as_tensor(state["running_var"], dtype=torch.float32).reshape(1, -1)
    if mean.shape != (1, 46) or variance.shape != (1, 46):
        raise ValueError(
            f"Normalizer {path} has shapes {tuple(mean.shape)} and "
            f"{tuple(variance.shape)}, expected (1, 46)."
        )
    enabled = bool(metadata.get("enabled", True))
    epsilon = float(metadata.get("epsilon", 1e-8))
    manifest = dict(checkpoint.get("environment_manifest") or {})
    return {
        "name": str(name),
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "mean": mean,
        "std": torch.sqrt(torch.clamp(variance, min=0.0)),
        "enabled": enabled,
        "epsilon": epsilon,
        "count": int(metadata.get("count", torch.as_tensor(state.get("count", 0)).item())),
        "environment_manifest": manifest,
    }


def normalize_states(states: torch.Tensor, normalizer: dict) -> torch.Tensor:
    if not normalizer["enabled"]:
        return states.float()
    mean = normalizer["mean"].to(states.device)
    std = normalizer["std"].to(states.device)
    return (states.float() - mean) / (std + float(normalizer["epsilon"]))


def score_states_actions(
    model: SafetyQNetwork,
    states: np.ndarray,
    actions: np.ndarray,
    normalizer: dict,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if states.ndim != 2 or states.shape[1] != 46:
        raise ValueError(f"Expected states [N, 46], got {states.shape}.")
    if actions.ndim != 2 or actions.shape[1] != 12:
        raise ValueError(f"Expected actions [N, 12], got {actions.shape}.")
    values = []
    with torch.no_grad():
        for start in range(0, states.shape[0], batch_size):
            stop = min(start + batch_size, states.shape[0])
            state_batch = torch.as_tensor(
                states[start:stop], dtype=torch.float32, device=device
            )
            action_batch = torch.as_tensor(
                actions[start:stop], dtype=torch.float32, device=device
            )
            state_batch = normalize_states(state_batch, normalizer)
            values.append(model(state_batch, action_batch).reshape(-1).cpu().numpy())
    return np.concatenate(values).astype(np.float64, copy=False)


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size:
        return {"samples": 0}
    return {
        "samples": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def roc_auc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    """Tie-aware probability that a positive receives the higher score."""

    positive_scores = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    negative_scores = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    if not positive_scores.size or not negative_scores.size:
        return float("nan")
    values = np.concatenate([positive_scores, negative_scores])
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    nr_positive = positive_scores.size
    rank_sum = float(np.sum(ranks[:nr_positive]))
    return (
        rank_sum - nr_positive * (nr_positive + 1) / 2.0
    ) / (nr_positive * negative_scores.size)


def threshold_metrics(
    positive_scores: np.ndarray, negative_scores: np.ndarray, threshold: float
) -> dict:
    positive_scores = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    negative_scores = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    return {
        "threshold": float(threshold),
        "recall_imminent_fall": (
            float(np.mean(positive_scores >= threshold))
            if positive_scores.size
            else None
        ),
        "normal_false_rejection_rate": (
            float(np.mean(negative_scores >= threshold))
            if negative_scores.size
            else None
        ),
    }


def threshold_for_recall(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    target_recall: float = 0.8,
) -> dict:
    positive_scores = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    if not positive_scores.size:
        return {
            "threshold": None,
            "recall_imminent_fall": None,
            "normal_false_rejection_rate": None,
        }
    required = max(1, int(math.ceil(float(target_recall) * positive_scores.size)))
    threshold = float(np.sort(positive_scores)[-required])
    return threshold_metrics(positive_scores, negative_scores, threshold)


def horizon_report(
    scores_by_trajectory: list[np.ndarray],
    failures_by_trajectory: list[np.ndarray],
    fall_by_trajectory: list[bool],
    horizon: int,
    epsilon: float,
) -> dict:
    positive_parts = []
    non_imminent_parts = []
    safe_episode_parts = []
    for scores, failures, trajectory_fell in zip(
        scores_by_trajectory, failures_by_trajectory, fall_by_trajectory
    ):
        labels = future_failure_labels(failures, horizon).astype(bool)
        positive_parts.append(scores[labels])
        non_imminent_parts.append(scores[~labels])
        if not trajectory_fell:
            safe_episode_parts.append(scores)
    positives = np.concatenate(positive_parts) if positive_parts else np.empty(0)
    non_imminent = (
        np.concatenate(non_imminent_parts) if non_imminent_parts else np.empty(0)
    )
    safe_episode = (
        np.concatenate(safe_episode_parts) if safe_episode_parts else np.empty(0)
    )
    # Prefer complete no-fall episodes for the plain-language "normal" group.
    # Also expose all non-imminent transitions so the choice is auditable.
    normal = safe_episode if safe_episode.size else non_imminent
    auc = roc_auc(positives, normal)
    candidate = threshold_for_recall(positives, normal, target_recall=0.8)
    separable = bool(
        positives.size
        and normal.size
        and auc >= 0.80
        and candidate["recall_imminent_fall"] is not None
        and candidate["recall_imminent_fall"] >= 0.80
        and candidate["normal_false_rejection_rate"] is not None
        and candidate["normal_false_rejection_rate"] <= 0.20
    )
    return {
        "horizon_steps": int(horizon),
        "imminent_fall": _summary(positives),
        "normal_safe_episodes": _summary(safe_episode),
        "all_non_imminent": _summary(non_imminent),
        "normal_group_used": (
            "transitions_from_complete_no_fall_episodes"
            if safe_episode.size
            else "all_non_imminent_transitions"
        ),
        "roc_auc": None if not np.isfinite(auc) else float(auc),
        "current_epsilon": threshold_metrics(positives, normal, epsilon),
        "threshold_at_minimum_80pct_recall": candidate,
        "separable_gate": {
            "passed": separable,
            "requirements": "ROC-AUC >= 0.80, recall >= 0.80, normal false rejection <= 0.20",
        },
    }


def diagnose(
    dataset_path: Path,
    qsafe_path: Path,
    normalizers: list[dict],
    horizons: tuple[int, ...],
    split: str | None,
    epsilon: float | None,
    device: torch.device,
    batch_size: int,
) -> dict:
    dataset = SafetyTrajectoryDataset(dataset_path)
    validate_legacy_dataset_contract(dataset.manifest["contract"])
    entries = dataset.entries(split)
    if not entries:
        raise ValueError(f"Dataset has no trajectories for split={split!r}.")
    model, metadata, optimizer_updates = load_legacy_qsafe(qsafe_path, device)
    checkpoint_epsilon = float(metadata.get("epsilon", DEFAULT_EPSILON))
    epsilon = checkpoint_epsilon if epsilon is None else float(epsilon)

    trajectories = []
    for entry in entries:
        arrays = dataset.load(entry)
        trajectories.append(
            {
                "entry": entry,
                "states": np.asarray(arrays["states"], dtype=np.float32),
                "actions": np.asarray(arrays["actions"], dtype=np.float32),
                "failures": np.asarray(arrays["failures"], dtype=np.float32),
            }
        )

    results = {}
    for normalizer in normalizers:
        scores_by_trajectory = [
            score_states_actions(
                model,
                item["states"],
                item["actions"],
                normalizer,
                device,
                batch_size,
            )
            for item in trajectories
        ]
        failures = [item["failures"] for item in trajectories]
        falls = [bool(item["entry"]["fall"]) for item in trajectories]
        reports = [
            horizon_report(scores_by_trajectory, failures, falls, horizon, epsilon)
            for horizon in horizons
        ]
        passed = [item["separable_gate"]["passed"] for item in reports]
        if all(passed):
            decision = "threshold_only_candidate"
        elif any(passed):
            decision = "partial_separation_not_enough_for_all_horizons"
        else:
            decision = "retrain_qsafe"
        results[normalizer["name"]] = {
            "normalizer": {
                key: normalizer[key]
                for key in ("name", "path", "sha256", "enabled", "epsilon", "count")
            },
            "all_scores": _summary(np.concatenate(scores_by_trajectory)),
            "horizons": reports,
            "decision": decision,
        }

    return {
        "diagnostic_version": 1,
        "dataset": {
            "path": str(dataset_path.resolve()),
            "split": split or "all",
            "statistics": dataset.statistics(),
            "selected_trajectories": len(entries),
            "selected_transitions": int(sum(item["length"] for item in entries)),
            "contract": dataset.manifest["contract"],
        },
        "qsafe": {
            "path": str(qsafe_path.resolve()),
            "sha256": _sha256(qsafe_path),
            "metadata": metadata,
            "optimizer_updates": optimizer_updates,
            "checkpoint_epsilon": checkpoint_epsilon,
            "evaluated_epsilon": epsilon,
            "output_activation": "tanh",
        },
        "discount_context": {
            str(horizon): {
                "gamma_power": float(float(metadata["gamma"]) ** horizon),
                "physical_seconds": float(0.02 * horizon),
            }
            for horizon in horizons
        },
        "results": results,
        "interpretation": {
            "threshold_only_candidate": (
                "All requested horizons separate imminent-fall and normal states; "
                "validate the proposed threshold on held-out trajectories before use."
            ),
            "partial_separation_not_enough_for_all_horizons": (
                "The critic only separates some horizons. A threshold can change "
                "intervention frequency but cannot create missing ranking information."
            ),
            "retrain_qsafe": (
                "No requested horizon meets the ranking/false-rejection gate; do not "
                "continue SAC fine-tuning with this checkpoint."
            ),
        },
    }


def _normalizer_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/policy.model.")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Use NAME=/path/to/policy.model.")
    return name.strip(), Path(raw_path).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare legacy-QSafe scores before falls with scores in complete "
            "normal episodes under one or more actor normalizers."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--qsafe", type=Path, required=True)
    parser.add_argument(
        "--normalizer",
        action="append",
        type=_normalizer_argument,
        required=True,
        metavar="NAME=POLICY_MODEL",
    )
    parser.add_argument("--horizons", default="5,10,25")
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    horizons = tuple(int(value) for value in args.horizons.split(",") if value)
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError("--horizons must contain positive integers.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    normalizers = [
        load_actor_normalizer(name, path) for name, path in args.normalizer
    ]
    report = diagnose(
        args.dataset.expanduser().resolve(),
        args.qsafe.expanduser().resolve(),
        normalizers,
        horizons,
        args.split,
        args.epsilon,
        device,
        args.batch_size,
    )
    output = (
        args.output.expanduser().resolve()
        if args.output
        else args.dataset.expanduser().resolve() / "legacy_qsafe_diagnostic.json"
    )
    _write_json(output, report)
    print(f"Wrote {output}")
    for name, result in report["results"].items():
        print(f"\n[{name}] {result['decision']}")
        for item in result["horizons"]:
            current = item["current_epsilon"]
            candidate = item["threshold_at_minimum_80pct_recall"]
            print(
                f"  H={item['horizon_steps']:>2}: AUC={item['roc_auc']}, "
                f"eps={report['qsafe']['evaluated_epsilon']:.6g} "
                f"recall={current['recall_imminent_fall']} "
                f"FPR={current['normal_false_rejection_rate']}; "
                f"candidate={candidate['threshold']} "
                f"recall={candidate['recall_imminent_fall']} "
                f"FPR={candidate['normal_false_rejection_rate']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
