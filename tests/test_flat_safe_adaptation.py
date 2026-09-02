from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.run_flat_safe_adaptation import (
    DEFAULT_ACTOR,
    DEFAULT_QSAFE,
    _task_flags,
    _train_command,
    _engineering_near_pass,
    _validate_flat_qsafe_metadata,
    _validate_legacy_flat_actor_manifest,
    _validate_legacy_flat_qsafe_metadata,
)


def _args():
    return SimpleNamespace(
        actor=Path("/actor/models"),
        qsafe=Path("/qsafe/qsafe.model"),
        interface="lo",
        steps=200_000,
        checkpoint_frequency=100_000,
        logging_frequency=1_000,
        qsafe_version=1,
        qsafe_gamma=0.7,
        qsafe_epsilon=0.1,
    )


def test_defaults_reuse_the_verified_v9_flat_artifacts():
    assert DEFAULT_ACTOR.name == "models"
    assert DEFAULT_ACTOR.parent.name == "isaac_sac_height_dr_v1"
    assert DEFAULT_QSAFE.name == "qsafe.model"
    assert DEFAULT_QSAFE.parent.parent.name == "isaac_sqrl_height_dr_v1"


def test_paired_train_commands_only_toggle_qsafe():
    args = _args()
    baseline = _train_command(args, 2, 33, "no_qsafe", "paired")
    protected = _train_command(args, 2, 33, "qsafe", "paired")

    baseline_treatment = "--algorithm.qsafe.enabled=false"
    protected_treatment = "--algorithm.qsafe.enabled=true"
    assert baseline_treatment in baseline
    assert protected_treatment in protected
    assert [value for value in baseline if value != baseline_treatment] == [
        value for value in protected if value != protected_treatment
    ]
    assert "--algorithm.learning_starts=1000" in baseline
    assert "--algorithm.finetune_actor_warmup_steps=10000" in baseline
    assert "--algorithm.finetune_actor_update_interval=10" in baseline
    assert "--algorithm.task_utd_ratio=1.0" in baseline
    assert "--algorithm.alpha_init=0.0002" in baseline
    assert "--algorithm.eval_policy=task" in baseline
    assert "--environment.action_profile=legacy_v1" in _task_flags()
    assert "--environment.foot_clearance_reward_scale=0.0" in _task_flags()
    assert "--environment.foot_clearance_target=0.07" in _task_flags()


def test_verified_v9_actor_contract_is_explicit():
    manifest = {
        "manifest_version": 9,
        "observation": {"version": "go2-observation-v3-body-velocity"},
        "action": {
            "version": "go2-action-v1",
            "pipeline_version": "sdk-absolute-position-v2",
            "scale": 0.25,
        },
    }
    _validate_legacy_flat_actor_manifest(manifest)
    manifest["action"]["version"] = "go2-action-v2-per-joint-scale"
    with pytest.raises(ValueError, match="manifest-v9/action-v1"):
        _validate_legacy_flat_actor_manifest(manifest)


def test_verified_legacy_qsafe_contract_is_explicit():
    metadata = {
        "observation_shape": [46],
        "action_shape": [12],
        "gamma": 0.7,
        "epsilon": 0.1,
    }
    _validate_legacy_flat_qsafe_metadata(metadata)
    metadata["qsafe_version"] = 2
    with pytest.raises(ValueError, match="calibrated flat QSafe v2"):
        _validate_legacy_flat_qsafe_metadata(metadata)


def test_calibrated_flat_qsafe_v2_contract_is_accepted_only_after_gate():
    metadata = {
        "qsafe_version": 2,
        "observation_shape": [230],
        "base_observation_shape": [46],
        "action_shape": [12],
        "history_length": 5,
        "control_dt": 0.02,
        "gamma": 0.9,
        "epsilon": 0.031,
        "environment_contract": {
            "observation": {"version": "go2-observation-v3-body-velocity"},
            "action": {
                "version": "go2-action-v1",
                "pipeline_version": "sdk-absolute-position-v2",
                "scale": 0.25,
            },
            "failure": {
                "version": "tilt-or-low-terrain-clearance-sustained-v3"
            },
        },
    }
    report = {
        "horizons": [5, 10, 25],
        "universal_qsafe_v2_pass": True,
        "selected": {
            "gamma_safe": 0.9,
            "epsilon": 0.031,
            "validation_pass": True,
            "test_pass": True,
        },
    }
    _validate_flat_qsafe_metadata(metadata, report)
    report["selected"]["test_pass"] = False
    with pytest.raises(ValueError, match="explicit engineering exception"):
        _validate_flat_qsafe_metadata(metadata, report)


def test_engineering_override_is_narrow_and_explicit():
    metadata = {
        "qsafe_version": 2,
        "observation_shape": [230],
        "base_observation_shape": [46],
        "action_shape": [12],
        "history_length": 5,
        "control_dt": 0.02,
        "gamma": 0.97,
        "epsilon": 0.017,
        "environment_contract": {
            "observation": {"version": "go2-observation-v3-body-velocity"},
            "action": {
                "version": "go2-action-v1",
                "pipeline_version": "sdk-absolute-position-v2",
                "scale": 0.25,
            },
            "failure": {
                "version": "tilt-or-low-terrain-clearance-sustained-v3"
            },
        },
    }
    report = {
        "artifact_status": "diagnostic_candidate",
        "data_gate_pass": True,
        "horizons": [5, 10, 25],
        "universal_qsafe_v2_pass": False,
        "selected": {
            "gamma_safe": 0.97,
            "epsilon": 0.017,
            "validation_pass": True,
            "test_pass": False,
            "test_by_horizon": {
                5: {"recall_future_failure": 0.91},
                10: {"recall_future_failure": 0.85},
                25: {
                    "recall_future_failure": 0.797,
                    "safe_action_false_rejection_rate": 0.034,
                    "fallback_rate": 0.036,
                },
            },
        },
    }
    assert _engineering_near_pass(report) is True
    with pytest.raises(ValueError, match="explicit engineering exception"):
        _validate_flat_qsafe_metadata(metadata, report)
    _validate_flat_qsafe_metadata(
        metadata, report, allow_diagnostic_near_pass=True
    )
    report["selected"]["test_by_horizon"][25][
        "recall_future_failure"
    ] = 0.794
    assert _engineering_near_pass(report) is False
    with pytest.raises(ValueError, match="explicit engineering exception"):
        _validate_flat_qsafe_metadata(
            metadata, report, allow_diagnostic_near_pass=True
        )
