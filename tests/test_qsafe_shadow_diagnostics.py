from pathlib import Path

import numpy as np

from rl_x.algorithms.qsafe.shadow_diagnostics import (
    ShadowQSafeRecorder,
    build_shadow_report,
)


def _synthetic_arrays():
    length = 60
    episode_id = np.repeat([0, 1], 30)
    failure = np.zeros(length, dtype=bool)
    failure[29] = True
    done = np.zeros(length, dtype=bool)
    done[[29, 59]] = True
    executed_q = np.full(length, 0.05, dtype=np.float32)
    executed_q[5:30] = 0.9
    candidate_q = np.repeat(executed_q[:, None], 4, axis=1)
    candidate_q[:, 1:] = 0.03
    return {
        "env_index": np.zeros(length, dtype=np.int32),
        "episode_id": episode_id,
        "failure": failure,
        "done": done,
        "executed_q": executed_q,
        "candidate_q": candidate_q,
        "candidate_best_action_l2": np.ones(length, dtype=np.float32),
        "observation_abs_z_p95": np.ones(length, dtype=np.float32),
        "observation_ood_fraction": np.zeros(length, dtype=np.float32),
    }


def test_shadow_report_separates_prefall_and_complete_safe_episodes():
    report = build_shadow_report(_synthetic_arrays(), epsilon=0.1)

    assert report["fall_episodes"] == 1
    assert report["complete_safe_episodes"] == 1
    assert report["horizons"]["25"]["recall_future_failure"] == 1.0
    assert report["horizons"]["25"]["normal_group"] == "complete_no_fall_episodes"
    assert report["horizons"]["25"]["roc_auc"] == 1.0
    assert report["candidate_pool"]["fallback_fraction"] == 0.0


def test_shadow_recorder_writes_recoverable_npz_and_report(tmp_path: Path):
    path = tmp_path / "shadow.npz"
    recorder = ShadowQSafeRecorder(path, nr_envs=1, epsilon=0.1)
    for step in range(3):
        recorder.add(
            global_step=step,
            states=np.zeros((1, 2), dtype=np.float32),
            next_states=np.ones((1, 2), dtype=np.float32),
            applied_actions=np.zeros((1, 1), dtype=np.float32),
            failure=np.asarray([step == 2]),
            terminated=np.asarray([step == 2]),
            truncated=np.asarray([False]),
            candidate_actions=np.zeros((1, 2, 1), dtype=np.float32),
            candidate_q=np.asarray([[0.2, 0.05]], dtype=np.float32),
            candidate_best_action_l2=np.asarray([0.4], dtype=np.float32),
            observation_abs_z_p95=np.asarray([1.2], dtype=np.float32),
            observation_ood_fraction=np.asarray([0.0], dtype=np.float32),
        )

    report = recorder.flush()

    assert path.is_file()
    assert recorder.report_path.is_file()
    assert report["transitions"] == 3
    with np.load(path) as arrays:
        assert arrays["candidate_q"].shape == (3, 2)
        assert arrays["candidate_actions"].shape == (3, 2, 1)
        assert arrays["next_state"].shape == (3, 2)
        assert arrays["failure"].tolist() == [False, False, True]


def test_shadow_report_supports_legacy_tanh_scores():
    arrays = _synthetic_arrays()
    arrays["executed_q"] = arrays["executed_q"] - 0.2
    arrays["candidate_q"] = arrays["candidate_q"] - 0.2

    report = build_shadow_report(
        arrays,
        epsilon=0.1,
        score_semantics="tanh_value",
    )

    assert report["score_semantics"] == "tanh_value"
    assert report["horizons"]["25"]["ece"] is None
    assert report["horizons"]["25"]["roc_auc"] == 1.0
