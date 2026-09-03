import json
from pathlib import Path

import numpy as np

from rl_x.algorithms.qsafe.dataset import SafetyTrajectoryDataset
from tools.build_target_shadow_qsafe_dataset import convert


def test_convert_shadow_uses_complete_stratified_diagnostic_episodes(
    tmp_path: Path,
):
    states = []
    next_states = []
    actions = []
    candidates = []
    failures = []
    done = []
    episode_ids = []
    for episode in range(6):
        fell = episode % 2 == 0
        for step in range(2):
            states.append([episode, step])
            next_states.append([episode, step + 1])
            actions.append([0.1])
            candidates.append([[0.1], [0.2]])
            failures.append(fell and step == 1)
            done.append(step == 1)
            episode_ids.append(episode)
    shadow = tmp_path / "shadow.npz"
    np.savez_compressed(
        shadow,
        env_index=np.zeros(12, dtype=np.int32),
        episode_id=np.asarray(episode_ids),
        state=np.asarray(states, dtype=np.float32),
        next_state=np.asarray(next_states, dtype=np.float32),
        applied_action=np.asarray(actions, dtype=np.float32),
        candidate_actions=np.asarray(candidates, dtype=np.float32),
        failure=np.asarray(failures),
        done=np.asarray(done),
    )
    contract_manifest = tmp_path / "source_manifest.json"
    contract_manifest.write_text(
        json.dumps({"contract": {"test_contract": True}}), encoding="utf-8"
    )

    counts = convert(
        shadow,
        tmp_path / "dataset",
        contract_manifest,
        "one_target_actor",
    )

    assert counts == {"train": 2, "validation": 2, "test": 2}
    statistics = SafetyTrajectoryDataset(tmp_path / "dataset").statistics()
    for split in ("train", "validation", "test"):
        assert statistics["by_split"][split]["fall"] == 1
        assert statistics["by_split"][split]["success"] == 1
