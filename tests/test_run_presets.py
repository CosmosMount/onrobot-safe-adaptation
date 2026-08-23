from pathlib import Path

import pytest

from src.config import RUN_PRESETS
from src.run import _artifact_flags


def test_framework_is_selected_by_training_backend():
    assert "--algorithm.name=sac_qsafe.pytorch" in RUN_PRESETS["pretrain"]
    for command in ("zero-shot", "finetune", "eval"):
        assert "--algorithm.name=sac_qsafe.flax" in RUN_PRESETS[command]


def test_transfer_commands_use_torch_pretrain_sidecars(tmp_path: Path):
    (tmp_path / "policy.model").touch()
    (tmp_path / "qsafe.model").touch()
    flags = _artifact_flags("finetune", str(tmp_path))
    assert flags == [
        f"--algorithm.pretrained_policy_path={tmp_path / 'policy.model'}",
        f"--algorithm.qsafe.checkpoint_path={tmp_path / 'qsafe.model'}",
    ]


def test_eval_rejects_a_missing_flax_checkpoint(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        _artifact_flags("eval", str(tmp_path))
