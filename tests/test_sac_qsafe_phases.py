import pytest

torch = pytest.importorskip("torch")

from rl_x.algorithms.sac_qsafe.pytorch.checkpoint import (
    load_checkpoint_bundle,
    save_checkpoint_bundle,
)
from rl_x.algorithms.sac_qsafe.pytorch.observation_normalizer import (
    ObservationNormalizer,
)
from rl_x.algorithms.sac_qsafe.pytorch.rollout import (
    PartitionedRolloutCounter,
    preserve_policy_outputs,
)


def test_normalizer_updates_then_freezes_exactly():
    normalizer = ObservationNormalizer(2)
    normalizer.normalize(torch.tensor([[1.0, 3.0], [3.0, 5.0]]), update=True)
    before = {key: value.clone() for key, value in normalizer.state_dict().items()}
    normalizer.freeze()
    normalizer.normalize(torch.tensor([[100.0, 100.0]]), update=True)
    for key, value in before.items():
        torch.testing.assert_close(normalizer.state_dict()[key], value)


def test_checkpoint_bundle_and_partition_counters(tmp_path):
    path = save_checkpoint_bundle(
        tmp_path / "step_0001",
        {"contract": "v1"},
        {"policy.pt": {"weight": torch.tensor([1.0])}},
    )
    manifest, artifacts = load_checkpoint_bundle(path)
    assert manifest == {"contract": "v1"}
    torch.testing.assert_close(artifacts["policy.pt"]["weight"], torch.tensor([1.0]))
    counter = PartitionedRolloutCounter(256, 64)
    assert counter.advance() == (256, 64)
    assert counter.advance() == (512, 128)


def test_preserved_policy_outputs_do_not_alias_reused_graph_storage():
    graph_storage = torch.tensor([[1.0, 2.0]])
    action, processed_action = preserve_policy_outputs(
        graph_storage, graph_storage + 1.0
    )

    graph_storage.fill_(99.0)

    torch.testing.assert_close(action, torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(processed_action, torch.tensor([[2.0, 3.0]]))
