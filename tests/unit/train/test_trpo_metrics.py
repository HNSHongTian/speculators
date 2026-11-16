"""Pre-gold prefix counts, weighting, boundaries, and unchanged training loss."""

from types import SimpleNamespace

import pytest
import torch

from speculators.train.trpo import step
from speculators.train.trpo.metrics import (
    add_tree_metrics,
    empty_tree_metrics,
    valid_response_suffix,
)
from speculators.train.utils import normalize_counted_metrics
from speculators.tree.types import PackedTree


def tree():
    return PackedTree(
        tokens=torch.tensor([2, 9, 3]),
        parent_indices=torch.tensor([0, 0, 1]),
        depths=torch.tensor([0, 0, 1]),
        path_scores=torch.tensor([-0.1, -0.4, -0.5]),
        pool_indices=torch.arange(3),
    )


def test_tree_prefix_not_any_node_hit():
    counts = empty_tree_metrics(3)
    add_tree_metrics(counts, tree(), [2, 3, 4])
    add_tree_metrics(counts, tree(), [8, 3, 4])
    metrics = normalize_counted_metrics(counts.copy())
    assert metrics["tree_accept_len"] == 2
    assert metrics["tree_accepted_draft_len"] == 1
    assert metrics["tree_accept_rate"] == pytest.approx(1 / 3)
    assert metrics["tree_position_0_accept_rate"] == 0.5
    assert metrics["tree_position_1_accept_rate"] == 1
    assert metrics["tree_position_1_survival_rate"] == 0.5
    assert metrics["tree_position_2_accept_rate"] == 0
    assert metrics["tree_full_accept_rate"] == 0


def test_weighted_reduction_and_empty_rank():
    ranks = [empty_tree_metrics(3) for _ in range(3)]
    add_tree_metrics(ranks[0], tree(), [2, 3])
    add_tree_metrics(ranks[1], tree(), [8])
    add_tree_metrics(ranks[1], tree(), [2])
    combined = {key: sum(rank[key] for rank in ranks) for key in ranks[0]}
    result = normalize_counted_metrics(combined, world_size=3)
    assert result["tree_accept_len"] == 2
    assert result["tree_accept_rate"] == 0.75
    assert result["tree_full_accept_rate"] == pytest.approx(2 / 3)
    assert result["tree_position_2_accept_rate"] == 0


@pytest.mark.parametrize(
    ("mask", "docs", "anchor", "expected"),
    [
        ([1, 1, 1, 1], [0, 0, 1, 1], 0, [2]),
        ([1, 1, 0, 1], [0, 0, 0, 0], 0, [2]),
        ([1, 0, 1, 1], [0, 0, 0, 0], 0, []),
        ([1, 1, 1, 1], [0, 0, 0, 0], 2, [4]),
    ],
)
def test_suffix_boundaries(mask, docs, anchor, expected):
    assert (
        valid_response_suffix(
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([mask]),
            torch.tensor([docs]),
            anchor=anchor,
            depth=3,
        )
        == expected
    )


def test_diagnostics_do_not_change_loss_or_gradients(monkeypatch):
    torch.manual_seed(9)
    logits = torch.randn(1, 3, 8, requires_grad=True)
    hidden = torch.zeros(1, 3, 4)
    draft = SimpleNamespace(
        block_size=3,
        config=SimpleNamespace(sample_from_anchor=True),
        markov_head=None,
        use_draft_vocab=False,
        _backbone_forward=lambda *a, **kw: (
            hidden,
            logits,
            torch.zeros_like(logits),
            torch.ones(1, 3),
            torch.arange(3),
        ),
    )
    args = {
        "hidden_states": hidden,
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "loss_mask": torch.ones(1, 4),
        "verifier_last_hidden_states": hidden,
        "document_ids": torch.zeros(1, 4, dtype=torch.long),
        "max_anchors": 1,
        "tree_top_k": 2,
        "tree_max_nodes": 3,
    }
    loss, metrics = step.compute_trpo_batch_loss(draft, **args)
    grad = torch.autograd.grad(loss, logits)[0]
    assert metrics["tree_accept_len_total"] == 1
    monkeypatch.setattr(step, "add_tree_metrics", lambda *a: None)
    baseline, _ = step.compute_trpo_batch_loss(draft, **args)
    baseline_grad = torch.autograd.grad(baseline, logits)[0]
    torch.testing.assert_close(loss, baseline, rtol=0, atol=0)
    torch.testing.assert_close(grad, baseline_grad, rtol=0, atol=0)


def test_empty_and_nonempty_batches_have_identical_metric_keys():
    hidden = torch.zeros(1, 3, 4)
    logits = torch.zeros(1, 3, 8, requires_grad=True)
    mask = torch.ones(1, 3)
    draft = SimpleNamespace(
        block_size=3,
        config=SimpleNamespace(sample_from_anchor=True),
        markov_head=None,
        use_draft_vocab=False,
        _backbone_forward=lambda *a, **kw: (
            hidden,
            logits,
            torch.zeros_like(logits),
            mask,
            torch.arange(3),
        ),
    )
    kwargs = {
        "hidden_states": hidden,
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "loss_mask": torch.ones(1, 4),
        "verifier_last_hidden_states": hidden,
        "document_ids": torch.zeros(1, 4, dtype=torch.long),
    }
    _, populated = step.compute_trpo_batch_loss(draft, **kwargs)
    mask.zero_()
    _, empty = step.compute_trpo_batch_loss(draft, **kwargs)
    assert list(empty) == list(populated)
