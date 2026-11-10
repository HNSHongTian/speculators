"""Step 2: actual-token alignment, complete blocks, and Markov gradients."""

import pytest
import torch

from speculators.models.dspark.bv_loss import build_bv_labels
from speculators.models.dspark.metrics import compute_metrics
from speculators.models.dspark.model_definitions import MarkovHead
from speculators.models.metrics import resolve_loss_config


@pytest.mark.parametrize("sample", [True, False])
def test_label_shift_and_padded_anchor(sample):
    ids = torch.arange(12)[None]
    indices = torch.tensor([2, 3, 4, 0, 1, 2])
    aligned = torch.tensor([[1, 1, 1, 0, 0, 0]])
    labels, valid = build_bv_labels(
        ids, torch.ones_like(ids), torch.zeros_like(ids), indices, aligned, 3, sample
    )
    assert labels[0].tolist() == ([3, 4, 5] if sample else [3, 4])
    assert valid.tolist() == [True, False]


@pytest.mark.parametrize("reason", ["document", "padding", "mask_gap", "tail", "eos"])
def test_invalid_prefixes_are_rejected_entirely(reason):
    ids = torch.arange(8)[None]
    mask, docs = torch.ones_like(ids), torch.zeros_like(ids)
    indices = torch.tensor([2, 3, 4])
    eos = ()
    if reason == "document":
        docs[:, 5:] = 1
    elif reason == "padding":
        docs[:, 5:] = -1
    elif reason == "mask_gap":
        mask[:, 4] = 0
    elif reason == "tail":
        indices = torch.tensor([5, 6, 7])
    elif reason == "eos":
        eos = (4,)
    _, valid = build_bv_labels(ids, mask, docs, indices, torch.ones(1, 3), 3, True, eos)
    assert not valid.any()


def test_eos_at_last_label_allowed_and_target_mask_is_shifted():
    ids = torch.arange(8)[None]
    mask = torch.zeros_like(ids)
    mask[:, 3:6] = 1
    labels, valid = build_bv_labels(
        ids,
        mask,
        torch.zeros_like(ids),
        torch.tensor([2, 3, 4]),
        torch.tensor([[1, 1, 1]]),
        3,
        True,
        (5,),
    )
    assert valid.all()
    assert labels.tolist() == [[3, 4, 5]]
    mask[:, 5] = 0
    _, valid = build_bv_labels(
        ids, mask, torch.zeros_like(ids), torch.tensor([2, 3, 4]), torch.ones(1, 3), 3
    )
    assert not valid.any()


@pytest.mark.parametrize("head_type", ["vanilla", "gated", "rnn"])
def test_bv_reaches_markov_and_backbone_and_ignores_ce_tv_decay(head_type):
    torch.manual_seed(4)
    head = MarkovHead(
        verifier_vocab_size=7,
        draft_vocab_size=7,
        markov_rank=4,
        hidden_size=8,
        head_type=head_type,
    )
    hidden = torch.randn(2, 3, 8, requires_grad=True)
    projection = torch.randn(8, 7)
    prev = torch.randint(7, (2, 3))
    logits = (
        hidden @ projection + head.block_bias(prev_token_ids=prev, hidden_states=hidden)
    ).reshape(1, 6, 7)
    teacher = torch.randn_like(logits, requires_grad=True)
    labels = torch.randint(7, (2, 3))
    kwargs = {
        "enable_bv_loss": True,
        "bv_labels": labels,
        "bv_valid_blocks": torch.tensor([True, False]),
        "bv_beta": torch.tensor(0.4),
        "bv_annealing": True,
    }
    losses = []
    for config, gamma in [("ce", 0.1), ('{"ce":0.1,"tv":0.9}', 100)]:
        loss, metrics = compute_metrics(
            logits,
            teacher,
            None,
            torch.ones(1, 6),
            3,
            resolve_loss_config(config),
            gamma=gamma,
            **kwargs,
        )
        losses.append(loss)
        assert metrics["bv_loss_total"] == 1
        assert "ce_loss_sum" not in metrics
    torch.testing.assert_close(*losses)
    losses[0].backward()
    assert hidden.grad.abs().sum() > 0
    assert head.markov_w1.weight.grad.abs().sum() > 0
    assert head.markov_w2.weight.grad.abs().sum() > 0
    assert teacher.grad is None
