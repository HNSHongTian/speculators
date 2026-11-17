"""Regression checks for loss/gradient equivalence of TRPO compute reuse."""

from types import SimpleNamespace

import pytest
import torch

from speculators.models.dflash.core import DFlashDraftModel
from speculators.train.trpo import step


@pytest.mark.parametrize("lambda_ce", [0.0, 0.1, 1.0])
def test_reused_gold_matches_independent_replay(monkeypatch, lambda_ce):
    torch.manual_seed(42)
    logits = torch.randn(1, 6, 8, requires_grad=True)
    hidden = torch.randn(1, 6, 4)
    weight = torch.randn(8, 8, requires_grad=True)
    markov = SimpleNamespace(
        head_type="vanilla",
        block_bias=lambda *, prev_token_ids, hidden_states: weight[prev_token_ids],
    )

    def backbone(*args, **kwargs):
        assert kwargs["compute_verifier_targets"] is False
        return hidden, logits, None, torch.ones(1, 6), torch.arange(6)

    draft = SimpleNamespace(
        block_size=3, config=SimpleNamespace(sample_from_anchor=True),
        markov_head=markov, use_draft_vocab=False, _backbone_forward=backbone,
    )
    kwargs = {
        "hidden_states": hidden, "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7]]),
        "loss_mask": torch.ones(1, 7), "verifier_last_hidden_states": hidden,
        "document_ids": torch.zeros(1, 7, dtype=torch.long),
        "max_anchors": 2, "max_trpo_anchors": 2, "tree_top_k": 2,
        "tree_max_nodes": 3, "lambda_ce": lambda_ce,
    }
    replay = step.replay_node_logprobs
    loss_fn = step.trpo_loss_from_groups
    calls = []

    def record_replay(*args, **kw):
        calls.append((args, kw))
        return replay(*args, **kw)

    monkeypatch.setattr(step, "replay_node_logprobs", record_replay)
    optimized, metrics = step.compute_trpo_batch_loss(draft, **kwargs)
    optimized_grads = torch.autograd.grad(optimized, (logits, weight))
    assert len(calls) == 2  # exactly one replay per anchor

    def independent_gold_loss(**kw):
        args, replay_kw = calls[-1]
        # Reproduce the old separate gold replay using a fresh autograd graph.
        gold = [1, 2, 3, 4, 5, 6, 7][
            int(replay_kw["anchor_token_id"]):
            int(replay_kw["anchor_token_id"]) + 3
        ]
        gold_indices = []
        for i in replay_kw["active_indices"]:
            path = []
            j = i
            while j >= 0:
                path.append(replay_kw["tree_tokens"][j])
                j = replay_kw["tree_parents"][j]
            if list(reversed(path)) == gold[:len(path)]:
                gold_indices.append(i)
        kw["gold_log_probs"] = replay(
            *args, **{**replay_kw, "active_indices": gold_indices}
        )
        return loss_fn(**kw)

    monkeypatch.setattr(step, "trpo_loss_from_groups", independent_gold_loss)
    baseline, baseline_metrics = step.compute_trpo_batch_loss(draft, **kwargs)
    baseline_grads = torch.autograd.grad(baseline, (logits, weight))
    torch.testing.assert_close(optimized, baseline)
    for actual, expected in zip(optimized_grads, baseline_grads, strict=True):
        torch.testing.assert_close(actual, expected)
    assert metrics == baseline_metrics


def test_backbone_can_skip_targets_without_changing_outputs_or_gradients():
    torch.manual_seed(7)
    head = torch.nn.Linear(4, 8)
    verifier_calls = []

    def verifier(x):
        verifier_calls.append(True)
        return head(x)

    model = SimpleNamespace(
        block_size=2, mask_token_id=0,
        config=SimpleNamespace(sample_from_anchor=True),
        _build_attention_mask=lambda *args: (
            None, None, torch.tensor([0]), torch.tensor([True]),
        ),
        embed_tokens=torch.nn.Embedding(8, 4),
        fc=torch.nn.Identity(), hidden_norm=torch.nn.Identity(),
        rotary_emb=lambda *args: None, verifier_lm_head=verifier,
        verifier_norm=torch.nn.Identity(), layers=[], norm=torch.nn.Identity(),
        lm_head=head,
    )
    args = (torch.randn(1, 4, 4), torch.tensor([[1, 2, 3, 4]]),
            torch.ones(1, 4), torch.randn(1, 4, 4), torch.zeros(1, 4))
    original = DFlashDraftModel._backbone_forward(model, *args, max_anchors=1)
    grad_before = torch.autograd.grad(original[1].sum(), head.weight)[0]
    skipped = DFlashDraftModel._backbone_forward(
        model, *args, max_anchors=1, compute_verifier_targets=False,
    )
    grad_after = torch.autograd.grad(skipped[1].sum(), head.weight)[0]
    assert len(verifier_calls) == 1
    assert original[2] is not None
    assert skipped[2] is None
    for i in (0, 1, 3, 4):
        torch.testing.assert_close(original[i], skipped[i], rtol=0, atol=0)
    torch.testing.assert_close(grad_before, grad_after, rtol=0, atol=0)
