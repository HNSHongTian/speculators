"""Unit coverage for inference-aligned TreeSpark training replay."""

import pytest
import torch

from speculators.models.dspark.model_definitions import MarkovHead
from speculators.models.dspark.tree_loss import (
    build_tree_loss_inputs,
    compute_tree_training_loss,
)


def _inputs(*, vocab=11, blocks=4, block_size=4, hidden_size=6):
    torch.manual_seed(7)
    base_logits = torch.randn(
        blocks, block_size, vocab, requires_grad=True, dtype=torch.float32
    )
    hidden = torch.randn(
        blocks, block_size, hidden_size, requires_grad=True, dtype=torch.float32
    )
    target = torch.randn(blocks, block_size, vocab)
    roots = torch.arange(1, blocks + 1)
    valid = torch.ones(blocks, block_size, dtype=torch.bool)
    return base_logits, hidden, target, roots, valid


def test_tree_replay_packs_budget_and_backpropagates_through_markov_head():
    base, hidden, target, roots, valid = _inputs()
    head = MarkovHead(
        verifier_vocab_size=11,
        draft_vocab_size=11,
        markov_rank=3,
        hidden_size=6,
        head_type="vanilla",
    )
    replay = build_tree_loss_inputs(
        markov_head=head,
        base_logits=base,
        hidden_states=hidden,
        target_logits=target,
        anchor_token_ids=roots,
        depth_valid_mask=valid,
        block_keep_mask=valid.any(-1),
        first_draft_slot=0,
        top_k=2,
        top_j=2,
        margin=1.0,
        hinge_cap=4.0,
        num_anchors=3,
        max_nodes=5,
        use_draft_vocab=False,
        d2t=None,
        grad_checkpoint=False,
    )
    assert replay is not None
    assert len(replay.node_logprobs) == 4
    assert replay.node_logprobs[0].shape == (3, 2)
    assert replay.node_logprobs[1].shape == (3, 4)
    packed_count = sum(mask.sum(-1) for mask in replay.node_packed_masks)
    assert torch.equal(packed_count, torch.full((3,), 5))

    loss, metrics = compute_tree_training_loss(
        replay,
        loss_decay_gamma=4.0,
        survival_alpha=0.1,
        coverage_alpha=0.1,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["tree_survival_loss_total"] > 0
    assert metrics["tree_coverage_loss_total"] > 0
    assert base.grad is not None
    assert base.grad.abs().sum() > 0
    assert head.markov_w1.weight.grad is not None
    assert head.markov_w1.weight.grad.abs().sum() > 0
    assert head.markov_w2.weight.grad is not None
    assert head.markov_w2.weight.grad.abs().sum() > 0


def test_rnn_replay_supports_reserved_slot_and_reduced_vocab():
    base, hidden, target, roots, valid = _inputs(vocab=7, block_size=4)
    head = MarkovHead(
        verifier_vocab_size=20,
        draft_vocab_size=7,
        markov_rank=3,
        hidden_size=6,
        head_type="rnn",
    )
    d2t = torch.tensor([0, 1, 3, 5, 7, 9, 11])
    replay = build_tree_loss_inputs(
        markov_head=head,
        base_logits=base,
        hidden_states=hidden,
        target_logits=target,
        anchor_token_ids=roots,
        depth_valid_mask=valid,
        block_keep_mask=valid.any(-1),
        first_draft_slot=1,
        top_k=2,
        top_j=3,
        margin=1.0,
        hinge_cap=4.0,
        num_anchors=2,
        max_nodes=4,
        use_draft_vocab=True,
        d2t=d2t,
        grad_checkpoint=False,
    )
    assert replay is not None
    assert len(replay.node_logprobs) == 3
    loss, _ = compute_tree_training_loss(
        replay,
        loss_decay_gamma=4.0,
        survival_alpha=1.0,
        coverage_alpha=1.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert hidden.grad is not None
    assert hidden.grad.abs().sum() > 0
    assert head.joint_proj.weight.grad is not None
    assert head.joint_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("head_type", ["vanilla", "gated", "rnn"])
def test_depth0_main_top1_cannot_be_pruned(head_type):
    base, hidden, target, roots, valid = _inputs()
    head = MarkovHead(
        verifier_vocab_size=11,
        draft_vocab_size=11,
        markov_rank=3,
        hidden_size=6,
        head_type=head_type,
    )
    with torch.no_grad():
        root_bias = head.block_bias(
            prev_token_ids=roots.unsqueeze(-1),
            hidden_states=hidden[:, :1],
        )[:, 0]
        reference_root_logits = base[:, 0] + root_bias
        target[:, 0] = reference_root_logits
        reference_top1 = reference_root_logits.argmax(dim=-1)

    replay = build_tree_loss_inputs(
        markov_head=head,
        base_logits=base,
        hidden_states=hidden,
        target_logits=target,
        anchor_token_ids=roots,
        depth_valid_mask=valid,
        block_keep_mask=valid.any(-1),
        first_draft_slot=0,
        top_k=2,
        top_j=2,
        margin=1.0,
        hinge_cap=4.0,
        num_anchors=4,
        max_nodes=5,
        use_draft_vocab=False,
        d2t=None,
        reference_top1_ids=reference_top1,
        grad_checkpoint=False,
    )
    assert replay is not None
    assert replay.argmax_survived[0].all()

    _, metrics = compute_tree_training_loss(
        replay,
        loss_decay_gamma=4.0,
        survival_alpha=0.1,
        coverage_alpha=0.1,
    )
    assert metrics["tree_depth0_main_top1_acc_sum"].item() == 4
    assert metrics["tree_depth0_replay_top1_acc_sum"].item() == 4
    assert metrics["tree_depth0_local_topk_hit_sum"].item() == 4
    assert metrics["tree_depth0_main_replay_agreement_sum"].item() == 4
    assert metrics["tree_depth0_top1_pack_violation_sum"].item() == 0
    assert metrics["tree_depth0_main_top1_survival_violation_sum"].item() == 0


def test_all_invalid_local_anchors_still_produce_zero_collective_safe_loss():
    base, hidden, target, roots, valid = _inputs(blocks=2, block_size=3)
    valid.zero_()
    replay = build_tree_loss_inputs(
        markov_head=None,
        base_logits=base,
        hidden_states=hidden,
        target_logits=target,
        anchor_token_ids=roots,
        depth_valid_mask=valid,
        block_keep_mask=valid.any(-1),
        first_draft_slot=0,
        top_k=2,
        top_j=2,
        margin=1.0,
        hinge_cap=4.0,
        num_anchors=2,
        max_nodes=3,
        use_draft_vocab=False,
        d2t=None,
        grad_checkpoint=False,
    )
    assert replay is not None
    loss, metrics = compute_tree_training_loss(
        replay,
        loss_decay_gamma=4.0,
        survival_alpha=0.1,
        coverage_alpha=0.1,
    )
    loss.backward()
    assert loss.item() == pytest.approx(0.0)
    assert metrics["tree_survival_loss_total"].item() == 0
    assert metrics["tree_coverage_loss_total"].item() == 0


@pytest.mark.parametrize(
    ("top_k", "top_j", "max_nodes", "message"),
    [
        (1, 2, 4, "top_k"),
        (2, 0, 4, "top_j"),
        (2, 2, 0, "max_nodes"),
    ],
)
def test_invalid_tree_policy_is_rejected(top_k, top_j, max_nodes, message):
    base, hidden, target, roots, valid = _inputs(blocks=2)
    with pytest.raises(ValueError, match=message):
        build_tree_loss_inputs(
            markov_head=None,
            base_logits=base,
            hidden_states=hidden,
            target_logits=target,
            anchor_token_ids=roots,
            depth_valid_mask=valid,
            block_keep_mask=valid.any(-1),
            first_draft_slot=0,
            top_k=top_k,
            top_j=top_j,
            margin=1.0,
            hinge_cap=4.0,
            num_anchors=2,
            max_nodes=max_nodes,
            use_draft_vocab=False,
            d2t=None,
            grad_checkpoint=False,
        )
