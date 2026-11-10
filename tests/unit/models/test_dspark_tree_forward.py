"""Tiny-backbone integration tests for TreeSpark training."""

import pytest
import torch
from transformers import Qwen3Config

from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.core import (
    DSparkDraftModel,
    _build_tree_anchor_selection_mask,
    _build_tree_depth_valid_mask,
)
from speculators.models.metrics import resolve_loss_config


def _tiny_model_and_batch():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        eos_token_id=31,
    )
    config._attn_implementation = "sdpa"
    model = DSparkDraftModel(
        DSparkSpeculatorConfig(
            transformer_layer_config=config,
            draft_vocab_size=32,
            block_size=3,
            aux_hidden_state_layer_ids=[0],
            mask_token_id=0,
            markov_rank=4,
            markov_head_type="vanilla",
            enable_confidence_head=False,
            sample_from_anchor=True,
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            if not torch.isfinite(parameter).all():
                parameter.normal_(std=0.1)
    batch = {
        "hidden_states": torch.randn(1, 16, 16),
        "input_ids": torch.randint(1, 30, (1, 16)),
        "loss_mask": torch.ones(1, 16),
        "verifier_last_hidden_states": torch.randn(1, 16, 16),
        "document_ids": torch.zeros(1, 16, dtype=torch.long),
    }
    return model, batch


def test_tree_training_forward_adds_metrics_and_gradients():
    model, batch = _tiny_model_and_batch()
    model.train()
    _, loss, metrics = model(
        **batch,
        max_anchors=3,
        loss_config=resolve_loss_config('{"ce": 0.1, "tv": 0.9}'),
        enable_tree_training=True,
        tree_loss_top_k=2,
        tree_loss_num_anchors=2,
        tree_loss_max_nodes=4,
        tree_rank_top_j=2,
        tree_survival_alpha=0.1,
        tree_rank_alpha=0.1,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["tree_survival_loss_total"] > 0
    assert metrics["tree_coverage_loss_total"] > 0
    assert metrics["tree_depth0_main_replay_agreement_sum"] == metrics[
        "tree_depth0_main_replay_agreement_total"
    ]
    assert metrics["tree_depth0_top1_pack_violation_sum"] == 0
    assert metrics["tree_depth0_main_top1_survival_violation_sum"] == 0
    assert metrics["tree_argmax_survival_depth_0_sum"] >= metrics[
        "tree_depth0_main_top1_acc_sum"
    ]
    assert model.fc.weight.grad is not None
    assert model.fc.weight.grad.abs().sum() > 0
    assert model.markov_head.markov_w1.weight.grad is not None
    assert model.markov_head.markov_w1.weight.grad.abs().sum() > 0


def test_tree_cli_values_are_forwarded_to_train_and_validation_calls():
    train_kwargs, val_kwargs = DSparkDraftModel.get_trainer_kwargs(
        loss_fn="ce",
        enable_tree_training=True,
        tree_loss_top_k=3,
        tree_loss_num_anchors=7,
        tree_loss_max_nodes=11,
        tree_rank_top_j=2,
        tree_rank_margin=0.5,
        tree_rank_hinge_cap=3.0,
        tree_survival_alpha=0.2,
        tree_rank_alpha=0.05,
    )
    assert train_kwargs["enable_tree_training"] is True
    assert train_kwargs["tree_loss_top_k"] == 3
    assert train_kwargs["tree_loss_num_anchors"] == 7
    assert train_kwargs["tree_loss_max_nodes"] == 11
    assert train_kwargs["tree_rank_top_j"] == 2
    assert train_kwargs["tree_rank_margin"] == pytest.approx(0.5)
    assert train_kwargs["tree_rank_hinge_cap"] == pytest.approx(3.0)
    assert train_kwargs["tree_survival_alpha"] == pytest.approx(0.2)
    assert train_kwargs["tree_rank_alpha"] == pytest.approx(0.05)
    assert "enable_tree_training" not in val_kwargs


def test_disabled_tree_training_does_not_build_replay(monkeypatch):
    model, batch = _tiny_model_and_batch()

    def fail(*args, **kwargs):
        raise AssertionError("disabled tree training must not build replay")

    monkeypatch.setattr("speculators.models.dspark.core.build_tree_loss_inputs", fail)
    _, loss, metrics = model(
        **batch,
        max_anchors=2,
        loss_config=resolve_loss_config("ce"),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert not any(name.startswith("tree_") for name in metrics)


def test_eval_mode_does_not_replay_tree_training(monkeypatch):
    model, batch = _tiny_model_and_batch()
    model.eval()

    def fail(*args, **kwargs):
        raise AssertionError("validation must not run stochastic tree replay")

    monkeypatch.setattr("speculators.models.dspark.core.build_tree_loss_inputs", fail)
    with torch.no_grad():
        _, loss, metrics = model(
            **batch,
            max_anchors=2,
            loss_config=resolve_loss_config("ce"),
            enable_tree_training=True,
        )
    assert torch.isfinite(loss)
    assert not any(name.startswith("tree_") for name in metrics)


def test_bv_and_tree_training_are_rejected_in_direct_forward():
    model, batch = _tiny_model_and_batch()
    with pytest.raises(ValueError, match="incompatible distributed loss scaling"):
        model(
            **batch,
            max_anchors=2,
            enable_bv_loss=True,
            enable_tree_training=True,
        )


def test_tree_depth_mask_uses_labels_and_stops_after_eos_or_document_boundary():
    input_ids = torch.tensor([[5, 6, 7, 31, 9, 10, 11, 12]])
    loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
    document_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1]])
    block_indices = torch.tensor([1, 2, 3, 4, 5, 6])
    aligned = torch.ones(1, 6, dtype=torch.bool)
    valid = _build_tree_depth_valid_mask(
        input_ids=input_ids,
        loss_mask=loss_mask,
        document_ids=document_ids,
        anchored_block_indices=block_indices,
        aligned_loss_mask=aligned,
        block_size=3,
        sample_from_anchor=True,
        eos_token_ids=(31,),
    )
    # First block predicts [7, EOS, 9]; the token after EOS is invalid.
    # Second block crosses from document 0 to document 1 at its second label.
    assert torch.equal(
        valid,
        torch.tensor([[True, True, False], [True, False, False]]),
    )


def test_tree_depth_mask_does_not_revive_after_a_masked_label():
    input_ids = torch.tensor([[5, 6, 7, 8, 9]])
    valid = _build_tree_depth_valid_mask(
        input_ids=input_ids,
        loss_mask=torch.tensor([[True, True, False, True, True]]),
        document_ids=torch.zeros_like(input_ids),
        anchored_block_indices=torch.tensor([0, 1, 2]),
        aligned_loss_mask=torch.ones(1, 3, dtype=torch.bool),
        block_size=3,
        sample_from_anchor=True,
        eos_token_ids=(),
    )
    assert torch.equal(valid, torch.tensor([[True, False, False]]))


def test_tree_anchor_selection_requires_valid_first_edge():
    input_ids = torch.tensor([[5, 6, 31, 8, 9, 10]])
    selection = _build_tree_anchor_selection_mask(
        input_ids=input_ids,
        loss_mask=torch.ones_like(input_ids, dtype=torch.bool),
        document_ids=torch.tensor([[0, 0, 0, 0, 1, 1]]),
        eos_token_ids=(31,),
    )
    # EOS is allowed as the first label (root 6), but an EOS root and a
    # cross-document first edge are not valid TreeSpark anchors.
    assert torch.equal(
        selection,
        torch.tensor([[True, True, False, False, True, False]]),
    )
