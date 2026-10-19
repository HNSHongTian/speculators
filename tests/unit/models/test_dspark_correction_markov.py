"""Unit coverage for collaborative DSpark Correction + Markov/RNN."""

import pytest
import torch
from pydantic import ValidationError
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import SpeculatorModel
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark_correction import DSparkCorrectionDraftModel
from speculators.models.dspark_correction_markov import (
    DSparkCorrectionMarkovDraftModel,
    DSparkCorrectionMarkovSpeculatorConfig,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _transformer_config() -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        attention_bias=False,
        attention_dropout=0.0,
        layer_types=["full_attention", "full_attention"],
        sliding_window=None,
        use_sliding_window=False,
    )
    config._attn_implementation = "eager"
    return config


def _speculators_config() -> SpeculatorsConfig:
    return SpeculatorsConfig(
        algorithm="dspark_correction_markov",
        proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
        default_proposal_method="greedy",
        verifier=VerifierConfig(name_or_path=None, architectures=[]),
    )


def _config(
    *,
    head_type: str = "vanilla",
    output_mode: str = "hidden",
    sample_from_anchor: bool = True,
    hidden_aux: bool = False,
    confidence_with_markov: bool = True,
    confidence_with_correction: bool = True,
) -> DSparkCorrectionMarkovSpeculatorConfig:
    return DSparkCorrectionMarkovSpeculatorConfig(
        transformer_layer_config=_transformer_config(),
        draft_vocab_size=32,
        block_size=3,
        aux_hidden_state_layer_ids=[1, 3],
        mask_token_id=1,
        sample_from_anchor=sample_from_anchor,
        markov_rank=8,
        markov_head_type=head_type,
        correction_output_mode=output_mode,
        correction_hidden_size=16,
        correction_rank=8,
        correction_num_layers=1,
        correction_num_heads=4,
        correction_hidden_aux_loss=hidden_aux,
        enable_confidence_head=True,
        confidence_head_with_markov=confidence_with_markov,
        confidence_head_with_correction=confidence_with_correction,
        speculators_config=_speculators_config(),
    )


def _initialize_frozen_verifier_weights(
    model: DSparkCorrectionMarkovDraftModel,
) -> None:
    with torch.no_grad():
        model.embed_tokens.weight.normal_(std=0.02)
        model.lm_head.weight.normal_(std=0.02)
        model.verifier_lm_head.weight.normal_(std=0.02)


def _training_inputs(
    model: DSparkCorrectionMarkovDraftModel,
) -> dict[str, torch.Tensor]:
    seq_len = 9
    hidden_size = model.config.transformer_layer_config.hidden_size
    return {
        "hidden_states": torch.randn(1, seq_len, 2 * hidden_size),
        "input_ids": torch.randint(2, 32, (1, seq_len)),
        "loss_mask": torch.ones(1, seq_len),
        "verifier_last_hidden_states": torch.randn(1, seq_len, hidden_size),
        "document_ids": torch.zeros(1, seq_len, dtype=torch.long),
    }


def test_registration_is_independent_from_existing_dspark_variants():
    assert SpeculatorModel.registry is not None
    assert SpeculatorModel.registry["dspark"] is DSparkDraftModel
    assert SpeculatorModel.registry["dspark_correction"] is DSparkCorrectionDraftModel
    assert (
        SpeculatorModel.registry["dspark_correction_markov"]
        is DSparkCorrectionMarkovDraftModel
    )


@pytest.mark.parametrize("head_type", ["vanilla", "gated", "rnn"])
def test_model_contains_correction_and_requested_markov_head(head_type):
    model = DSparkCorrectionMarkovDraftModel(_config(head_type=head_type))

    assert model.correction_head is not None
    assert model.markov_head is not None
    assert model.markov_head.head_type == head_type
    assert model.correction_markov_gate.in_features == 16
    assert model.correction_markov_scale.item() == 0.0


def test_zero_global_scale_preserves_correction_logits_exactly():
    model = DSparkCorrectionMarkovDraftModel(_config())
    correction_logits = torch.randn(2, 3, 32)
    correction_states = torch.randn(2, 3, 16)
    previous_token_ids = torch.randint(0, 32, (2, 3))
    hidden_states = torch.randn(2, 3, 16)

    final, gate, _, _ = model._apply_collaborative_markov(
        correction_logits,
        correction_states,
        previous_token_ids,
        hidden_states,
    )

    assert torch.equal(final, correction_logits)
    assert torch.count_nonzero(gate) == 0


@pytest.mark.parametrize("head_type", ["vanilla", "gated", "rnn"])
def test_collaboration_formula_matches_manual_computation(head_type):
    torch.manual_seed(11)
    model = DSparkCorrectionMarkovDraftModel(_config(head_type=head_type))
    with torch.no_grad():
        model.correction_markov_scale.fill_(0.7)
        model.correction_markov_gate.weight.normal_(std=0.02)
    correction_logits = torch.randn(2, 3, 32)
    correction_states = torch.randn(2, 3, 16)
    previous_token_ids = torch.randint(0, 32, (2, 3))
    hidden_states = torch.randn(2, 3, 16)

    final, gate, prev_emb, markov_bias = model._apply_collaborative_markov(
        correction_logits,
        correction_states,
        previous_token_ids,
        hidden_states,
    )
    expected_bias = model.markov_head.block_bias(
        prev_token_ids=previous_token_ids,
        hidden_states=hidden_states,
        prev_emb=model.markov_head.prev_embeddings(previous_token_ids),
    )
    expected_gate = torch.tanh(model.correction_markov_scale) * torch.sigmoid(
        model.correction_markov_gate(correction_states)
    )

    assert torch.allclose(
        prev_emb, model.markov_head.prev_embeddings(previous_token_ids)
    )
    assert torch.allclose(markov_bias, expected_bias)
    assert torch.allclose(gate, expected_gate)
    assert torch.allclose(final, correction_logits + expected_gate * expected_bias)


def test_rnn_state_is_independent_between_blocks():
    torch.manual_seed(13)
    model = DSparkCorrectionMarkovDraftModel(_config(head_type="rnn"))
    previous_token_ids = torch.randint(0, 32, (2, 3))
    hidden_states = torch.randn(2, 3, 16)
    original = model.markov_head.block_bias(
        prev_token_ids=previous_token_ids,
        hidden_states=hidden_states,
    )
    changed_ids = previous_token_ids.clone()
    changed_hidden = hidden_states.clone()
    changed_ids[0] = torch.randint(0, 32, (3,))
    changed_hidden[0] += 100.0
    changed = model.markov_head.block_bias(
        prev_token_ids=changed_ids,
        hidden_states=changed_hidden,
    )

    assert torch.allclose(original[1], changed[1])
    assert not torch.allclose(original[0], changed[0])


@pytest.mark.parametrize("head_type", ["vanilla", "gated", "rnn"])
def test_combined_refinement_does_not_leak_future_slots(head_type):
    """A future hidden/token change must not alter earlier combined logits."""
    torch.manual_seed(15)
    model = DSparkCorrectionMarkovDraftModel(_config(head_type=head_type))
    _initialize_frozen_verifier_weights(model)
    with torch.no_grad():
        model.correction_head.correction_up.weight.normal_(std=0.02)
        model.correction_markov_scale.fill_(0.5)

    hidden = torch.randn(1, 3, 16)
    base_logits = model.lm_head(hidden).detach()
    targets = torch.randn(1, 3, 32)
    previous_ids = torch.randint(2, 32, (1, 3))

    correction_logits, correction_states, _ = model._apply_correction(
        hidden, base_logits, targets, previous_ids
    )
    before, _, _, _ = model._apply_collaborative_markov(
        correction_logits, correction_states, previous_ids, hidden
    )

    changed_hidden = hidden.clone()
    changed_targets = targets.clone()
    changed_ids = previous_ids.clone()
    changed_hidden[:, 2] += 100.0
    changed_targets[:, 2] += 100.0
    changed_ids[:, 2] = (changed_ids[:, 2] + 1) % 32
    changed_base_logits = model.lm_head(changed_hidden).detach()
    changed_correction_logits, changed_states, _ = model._apply_correction(
        changed_hidden, changed_base_logits, changed_targets, changed_ids
    )
    after, _, _, _ = model._apply_collaborative_markov(
        changed_correction_logits, changed_states, changed_ids, changed_hidden
    )

    assert torch.allclose(before[:, :2], after[:, :2], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(before[:, 2], after[:, 2])


@pytest.mark.parametrize("head_type", ["vanilla", "gated", "rnn"])
@pytest.mark.parametrize("output_mode", ["hidden", "logits"])
def test_training_forward_backward_all_head_and_correction_modes(
    head_type, output_mode
):
    torch.manual_seed(17)
    model = DSparkCorrectionMarkovDraftModel(
        _config(head_type=head_type, output_mode=output_mode)
    )
    _initialize_frozen_verifier_weights(model)
    with torch.no_grad():
        model.correction_markov_scale.fill_(0.5)

    _, loss, metrics = model(**_training_inputs(model), max_anchors=2)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "accept_len_sum" in metrics
    assert "correction_markov_gate_sum" in metrics
    assert "correction_markov_bias_norm_sum" in metrics
    assert "correction_markov_scale_sum" in metrics
    loss.backward()
    assert model.correction_head.correction_up.weight.grad is not None
    assert model.markov_head.markov_w2.weight.grad is not None
    assert torch.isfinite(model.markov_head.markov_w2.weight.grad).all()
    if head_type == "rnn":
        assert model.markov_head.joint_proj.weight.grad is not None
    assert model.correction_markov_gate.weight.grad is not None
    assert model.correction_markov_scale.grad is not None
    assert model.confidence_head is not None
    assert model.confidence_head.proj.weight.grad is not None


@pytest.mark.parametrize("output_mode", ["hidden", "logits"])
def test_reserved_anchor_alignment_with_rnn(output_mode):
    model = DSparkCorrectionMarkovDraftModel(
        _config(
            head_type="rnn",
            output_mode=output_mode,
            sample_from_anchor=False,
        )
    )
    _initialize_frozen_verifier_weights(model)
    _, loss, metrics = model(**_training_inputs(model), max_anchors=2)

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["accept_len_sum"])


@pytest.mark.parametrize(
    ("with_markov", "with_correction", "expected_dim"),
    [(False, False, 16), (True, False, 24), (False, True, 32), (True, True, 40)],
)
def test_confidence_feature_dimensions(with_markov, with_correction, expected_dim):
    model = DSparkCorrectionMarkovDraftModel(
        _config(
            confidence_with_markov=with_markov,
            confidence_with_correction=with_correction,
        )
    )

    assert model.confidence_head is not None
    assert model.confidence_head.proj.in_features == expected_dim


def test_hidden_auxiliary_loss_is_reported():
    model = DSparkCorrectionMarkovDraftModel(_config(hidden_aux=True))
    _initialize_frozen_verifier_weights(model)

    _, loss, metrics = model(**_training_inputs(model), max_anchors=2)

    assert torch.isfinite(loss)
    assert "correction_hidden_aux_loss_sum" in metrics


def test_config_requires_positive_markov_rank():
    with pytest.raises(ValidationError, match="markov_rank"):
        DSparkCorrectionMarkovSpeculatorConfig(
            transformer_layer_config=_transformer_config(),
            markov_rank=0,
        )


def test_config_rejects_invalid_correction_geometry():
    with pytest.raises(ValidationError, match="divisible"):
        DSparkCorrectionMarkovSpeculatorConfig(
            transformer_layer_config=_transformer_config(),
            correction_hidden_size=15,
            correction_num_heads=4,
        )


def test_config_rejects_hidden_auxiliary_loss_in_logits_mode():
    with pytest.raises(ValidationError, match="hidden mode"):
        DSparkCorrectionMarkovSpeculatorConfig(
            transformer_layer_config=_transformer_config(),
            correction_output_mode="logits",
            correction_hidden_aux_loss=True,
        )


def test_checkpoint_round_trip_uses_combined_registry(tmp_path):
    model = DSparkCorrectionMarkovDraftModel(_config(head_type="rnn"))
    _initialize_frozen_verifier_weights(model)
    model.save_pretrained(tmp_path)

    restored = SpeculatorModel.from_pretrained(tmp_path, local_files_only=True)

    assert isinstance(restored, DSparkCorrectionMarkovDraftModel)
    assert restored.config.speculators_model_type == "dspark_correction_markov"
    assert restored.correction_head is not None
    assert restored.markov_head.head_type == "rnn"
    assert restored.correction_markov_gate is not None
