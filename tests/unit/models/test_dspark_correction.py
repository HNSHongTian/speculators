"""Unit coverage for the isolated DSpark Correction model."""

import pytest
import torch
from pydantic import ValidationError
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import SpeculatorModel
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark_correction import (
    DSparkCorrectionDraftModel,
    DSparkCorrectionSpeculatorConfig,
)
from speculators.models.dspark_correction.model_definitions import CausalCorrectionHead
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
        algorithm="dspark_correction",
        proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
        default_proposal_method="greedy",
        verifier=VerifierConfig(name_or_path=None, architectures=[]),
    )


def _config(
    *,
    output_mode: str = "hidden",
    sample_from_anchor: bool = True,
    hidden_aux: bool = False,
) -> DSparkCorrectionSpeculatorConfig:
    return DSparkCorrectionSpeculatorConfig(
        transformer_layer_config=_transformer_config(),
        draft_vocab_size=32,
        block_size=3,
        aux_hidden_state_layer_ids=[1, 3],
        mask_token_id=1,
        sample_from_anchor=sample_from_anchor,
        correction_output_mode=output_mode,
        correction_hidden_size=16,
        correction_rank=8,
        correction_num_layers=1,
        correction_num_heads=4,
        correction_hidden_aux_loss=hidden_aux,
        enable_confidence_head=True,
        confidence_head_with_correction=True,
        speculators_config=_speculators_config(),
    )


def _initialize_frozen_verifier_weights(model: DSparkCorrectionDraftModel) -> None:
    with torch.no_grad():
        model.embed_tokens.weight.normal_(std=0.02)
        model.lm_head.weight.normal_(std=0.02)
        model.verifier_lm_head.weight.normal_(std=0.02)


def _training_inputs(model: DSparkCorrectionDraftModel) -> dict[str, torch.Tensor]:
    seq_len = 9
    hidden_size = model.config.transformer_layer_config.hidden_size
    return {
        "hidden_states": torch.randn(1, seq_len, 2 * hidden_size),
        "input_ids": torch.randint(2, 32, (1, seq_len)),
        "loss_mask": torch.ones(1, seq_len),
        "verifier_last_hidden_states": torch.randn(1, seq_len, hidden_size),
        "document_ids": torch.zeros(1, seq_len, dtype=torch.long),
    }


def test_registration_does_not_replace_native_dspark():
    assert SpeculatorModel.registry is not None
    assert SpeculatorModel.registry["dspark"] is DSparkDraftModel
    assert SpeculatorModel.registry["dspark_correction"] is DSparkCorrectionDraftModel


def test_correction_head_starts_as_noop_in_both_modes():
    for output_mode, output_size in (("hidden", 16), ("logits", 32)):
        head = CausalCorrectionHead(
            input_hidden_size=16,
            block_size=3,
            correction_hidden_size=16,
            correction_rank=8,
            num_layers=1,
            num_heads=4,
            gate_bias=0.0,
            output_mode=output_mode,
            draft_vocab_size=32,
        )
        kwargs = {}
        if output_mode == "logits":
            kwargs = {
                "previous_logits": torch.randn(2, 3, 32),
                "previous_logits_mask": torch.ones(2, 3, dtype=torch.bool),
            }
        residual, states = head(
            torch.randn(2, 3, 16),
            torch.randn(2, 3, 16),
            torch.arange(3).expand(2, -1),
            **kwargs,
        )
        assert residual.shape == (2, 3, output_size)
        assert states.shape == (2, 3, 16)
        assert torch.count_nonzero(residual) == 0


def test_correction_attention_is_causal():
    torch.manual_seed(3)
    head = CausalCorrectionHead(
        input_hidden_size=16,
        block_size=3,
        correction_hidden_size=16,
        correction_rank=8,
        num_layers=1,
        num_heads=4,
        gate_bias=0.0,
        output_mode="hidden",
        draft_vocab_size=32,
    )
    with torch.no_grad():
        head.correction_up.weight.normal_(std=0.02)
    previous = torch.randn(1, 3, 16)
    hidden = torch.randn(1, 3, 16)
    positions = torch.arange(3).unsqueeze(0)
    before, _ = head(previous, hidden, positions)
    changed = hidden.clone()
    changed[:, 2] += 100.0
    after, _ = head(previous, changed, positions)
    assert torch.allclose(before[:, :2], after[:, :2], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(before[:, 2], after[:, 2])


@pytest.mark.parametrize("output_mode", ["hidden", "logits"])
@pytest.mark.parametrize("sample_from_anchor", [True, False])
def test_training_forward_backward(output_mode, sample_from_anchor):
    torch.manual_seed(7)
    model = DSparkCorrectionDraftModel(
        _config(output_mode=output_mode, sample_from_anchor=sample_from_anchor)
    )
    _initialize_frozen_verifier_weights(model)

    _, loss, metrics = model(**_training_inputs(model), max_anchors=2)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "accept_len_sum" in metrics
    loss.backward()
    assert model.markov_head is None
    assert model.correction_head.correction_up.weight.grad is not None
    assert torch.isfinite(model.correction_head.correction_up.weight.grad).all()
    assert model.confidence_head is not None
    assert model.confidence_head.proj.weight.grad is not None


def test_hidden_auxiliary_loss_is_reported():
    model = DSparkCorrectionDraftModel(_config(hidden_aux=True))
    _initialize_frozen_verifier_weights(model)

    _, loss, metrics = model(**_training_inputs(model), max_anchors=2)

    assert torch.isfinite(loss)
    assert "correction_hidden_aux_loss_sum" in metrics
    assert metrics["correction_hidden_aux_loss_sum"] >= 0


def test_config_rejects_invalid_head_geometry():
    with pytest.raises(ValidationError, match="divisible"):
        DSparkCorrectionSpeculatorConfig(
            transformer_layer_config=_transformer_config(),
            correction_hidden_size=15,
            correction_num_heads=4,
        )


def test_config_rejects_hidden_auxiliary_loss_in_logits_mode():
    with pytest.raises(ValidationError, match="hidden mode"):
        DSparkCorrectionSpeculatorConfig(
            transformer_layer_config=_transformer_config(),
            correction_output_mode="logits",
            correction_hidden_aux_loss=True,
        )


def test_checkpoint_round_trip_uses_correction_registry(tmp_path):
    model = DSparkCorrectionDraftModel(_config())
    _initialize_frozen_verifier_weights(model)
    model.save_pretrained(tmp_path)

    restored = SpeculatorModel.from_pretrained(tmp_path, local_files_only=True)

    assert isinstance(restored, DSparkCorrectionDraftModel)
    assert restored.config.speculators_model_type == "dspark_correction"
    assert restored.markov_head is None
    assert restored.correction_head is not None
