"""Tests for the training-only DFlare adaptations of DSpark."""

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import SpeculatorModel
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark_deep import (
    DSparkDeepDraftModel,
    DSparkDeepSpeculatorConfig,
)
from speculators.models.dspark_layerwise import (
    DSparkLayerwiseDraftModel,
    DSparkLayerwiseSpeculatorConfig,
)
from speculators.models.dspark_layerwise.model_definitions import (
    Qwen3DSparkLayerwiseAttention,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _transformer_config(num_layers: int) -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        attention_bias=False,
        attention_dropout=0.0,
        layer_types=["full_attention"] * num_layers,
        sliding_window=None,
        use_sliding_window=False,
    )
    config._attn_implementation = "eager"
    return config


def _speculators_config() -> SpeculatorsConfig:
    return SpeculatorsConfig(
        algorithm="dspark",
        proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=2)],
        default_proposal_method="greedy",
        verifier=VerifierConfig(name_or_path=None, architectures=[]),
    )


def _layerwise_config(
    *, num_layers: int = 3, target_layers: tuple[int, ...] = (1, 3, 5)
) -> DSparkLayerwiseSpeculatorConfig:
    return DSparkLayerwiseSpeculatorConfig(
        transformer_layer_config=_transformer_config(num_layers),
        draft_vocab_size=32,
        block_size=2,
        aux_hidden_state_layer_ids=list(target_layers),
        mask_token_id=1,
        markov_rank=4,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        speculators_config=_speculators_config(),
    )


def _deep_config(num_layers: int) -> DSparkDeepSpeculatorConfig:
    return DSparkDeepSpeculatorConfig(
        transformer_layer_config=_transformer_config(num_layers),
        draft_vocab_size=32,
        block_size=2,
        aux_hidden_state_layer_ids=[1, 3, 5],
        mask_token_id=1,
        markov_rank=4,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        minimum_draft_layers=6,
        speculators_config=_speculators_config(),
    )


def _initialize_frozen_verifier_weights(model: DSparkDraftModel) -> None:
    """The real trainer loads these weights from the verifier checkpoint."""
    with torch.no_grad():
        model.embed_tokens.weight.normal_(std=0.02)
        model.lm_head.weight.normal_(std=0.02)
        model.verifier_lm_head.weight.normal_(std=0.02)


def test_variants_are_registered_without_replacing_dspark():
    assert SpeculatorModel.registry is not None
    assert SpeculatorModel.registry["dspark"] is DSparkDraftModel
    assert SpeculatorModel.registry["dspark_layerwise"] is DSparkLayerwiseDraftModel
    assert SpeculatorModel.registry["dspark_deep"] is DSparkDeepDraftModel


def test_layerwise_fusion_shape_probabilities_and_gradients():
    model = DSparkLayerwiseDraftModel(_layerwise_config())
    hidden_size = model.config.transformer_layer_config.hidden_size
    hidden_states = torch.randn(
        2,
        7,
        model.num_target_layers * hidden_size,
        requires_grad=True,
    )

    fused = model._fuse_target_hidden_states(hidden_states)
    probs = torch.softmax(model.layer_fusion_weights, dim=-1)

    assert fused.shape == (2, 7, model.num_draft_layers, hidden_size)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(model.num_draft_layers))
    assert not torch.allclose(probs[0], probs[-1])

    (fused * torch.randn_like(fused)).sum().backward()
    assert model.layer_fusion_weights.grad is not None
    assert torch.isfinite(model.layer_fusion_weights.grad).all()
    assert model.layer_fusion_weights.grad.abs().sum() > 0


def test_layerwise_fusion_rejects_hidden_state_width_mismatch():
    model = DSparkLayerwiseDraftModel(_layerwise_config())
    hidden_size = model.config.transformer_layer_config.hidden_size
    wrong_width = model.num_target_layers * hidden_size - 1

    with pytest.raises(ValueError, match="target hidden-state width"):
        model._fuse_target_hidden_states(torch.randn(1, 4, wrong_width))


def test_layerwise_attention_uses_independent_target_kv_projections():
    model = DSparkLayerwiseDraftModel(_layerwise_config(num_layers=2))
    attention = model.layers[0].self_attn

    assert isinstance(attention, Qwen3DSparkLayerwiseAttention)
    assert attention.k_proj_target is not attention.k_proj
    assert attention.v_proj_target is not attention.v_proj
    assert (
        attention.k_proj_target.weight.data_ptr() != attention.k_proj.weight.data_ptr()
    )
    assert (
        attention.v_proj_target.weight.data_ptr() != attention.v_proj.weight.data_ptr()
    )


def test_layerwise_training_forward_and_backward():
    torch.manual_seed(7)
    model = DSparkLayerwiseDraftModel(_layerwise_config(num_layers=2))
    _initialize_frozen_verifier_weights(model)

    seq_len = 8
    hidden_size = model.config.transformer_layer_config.hidden_size
    hidden_states = torch.randn(1, seq_len, model.num_target_layers * hidden_size)
    input_ids = torch.randint(2, 32, (1, seq_len))
    loss_mask = torch.ones(1, seq_len)
    verifier_last_hidden_states = torch.randn(1, seq_len, hidden_size)
    document_ids = torch.zeros(1, seq_len, dtype=torch.long)

    _, loss, metrics = model(
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        verifier_last_hidden_states=verifier_last_hidden_states,
        document_ids=document_ids,
        max_anchors=2,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "accept_len_sum" in metrics

    loss.backward()
    assert model.layer_fusion_weights.grad is not None
    assert model.layers[0].self_attn.k_proj_target.weight.grad is not None
    assert model.markov_head is not None
    assert model.markov_head.markov_w2.weight.grad is not None
    assert model.confidence_head is not None
    assert model.confidence_head.proj.weight.grad is not None


def test_deep_variant_keeps_original_dspark_backbone_and_heads():
    model = DSparkDeepDraftModel(_deep_config(num_layers=7))

    assert isinstance(model, DSparkDraftModel)
    assert len(model.layers) == 7
    assert hasattr(model, "fc")
    assert not hasattr(model, "layer_fusion_weights")
    assert model.markov_head is not None
    assert model.confidence_head is not None


def test_deep_variant_rejects_non_scaled_depth():
    with pytest.raises(ValueError, match="--num-layers 7"):
        DSparkDeepDraftModel(_deep_config(num_layers=5))


def test_layerwise_checkpoint_round_trip_uses_registered_model(tmp_path):
    model = DSparkLayerwiseDraftModel(_layerwise_config(num_layers=2))
    model.save_pretrained(tmp_path)

    restored = SpeculatorModel.from_pretrained(tmp_path, local_files_only=True)

    assert isinstance(restored, DSparkLayerwiseDraftModel)
    assert restored.config.speculators_model_type == "dspark_layerwise"
    assert torch.equal(
        restored.layer_fusion_weights,
        model.layer_fusion_weights,
    )
