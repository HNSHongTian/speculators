"""Step 2 integration: actual tiny DSpark backbone, no downloads/services.

Use BV_TEST_DEVICE=npu:0 (torch_npu installed) or cuda:0 on deployment hardware.
"""

import os

import pytest
import torch
from transformers import Qwen3Config

from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.metrics import resolve_loss_config


def tiny_model(sample=True, confidence=False):
    device = os.environ.get("BV_TEST_DEVICE", "cpu")
    if device.startswith("npu"):
        import torch_npu  # noqa: F401, PLC0415
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
            markov_head_type="rnn",
            enable_confidence_head=confidence,
            sample_from_anchor=sample,
        )
    ).to(device)
    with torch.no_grad():
        for param in model.parameters():
            if not torch.isfinite(param).all():
                param.normal_(std=0.1)
    batch = {
        "hidden_states": torch.randn(1, 16, 16, device=device),
        "input_ids": torch.randint(1, 30, (1, 16), device=device),
        "loss_mask": torch.ones(1, 16, device=device),
        "verifier_last_hidden_states": torch.randn(1, 16, 16, device=device),
        "document_ids": torch.zeros(1, 16, dtype=torch.long, device=device),
    }
    return model, batch


@pytest.mark.parametrize("sample", [True, False])
@pytest.mark.parametrize("confidence", [True, False])
def test_real_backbone_bv_forward_backward(sample, confidence):
    model, batch = tiny_model(sample, confidence)
    _, loss, metrics = model(
        **batch,
        max_anchors=2,
        enable_bv_loss=True,
        bv_beta=batch["hidden_states"].new_tensor(0.2),
        bv_annealing=True,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["bv_valid_blocks"] == 2
    for grad in [
        model.fc.weight.grad,
        model.markov_head.markov_w1.weight.grad,
        model.markov_head.markov_w2.weight.grad,
    ]:
        assert grad.abs().sum() > 0
        assert torch.isfinite(grad).all()
    assert model.verifier_lm_head.weight.grad is None
    assert model.verifier_norm.weight.grad is None
    if confidence:
        assert model.confidence_head.proj.weight.grad.abs().sum() > 0


def test_disabled_forward_never_builds_bv_labels(monkeypatch):
    model, batch = tiny_model()

    def fail(*args, **kwargs):
        raise AssertionError("disabled BV must not build labels")

    monkeypatch.setattr("speculators.models.dspark.core.build_bv_labels", fail)
    _, loss, metrics = model(
        **batch, max_anchors=2, loss_config=resolve_loss_config("ce")
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert not any(k.startswith("bv_") for k in metrics)


def test_reduced_vocabulary_rejected_before_forward():
    model, batch = tiny_model()
    model.use_draft_vocab = True
    with pytest.raises(ValueError, match="full verifier vocabulary"):
        model(**batch, max_anchors=2, enable_bv_loss=True)
