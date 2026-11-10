"""Meta-device parameter audit and dense-equivalent training matmul FLOPs.

Defaults reproduce the user's five-layer Qwen3 DSpark configuration. No model
weights, external services, or large tensor allocations are needed. FLOPs omit
loss functions, elementwise operations, optimizer work and teacher service work.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen3Config

from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel


def default_config():
    transformer = Qwen3Config(
        hidden_size=3584,
        intermediate_size=18432,
        num_hidden_layers=5,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=131072,
        attention_bias=False,
        tie_word_embeddings=False,
        max_position_embeddings=262144,
    )
    transformer._attn_implementation = "sdpa"
    return DSparkSpeculatorConfig(
        transformer_layer_config=transformer,
        aux_hidden_state_layer_ids=[1, 10, 19, 28, 37],
        block_size=7,
        draft_vocab_size=131072,
        mask_token_id=139,
        markov_rank=256,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        sample_from_anchor=True,
    )


def audit(config, seq_len, max_anchors):
    c = config.transformer_layer_config
    if c.model_type != "qwen3" or c.attention_bias:
        raise ValueError("This formula requires Qwen3 without attention bias")
    if config.markov_head_type != "vanilla" or config.markov_rank <= 0:
        raise ValueError("This formula requires a positive-rank vanilla Markov head")
    with torch.device("meta"):
        model = DSparkDraftModel(config)
    groups = {
        name: {
            "parameters": sum(p.numel() for p in module.parameters()),
            "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad),
        }
        for name, module in model.named_children()
    }
    hidden = c.hidden_size
    query_dim = c.num_attention_heads * c.head_dim
    kv_dim = c.num_key_value_heads * c.head_dim
    draft_tokens = max_anchors * config.block_size
    fusion_layers = len(config.aux_hidden_state_layer_ids)
    vocab = config.draft_vocab_size
    rank = config.markov_rank
    confidence_dim = hidden + (rank if config.confidence_head_with_markov else 0)
    # RMSNorm weights: 2H + 2d per decoder, and 2H outside the decoders.
    per_layer = (
        2 * hidden * (query_dim + kv_dim)
        + 3 * hidden * c.intermediate_size
        + 2 * hidden
        + 2 * c.head_dim
    )
    expected_trainable = (
        c.num_hidden_layers * per_layer
        + fusion_layers * hidden**2
        + 2 * hidden
        + rank * (c.vocab_size + vocab)
        + (confidence_dim + 1 if config.enable_confidence_head else 0)
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != expected_trainable:
        raise AssertionError(f"Formula={expected_trainable}, actual={trainable}")
    forward = {
        "fusion": 2 * seq_len * fusion_layers * hidden**2,
        "decoder_projections_mlp": c.num_hidden_layers
        * (
            4 * draft_tokens * hidden * query_dim
            + 4 * (seq_len + draft_tokens) * hidden * kv_dim
            + 6 * draft_tokens * hidden * c.intermediate_size
        ),
        "attention_dense_equivalent": (
            c.num_hidden_layers
            * 4
            * draft_tokens
            * (seq_len + draft_tokens)
            * query_dim
        ),
        "draft_lm_head": 2 * draft_tokens * hidden * vocab,
        "verifier_lm_head": 2 * seq_len * hidden * vocab,
        "markov_projection": 2 * draft_tokens * rank * vocab,
        "confidence_projection": (
            2 * draft_tokens * confidence_dim if config.enable_confidence_head else 0
        ),
    }
    multipliers = {
        "fusion": 2,  # Frozen input: forward and dW, no dX.
        "decoder_projections_mlp": 3,
        "attention_dense_equivalent": 3,
        "draft_lm_head": 2,  # Frozen weight: forward and dX, no dW.
        "verifier_lm_head": 1,  # no_grad.
        "markov_projection": 3,
        "confidence_projection": 3,
    }
    training = {key: value * multipliers[key] for key, value in forward.items()}
    return {
        "seq_len": seq_len,
        "max_anchors": max_anchors,
        "block_size": config.block_size,
        "draft_query_tokens": draft_tokens,
        "modules": groups,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": trainable,
        "parameters_excluding_training_only_verifier_head_and_norm": sum(
            p.numel()
            for name, p in model.named_parameters()
            if name not in {"verifier_lm_head.weight", "verifier_norm.weight"}
        ),
        "forward_flops": forward,
        "forward_backward_flops": training,
        "forward_tflop": sum(forward.values()) / 1e12,
        "forward_backward_tflop": sum(training.values()) / 1e12,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional saved DSpark config.json")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--max-anchors", type=int, default=512)
    args = parser.parse_args()
    if args.seq_len <= 0 or args.max_anchors <= 0:
        parser.error("seq-len and max-anchors must be positive")
    config = (
        DSparkSpeculatorConfig(**json.loads(args.config.read_text()))
        if args.config
        else default_config()
    )
    print(json.dumps(audit(config, args.seq_len, args.max_anchors), indent=2))


if __name__ == "__main__":
    main()
