"""Causal Correction modules for the isolated DSpark Correction variant."""

from typing import Literal

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention, silu

__all__ = ["CausalCorrectionHead"]


class _TinyCausalAttention(nn.Module):
    """Small causal self-attention used only inside one draft block."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("Correction hidden size must divide evenly across heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = value.shape
        return value.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self._split_heads(self.q_proj(hidden_states))
        key = self._split_heads(self.k_proj(hidden_states))
        value = self._split_heads(self.v_proj(hidden_states))
        output = scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=query.shape[-2] > 1,
        )
        output = output.transpose(1, 2).contiguous().view_as(hidden_states)
        return self.o_proj(output)


class _TinyCausalLayer(nn.Module):
    """Pre-norm attention/MLP block for Correction."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.input_norm = nn.RMSNorm(hidden_size)
        self.attention = _TinyCausalAttention(hidden_size, num_heads)
        self.post_attention_norm = nn.RMSNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.down_proj = nn.Linear(4 * hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(self.input_norm(hidden_states))
        normed = self.post_attention_norm(hidden_states)
        mlp_output = self.down_proj(silu(self.gate_proj(normed)) * self.up_proj(normed))
        return hidden_states + mlp_output


class CausalCorrectionHead(nn.Module):
    """Predict a gated hidden or logit residual for parallel DFlash outputs.

    Each position combines the previous-token embedding, current DFlash hidden state,
    block-relative position, and (in logits mode) the previous target distribution.
    Causal attention then carries information only from earlier positions in the same
    block. The output up-projection is zero-initialized so a fresh model begins from
    the unmodified DFlash prediction path.
    """

    def __init__(
        self,
        *,
        input_hidden_size: int,
        block_size: int,
        correction_hidden_size: int,
        correction_rank: int,
        num_layers: int,
        num_heads: int,
        gate_bias: float,
        output_mode: Literal["hidden", "logits"],
        draft_vocab_size: int,
    ) -> None:
        super().__init__()
        if correction_hidden_size % num_heads:
            raise ValueError(
                "correction_hidden_size must be divisible by correction_num_heads"
            )
        self.output_mode = output_mode
        self.draft_vocab_size = draft_vocab_size
        self.hidden_proj = nn.Linear(
            input_hidden_size, correction_hidden_size, bias=False
        )
        self.token_proj = nn.Linear(
            input_hidden_size, correction_hidden_size, bias=False
        )
        self.position_embedding = nn.Embedding(block_size, correction_hidden_size)
        self.previous_logits_down: nn.Linear | None = None
        self.previous_logits_proj: nn.Linear | None = None
        if output_mode == "logits":
            self.previous_logits_down = nn.Linear(
                draft_vocab_size, correction_rank, bias=False
            )
            self.previous_logits_proj = nn.Linear(
                correction_rank, correction_hidden_size, bias=False
            )
        self.layers = nn.ModuleList(
            [
                _TinyCausalLayer(correction_hidden_size, num_heads)
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.RMSNorm(correction_hidden_size)
        self.correction_down = nn.Linear(
            correction_hidden_size, correction_rank, bias=False
        )
        output_size = input_hidden_size if output_mode == "hidden" else draft_vocab_size
        self.correction_up = nn.Linear(correction_rank, output_size, bias=False)
        self.residual_gate = nn.Linear(correction_hidden_size, 1)
        self._reset_residual_path(gate_bias)

    def _reset_residual_path(self, gate_bias: float) -> None:
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.correction_up.weight)
        nn.init.zeros_(self.residual_gate.weight)
        nn.init.constant_(self.residual_gate.bias, gate_bias)

    def _encode_previous_logits(
        self,
        previous_logits: torch.Tensor,
        previous_logits_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.previous_logits_down is None:
            raise RuntimeError("Logit Correction is missing its rank projection")
        probabilities = torch.softmax(previous_logits.float(), dim=-1)
        rank_features = self.previous_logits_down(
            probabilities.to(self.previous_logits_down.weight.dtype)
        )
        return rank_features * previous_logits_mask.to(
            device=rank_features.device,
            dtype=rank_features.dtype,
        ).unsqueeze(-1)

    def forward(
        self,
        previous_token_embeddings: torch.Tensor,
        dflash_hidden: torch.Tensor,
        block_positions: torch.Tensor,
        *,
        previous_logits: torch.Tensor | None = None,
        previous_logits_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_shape = dflash_hidden.shape[:-1]
        if previous_token_embeddings.shape != dflash_hidden.shape:
            raise ValueError("Previous-token embeddings must match DFlash hidden")
        if block_positions.shape != prefix_shape:
            raise ValueError("Block positions must match Correction token dimensions")

        dtype = self.hidden_proj.weight.dtype
        hidden_states = (
            self.hidden_proj(dflash_hidden.to(dtype))
            + self.token_proj(previous_token_embeddings.to(dtype))
            + self.position_embedding(block_positions.long()).to(dtype)
        )
        if self.output_mode == "logits":
            if previous_logits is None or previous_logits_mask is None:
                raise ValueError("Logit Correction requires previous logits and mask")
            expected = (*prefix_shape, self.draft_vocab_size)
            if previous_logits.shape != expected:
                raise ValueError(
                    f"Expected previous logits shape {expected}, "
                    f"got {tuple(previous_logits.shape)}"
                )
            if previous_logits_mask.shape != prefix_shape:
                raise ValueError("Previous-logits mask must match token dimensions")
            if self.previous_logits_proj is None:
                raise RuntimeError("Logit Correction is missing its feature projection")
            rank_features = self._encode_previous_logits(
                previous_logits, previous_logits_mask
            )
            hidden_states = hidden_states + self.previous_logits_proj(rank_features)
        elif previous_logits is not None or previous_logits_mask is not None:
            raise ValueError("Previous logits are accepted only in logits mode")

        for layer in self.layers:
            hidden_states = layer(hidden_states)
        causal_states = self.output_norm(hidden_states)
        low_rank = silu(self.correction_down(causal_states))
        residual = self.correction_up(low_rank)
        residual = residual * torch.sigmoid(self.residual_gate(causal_states))
        return residual, causal_states
