"""DSpark training model with DFlare-style layer-wise target conditioning."""

from typing import ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.utils import get_base_indices_for_anchored_blocks
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark_layerwise.config import (
    DSparkLayerwiseSpeculatorConfig,
)
from speculators.models.dspark_layerwise.model_definitions import (
    Qwen3DSparkLayerwiseDecoderLayer,
)

__all__ = [
    "DSparkLayerwiseDraftModel",
]

_TARGET_HIDDEN_NDIM = 3


@SpeculatorModel.register("dspark_layerwise")
class DSparkLayerwiseDraftModel(DSparkDraftModel):
    """DSpark whose draft layers learn distinct mixtures of target layers."""

    config_class: ClassVar[type[DSparkLayerwiseSpeculatorConfig]] = (  # type: ignore[misc,assignment]
        DSparkLayerwiseSpeculatorConfig
    )
    _no_split_modules = ["Qwen3DSparkLayerwiseDecoderLayer"]

    def __init__(self, config: DSparkLayerwiseSpeculatorConfig) -> None:
        super().__init__(config=config)

        tl_config = config.transformer_layer_config
        self.num_draft_layers = tl_config.num_hidden_layers
        self.num_target_layers = len(self.target_layer_ids)
        if self.num_target_layers == 0:
            raise ValueError(
                "dspark_layerwise requires at least one aux_hidden_state_layer_id."
            )

        # Replace the inherited DFlash layers and shared FC with the DFlare-style
        # layer-wise backbone. The original DSpark source remains untouched.
        self.layers = nn.ModuleList(
            [
                Qwen3DSparkLayerwiseDecoderLayer(tl_config, layer_idx)  # type: ignore[arg-type]
                for layer_idx in range(self.num_draft_layers)
            ]
        )
        self.layers.apply(self._init_weights)
        del self.fc

        self.layer_fusion_weights = nn.Parameter(
            torch.zeros(self.num_draft_layers, self.num_target_layers)
        )
        self._init_fusion_weights(config.fusion_init_scale)

    def _init_fusion_weights(self, scale: float) -> None:
        """Give successive draft layers different initial target-layer preferences."""
        with torch.no_grad():
            self.layer_fusion_weights.zero_()
            for draft_idx in range(self.num_draft_layers):
                target_idx = min(
                    self.num_target_layers - 1,
                    int((draft_idx / self.num_draft_layers) * self.num_target_layers),
                )
                self.layer_fusion_weights[draft_idx, target_idx] = scale

    def _fuse_target_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return layer-specific target context with shape ``[B, S, D, H]``."""
        if hidden_states.ndim != _TARGET_HIDDEN_NDIM:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, target_layers * "
                f"hidden_size], got {tuple(hidden_states.shape)}."
            )

        hidden_size = self.config.transformer_layer_config.hidden_size
        expected_width = self.num_target_layers * hidden_size
        if hidden_states.shape[-1] != expected_width:
            raise ValueError(
                "The target hidden-state width does not match the configured target "
                f"layers: got {hidden_states.shape[-1]}, expected "
                f"{self.num_target_layers} * {hidden_size} = {expected_width}. "
                "Use identical --target-layer-ids for hidden-state generation and "
                "training."
            )

        batch_size, seq_len, _ = hidden_states.shape
        target_by_layer = hidden_states.reshape(
            batch_size, seq_len, self.num_target_layers, hidden_size
        )
        fusion_probs = torch.softmax(self.layer_fusion_weights, dim=-1)
        fused_hidden = torch.einsum("bsth,dt->bsdh", target_by_layer, fusion_probs)
        return self.hidden_norm(fused_hidden)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DSparkLayerwiseDraftModel":
        enable_confidence_head = kwargs.get("enable_confidence_head")
        confidence_with_markov = kwargs.get("confidence_head_with_markov")
        config = DSparkLayerwiseSpeculatorConfig(
            **cls._build_base_config_kwargs("dspark", verifier_config, **kwargs),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_confidence_head=(
                True if enable_confidence_head is None else enable_confidence_head
            ),
            confidence_head_with_markov=(
                True if confidence_with_markov is None else confidence_with_markov
            ),
            fusion_init_scale=kwargs.get("fusion_init_scale", 2.0),
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    def _backbone_forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        """Run the DSpark backbone with one fused target view per draft layer."""
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]
        num_anchors = kwargs.pop("max_anchors", 3072)
        if position_ids is None:
            position_ids = torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid = (
            self._build_attention_mask(loss_mask, num_anchors, document_ids, device)
        )

        mask_tokens_size = num_anchors * self.block_size
        mask_token_ids = torch.full(
            (1, mask_tokens_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        noise_embedding = self.embed_tokens(mask_token_ids)

        fused_target_hidden = self._fuse_target_hidden_states(hidden_states)

        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[0, anchor_positions], self.block_size
        )
        position_ids = torch.cat([position_ids, mask_position_ids.unsqueeze(0)], dim=1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )

        with torch.no_grad():
            verifier_logits = self.verifier_lm_head(
                self.verifier_norm(verifier_last_hidden_states)
            )
            if not self.config.sample_from_anchor:
                verifier_logits = torch.roll(verifier_logits, 1, dims=1)
            targets = verifier_logits[:, anchored_block_indices]

        for layer_idx, layer in enumerate(self.layers):
            noise_embedding = layer(
                hidden_states=noise_embedding,
                target_hidden=fused_target_hidden[:, :, layer_idx, :],
                attention_mask=(
                    sliding_window_attn_mask
                    if layer_idx in self.sliding_window_indices
                    else full_attn_mask
                ),
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden = self.norm(noise_embedding)
        logits = self.lm_head(hidden)
        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )
        if not self.config.sample_from_anchor:
            aligned_loss_mask[:, :: self.block_size] = 0

        return hidden, logits, targets, aligned_loss_mask, anchored_block_indices
