"""DSpark with collaborative causal Correction and Markov/RNN refinement."""

from typing import ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dspark.metrics import compute_metrics
from speculators.models.dspark.model_definitions import ConfidenceHead, MarkovHead
from speculators.models.dspark_correction.core import DSparkCorrectionDraftModel
from speculators.models.dspark_correction_markov.config import (
    DSparkCorrectionMarkovSpeculatorConfig,
)
from speculators.models.metrics import LossConfig, kl_div_loss
from speculators.models.utils import conditional_torch_compile

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}

__all__ = ["DSparkCorrectionMarkovDraftModel"]


@SpeculatorModel.register("dspark_correction_markov")
class DSparkCorrectionMarkovDraftModel(DSparkCorrectionDraftModel):
    """Correction-first DSpark with a gated native Markov or RNN residual."""

    config_class: ClassVar[type[DSparkCorrectionMarkovSpeculatorConfig]] = (  # type: ignore[misc,assignment]
        DSparkCorrectionMarkovSpeculatorConfig
    )

    def __init__(self, config: DSparkCorrectionMarkovSpeculatorConfig) -> None:
        # The parent constructs the unchanged DFlash backbone and the shared
        # Correction head. This class then restores the native DSpark Markov head.
        super().__init__(config=config)  # type: ignore[arg-type]
        hidden_size = config.transformer_layer_config.hidden_size
        self.markov_head = MarkovHead(
            verifier_vocab_size=self.verifier_vocab_size,
            draft_vocab_size=self.draft_vocab_size,
            markov_rank=config.markov_rank,
            hidden_size=hidden_size,
            head_type=config.markov_head_type,
        )
        self.correction_markov_gate = nn.Linear(config.correction_hidden_size, 1)
        self.correction_markov_scale = nn.Parameter(
            torch.tensor(float(config.correction_markov_scale_init))
        )
        nn.init.zeros_(self.correction_markov_gate.weight)
        nn.init.constant_(
            self.correction_markov_gate.bias, config.correction_markov_gate_bias
        )

        self.confidence_head: ConfidenceHead | None = None
        if config.enable_confidence_head:
            input_dim = hidden_size
            if config.confidence_head_with_markov:
                input_dim += config.markov_rank
            if config.confidence_head_with_correction:
                input_dim += config.correction_hidden_size
            self.confidence_head = ConfidenceHead(input_dim)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DSparkCorrectionMarkovDraftModel":
        """Build the isolated collaborative model from shared training args."""
        kwargs = dict(kwargs)
        if kwargs.get("sample_from_anchor") is None:
            kwargs["sample_from_anchor"] = True
        enable_confidence = kwargs.get("enable_confidence_head")
        confidence_with_markov = kwargs.get("confidence_head_with_markov")
        config = cls.config_class(
            **cls._build_base_config_kwargs(
                "dspark_correction_markov", verifier_config, **kwargs
            ),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_confidence_head=(
                True if enable_confidence is None else enable_confidence
            ),
            confidence_head_with_markov=(
                True if confidence_with_markov is None else confidence_with_markov
            ),
            correction_output_mode=kwargs.get("correction_output_mode", "hidden"),
            correction_hidden_size=kwargs.get("correction_hidden_size", 512),
            correction_rank=kwargs.get("correction_rank", 256),
            correction_num_layers=kwargs.get("correction_num_layers", 1),
            correction_num_heads=kwargs.get("correction_num_heads", 8),
            correction_gate_bias=kwargs.get("correction_gate_bias", 0.0),
            correction_markov_gate_bias=kwargs.get("correction_markov_gate_bias", -2.0),
            correction_markov_scale_init=kwargs.get(
                "correction_markov_scale_init", 0.0
            ),
            confidence_head_with_correction=kwargs.get(
                "confidence_head_with_correction", True
            ),
            correction_hidden_aux_loss=kwargs.get("correction_hidden_aux_loss", False),
            correction_hidden_aux_weight=kwargs.get(
                "correction_hidden_aux_weight", 0.1
            ),
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    def _apply_collaborative_markov(
        self,
        correction_logits: torch.Tensor,
        correction_states: torch.Tensor,
        previous_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply MMuse-style gated Markov bias after Correction."""
        prev_emb = self.markov_head.prev_embeddings(previous_token_ids)
        markov_bias = self.markov_head.block_bias(
            prev_token_ids=previous_token_ids,
            hidden_states=hidden_states,
            prev_emb=prev_emb,
        )
        gate_dtype = self.correction_markov_gate.weight.dtype
        local_gate = torch.sigmoid(
            self.correction_markov_gate(correction_states.to(gate_dtype))
        )
        collaboration_gate = torch.tanh(self.correction_markov_scale) * local_gate
        final_logits = correction_logits + (
            collaboration_gate.to(markov_bias.dtype) * markov_bias
        )
        return final_logits, collaboration_gate, prev_emb, markov_bias

    @staticmethod
    def _add_collaboration_metrics(
        metrics: dict,
        collaboration_gate: torch.Tensor,
        markov_bias: torch.Tensor,
        loss_mask: torch.Tensor,
        global_scale: torch.Tensor,
    ) -> None:
        mask = loss_mask.to(collaboration_gate.dtype).unsqueeze(-1)
        count = mask.sum().clamp_min(1.0)
        metrics["correction_markov_gate_sum"] = (
            collaboration_gate.detach() * mask
        ).sum()
        metrics["correction_markov_gate_total"] = count
        bias_norm = markov_bias.detach().float().norm(dim=-1, keepdim=True)
        metrics["correction_markov_bias_norm_sum"] = (bias_norm * mask.float()).sum()
        metrics["correction_markov_bias_norm_total"] = count.float()
        metrics["correction_markov_scale_sum"] = global_scale.detach().float().clone()
        metrics["correction_markov_scale_total"] = torch.ones(
            (), device=global_scale.device
        )

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 3072,
        confidence_head_alpha: float = 1.0,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        **kwargs,
    ):
        hidden, base_logits, targets, aligned_loss_mask, anchored_block_indices = (
            self._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                max_anchors=max_anchors,
                **kwargs,
            )
        )
        block_size = self.block_size
        num_blocks = hidden.shape[1] // block_size
        hidden_blocks = hidden.view(num_blocks, block_size, -1)
        base_logits_blocks = base_logits.view(num_blocks, block_size, -1)
        target_blocks = targets.view(num_blocks, block_size, -1)
        block_tokens = input_ids[0, anchored_block_indices].view(num_blocks, block_size)
        previous_token_ids = self._previous_token_ids(
            block_tokens, sample_from_anchor=self.config.sample_from_anchor
        )

        correction_logits, correction_states, corrected_hidden = self._apply_correction(
            hidden_blocks,
            base_logits_blocks,
            target_blocks,
            previous_token_ids,
        )
        final_blocks, collaboration_gate, prev_emb, markov_bias = (
            self._apply_collaborative_markov(
                correction_logits,
                correction_states,
                previous_token_ids,
                hidden_blocks,
            )
        )
        logits = final_blocks.reshape(1, num_blocks * block_size, -1)

        confidence_logits = None
        if self.confidence_head is not None:
            confidence_features = [hidden_blocks]
            if self.config.confidence_head_with_markov:
                confidence_features.append(prev_emb.to(hidden_blocks.dtype))
            if self.config.confidence_head_with_correction:
                confidence_features.append(correction_states.to(hidden_blocks.dtype))
            confidence_logits = self.confidence_head(
                torch.cat(confidence_features, dim=-1)
            ).reshape(1, num_blocks * block_size)

        loss, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            aligned_loss_mask,
            block_size,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            gamma=gamma,
            confidence_head_alpha=confidence_head_alpha,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
        )
        self._add_collaboration_metrics(
            metrics,
            collaboration_gate,
            markov_bias,
            aligned_loss_mask.view(num_blocks, block_size),
            torch.tanh(self.correction_markov_scale),
        )

        if self.config.correction_hidden_aux_loss:
            if corrected_hidden is None:
                raise RuntimeError("Hidden auxiliary loss requires hidden mode")
            with torch.no_grad():
                verifier_targets = self.verifier_norm(
                    verifier_last_hidden_states.to(self.verifier_norm.weight.dtype)
                )
                if not self.config.sample_from_anchor:
                    verifier_targets = torch.roll(verifier_targets, 1, dims=1)
                verifier_targets = verifier_targets[:, anchored_block_indices].view_as(
                    corrected_hidden
                )
            auxiliary_loss = self._hidden_alignment_loss(
                corrected_hidden,
                verifier_targets,
                aligned_loss_mask.view(num_blocks, block_size),
            )
            loss = loss + self.config.correction_hidden_aux_weight * auxiliary_loss
            metrics["loss_sum"] = loss.detach().clone()
            metrics["correction_hidden_aux_loss_sum"] = auxiliary_loss.detach().clone()
            metrics["correction_hidden_aux_loss_total"] = torch.ones(
                (), device=loss.device
            )
        return None, loss, metrics
