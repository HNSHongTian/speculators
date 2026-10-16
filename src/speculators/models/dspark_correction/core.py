"""DSpark with an isolated MMuse-inspired causal Correction head."""

from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark.metrics import compute_metrics
from speculators.models.dspark.model_definitions import ConfidenceHead
from speculators.models.dspark_correction.config import (
    DSparkCorrectionSpeculatorConfig,
)
from speculators.models.dspark_correction.model_definitions import CausalCorrectionHead
from speculators.models.metrics import LossConfig, kl_div_loss
from speculators.models.utils import conditional_torch_compile

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}

__all__ = ["DSparkCorrectionDraftModel"]


@SpeculatorModel.register("dspark_correction")
class DSparkCorrectionDraftModel(DSparkDraftModel):
    """DFlash/DSpark training path with Correction replacing the Markov head."""

    config_class: ClassVar[type[DSparkCorrectionSpeculatorConfig]] = (  # type: ignore[misc,assignment]
        DSparkCorrectionSpeculatorConfig
    )

    def __init__(self, config: DSparkCorrectionSpeculatorConfig) -> None:
        # Deliberately bypass DSparkDraftModel.__init__: the original DSpark class
        # must keep constructing MarkovHead, while this isolated variant replaces it.
        DFlashDraftModel.__init__(self, config=config)
        hidden_size = config.transformer_layer_config.hidden_size
        self.markov_head = None
        self.correction_head = CausalCorrectionHead(
            input_hidden_size=hidden_size,
            block_size=self.block_size,
            correction_hidden_size=config.correction_hidden_size,
            correction_rank=config.correction_rank,
            num_layers=config.correction_num_layers,
            num_heads=config.correction_num_heads,
            gate_bias=config.correction_gate_bias,
            output_mode=config.correction_output_mode,
            draft_vocab_size=self.draft_vocab_size,
        )
        self.confidence_head: ConfidenceHead | None = None
        if config.enable_confidence_head:
            correction_width = (
                config.correction_hidden_size
                if config.confidence_head_with_correction
                else 0
            )
            self.confidence_head = ConfidenceHead(hidden_size + correction_width)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DSparkCorrectionDraftModel":
        """Build the isolated Correction variant from the shared training CLI."""
        kwargs = dict(kwargs)
        if kwargs.get("sample_from_anchor") is None:
            kwargs["sample_from_anchor"] = True
        enable_confidence = kwargs.get("enable_confidence_head")
        config = cls.config_class(
            **cls._build_base_config_kwargs(
                "dspark_correction", verifier_config, **kwargs
            ),
            markov_rank=0,
            confidence_head_with_markov=False,
            enable_confidence_head=(
                True if enable_confidence is None else enable_confidence
            ),
            correction_output_mode=kwargs.get("correction_output_mode", "hidden"),
            correction_hidden_size=kwargs.get("correction_hidden_size", 512),
            correction_rank=kwargs.get("correction_rank", 256),
            correction_num_layers=kwargs.get("correction_num_layers", 1),
            correction_num_heads=kwargs.get("correction_num_heads", 8),
            correction_gate_bias=kwargs.get("correction_gate_bias", 0.0),
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

    @staticmethod
    def _previous_token_ids(
        block_tokens: torch.Tensor, *, sample_from_anchor: bool
    ) -> torch.Tensor:
        if sample_from_anchor:
            return block_tokens
        return torch.cat([block_tokens[:, :1], block_tokens[:, :-1]], dim=1)

    def _apply_correction(
        self,
        hidden_blocks: torch.Tensor,
        base_logits_blocks: torch.Tensor,
        target_blocks: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return corrected logits, full-block causal states and optional hidden."""
        num_blocks, block_size, hidden_size = hidden_blocks.shape
        start_position = 0 if self.config.sample_from_anchor else 1
        active_hidden = hidden_blocks[:, start_position:]
        active_previous_ids = previous_token_ids[:, start_position:]
        positions = torch.arange(
            start_position,
            block_size,
            device=hidden_blocks.device,
            dtype=torch.long,
        ).unsqueeze(0)
        positions = positions.expand(num_blocks, -1)
        with torch.no_grad():
            previous_embeddings = self.embed_tokens(active_previous_ids)

        correction_kwargs = {}
        if self.config.correction_output_mode == "logits":
            if start_position == 0:
                previous_logits = torch.cat(
                    [
                        torch.zeros_like(target_blocks[:, :1]),
                        target_blocks[:, :-1],
                    ],
                    dim=1,
                )
                previous_mask = positions > 0
            else:
                previous_logits = target_blocks[:, :-1]
                previous_mask = torch.ones_like(positions, dtype=torch.bool)
            correction_kwargs = {
                "previous_logits": previous_logits,
                "previous_logits_mask": previous_mask,
            }

        residual, active_states = self.correction_head(
            previous_embeddings,
            active_hidden,
            positions,
            **correction_kwargs,
        )
        corrected_hidden = None
        if self.config.correction_output_mode == "hidden":
            active_corrected = active_hidden + residual.to(active_hidden.dtype)
            corrected_hidden = (
                active_corrected
                if start_position == 0
                else torch.cat(
                    [hidden_blocks[:, :start_position], active_corrected], dim=1
                )
            )
            logits = self.lm_head(
                corrected_hidden.reshape(1, num_blocks * block_size, hidden_size).to(
                    self.lm_head.weight.dtype
                )
            ).view(num_blocks, block_size, -1)
        else:
            active_logits = base_logits_blocks[:, start_position:] + residual.to(
                base_logits_blocks.dtype
            )
            logits = (
                active_logits
                if start_position == 0
                else torch.cat(
                    [base_logits_blocks[:, :start_position], active_logits], dim=1
                )
            )

        states = active_states
        if start_position:
            states = torch.cat(
                [
                    active_states.new_zeros(
                        num_blocks, start_position, active_states.shape[-1]
                    ),
                    active_states,
                ],
                dim=1,
            )
        return logits, states, corrected_hidden

    @staticmethod
    def _hidden_alignment_loss(
        corrected_hidden: torch.Tensor,
        verifier_hidden: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        per_token = torch.nn.functional.smooth_l1_loss(
            corrected_hidden.float(),
            verifier_hidden.float(),
            reduction="none",
        ).mean(dim=-1)
        mask = loss_mask.to(per_token.dtype)
        return (per_token * mask).sum() / mask.sum().clamp_min(1.0)

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
        corrected_logits, correction_states, corrected_hidden = self._apply_correction(
            hidden_blocks,
            base_logits_blocks,
            target_blocks,
            previous_token_ids,
        )
        logits = corrected_logits.reshape(1, num_blocks * block_size, -1)

        confidence_logits = None
        if self.confidence_head is not None:
            confidence_features = hidden_blocks
            if self.config.confidence_head_with_correction:
                confidence_features = torch.cat(
                    [hidden_blocks, correction_states.to(hidden_blocks.dtype)], dim=-1
                )
            confidence_logits = self.confidence_head(confidence_features).reshape(
                1, num_blocks * block_size
            )

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
