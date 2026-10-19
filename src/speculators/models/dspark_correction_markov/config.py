from typing import Literal

from pydantic import Field, model_validator

from speculators import SpeculatorModelConfig
from speculators.models.dspark.config import DSparkSpeculatorConfig

__all__ = ["DSparkCorrectionMarkovSpeculatorConfig"]


@SpeculatorModelConfig.register("dspark_correction_markov")
class DSparkCorrectionMarkovSpeculatorConfig(DSparkSpeculatorConfig):
    """DSpark with both causal Correction and the native Markov/RNN head."""

    speculators_model_type: Literal["dspark_correction_markov"] = (  # type: ignore[assignment]
        "dspark_correction_markov"
    )
    architectures: list[str] = Field(
        default_factory=lambda: ["DSparkCorrectionMarkovDraftModel"],
        description="Model architectures that can load these weights",
    )

    correction_output_mode: Literal["hidden", "logits"] = Field(
        default="hidden",
        description=(
            "Correction output space. 'hidden' adds a residual before the LM head; "
            "'logits' adds a low-rank vocabulary residual to DFlash logits."
        ),
    )
    correction_hidden_size: int = Field(default=512, gt=0)
    correction_rank: int = Field(default=256, gt=0)
    correction_num_layers: int = Field(default=1, gt=0)
    correction_num_heads: int = Field(default=8, gt=0)
    correction_gate_bias: float = Field(default=0.0)
    correction_markov_gate_bias: float = Field(
        default=-2.0,
        description="Initial local gate bias for the collaborative Markov residual.",
    )
    correction_markov_scale_init: float = Field(
        default=0.0,
        description="Initial pre-tanh global scale for the Markov residual.",
    )
    confidence_head_with_correction: bool = Field(
        default=True,
        description="Append Correction causal states to confidence features.",
    )
    correction_hidden_aux_loss: bool = Field(
        default=False,
        description="Align corrected hidden states with verifier pre-LM hidden.",
    )
    correction_hidden_aux_weight: float = Field(default=0.1, ge=0.0)

    @model_validator(mode="after")
    def validate_combined_heads(self) -> "DSparkCorrectionMarkovSpeculatorConfig":
        if self.markov_rank <= 0:
            raise ValueError(
                "dspark_correction_markov requires markov_rank to be greater than 0"
            )
        if self.correction_hidden_size % self.correction_num_heads:
            raise ValueError(
                "correction_hidden_size must be divisible by correction_num_heads"
            )
        if self.correction_hidden_aux_loss and self.correction_output_mode != "hidden":
            raise ValueError(
                "correction_hidden_aux_loss is supported only in hidden mode"
            )
        return self
