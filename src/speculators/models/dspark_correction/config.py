from typing import Literal

from pydantic import Field, model_validator

from speculators import SpeculatorModelConfig
from speculators.models.dspark.config import DSparkSpeculatorConfig

__all__ = ["DSparkCorrectionSpeculatorConfig"]


@SpeculatorModelConfig.register("dspark_correction")
class DSparkCorrectionSpeculatorConfig(DSparkSpeculatorConfig):
    """DSpark variant whose sequential refinement is a causal Correction head.

    The original DSpark model remains registered as ``dspark`` and continues to use
    its Markov head. This isolated variant reuses the DFlash backbone and DSpark
    training objective while replacing Markov refinement with Correction.
    """

    speculators_model_type: Literal["dspark_correction"] = "dspark_correction"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DSparkCorrectionDraftModel"],
        description="Model architectures that can load these weights",
    )

    # These inherited DSpark fields are intentionally disabled in this variant.
    markov_rank: int = Field(
        default=0,
        ge=0,
        description="Disabled for dspark_correction; Correction replaces Markov.",
    )
    confidence_head_with_markov: bool = Field(
        default=False,
        description=(
            "Disabled for dspark_correction; use confidence_head_with_correction."
        ),
    )

    correction_output_mode: Literal["hidden", "logits"] = Field(
        default="hidden",
        description=(
            "Correction output space. 'hidden' adds a residual before the LM head; "
            "'logits' adds a low-rank vocabulary bias to the DFlash base logits."
        ),
    )
    correction_hidden_size: int = Field(
        default=512,
        gt=0,
        description="Internal hidden width of the causal Correction head.",
    )
    correction_rank: int = Field(
        default=256,
        gt=0,
        description="Low-rank bottleneck used by the Correction output projection.",
    )
    correction_num_layers: int = Field(
        default=1,
        gt=0,
        description="Number of causal Transformer layers in Correction.",
    )
    correction_num_heads: int = Field(
        default=8,
        gt=0,
        description="Number of attention heads in each Correction layer.",
    )
    correction_gate_bias: float = Field(
        default=0.0,
        description="Initial bias of the sigmoid Correction residual gate.",
    )
    confidence_head_with_correction: bool = Field(
        default=True,
        description=(
            "Concatenate the Correction causal state with DFlash hidden states "
            "before the DSpark confidence head."
        ),
    )
    correction_hidden_aux_loss: bool = Field(
        default=False,
        description=(
            "Add SmoothL1 alignment between corrected hidden states and verifier "
            "pre-LM hidden states. Supported only in hidden output mode."
        ),
    )
    correction_hidden_aux_weight: float = Field(
        default=0.1,
        ge=0.0,
        description="Weight of the optional Correction hidden-alignment loss.",
    )

    @model_validator(mode="after")
    def validate_correction_options(self) -> "DSparkCorrectionSpeculatorConfig":
        if self.correction_hidden_size % self.correction_num_heads:
            raise ValueError(
                "correction_hidden_size must be divisible by correction_num_heads"
            )
        if self.correction_hidden_aux_loss and self.correction_output_mode != "hidden":
            raise ValueError(
                "correction_hidden_aux_loss is supported only in hidden mode"
            )
        if self.markov_rank != 0 or self.confidence_head_with_markov:
            raise ValueError(
                "dspark_correction replaces Markov; set markov_rank=0 and "
                "confidence_head_with_markov=False"
            )
        return self
