from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dspark.config import DSparkSpeculatorConfig

__all__ = [
    "DSparkDeepSpeculatorConfig",
]


@SpeculatorModelConfig.register("dspark_deep")
class DSparkDeepSpeculatorConfig(DSparkSpeculatorConfig):
    """Independent DSpark model type for draft-depth scaling experiments."""

    speculators_model_type: Literal["dspark_deep"] = "dspark_deep"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DSparkDeepDraftModel"],
        description="Model architectures that can load these weights",
    )
    minimum_draft_layers: int = Field(
        default=6,
        ge=2,
        description=(
            "Minimum decoder depth accepted by this depth-scaling variant. "
            "DFlare-style experiments normally use seven draft layers."
        ),
    )
