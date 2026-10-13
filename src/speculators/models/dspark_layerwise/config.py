from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dspark.config import DSparkSpeculatorConfig

__all__ = [
    "DSparkLayerwiseSpeculatorConfig",
]


@SpeculatorModelConfig.register("dspark_layerwise")
class DSparkLayerwiseSpeculatorConfig(DSparkSpeculatorConfig):
    """DSpark config with DFlare-style per-draft-layer target fusion.

    The DSpark Markov and confidence heads are inherited unchanged.  Only the
    DFlash backbone conditioning path is replaced by adaptive layer-wise fusion
    and heterogeneous target/draft KV projections.
    """

    speculators_model_type: Literal["dspark_layerwise"] = "dspark_layerwise"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DSparkLayerwiseDraftModel"],
        description="Model architectures that can load these weights",
    )
    fusion_init_scale: float = Field(
        default=2.0,
        ge=0.0,
        description=(
            "Initial logit assigned to the target layer preferred by each draft "
            "layer. Other fusion logits start at zero."
        ),
    )
