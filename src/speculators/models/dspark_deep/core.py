"""Depth-scaled DSpark training model.

This variant intentionally keeps DSpark's original shared target projection and
attention implementation.  Its only experimental variable is decoder depth.
"""

from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark_deep.config import DSparkDeepSpeculatorConfig

__all__ = [
    "DSparkDeepDraftModel",
]


@SpeculatorModel.register("dspark_deep")
class DSparkDeepDraftModel(DSparkDraftModel):
    """DSpark isolated under a new model type with an enforced deeper decoder."""

    config_class: ClassVar[type[DSparkDeepSpeculatorConfig]] = (  # type: ignore[misc,assignment]
        DSparkDeepSpeculatorConfig
    )

    def __init__(self, config: DSparkDeepSpeculatorConfig) -> None:
        depth = config.transformer_layer_config.num_hidden_layers
        if depth < config.minimum_draft_layers:
            raise ValueError(
                "dspark_deep requires transformer_layer_config.num_hidden_layers "
                f">= {config.minimum_draft_layers}, got {depth}. Pass "
                "--num-layers 7 for the DFlare-style depth experiment."
            )
        super().__init__(config=config)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DSparkDeepDraftModel":
        enable_confidence_head = kwargs.get("enable_confidence_head")
        confidence_with_markov = kwargs.get("confidence_head_with_markov")
        config = DSparkDeepSpeculatorConfig(
            **cls._build_base_config_kwargs("dspark", verifier_config, **kwargs),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_confidence_head=(
                True if enable_confidence_head is None else enable_confidence_head
            ),
            confidence_head_with_markov=(
                True if confidence_with_markov is None else confidence_with_markov
            ),
            minimum_draft_layers=kwargs.get("minimum_draft_layers", 6),
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model
