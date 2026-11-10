from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.bv_loss import build_bv_labels
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.loss_config import resolve_dspark_loss_config
from speculators.models.dspark.metrics import compute_metrics
from speculators.models.dspark.model_definitions import ConfidenceHead, MarkovHead
from speculators.models.dspark.tree_loss import (
    build_tree_loss_inputs,
    compute_tree_training_loss,
)
from speculators.models.metrics import LossConfig, kl_div_loss
from speculators.models.utils import conditional_torch_compile

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}

__all__ = [
    "DSparkDraftModel",
]


@torch.no_grad()
def _build_tree_anchor_selection_mask(
    *,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    document_ids: torch.Tensor,
    eos_token_ids: tuple[int, ...],
) -> torch.Tensor:
    """Select roots whose first predicted label is valid for tree training.

    DSpark slot zero predicts the token after the anchor.  Requiring both the
    anchor and that label to be trainable keeps the base DSpark objective and
    TreeSpark replay on the same population.  Document and EOS checks prevent
    roots whose first edge can never be verified.
    """
    selection_mask = torch.zeros_like(loss_mask, dtype=torch.bool)
    if loss_mask.shape[1] < 2:  # noqa: PLR2004
        return selection_mask

    root_docs = document_ids[:, :-1]
    label_docs = document_ids[:, 1:]
    valid = (
        loss_mask[:, :-1].bool()
        & loss_mask[:, 1:].bool()
        & (root_docs == label_docs)
        & (root_docs >= 0)
    )
    for eos_token_id in eos_token_ids:
        valid &= input_ids[:, :-1] != int(eos_token_id)
    selection_mask[:, :-1] = valid
    return selection_mask


@torch.no_grad()
def _build_tree_depth_valid_mask(
    *,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    document_ids: torch.Tensor,
    anchored_block_indices: torch.Tensor,
    aligned_loss_mask: torch.Tensor,
    block_size: int,
    sample_from_anchor: bool,
    eos_token_ids: tuple[int, ...],
) -> torch.Tensor:
    """Align tree supervision with the token predicted by each draft slot."""
    block_indices = anchored_block_indices.reshape(-1, block_size)
    first_slot = 0 if sample_from_anchor else 1
    label_indices = block_indices + int(sample_from_anchor)
    in_bounds = (label_indices >= 0) & (label_indices < input_ids.shape[1])
    safe_labels = label_indices.clamp(0, input_ids.shape[1] - 1)
    anchor_docs = document_ids[0, block_indices[:, :1]]
    prediction_indices = (safe_labels - 1).clamp(min=0)
    valid = (
        in_bounds
        & loss_mask[0, safe_labels].bool()
        & (document_ids[0, safe_labels] == anchor_docs)
        & (document_ids[0, prediction_indices] == anchor_docs)
        & (anchor_docs >= 0)
    )
    valid &= aligned_loss_mask.reshape(-1, block_size).bool().any(-1, keepdim=True)
    valid[:, :first_slot] = False

    if first_slot < block_size and eos_token_ids:
        labels = input_ids[0, safe_labels[:, first_slot:]]
        root = input_ids[0, block_indices[:, :1]]
        preceding = torch.cat([root, labels[:, :-1]], dim=-1)
        no_prior_eos = torch.ones_like(preceding, dtype=torch.bool)
        for eos_token_id in eos_token_ids:
            no_prior_eos &= preceding != int(eos_token_id)
        no_prior_eos = no_prior_eos.to(torch.int64).cumprod(dim=-1).bool()
        valid[:, first_slot:] &= no_prior_eos
    if first_slot < block_size:
        # A node can only be supervised when its complete path from the anchor
        # is valid.  This also prevents a later loss-mask island from reviving a
        # path after an invalid label or document boundary.
        valid[:, first_slot:] = (
            valid[:, first_slot:].to(torch.int64).cumprod(dim=-1).bool()
        )
    return valid


@SpeculatorModel.register("dspark")
class DSparkDraftModel(DFlashDraftModel):
    """DFlash backbone plus a Markov logit-bias head and a confidence head.

    After the base draft logits are produced, the Markov head biases position
    ``k`` using the previous block token and the confidence head predicts each
    position's acceptance probability. Everything else is inherited from DFlash.
    """

    config_class: ClassVar[type[DSparkSpeculatorConfig]] = DSparkSpeculatorConfig  # type: ignore[misc,assignment]

    def __init__(self, config: DSparkSpeculatorConfig) -> None:
        super().__init__(config=config)

        hidden_size = config.transformer_layer_config.hidden_size

        self.markov_head: MarkovHead | None = None
        if config.markov_rank > 0:
            self.markov_head = MarkovHead(
                verifier_vocab_size=self.verifier_vocab_size,
                draft_vocab_size=self.draft_vocab_size,
                markov_rank=config.markov_rank,
                hidden_size=hidden_size,
                head_type=config.markov_head_type,
            )

        self.confidence_head: ConfidenceHead | None = None
        if config.enable_confidence_head:
            if config.confidence_head_with_markov and self.markov_head is None:
                raise ValueError(
                    "confidence_head_with_markov=True requires markov_rank > 0."
                )
            input_dim = hidden_size + (
                config.markov_rank if config.confidence_head_with_markov else 0
            )
            self.confidence_head = ConfidenceHead(input_dim)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DSparkDraftModel":
        """Create a DSpark model from training arguments (mirrors DFlash)."""
        enable_confidence_head_arg = kwargs.get("enable_confidence_head")
        confidence_head_with_markov_arg = kwargs.get("confidence_head_with_markov")
        config = DSparkSpeculatorConfig(
            **cls._build_base_config_kwargs("dspark", verifier_config, **kwargs),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_confidence_head=(
                True
                if enable_confidence_head_arg is None
                else enable_confidence_head_arg
            ),
            confidence_head_with_markov=(
                True
                if confidence_head_with_markov_arg is None
                else confidence_head_with_markov_arg
            ),
            tree_training_enabled=kwargs.get("enable_tree_training", False),
            tree_loss_top_k=kwargs.get("tree_loss_top_k", 4),
            tree_loss_num_anchors=kwargs.get("tree_loss_num_anchors", 128),
            tree_loss_max_nodes=kwargs.get("tree_loss_max_nodes", 32),
            tree_rank_margin=kwargs.get("tree_rank_margin", 1.0),
            tree_rank_hinge_cap=kwargs.get("tree_rank_hinge_cap", 4.0),
            tree_rank_top_j=kwargs.get("tree_rank_top_j", 4),
            tree_survival_alpha=kwargs.get("tree_survival_alpha", 0.1),
            tree_rank_alpha=kwargs.get("tree_rank_alpha", 0.1),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve DSpark's compound loss from ``--loss-fn``."""
        loss_config, bv_weight, ce_weight, tv_weight = resolve_dspark_loss_config(
            kwargs["loss_fn"], enable_bv_loss=kwargs.get("enable_bv_loss", False)
        )
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 3072)
        confidence_head_alpha = kwargs.get("confidence_head_alpha", 1.0)
        per_position_loss_weight = kwargs.get(
            "per_position_loss_weight", "fixed-exp-decay"
        )
        dpace_alpha = kwargs.get("dpace_alpha", 0.5)
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "confidence_head_alpha": confidence_head_alpha,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
        }
        tree_training = {}
        if kwargs.get("enable_tree_training", False):
            tree_training.update(
                enable_tree_training=True,
                tree_loss_top_k=kwargs.get("tree_loss_top_k", 4),
                tree_loss_num_anchors=kwargs.get("tree_loss_num_anchors", 128),
                tree_loss_max_nodes=kwargs.get("tree_loss_max_nodes", 32),
                tree_rank_margin=kwargs.get("tree_rank_margin", 1.0),
                tree_rank_hinge_cap=kwargs.get("tree_rank_hinge_cap", 4.0),
                tree_rank_top_j=kwargs.get("tree_rank_top_j", 4),
                tree_survival_alpha=kwargs.get("tree_survival_alpha", 0.1),
                tree_rank_alpha=kwargs.get("tree_rank_alpha", 0.1),
            )
        if kwargs.get("enable_bv_loss", False):
            shared.update(
                enable_bv_loss=True,
                bv_chunk_size=kwargs.get("bv_chunk_size", 16),
                bv_loss_weight=bv_weight,
                ce_loss_weight=ce_weight,
                tv_loss_weight=tv_weight,
            )
        train_call_kwargs = dict(shared)
        train_call_kwargs.update(tree_training)
        return train_call_kwargs, dict(shared)

    @conditional_torch_compile
    def forward(  # noqa: C901, PLR0917
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_hidden*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 3072,
        confidence_head_alpha: float = 1.0,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        enable_bv_loss: bool = False,
        bv_beta: torch.Tensor | None = None,
        bv_annealing: bool = False,
        bv_chunk_size: int = 16,
        bv_loss_weight: float = 1.0,
        ce_loss_weight: float = 0.0,
        tv_loss_weight: float = 0.0,
        enable_tree_training: bool = False,
        tree_loss_top_k: int = 4,
        tree_loss_num_anchors: int = 128,
        tree_loss_max_nodes: int = 32,
        tree_rank_margin: float = 1.0,
        tree_rank_hinge_cap: float = 4.0,
        tree_rank_top_j: int = 4,
        tree_survival_alpha: float = 0.1,
        tree_rank_alpha: float = 0.1,
        **kwargs,
    ):
        if enable_bv_loss and enable_tree_training:
            raise ValueError(
                "BV and TreeSpark training use incompatible distributed loss scaling"
            )
        if enable_bv_loss and self.use_draft_vocab:
            raise ValueError("BV loss requires the full verifier vocabulary")
        eos = getattr(self.config.transformer_layer_config, "eos_token_id", None)
        eos_ids = () if eos is None else tuple(eos if isinstance(eos, list) else [eos])
        tree_alignment_enabled = bool(
            (enable_tree_training or self.config.tree_training_enabled)
            and not enable_bv_loss
        )
        backbone_kwargs = dict(kwargs)
        if tree_alignment_enabled:
            backbone_kwargs["anchor_selection_mask"] = (
                _build_tree_anchor_selection_mask(
                    input_ids=input_ids,
                    loss_mask=loss_mask,
                    document_ids=document_ids,
                    eos_token_ids=eos_ids,
                )
            )
        hidden, logits, targets, aligned_loss_mask, anchored_block_indices = (
            self._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                max_anchors=max_anchors,
                **backbone_kwargs,
            )
        )

        # DSpark: add the Markov logit bias and predict per-position confidence.
        num_blocks = max_anchors
        block = self.block_size
        mask_tokens_size = num_blocks * block
        # Ground-truth block tokens (verifier vocab); position 0 is the anchor.
        block_tokens = input_ids[0, anchored_block_indices].view(num_blocks, block)
        if self.config.sample_from_anchor:
            # With sample_from_anchor=True (DSpark default), slot k predicts
            # token p+k+1 and the inference Markov chain conditions slot k's
            # bias on the token at the previous position p+k.
            prev_token_ids = block_tokens
        else:
            # With sample_from_anchor=False (Dflash default), slot k predicts
            # token p+k, so the previous token within the block is
            # block_tokens[:, k-1] (shifted).
            prev_token_ids = torch.cat(
                [block_tokens[:, :1], block_tokens[:, :-1]], dim=1
            )  # [num_blocks, block]
        hidden_blocks = hidden.view(num_blocks, block, -1)
        base_logits = logits.view(num_blocks, block, -1)
        target_blocks = targets.view(num_blocks, block, -1)

        depth_valid_mask = None
        metrics_loss_mask = aligned_loss_mask
        if tree_alignment_enabled:
            depth_valid_mask = _build_tree_depth_valid_mask(
                input_ids=input_ids,
                loss_mask=loss_mask,
                document_ids=document_ids,
                anchored_block_indices=anchored_block_indices,
                aligned_loss_mask=aligned_loss_mask,
                block_size=block,
                sample_from_anchor=self.config.sample_from_anchor,
                eos_token_ids=eos_ids,
            )
            # Use the predicted-label mask for the base CE/TV/confidence terms
            # and their position metrics.  This is the same population used by
            # TreeSpark before its optional anchor subsample.
            metrics_loss_mask = depth_valid_mask.reshape(1, -1)

        confidence_logits = None
        prev_emb = None
        corrected_logits = base_logits
        if self.markov_head is not None:
            prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
            markov_bias = self.markov_head.block_bias(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_blocks,
                prev_emb=prev_emb,
            )
            corrected_logits = base_logits + markov_bias
        logits = corrected_logits.view(1, mask_tokens_size, -1)

        tree_loss = None
        tree_metrics = {}
        if enable_tree_training and self.training:
            if depth_valid_mask is None:
                raise RuntimeError("TreeSpark training requires an aligned depth mask")
            first_draft_slot = 0 if self.config.sample_from_anchor else 1
            tree_inputs = build_tree_loss_inputs(
                markov_head=self.markov_head,
                base_logits=base_logits,
                hidden_states=hidden_blocks,
                target_logits=target_blocks,
                anchor_token_ids=block_tokens[:, 0],
                depth_valid_mask=depth_valid_mask,
                block_keep_mask=depth_valid_mask.any(dim=-1),
                first_draft_slot=first_draft_slot,
                top_k=tree_loss_top_k,
                top_j=tree_rank_top_j,
                margin=tree_rank_margin,
                hinge_cap=tree_rank_hinge_cap,
                num_anchors=tree_loss_num_anchors,
                max_nodes=tree_loss_max_nodes,
                use_draft_vocab=self.use_draft_vocab,
                d2t=self.d2t,
                reference_top1_ids=corrected_logits[
                    :, first_draft_slot
                ].detach().argmax(dim=-1),
            )
            if tree_inputs is not None:
                tree_loss, tree_metrics = compute_tree_training_loss(
                    tree_inputs,
                    loss_decay_gamma=gamma,
                    survival_alpha=tree_survival_alpha,
                    coverage_alpha=tree_rank_alpha,
                )
                eligible = depth_valid_mask.any(dim=-1)
                tree_metrics["tree_anchor_eligible_ratio_sum"] = (
                    eligible.to(torch.float32).sum().detach()
                )
                tree_metrics["tree_anchor_eligible_ratio_total"] = (
                    eligible.new_tensor(eligible.numel(), dtype=torch.float32)
                )

        if self.confidence_head is not None:
            # confidence_head_with_markov requires markov_rank > 0 (enforced in
            # __init__), so prev_emb is always set when the flag is on.
            if self.config.confidence_head_with_markov and prev_emb is not None:
                conf_features = torch.cat(
                    [hidden_blocks, prev_emb.to(hidden_blocks.dtype)], dim=-1
                )
            else:
                conf_features = hidden_blocks
            confidence_logits = self.confidence_head(conf_features).reshape(
                1, mask_tokens_size
            )

        bv_kwargs = {}
        if enable_bv_loss:
            eos = getattr(self.config.transformer_layer_config, "eos_token_id", None)
            eos_ids = (
                () if eos is None else tuple(eos if isinstance(eos, list) else [eos])
            )
            labels, valid_blocks = build_bv_labels(
                input_ids,
                loss_mask,
                document_ids,
                anchored_block_indices,
                aligned_loss_mask,
                block,
                self.config.sample_from_anchor,
                eos_ids,
            )
            bv_kwargs = {
                "enable_bv_loss": True,
                "bv_labels": labels,
                "bv_valid_blocks": valid_blocks,
                "bv_beta": bv_beta,
                "bv_annealing": bv_annealing,
                "bv_chunk_size": bv_chunk_size,
                "bv_loss_weight": bv_loss_weight,
                "ce_loss_weight": ce_loss_weight,
                "tv_loss_weight": tv_loss_weight,
            }
        loss, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            metrics_loss_mask,
            self.block_size,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            gamma=gamma,
            confidence_head_alpha=confidence_head_alpha,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
            **bv_kwargs,
        )
        if tree_loss is not None:
            loss = loss + tree_loss
            metrics["loss_sum"] = metrics["loss_sum"] + tree_loss.detach()
            metrics.update(tree_metrics)
        return None, loss, metrics
