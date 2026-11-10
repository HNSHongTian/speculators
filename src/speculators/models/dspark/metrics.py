"""Loss and metrics for the DSpark draft model.

loss = compound_loss(logits, targets) + conf_alpha * BCE(confidence, accept_rate)

The confidence target ``accept_rate = sum_v min(q_v, p_v) = 1 - d_TV`` is the
analytical acceptance rate (the overlap ``tv_loss`` already computes).
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch.nn.functional import binary_cross_entropy_with_logits, softmax

from speculators.models.dspark.bv_loss import block_ce_loss, block_tv_loss, bv_loss
from speculators.models.metrics import (
    LossConfig,
    compound_loss,
    compute_accuracy_multi_step,
    dflash_loss_decay,
    dpace_loss_decay,
)

__all__ = [
    "compute_metrics",
]

_EPS = 1e-8


def _masked_decayed_mean(
    elementwise: torch.Tensor,  # [1, T]
    loss_mask: torch.Tensor,  # [1, T]
    pos_idx: torch.Tensor,  # [1, T]
    decay_fn: Callable[..., torch.Tensor] | None,
) -> torch.Tensor:
    """Masked, optionally position-decayed mean of a precomputed per-position term."""
    loss_mask = loss_mask.to(elementwise.dtype)
    weighted = elementwise * loss_mask
    if decay_fn is not None:
        weighted = weighted * decay_fn(
            pos_idx.to(weighted.dtype), elementwise_loss=elementwise
        )
    denominator = loss_mask.sum(dim=1) + _EPS
    return (weighted.sum(dim=1) / denominator).mean()


def compute_metrics(  # noqa: C901
    logits: torch.Tensor,  # [1, T, draft_vocab_size] (Markov-corrected)
    targets: torch.Tensor,  # [1, T, draft_vocab_size]
    confidence_logits: torch.Tensor | None,  # [1, T] or None
    loss_mask: torch.Tensor,  # [1, T]
    block_size: int,
    loss_config: LossConfig,
    gamma: float = 4.0,
    confidence_head_alpha: float = 1.0,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    sample_from_anchor: bool = True,
    enable_bv_loss: bool = False,
    bv_labels: torch.Tensor | None = None,
    bv_valid_blocks: torch.Tensor | None = None,
    bv_beta: torch.Tensor | None = None,
    bv_annealing: bool = False,
    bv_chunk_size: int = 16,
    bv_loss_weight: float = 1.0,
    ce_loss_weight: float = 0.0,
    tv_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Compute the DSpark loss and a metrics dict (``*_sum``/``*_total`` pairs)."""

    device = logits.device
    seq_len = logits.shape[1]
    pos_idx = (torch.arange(seq_len, device=device) % block_size).unsqueeze(0)
    start_pos = 0 if sample_from_anchor else 1
    if per_position_loss_weight == "dpace":
        decay_fn = partial(
            dpace_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
        )
    else:
        decay_fn = partial(
            dflash_loss_decay, gamma=gamma, sample_from_anchor=sample_from_anchor
        )

    bv_metrics = {}
    if enable_bv_loss:
        if bv_labels is None or bv_valid_blocks is None:
            raise ValueError("BV requires actual labels and complete-block validity")
        if per_position_loss_weight == "dpace":
            raise ValueError("BV loss cannot use D-PACE position weights")
        if bv_beta is None:
            bv_beta = logits.new_tensor(1.0, dtype=torch.float32)
        block_logits = logits.reshape(-1, block_size, logits.shape[-1])[:, start_pos:]
        block_targets = targets.reshape(-1, block_size, targets.shape[-1])[
            :, start_pos:
        ]
        raw_bv_loss, bv_metrics = bv_loss(
            block_logits,
            block_targets,
            bv_labels,
            bv_valid_blocks,
            bv_beta,
            annealing=bv_annealing,
            chunk_size=bv_chunk_size,
        )
        term_losses = {}
        # Diagnostics/confidence use the same complete blocks as the BV main loss.
        loss_mask = bv_valid_blocks[:, None].expand(-1, block_size).reshape(1, -1)
        loss_mask = loss_mask & (pos_idx >= start_pos)
        count = bv_metrics["bv_loss_total"]
        weighted_bv_sum = bv_loss_weight * bv_metrics["bv_loss_sum"]
        loss = bv_loss_weight * raw_bv_loss
        bv_metrics.update(
            bv_weighted_loss_sum=weighted_bv_sum,
            bv_weighted_loss_total=count.clone(),
            main_loss_sum=weighted_bv_sum.clone(),
            main_loss_total=count.clone(),
        )
        token_positions = torch.arange(block_size, device=device)[start_pos:]
        token_weights = dflash_loss_decay(
            token_positions.float(),
            gamma=gamma,
            sample_from_anchor=sample_from_anchor,
        )
        if ce_loss_weight > 0:
            ce, ce_sum = block_ce_loss(
                block_logits,
                block_targets,
                bv_valid_blocks,
                token_weights,
                chunk_size=bv_chunk_size,
            )
            loss = loss + ce_loss_weight * ce
            bv_metrics.update(
                ce_loss_sum=ce_sum,
                ce_loss_total=count.clone(),
                ce_weighted_loss_sum=ce_loss_weight * ce_sum,
                ce_weighted_loss_total=count.clone(),
            )
            bv_metrics["main_loss_sum"] += ce_loss_weight * ce_sum
        if tv_loss_weight > 0:
            tv, tv_sum = block_tv_loss(
                block_logits,
                block_targets,
                bv_valid_blocks,
                token_weights,
                chunk_size=bv_chunk_size,
            )
            loss = loss + tv_loss_weight * tv
            bv_metrics.update(
                tv_loss_sum=tv_sum,
                tv_loss_total=count.clone(),
                tv_weighted_loss_sum=tv_loss_weight * tv_sum,
                tv_weighted_loss_total=count.clone(),
            )
            bv_metrics["main_loss_sum"] += tv_loss_weight * tv_sum
        for name, weight in (
            ("bv", bv_loss_weight),
            ("ce", ce_loss_weight),
            ("tv", tv_loss_weight),
        ):
            bv_metrics[f"{name}_loss_weight_sum"] = count * weight
            bv_metrics[f"{name}_loss_weight_total"] = count.clone()
    else:
        loss, term_losses = compound_loss(
            logits,
            targets,
            loss_mask,
            pos_idx,
            loss_config=loss_config,
            decay_fn=decay_fn,
        )

    # Analytical per-position acceptance rate = distributional overlap.
    with torch.no_grad():
        draft_p = softmax(logits.float(), dim=-1)
        target_p = softmax(targets.float(), dim=-1)
        accept_rate = torch.minimum(draft_p, target_p).sum(dim=-1)  # [1, T]
        # Per-block cumulative acceptance product over the draft slots (slot 0
        # is the anchor), shared by the accept-length and calibration metrics.
        num_blocks = seq_len // block_size
        accept_blocks = accept_rate.view(num_blocks, block_size)
        draft_mask = loss_mask.to(accept_rate.dtype).view(num_blocks, block_size)[
            :, start_pos:
        ]
        accept_prefix = (accept_blocks[:, start_pos:] * draft_mask).cumprod(dim=-1)

    metrics: dict[str, Any] = {}
    if confidence_logits is not None:
        c_star = accept_rate.detach().to(confidence_logits.dtype)
        bce = binary_cross_entropy_with_logits(
            confidence_logits, c_star, reduction="none"
        )  # [1, T]
        conf_loss = _masked_decayed_mean(bce, loss_mask, pos_idx, decay_fn)
        loss = loss + confidence_head_alpha * conf_loss

        with torch.no_grad():
            mask_f = loss_mask.to(accept_rate.dtype)
            mask_total = mask_f.sum().clamp_min(1.0)
            conf_prob = confidence_logits.float().sigmoid()
            metrics["confidence_loss_sum"] = conf_loss.detach().clone()
            metrics["confidence_loss_total"] = torch.ones((), device=device)
            metrics["confidence_abs_error_sum"] = (
                (conf_prob - accept_rate).abs() * mask_f
            ).sum()
            metrics["confidence_abs_error_total"] = mask_total
            # Mean predicted vs. observed acceptance — a calibration sanity check.
            metrics["confidence_pred_mean_sum"] = (conf_prob * mask_f).sum()
            metrics["confidence_pred_mean_total"] = mask_total.clone()
            # Calibration of the cumulative acceptance product, which is what
            # dynamic draft-length thresholding consumes (signed pred - target).
            conf_prefix = (
                conf_prob.view(num_blocks, block_size)[:, start_pos:] * draft_mask
            ).cumprod(dim=-1)
            metrics["confidence_cumprod_bias_sum"] = (
                (conf_prefix - accept_prefix) * draft_mask
            ).sum()
            metrics["confidence_cumprod_bias_total"] = draft_mask.sum().clamp_min(1.0)

    ones = torch.ones((), device=device)
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    if enable_bv_loss:
        metrics.update(bv_metrics)
        count = bv_metrics["bv_loss_total"]
        # Log the actual objective, not the annealing backward surrogate.
        metrics["loss_sum"] = bv_metrics.get(
            "main_loss_sum", bv_metrics["bv_loss_sum"]
        ).clone()
        if confidence_logits is not None:
            metrics["loss_sum"] += confidence_head_alpha * conf_loss.detach() * count
            metrics["confidence_loss_weight_sum"] = count * confidence_head_alpha
            metrics["confidence_loss_weight_total"] = count.clone()
        metrics["loss_total"] = count.clone()
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones.clone()

    # Mean acceptance rate of the (Markov-corrected) drafter.
    with torch.no_grad():
        mask_f = loss_mask.to(accept_rate.dtype)
        metrics["accept_rate_sum"] = (accept_rate * mask_f).sum()
        metrics["accept_rate_total"] = mask_f.sum().clamp_min(1.0)

    # Expected accepted draft length per block (DSpark's tau): the cumulative
    # acceptance product summed over draft slots, plus the always-emitted anchor.
    with torch.no_grad():
        per_block_len = accept_prefix.sum(dim=-1) + 1.0
        block_valid = (draft_mask.sum(dim=-1) > 0).to(accept_rate.dtype)
        metrics["accept_len_sum"] = (per_block_len * block_valid).sum()
        metrics["accept_len_total"] = block_valid.sum().clamp_min(1.0)

    # Per-position greedy accuracy
    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)
    if enable_bv_loss:
        predicted = pred_ids.reshape(-1, block_size)[:, start_pos:]
        correct_labels = (predicted == bv_labels) & bv_valid_blocks[:, None]
        metrics["bv_sampled_label_acc_sum"] = correct_labels.sum().float()
        metrics["bv_sampled_label_acc_total"] = (
            bv_valid_blocks.sum().float() * predicted.shape[1]
        )
    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )
    metrics["full_acc_sum"] = correct_per_pos[start_pos:].sum()
    metrics["full_acc_total"] = total_per_pos[start_pos:].sum()
    for pos in range(start_pos, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]

    return loss, metrics
