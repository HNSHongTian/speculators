"""E2E-TV on fixed sampled histories (not a score-function estimator)."""

import torch
import torch.distributed as dist
from torch.nn.functional import binary_cross_entropy_with_logits


def specopd_loss(overlap, valid, confidence=None, confidence_alpha=1.0):
    """Position-normalized loss; DDP averages globally counted gradients."""
    if overlap.shape != valid.shape or overlap.ndim != 2:  # noqa: PLR2004
        raise ValueError("Expected [branches, positions] overlap and valid mask")
    if torch.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("SpecOPD validity must be prefix closed")
    overlap = overlap.float()
    survival = torch.where(valid, overlap, torch.ones_like(overlap)).cumprod(-1)
    count = valid.sum().to(overlap.dtype)
    main = ((1 - survival) * valid).sum()
    conf = main * 0
    if confidence is not None:
        conf = (
            binary_cross_entropy_with_logits(
                confidence.float(), overlap.detach(), reduction="none"
            )
            * valid
        ).sum()
    numerator = main + confidence_alpha * conf
    global_count = count.detach().clone()
    world = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_count)
        world = dist.get_world_size()
    loss = numerator * world / global_count.clamp_min(1)
    metrics = {
        "loss_sum": numerator.detach(),
        "loss_total": count.detach(),
        "specopd_e2e_loss_sum": main.detach(),
        "specopd_e2e_loss_total": count.detach(),
        "specopd_confidence_loss_sum": conf.detach(),
        "specopd_confidence_loss_total": count.detach(),
        "specopd_overlap_sum": (overlap.detach() * valid).sum(),
        "specopd_overlap_total": count.detach(),
        "specopd_surrogate_length_sum": (survival.detach() * valid).sum(),
        "specopd_surrogate_length_total": valid.any(-1).sum().float(),
        "specopd_supervised_tokens": count.detach(),
        "specopd_branches": valid.any(-1).sum().float(),
        "specopd_global_positions": global_count,
    }
    for i in range(overlap.shape[1]):
        metrics[f"specopd_survival_{i + 1}_sum"] = (
            survival[:, i].detach() * valid[:, i]
        ).sum()
        metrics[f"specopd_survival_{i + 1}_total"] = valid[:, i].sum().float()
    return loss, metrics
