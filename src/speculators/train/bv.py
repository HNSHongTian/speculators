"""Training-only BV scheduling, resume validation and block-count reduction."""

import math

import torch
import torch.distributed as dist

from speculators.train.distributed import get_dp_group

BV_RECIPE = "integrated-log-v1-vocab-floor-complete-blocks"


def bv_progress(
    epoch: int, local_step: int, steps_in_epoch: int, anneal_epochs: float
) -> tuple[float, bool]:
    if not math.isfinite(anneal_epochs) or anneal_epochs < 0:
        raise ValueError("bv_anneal_epochs must be finite and nonnegative")
    if steps_in_epoch <= 0 or not 1 <= local_step <= steps_in_epoch:
        raise ValueError("BV progress requires a valid full-epoch step index")
    progress = epoch + (local_step - 1) / steps_in_epoch
    annealing = anneal_epochs > 0 and progress < anneal_epochs
    beta = min(progress / anneal_epochs, 1.0) if anneal_epochs else 1.0
    return beta, annealing


def validate_bv_resume(saved: dict, recipe: dict) -> None:
    if saved.get("recipe") != recipe:
        raise ValueError(
            "BV checkpoint recipe/data configuration differs or is missing; "
            "use a new save directory for a new experiment"
        )


def scale_bv_for_backward(
    local_mean: torch.Tensor, local_count: torch.Tensor
) -> tuple[torch.Tensor, bool]:
    """Compensate averaged DP gradients for unequal valid-block counts.

    All ranks call this, including empty ranks. Backward must still run on an
    all-empty batch to complete DDP's reducer; only optimizer/scheduler skip.
    """
    count = local_count.detach().clone()
    world = 1
    if dist.is_available() and dist.is_initialized():
        group = get_dp_group()
        world = dist.get_world_size(group)
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=group)
    scaled = local_mean * (local_count * world / count.clamp_min(1))
    return scaled, bool(count.item() > 0)
