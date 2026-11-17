"""TRPO loss combination helpers."""

from __future__ import annotations

import torch


def trpo_loss_from_groups(
    *,
    advantages: torch.Tensor,
    log_probs: torch.Tensor,
    gold_log_probs: torch.Tensor | None = None,
    lambda_ce: float = 0.1,
    tensor_metrics: bool = False,
) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
    """L = -mean(A * logπ) + λ_ce * NLL(gold).

    ``tensor_metrics`` defers host synchronization for batch-level aggregation.
    Returned metric tensors are detached; the returned loss retains its graph.
    """
    if advantages.numel() == 0:
        pg = log_probs.new_zeros(())
    else:
        pg = -(advantages * log_probs).mean()

    if gold_log_probs is None or gold_log_probs.numel() == 0 or lambda_ce == 0.0:
        ce = log_probs.new_zeros(())
    else:
        ce = -gold_log_probs.mean()

    loss = pg + float(lambda_ce) * ce
    metrics = {
        "trpo_pg": pg.detach(),
        "trpo_ce": ce.detach(),
        "trpo_loss": loss.detach(),
    }
    if not tensor_metrics:
        values = torch.stack(list(metrics.values())).cpu().tolist()
        metrics = dict(zip(metrics, values, strict=True))
    metrics["trpo_lambda_ce"] = float(lambda_ce)
    return loss, metrics
