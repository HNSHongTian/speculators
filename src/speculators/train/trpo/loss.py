"""TRPO loss combination helpers."""

from __future__ import annotations

import torch


def trpo_loss_from_groups(
    *,
    advantages: torch.Tensor,
    log_probs: torch.Tensor,
    gold_log_probs: torch.Tensor | None = None,
    lambda_ce: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L = -mean(A * logπ) + λ_ce * NLL(gold)."""
    metrics: dict[str, float] = {}
    if advantages.numel() == 0:
        pg = log_probs.new_zeros(())
    else:
        pg = -(advantages * log_probs).mean()
    metrics["trpo_pg"] = float(pg.detach().item())

    if gold_log_probs is None or gold_log_probs.numel() == 0 or lambda_ce == 0.0:
        ce = log_probs.new_zeros(())
    else:
        ce = -gold_log_probs.mean()
    metrics["trpo_ce"] = float(ce.detach().item())
    metrics["trpo_lambda_ce"] = float(lambda_ce)

    loss = pg + float(lambda_ce) * ce
    metrics["trpo_loss"] = float(loss.detach().item())
    return loss, metrics
