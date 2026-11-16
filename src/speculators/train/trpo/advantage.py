"""Per-depth relative advantages over top-k ∪ gold."""

from __future__ import annotations

import torch

from speculators.train.trpo.reward import TrainTree


def normalize_group(
    values: list[float],
    *,
    eps: float = 1e-6,
) -> list[float]:
    if not values:
        return []
    t = torch.tensor(values, dtype=torch.float32)
    if t.numel() == 1:
        return [0.0]
    std = float(t.std(unbiased=False).item())
    if std <= eps:
        return [0.0] * len(values)
    mean = float(t.mean().item())
    return [float((v - mean) / std) for v in values]


def compute_depth_advantages(
    tree: TrainTree,
    rewards: list[float],
    *,
    eps: float = 1e-6,
) -> list[float]:
    """A(v) for nodes in G_d = depth top-k ∪ gold(d); others stay 0."""
    n = len(tree.tokens)
    advantages = [0.0] * n
    gold_by_depth: dict[int, list[int]] = {}
    for i, (d, g) in enumerate(zip(tree.depths, tree.is_gold, strict=True)):
        if g:
            gold_by_depth.setdefault(d, []).append(i)

    depths = sorted(set(tree.depths) | set(tree.depth_topk) | set(gold_by_depth))
    for d in depths:
        group: list[int] = []
        seen: set[int] = set()
        for i in tree.depth_topk.get(d, []):
            if i not in seen:
                group.append(i)
                seen.add(i)
        for i in gold_by_depth.get(d, []):
            if i not in seen:
                group.append(i)
                seen.add(i)
        if not group:
            continue
        group_r = [rewards[i] for i in group]
        group_a = normalize_group(group_r, eps=eps)
        for i, a in zip(group, group_a, strict=True):
            advantages[i] = a
    return advantages
