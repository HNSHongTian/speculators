"""Counted response-prefix diagnostics on the original, pre-gold packed tree.

These are training proxies against stored response tokens, not an online
verifier run or DSpark's distribution-overlap acceptance estimate.
"""

import torch

from speculators.tree.types import PackedTree


def empty_tree_metrics(depth: int) -> dict[str, float]:
    names = [
        "tree_accept_len",
        "tree_accepted_draft_len",
        "tree_accept_rate",
        "tree_valid_depth_mean",
        "tree_full_accept_rate",
    ]
    for position in range(depth):
        names.extend(
            [
                f"tree_position_{position}_accept_rate",
                f"tree_position_{position}_survival_rate",
            ]
        )
    return {f"{name}_{suffix}": 0.0 for name in names for suffix in ("sum", "total")}


@torch.no_grad()
def valid_response_suffix(input_ids, loss_mask, document_ids, *, anchor, depth):
    """Stop at sequence end, document boundary, or first unsupervised token."""
    stop = min(anchor + 1 + depth, input_ids.numel())
    ids = input_ids.reshape(-1)[anchor + 1 : stop].detach().cpu().tolist()
    masks = loss_mask.reshape(-1)[anchor + 1 : stop].detach().cpu().tolist()
    docs = document_ids.reshape(-1)[anchor:stop].detach().cpu().tolist()
    suffix = []
    for token, mask, document in zip(ids, masks, docs[1:], strict=True):
        if mask <= 0 or document != docs[0]:
            break
        suffix.append(int(token))
    return suffix


@torch.no_grad()
def add_tree_metrics(counts: dict[str, float], tree: PackedTree, suffix: list[int]):
    """Accumulate numerator/denominator, preserving weighting across ranks."""
    horizon = len(suffix)
    if not horizon:
        return
    tokens = tree.tokens.detach().cpu().tolist()
    parents = tree.parent_indices.detach().cpu().tolist()
    children = {}
    for node, (parent, token) in enumerate(zip(parents, tokens, strict=True), start=1):
        children.setdefault(parent, []).append((node, token))
    # Keep all matching children (also correct if input contains duplicates).
    frontier, accepted = {0}, 0
    for token in suffix:
        frontier = {
            node
            for parent in frontier
            for node, value in children.get(parent, [])
            if value == token
        }
        if not frontier:
            break
        accepted += 1

    def add(name, numerator, denominator):
        counts[f"{name}_sum"] += float(numerator)
        counts[f"{name}_total"] += float(denominator)

    add("tree_accepted_draft_len", accepted, 1)
    add("tree_accept_len", accepted + 1, 1)
    add("tree_accept_rate", accepted, horizon)
    add("tree_valid_depth_mean", horizon, 1)
    add("tree_full_accept_rate", accepted == horizon, 1)
    for position in range(horizon):
        matched = accepted > position
        add(f"tree_position_{position}_accept_rate", matched, accepted >= position)
        add(f"tree_position_{position}_survival_rate", matched, 1)
