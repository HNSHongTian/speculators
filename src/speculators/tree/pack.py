"""Tree packing strategies: global score cutoff vs DeepTree (leaf-up)."""

from __future__ import annotations

from typing import Literal

import torch

from speculators.tree.types import PackedTree

PackMode = Literal["score", "leaf-up"]


def pack_tree_nodes(
    *,
    pool_tokens: torch.Tensor,
    pool_parents: torch.Tensor,
    pool_depths: torch.Tensor,
    pool_scores: torch.Tensor,
    max_nodes: int,
    top_k: int,
    mode: PackMode = "leaf-up",
) -> PackedTree:
    """Pack a candidate pool into a prefix-closed tree of size ``<= max_nodes``.

    Pool tensors are 1-D on the same device. ``pool_parents[i]`` is the pool
    index of the parent, or ``-1`` when the parent is the anchor.
    """
    if int(max_nodes) < 1:
        raise ValueError("max_nodes must be >= 1")
    num_pool = int(pool_tokens.numel())
    if num_pool == 0:
        device = pool_tokens.device
        empty = torch.empty(0, dtype=torch.long, device=device)
        return PackedTree(
            tokens=empty,
            parent_indices=empty,
            depths=empty,
            path_scores=torch.empty(0, dtype=pool_scores.dtype, device=device),
            pool_indices=empty,
        )

    if mode == "score":
        packed_idx = _pack_score(
            pool_depths=pool_depths,
            pool_scores=pool_scores,
            max_nodes=max_nodes,
        )
    elif mode == "leaf-up":
        packed_idx = _pack_leaf_up(
            pool_parents=pool_parents,
            pool_depths=pool_depths,
            pool_scores=pool_scores,
            max_nodes=max_nodes,
            top_k=top_k,
        )
    else:
        raise ValueError(f"Unknown pack mode: {mode}")

    return _materialize_packed(
        packed_idx=packed_idx,
        pool_tokens=pool_tokens,
        pool_parents=pool_parents,
        pool_depths=pool_depths,
        pool_scores=pool_scores,
    )


def _pack_score(
    *,
    pool_depths: torch.Tensor,
    pool_scores: torch.Tensor,
    max_nodes: int,
) -> torch.Tensor:
    num_pool = int(pool_scores.numel())
    num_selected = min(int(max_nodes), num_pool)
    order = torch.argsort(pool_depths, stable=True)
    order = order[torch.argsort(pool_scores[order], descending=True, stable=True)]
    return order[:num_selected].sort().values


def _pack_leaf_up(
    *,
    pool_parents: torch.Tensor,
    pool_depths: torch.Tensor,
    pool_scores: torch.Tensor,
    max_nodes: int,
    top_k: int,
) -> torch.Tensor:
    """DeepTree: bottom-up, per-depth top-k paths, root-to-leaf fill."""
    device = pool_scores.device
    num_pool = int(pool_scores.numel())
    k = max(1, int(top_k))
    max_depth = int(pool_depths.max().item()) if num_pool else -1

    packed_set: set[int] = set()
    parents_cpu = pool_parents.detach().cpu().tolist()
    depths_cpu = pool_depths.detach().cpu().tolist()
    scores_cpu = pool_scores.detach().float().cpu().tolist()

    for depth in range(max_depth, -1, -1):
        if len(packed_set) >= int(max_nodes):
            break
        candidates = [i for i, d in enumerate(depths_cpu) if d == depth]
        candidates.sort(key=lambda i: scores_cpu[i], reverse=True)
        for pool_i in candidates[:k]:
            if pool_i in packed_set:
                continue
            path: list[int] = []
            cur = pool_i
            while cur >= 0:
                path.append(cur)
                cur = parents_cpu[cur]
            # root-to-leaf
            for node in reversed(path):
                if node in packed_set:
                    continue
                if len(packed_set) >= int(max_nodes):
                    break
                packed_set.add(node)
            if len(packed_set) >= int(max_nodes):
                break

    if not packed_set:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.tensor(sorted(packed_set), dtype=torch.long, device=device)


def _materialize_packed(
    *,
    packed_idx: torch.Tensor,
    pool_tokens: torch.Tensor,
    pool_parents: torch.Tensor,
    pool_depths: torch.Tensor,
    pool_scores: torch.Tensor,
) -> PackedTree:
    device = pool_tokens.device
    num_pool = int(pool_tokens.numel())
    num_selected = int(packed_idx.numel())
    remap = torch.zeros(num_pool, dtype=torch.long, device=device)
    if num_selected:
        remap[packed_idx] = torch.arange(
            1, num_selected + 1, dtype=torch.long, device=device
        )
    selected_parents = pool_parents[packed_idx]
    parent_indices = torch.zeros(num_selected, dtype=torch.long, device=device)
    non_root = selected_parents >= 0
    if bool(non_root.any()):
        parent_indices[non_root] = remap[selected_parents[non_root]]
    return PackedTree(
        tokens=pool_tokens[packed_idx],
        parent_indices=parent_indices,
        depths=pool_depths[packed_idx],
        path_scores=pool_scores[packed_idx],
        pool_indices=packed_idx,
    )
