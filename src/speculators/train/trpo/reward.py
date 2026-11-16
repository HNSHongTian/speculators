"""Node rewards and gold-path merge for TRPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from speculators.tree.types import PackedTree


@dataclass
class TrainTree:
    """Packed draft tree plus optional gold nodes (training only)."""

    tokens: list[int]
    parents: list[int]  # -1 = anchor; else index into this list
    depths: list[int]
    path_scores: list[float]
    is_gold: list[bool]
    # For each depth, pool/top-k node indices in this list (draft only).
    depth_topk: dict[int, list[int]]


def merge_gold_path(
    packed: PackedTree,
    *,
    gold_token_ids: list[int],
    top_k: int,
) -> TrainTree:
    """Attach a gold chain after packing. Dedup same-parent same-token."""
    tokens = [int(x) for x in packed.tokens.detach().cpu().tolist()]
    # packed parent_indices use 0=anchor, i+1=node i → convert to list index / -1
    raw_parents = [int(x) for x in packed.parent_indices.detach().cpu().tolist()]
    parents = [(-1 if p <= 0 else p - 1) for p in raw_parents]
    depths = [int(x) for x in packed.depths.detach().cpu().tolist()]
    scores = [float(x) for x in packed.path_scores.detach().cpu().tolist()]
    is_gold = [False] * len(tokens)

    # Per-depth draft top-k by path_score among packed draft nodes.
    by_depth: dict[int, list[int]] = {}
    for i, d in enumerate(depths):
        by_depth.setdefault(d, []).append(i)
    depth_topk: dict[int, list[int]] = {}
    for d, idxs in by_depth.items():
        idxs_sorted = sorted(idxs, key=lambda i: scores[i], reverse=True)
        depth_topk[d] = idxs_sorted[: max(1, int(top_k))]

    # Walk / extend gold chain from anchor.
    parent_idx = -1
    for depth, gold_tok in enumerate(gold_token_ids):
        existing = None
        for i, (t, p, dg) in enumerate(zip(tokens, parents, depths, strict=True)):
            if p == parent_idx and t == int(gold_tok) and dg == depth:
                existing = i
                break
        if existing is None:
            tokens.append(int(gold_tok))
            parents.append(parent_idx)
            depths.append(depth)
            # Gold path_score unused for packing; place high for stable ties.
            scores.append(0.0)
            is_gold.append(True)
            parent_idx = len(tokens) - 1
        else:
            # Draft already proposed the gold token — still mark for G_d / CE.
            is_gold[existing] = True
            parent_idx = existing

    return TrainTree(
        tokens=tokens,
        parents=parents,
        depths=depths,
        path_scores=scores,
        is_gold=is_gold,
        depth_topk=depth_topk,
    )


def compute_node_rewards(
    tree: TrainTree,
    *,
    gold_token_ids: list[int],
) -> list[float]:
    """R(v) = max accept length through v, else 0 if prefix already mismatches."""
    n = len(tree.tokens)
    if n == 0:
        return []

    # Build children lists.
    children: list[list[int]] = [[] for _ in range(n)]
    roots: list[int] = []
    for i, p in enumerate(tree.parents):
        if p < 0:
            roots.append(i)
        else:
            children[p].append(i)

    # Prefix match length from root for each node (0 if unreachable / mismatch).
    prefix_ok = [False] * n
    accept_from_root = [0] * n  # matched tokens on path root→v inclusive if ok

    def _match_depth(node: int) -> int | None:
        """Return matched length along unique path if continuous match else None."""
        path: list[int] = []
        cur = node
        while cur >= 0:
            path.append(cur)
            cur = tree.parents[cur]
        path.reverse()
        matched = 0
        for depth, idx in enumerate(path):
            if depth >= len(gold_token_ids):
                return None
            if tree.tokens[idx] != int(gold_token_ids[depth]):
                return None
            matched = depth + 1
        return matched

    for i in range(n):
        m = _match_depth(i)
        if m is None:
            prefix_ok[i] = False
            accept_from_root[i] = 0
        else:
            prefix_ok[i] = True
            accept_from_root[i] = m

    # R(v) = max over leaves/ends in subtree of accept length, if v itself ok.
    # Equivalently: max accept_from_root over nodes in subtree that are ok
    # (including v). For a correct prefix node, deeper correct descendants
    # raise the max.
    rewards = [0.0] * n

    def _dfs(node: int) -> float:
        if not prefix_ok[node]:
            rewards[node] = 0.0
            for c in children[node]:
                _dfs(c)
            return 0.0
        best = float(accept_from_root[node])
        for c in children[node]:
            best = max(best, _dfs(c))
        rewards[node] = best
        return best

    for r in roots:
        _dfs(r)
    return rewards
