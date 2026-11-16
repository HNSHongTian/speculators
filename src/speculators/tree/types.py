from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TreeNode:
    """One candidate / packed tree node (non-root)."""

    token_id: int
    parent: int  # pool index of parent, or -1 if child of anchor
    depth: int
    path_score: float


@dataclass
class PackedTree:
    """Prefix-closed packed tree ready for verify / TRPO."""

    tokens: torch.Tensor  # [N]
    parent_indices: torch.Tensor  # [N], packed indexing: 0=anchor, i+1=node i
    depths: torch.Tensor  # [N]
    path_scores: torch.Tensor  # [N]
    pool_indices: torch.Tensor  # [N], indices into the original candidate pool
