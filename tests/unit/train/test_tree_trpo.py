"""Unit tests for DeepTree pack + TRPO reward/advantage."""

from __future__ import annotations

import torch

from speculators.train.trpo.advantage import compute_depth_advantages, normalize_group
from speculators.train.trpo.reward import compute_node_rewards, merge_gold_path
from speculators.tree.pack import pack_tree_nodes
from speculators.tree.types import PackedTree


def test_normalize_group_zero_variance():
    assert normalize_group([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]
    assert normalize_group([1.0]) == [0.0]


def test_deeptree_prefers_deep_paths():
    # pool: shallow high score vs deep path
    # indices: 0(depth0), 1(depth1 child of 0), 2(depth2 child of 1),
    #          3(depth0 high score sibling)
    pool_tokens = torch.tensor([10, 11, 12, 99])
    pool_parents = torch.tensor([-1, 0, 1, -1])
    pool_depths = torch.tensor([0, 1, 2, 0])
    pool_scores = torch.tensor([-0.5, -1.0, -1.5, -0.1])  # 99 is shallow best
    packed = pack_tree_nodes(
        pool_tokens=pool_tokens,
        pool_parents=pool_parents,
        pool_depths=pool_depths,
        pool_scores=pool_scores,
        max_nodes=3,
        top_k=1,
        mode="leaf-up",
    )
    # depth=2 top-1 is node 2 → path 0,1,2 fills budget; shallow 99 dropped
    assert set(packed.tokens.tolist()) == {10, 11, 12}


def test_gold_merge_and_rewards():
    packed = PackedTree(
        tokens=torch.tensor([1, 9]),  # wrong branch at depth1
        parent_indices=torch.tensor([0, 1]),  # 0=anchor, 1=first node
        depths=torch.tensor([0, 1]),
        path_scores=torch.tensor([-0.1, -1.0]),
        pool_indices=torch.tensor([0, 1]),
    )
    gold = [1, 2, 3]
    tree = merge_gold_path(packed, gold_token_ids=gold, top_k=4)
    # depth0 token 1 exists; gold extends 2,3
    assert 2 in tree.tokens and 3 in tree.tokens
    rewards = compute_node_rewards(tree, gold_token_ids=gold)
    advantages = compute_depth_advantages(tree, rewards)
    # wrong node 9 should have R=0; gold nodes positive R
    wrong = tree.tokens.index(9)
    assert rewards[wrong] == 0.0
    gold_idx = [i for i, g in enumerate(tree.is_gold) if g]
    assert gold_idx
    assert all(rewards[i] > 0 for i in gold_idx)
    assert len(advantages) == len(tree.tokens)
