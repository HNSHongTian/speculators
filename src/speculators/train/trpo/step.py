"""One-batch TRPO objective on top of a DSpark/DFlash draft backbone."""

from __future__ import annotations

from typing import Any

import torch

from speculators.train.trpo.advantage import compute_depth_advantages
from speculators.train.trpo.expand import expand_and_pack_tree, replay_node_logprobs
from speculators.train.trpo.loss import trpo_loss_from_groups
from speculators.train.trpo.metrics import (
    add_tree_metrics,
    empty_tree_metrics,
    valid_response_suffix,
)
from speculators.train.trpo.reward import compute_node_rewards, merge_gold_path
from speculators.tree.pack import PackMode


def _sample_from_anchor(draft: Any) -> bool:
    return bool(getattr(getattr(draft, "config", None), "sample_from_anchor", False))


def _first_draft_slot(draft: Any) -> int:
    return 0 if _sample_from_anchor(draft) else 1


def _proposal_tokens(draft: Any) -> int:
    block = int(draft.block_size)
    return block if _sample_from_anchor(draft) else max(1, block - 1)


def _gold_tokens_for_anchor(
    input_ids: torch.Tensor,
    *,
    anchor: int,
    proposal_tokens: int,
    sample_from_anchor: bool,
) -> list[int]:
    """Ground-truth continuation tokens used as gold + accept target."""
    # sample_from_anchor=True: slot k predicts token at anchor+k+1
    # sample_from_anchor=False: slot k (k>=1) predicts token at anchor+k
    start = int(anchor) + (1 if sample_from_anchor else 1)
    # For False, first draft slot is 1 predicting anchor+1 — same start.
    seq = input_ids.view(-1)
    out: list[int] = []
    for i in range(int(proposal_tokens)):
        pos = start + i
        if pos >= int(seq.numel()):
            break
        out.append(int(seq[pos].item()))
    return out


def compute_trpo_batch_loss(
    draft: Any,
    *,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    verifier_last_hidden_states: torch.Tensor,
    document_ids: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    max_anchors: int = 64,
    max_trpo_anchors: int = 8,
    tree_top_k: int = 4,
    tree_max_nodes: int = 32,
    tree_pack_mode: PackMode = "leaf-up",
    lambda_ce: float = 0.1,
    **kwargs: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run backbone once, then TRPO over a subset of anchors."""
    del kwargs
    hidden, base_logits, _targets, aligned_loss_mask, anchored_block_indices = (
        draft._backbone_forward(  # noqa: SLF001
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            position_ids,
            max_anchors=max_anchors,
            compute_verifier_targets=False,
        )
    )

    block = int(draft.block_size)
    num_blocks = int(base_logits.shape[1] // block)
    base_blocks = base_logits.view(num_blocks, block, -1)
    hidden_blocks = hidden.view(num_blocks, block, -1)
    mask_blocks = aligned_loss_mask.view(num_blocks, block)

    # Recover anchor positions from block indices (slot 0 of each block).
    anchor_positions = anchored_block_indices.view(num_blocks, block)[:, 0]
    valid = mask_blocks.any(dim=-1)

    valid_idx = torch.nonzero(valid, as_tuple=False).view(-1)
    # Fixed keys even for empty ranks, so distributed metric reductions align.
    diagnostics = empty_tree_metrics(_proposal_tokens(draft))
    empty_metrics = dict.fromkeys(
        ("trpo_loss", "trpo_anchors", "trpo_nodes", "trpo_gold_nodes",
         "trpo_mean_reward_gold", "trpo_pg", "trpo_ce"), 0.0,
    )
    if valid_idx.numel() == 0:
        zero = base_logits.sum() * 0.0
        return zero, {**empty_metrics, **diagnostics}

    k = min(int(max_trpo_anchors), int(valid_idx.numel()))
    # Deterministic subset: first k valid (already sorted by select_anchors).
    chosen = valid_idx[:k]

    first_slot = _first_draft_slot(draft)
    prop_len = _proposal_tokens(draft)
    sfa = _sample_from_anchor(draft)

    losses: list[torch.Tensor] = []
    loss_statistics: list[torch.Tensor] = []
    metrics_acc: dict[str, float] = {
        **empty_metrics,
        "trpo_anchors": float(k),
        "trpo_nodes": 0.0,
        "trpo_gold_nodes": 0.0,
        "trpo_mean_reward_gold": 0.0,
    }
    gold_reward_sum = 0.0
    gold_reward_count = 0

    for bi in chosen.tolist():
        anchor = int(anchor_positions[bi].item())
        gold = _gold_tokens_for_anchor(
            input_ids,
            anchor=anchor,
            proposal_tokens=prop_len,
            sample_from_anchor=sfa,
        )
        if len(gold) < 1:
            continue

        packed = expand_and_pack_tree(
            draft,
            anchor_token_id=int(input_ids[0, anchor].item()),
            block_base_logits=base_blocks[bi],
            block_hidden=hidden_blocks[bi],
            first_draft_slot=first_slot,
            max_proposal_tokens=min(prop_len, len(gold)),
            top_k=tree_top_k,
            max_nodes=tree_max_nodes,
            pack_mode=tree_pack_mode,
        )
        # Inspect only the model's packed tree, before injecting any gold nodes.
        suffix = valid_response_suffix(
            input_ids, loss_mask, document_ids, anchor=anchor, depth=prop_len,
        )
        add_tree_metrics(diagnostics, packed, suffix)
        tree = merge_gold_path(packed, gold_token_ids=gold, top_k=tree_top_k)
        rewards = compute_node_rewards(tree, gold_token_ids=gold)
        advantages = compute_depth_advantages(tree, rewards)

        # Active set = union of all G_d (nodes with nonzero group membership).
        active: list[int] = []
        seen: set[int] = set()
        for d, idxs in tree.depth_topk.items():
            for i in idxs:
                if i not in seen:
                    active.append(i)
                    seen.add(i)
        for i, g in enumerate(tree.is_gold):
            if g and i not in seen:
                active.append(i)
                seen.add(i)
            if g:
                gold_reward_sum += rewards[i]
                gold_reward_count += 1

        if not active:
            continue

        adv_t = torch.tensor(
            [advantages[i] for i in active],
            dtype=torch.float32,
            device=base_logits.device,
        )
        logps = replay_node_logprobs(
            draft,
            tree_tokens=tree.tokens,
            tree_parents=tree.parents,
            tree_depths=tree.depths,
            active_indices=active,
            anchor_token_id=int(input_ids[0, anchor].item()),
            block_base_logits=base_blocks[bi],
            block_hidden=hidden_blocks[bi],
            first_draft_slot=first_slot,
        )
        # Reuse the same graph: PG and CE gradients accumulate automatically.
        gold_positions = [j for j, i in enumerate(active) if tree.is_gold[i]]
        gold_logps = logps[gold_positions] if gold_positions else None

        loss_i, m_i = trpo_loss_from_groups(
            advantages=adv_t,
            log_probs=logps,
            gold_log_probs=gold_logps,
            lambda_ce=lambda_ce,
            tensor_metrics=True,
        )
        losses.append(loss_i)
        metrics_acc["trpo_nodes"] += float(len(active))
        metrics_acc["trpo_gold_nodes"] += float(len(gold_positions))
        loss_statistics.append(torch.stack([m_i["trpo_pg"], m_i["trpo_ce"]]))

    if not losses:
        zero = base_logits.sum() * 0.0
        return zero, {**metrics_acc, "trpo_loss": 0.0, **diagnostics}

    loss = torch.stack(losses).mean()
    n = float(len(losses))
    # One host transfer for batch loss statistics, instead of per-anchor reads.
    pg_ce = torch.stack(loss_statistics).mean(dim=0)
    loss_value, pg_value, ce_value = torch.cat(
        [loss.detach().reshape(1), pg_ce]
    ).cpu().tolist()
    metrics_acc["trpo_loss"] = loss_value
    metrics_acc["trpo_pg"] = pg_value
    metrics_acc["trpo_ce"] = ce_value
    metrics_acc["trpo_nodes"] = metrics_acc["trpo_nodes"] / n
    metrics_acc["trpo_gold_nodes"] = metrics_acc["trpo_gold_nodes"] / n
    if gold_reward_count:
        metrics_acc["trpo_mean_reward_gold"] = gold_reward_sum / gold_reward_count
    # Keep a tiny graph attachment so unused backbone params stay connected
    # when all advantages are zero (still fine to step).
    loss = loss + 0.0 * base_logits.float().sum() * 0.0
    return loss, {**metrics_acc, **diagnostics}
