"""Expand a DSpark-style Markov tree pool from one backbone block."""

from __future__ import annotations

from typing import Any

import torch

from speculators.tree.pack import PackMode, pack_tree_nodes
from speculators.tree.types import PackedTree


def _draft_ids_to_target(draft: Any, draft_ids: torch.Tensor) -> torch.Tensor:
    if not getattr(draft, "use_draft_vocab", False) or getattr(draft, "d2t", None) is None:
        return draft_ids
    return (draft_ids + draft.d2t.to(draft_ids.device)).long()


def _frontier_step_logits(
    draft: Any,
    *,
    depth_logits: torch.Tensor,
    depth_hidden: torch.Tensor,
    frontier_target_tokens: torch.Tensor,
    frontier_states: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    markov_head = getattr(draft, "markov_head", None)
    width = int(frontier_target_tokens.numel())
    if markov_head is None:
        return depth_logits.unsqueeze(0).expand(width, -1), None

    step_hidden = depth_hidden.unsqueeze(0).expand(width, -1)
    if getattr(markov_head, "head_type", None) == "rnn":
        previous = markov_head.prev_embeddings(frontier_target_tokens.long())
        previous = previous.to(markov_head.markov_w2.weight.dtype)
        if frontier_states is None:
            frontier_states = previous.new_zeros(width, markov_head.markov_rank)
        else:
            frontier_states = frontier_states.to(previous.dtype)
        recurrent_input = torch.cat(
            [frontier_states, previous, step_hidden.to(previous.dtype)],
            dim=-1,
        )
        gate_raw, candidate_raw, output_raw = markov_head.joint_proj(
            recurrent_input
        ).chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        new_states = gate * frontier_states + (1.0 - gate) * torch.tanh(candidate_raw)
        bias = markov_head.markov_w2(torch.tanh(output_raw))
        return depth_logits.unsqueeze(0) + bias, new_states

    bias = markov_head.block_bias(
        prev_token_ids=frontier_target_tokens.reshape(-1, 1),
        hidden_states=step_hidden.unsqueeze(1),
    )[:, 0, :]
    return depth_logits.unsqueeze(0) + bias, None


def expand_and_pack_tree(
    draft: Any,
    *,
    anchor_token_id: int,
    block_base_logits: torch.Tensor,
    block_hidden: torch.Tensor,
    first_draft_slot: int,
    max_proposal_tokens: int,
    top_k: int,
    max_nodes: int,
    pack_mode: PackMode = "leaf-up",
) -> PackedTree:
    """Build candidate pool then pack. Proposal path is inference-style."""
    if int(top_k) < 2:
        raise ValueError("tree expansion requires top_k >= 2")
    device = block_base_logits.device
    first = int(first_draft_slot)
    stop = first + int(max_proposal_tokens)
    depth_logits = block_base_logits[first:stop, :]
    depth_hidden = block_hidden[first:stop, :]
    if depth_logits.shape[0] != int(max_proposal_tokens):
        raise RuntimeError(
            f"expected {max_proposal_tokens} draft depths, got {depth_logits.shape[0]}"
        )

    vocab_size = int(depth_logits.shape[-1])
    k = min(int(top_k), vocab_size)
    use_rnn = getattr(getattr(draft, "markov_head", None), "head_type", None) == "rnn"

    pool_tokens = torch.empty(0, dtype=torch.long, device=device)
    pool_scores = torch.empty(0, dtype=torch.float32, device=device)
    pool_parents = torch.empty(0, dtype=torch.long, device=device)
    pool_depths = torch.empty(0, dtype=torch.long, device=device)

    frontier_tokens = torch.tensor([anchor_token_id], dtype=torch.long, device=device)
    frontier_scores = torch.zeros(1, dtype=torch.float32, device=device)
    frontier_pool = torch.full((1,), -1, dtype=torch.long, device=device)
    frontier_states = None

    if use_rnn and first > 0:
        for reserved_slot in range(first):
            _, frontier_states = _frontier_step_logits(
                draft,
                depth_logits=block_base_logits[reserved_slot, :],
                depth_hidden=block_hidden[reserved_slot, :],
                frontier_target_tokens=frontier_tokens,
                frontier_states=frontier_states,
            )

    with torch.no_grad():
        for depth in range(int(max_proposal_tokens)):
            width = int(frontier_tokens.numel())
            corrected_logits, new_states = _frontier_step_logits(
                draft,
                depth_logits=depth_logits[depth],
                depth_hidden=depth_hidden[depth],
                frontier_target_tokens=frontier_tokens,
                frontier_states=frontier_states,
            )
            log_probs = torch.log_softmax(corrected_logits.float(), dim=-1)
            top_values, top_draft_ids = log_probs.topk(k, dim=-1)
            top_target_ids = _draft_ids_to_target(draft, top_draft_ids)
            candidate_scores = frontier_scores.unsqueeze(1) + top_values
            num_candidates = width * k
            pool_start = int(pool_tokens.numel())

            pool_parents = torch.cat(
                [pool_parents, frontier_pool.repeat_interleave(k)]
            )
            pool_tokens = torch.cat([pool_tokens, top_target_ids.reshape(-1)])
            pool_scores = torch.cat([pool_scores, candidate_scores.reshape(-1)])
            pool_depths = torch.cat(
                [
                    pool_depths,
                    torch.full(
                        (num_candidates,), depth, dtype=torch.long, device=device
                    ),
                ]
            )

            keep = min(k, num_candidates)
            selected = candidate_scores.reshape(-1).topk(keep).indices
            frontier_tokens = top_target_ids.reshape(-1)[selected]
            frontier_scores = candidate_scores.reshape(-1)[selected]
            frontier_pool = pool_start + selected
            if use_rnn and new_states is not None:
                frontier_states = new_states[selected // k]

    return pack_tree_nodes(
        pool_tokens=pool_tokens,
        pool_parents=pool_parents,
        pool_depths=pool_depths,
        pool_scores=pool_scores,
        max_nodes=max_nodes,
        top_k=top_k,
        mode=pack_mode,
    )


def replay_node_logprobs(
    draft: Any,
    *,
    tree_tokens: list[int],
    tree_parents: list[int],
    tree_depths: list[int],
    active_indices: list[int],
    anchor_token_id: int,
    block_base_logits: torch.Tensor,
    block_hidden: torch.Tensor,
    first_draft_slot: int,
) -> torch.Tensor:
    """Return logπ(token|parent) for ``active_indices`` with grad enabled."""
    if not active_indices:
        return block_base_logits.new_zeros(0)

    device = block_base_logits.device
    markov_head = getattr(draft, "markov_head", None)
    logps: list[torch.Tensor] = []

    for node_i in active_indices:
        depth = int(tree_depths[node_i])
        slot = int(first_draft_slot) + depth
        parent = tree_parents[node_i]
        prev_tok = (
            int(anchor_token_id) if parent < 0 else int(tree_tokens[parent])
        )
        depth_logits = block_base_logits[slot]
        depth_hidden = block_hidden[slot]
        prev = torch.tensor([prev_tok], dtype=torch.long, device=device)
        if markov_head is None:
            logits = depth_logits
        else:
            bias = markov_head.block_bias(
                prev_token_ids=prev.reshape(1, 1),
                hidden_states=depth_hidden.unsqueeze(0).unsqueeze(0),
            )[0, 0]
            logits = depth_logits + bias
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        tok = int(tree_tokens[node_i])
        # Map target token → draft id when using draft vocab.
        if getattr(draft, "use_draft_vocab", False) and getattr(draft, "t2d", None) is not None:
            t2d = draft.t2d.to(device)
            if tok < 0 or tok >= t2d.numel() or not bool(t2d[tok]):
                # Gold outside draft vocab: zero contrib (masked upstream ideally).
                logps.append(log_probs.new_zeros(()))
                continue
            # Find draft id: target = draft_id + d2t[draft_id]
            d2t = draft.d2t.to(device)
            draft_ids = torch.arange(d2t.numel(), device=device)
            matches = (draft_ids + d2t) == tok
            if not bool(matches.any()):
                logps.append(log_probs.new_zeros(()))
                continue
            draft_id = int(draft_ids[matches][0].item())
        else:
            draft_id = tok
        logps.append(log_probs[draft_id])

    return torch.stack(logps)
