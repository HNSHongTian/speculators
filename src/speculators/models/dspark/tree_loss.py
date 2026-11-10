"""Inference-aligned TreeSpark objectives for DSpark training.

The replay in this module mirrors tree deployment: every frontier parent is
expanded with its local top-k children, the next frontier is pruned by joint
path log-probability, and a global top-N budget chooses the nodes that would be
verified.  Candidate identities and topology are detached; conditional draft
probabilities remain differentiable through the DSpark backbone and Markov
head.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from speculators.models.dspark.model_definitions import MarkovHead

_EPS = 1e-6


@dataclass
class TreeLossInputs:
    """Per-depth tensors produced by replaying the deployed tree policy."""

    node_logprobs: list[torch.Tensor] = field(default_factory=list)
    node_masses: list[torch.Tensor] = field(default_factory=list)
    node_valids: list[torch.Tensor] = field(default_factory=list)
    node_packed_masks: list[torch.Tensor | None] = field(default_factory=list)
    rank_hinges: list[torch.Tensor] = field(default_factory=list)
    rank_weights: list[torch.Tensor] = field(default_factory=list)
    depth_valids: list[torch.Tensor] = field(default_factory=list)
    argmax_survived: list[torch.Tensor] = field(default_factory=list)
    diagnostics: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )


def _subsample_anchors(
    block_keep_mask: torch.Tensor, num_anchors: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly select a fixed-width set of valid anchors without replacement."""
    num_blocks = int(block_keep_mask.numel())
    num_selected = min(int(num_anchors), num_blocks)
    random_values = torch.rand(num_blocks, device=block_keep_mask.device)
    random_values = torch.where(
        block_keep_mask,
        random_values,
        torch.full_like(random_values, 2.0),
    )
    selected = random_values.topk(num_selected, largest=False).indices
    keep_count = block_keep_mask.sum().clamp(max=num_selected)
    selected_valid = torch.arange(num_selected, device=selected.device) < keep_count
    selected = torch.where(selected_valid, selected, torch.zeros_like(selected))
    return selected, selected_valid


def _draft_to_target_ids(
    draft_ids: torch.Tensor,
    *,
    use_draft_vocab: bool,
    d2t: torch.Tensor | None,
) -> torch.Tensor:
    if not use_draft_vocab:
        return draft_ids.long()
    if d2t is None:
        raise ValueError("Tree training requires a loaded draft-to-target mapping")
    flat_ids = draft_ids.reshape(-1)
    offsets = d2t.to(flat_ids.device).index_select(0, flat_ids)
    return (flat_ids + offsets).reshape_as(draft_ids).long()


def _parent_conditioned_logits(
    base_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    parent_tokens: torch.Tensor,
    parent_states: torch.Tensor | None,
    *,
    markov_head: MarkovHead | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one lightweight Markov step to every parent at one tree depth."""
    num_anchors, frontier_width = parent_tokens.shape
    expanded_hidden = hidden_states.unsqueeze(1).expand(
        num_anchors, frontier_width, hidden_states.shape[-1]
    )
    if markov_head is None:
        logits = base_logits.unsqueeze(1).expand(
            num_anchors, frontier_width, base_logits.shape[-1]
        )
        return logits, base_logits.new_empty((0,), dtype=torch.float32)

    if markov_head.head_type == "rnn":
        previous = markov_head.prev_embeddings(parent_tokens).to(
            markov_head.markov_w2.weight.dtype
        )
        if parent_states is None or parent_states.numel() == 0:
            parent_states = previous.new_zeros(
                num_anchors, frontier_width, markov_head.markov_rank
            )
        else:
            parent_states = parent_states.to(previous.dtype)
        recurrent_input = torch.cat(
            [parent_states, previous, expanded_hidden.to(previous.dtype)], dim=-1
        )
        gate_raw, candidate_raw, output_raw = markov_head.joint_proj(
            recurrent_input
        ).chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        new_states = gate * parent_states + (1.0 - gate) * torch.tanh(candidate_raw)
        bias = markov_head.markov_w2(torch.tanh(output_raw))
        return base_logits.unsqueeze(1) + bias, new_states

    flat_tokens = parent_tokens.reshape(-1, 1)
    flat_hidden = expanded_hidden.reshape(-1, 1, expanded_hidden.shape[-1])
    bias = markov_head.block_bias(
        prev_token_ids=flat_tokens,
        hidden_states=flat_hidden,
    )[:, 0, :]
    bias = bias.reshape(num_anchors, frontier_width, -1)
    return base_logits.unsqueeze(1) + bias, base_logits.new_empty(
        (0,), dtype=torch.float32
    )


def _tree_depth_segment(  # noqa: PLR0917
    base_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    parent_tokens: torch.Tensor,
    parent_states: torch.Tensor,
    target_top_ids: torch.Tensor,
    target_top_probs: torch.Tensor,
    *,
    markov_head: MarkovHead | None,
    top_k: int,
    margin: float,
    hinge_cap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand one depth and compute its differentiable coverage hinge."""
    corrected_logits, new_states = _parent_conditioned_logits(
        base_logits,
        hidden_states,
        parent_tokens,
        parent_states,
        markov_head=markov_head,
    )
    logits_float = corrected_logits.float()
    top_values, top_ids = logits_float.topk(top_k, dim=-1)
    node_logprobs = functional.log_softmax(logits_float, dim=-1).gather(-1, top_ids)

    frontier_width = int(parent_tokens.shape[-1])
    num_target_tokens = int(target_top_ids.shape[-1])
    gather_ids = target_top_ids.unsqueeze(1).expand(
        -1, frontier_width, num_target_tokens
    )
    target_logits = logits_float.gather(-1, gather_ids)
    boundary = top_values[..., -1:]
    hinge = (float(margin) - target_logits + boundary).clamp_min(0.0)
    if hinge_cap > 0:
        hinge = hinge.clamp_max(float(hinge_cap))
    rank_hinge = (hinge * target_top_probs.unsqueeze(1)).sum(dim=-1)
    return (
        node_logprobs.flatten(1),
        top_ids.flatten(1),
        rank_hinge,
        new_states,
    )


@torch.no_grad()
def _pack_and_walk(
    inputs: TreeLossInputs,
    *,
    pool_tokens: list[torch.Tensor],
    pool_scores: list[torch.Tensor],
    pool_parents: list[torch.Tensor],
    target_argmax: list[torch.Tensor],
    max_nodes: int,
) -> None:
    """Apply the deployed global budget and replay greedy target acceptance."""
    scores = torch.cat(pool_scores, dim=-1)
    tokens = torch.cat(pool_tokens, dim=-1)
    parents = torch.cat(pool_parents, dim=-1)
    num_anchors, num_pool = scores.shape

    num_selected = min(int(max_nodes), int(num_pool))
    packed_indices = scores.topk(num_selected, dim=-1).indices
    packed_mask = torch.zeros_like(scores, dtype=torch.bool)
    packed_mask.scatter_(-1, packed_indices, True)

    previous_pool_node = torch.full(
        (num_anchors,), -1, dtype=torch.long, device=scores.device
    )
    alive = torch.ones(num_anchors, dtype=torch.bool, device=scores.device)
    pool_offset = 0
    for depth, depth_tokens in enumerate(pool_tokens):
        num_candidates = int(depth_tokens.shape[-1])
        sl = slice(pool_offset, pool_offset + num_candidates)
        packed_depth = packed_mask[:, sl]
        inputs.node_packed_masks.append(packed_depth)
        matches = (
            alive.unsqueeze(-1)
            & (parents[:, sl] == previous_pool_node.unsqueeze(-1))
            & (tokens[:, sl] == target_argmax[depth].unsqueeze(-1))
            & packed_depth
        )
        survived = matches.any(dim=-1)
        match_index = matches.float().argmax(dim=-1)
        previous_pool_node = torch.where(
            survived,
            torch.full_like(previous_pool_node, pool_offset) + match_index,
            previous_pool_node,
        )
        alive = alive & survived
        inputs.argmax_survived.append(alive.clone())
        pool_offset += num_candidates


@torch.compiler.disable
def build_tree_loss_inputs(  # noqa: C901
    *,
    markov_head: MarkovHead | None,
    base_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    target_logits: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    depth_valid_mask: torch.Tensor,
    block_keep_mask: torch.Tensor,
    first_draft_slot: int,
    top_k: int,
    top_j: int,
    margin: float,
    hinge_cap: float,
    num_anchors: int,
    max_nodes: int,
    use_draft_vocab: bool,
    d2t: torch.Tensor | None,
    reference_top1_ids: torch.Tensor | None = None,
    grad_checkpoint: bool = True,
) -> TreeLossInputs | None:
    """Replay TreeSpark expansion, pruning and packing on sampled anchors.

    Inputs are block-major: ``[num_blocks, block_size, ...]``.  For the
    DFlash-compatible ``sample_from_anchor=False`` layout, slot zero is skipped
    as a proposal and replayed only to initialize the RNN state, matching the
    existing tree inference path.
    """
    num_blocks, block_size, vocab_size = base_logits.shape
    first_draft_slot = int(first_draft_slot)
    num_depths = int(block_size) - first_draft_slot
    if num_depths <= 0 or num_anchors <= 0:
        return None
    if top_k < 2:  # noqa: PLR2004
        raise ValueError("Tree training requires tree_loss_top_k >= 2")
    if max_nodes < 1:
        raise ValueError("Tree training requires tree_loss_max_nodes >= 1")
    if top_j < 1:
        raise ValueError("Tree training requires tree_rank_top_j >= 1")

    top_k = max(1, min(int(top_k), int(vocab_size)))
    top_j = max(1, min(int(top_j), int(vocab_size)))
    selected, selected_valid = _subsample_anchors(block_keep_mask, num_anchors)
    base_sub = base_logits.index_select(0, selected)
    hidden_sub = hidden_states.index_select(0, selected)
    target_sub = target_logits.index_select(0, selected).detach()
    valid_sub = depth_valid_mask.index_select(0, selected)
    root_tokens = anchor_token_ids.index_select(0, selected).long()
    reference_top1_sub = (
        None
        if reference_top1_ids is None
        else reference_top1_ids.index_select(0, selected).detach().long()
    )
    num_selected = int(selected.numel())

    frontier_tokens = root_tokens.unsqueeze(-1)
    frontier_scores = base_logits.new_zeros((num_selected, 1), dtype=torch.float32)
    frontier_mass = base_logits.new_ones((num_selected, 1), dtype=torch.float32)
    frontier_pool = torch.full(
        (num_selected, 1), -1, dtype=torch.long, device=base_logits.device
    )
    frontier_states: torch.Tensor | None = None
    if markov_head is not None and markov_head.head_type == "rnn":
        frontier_states = hidden_states.new_zeros(
            num_selected, 1, markov_head.markov_rank
        )
        for reserved_slot in range(first_draft_slot):
            _, frontier_states = _parent_conditioned_logits(
                base_sub[:, reserved_slot],
                hidden_sub[:, reserved_slot],
                frontier_tokens,
                frontier_states,
                markov_head=markov_head,
            )
            frontier_states = frontier_states.detach()

    inputs = TreeLossInputs()
    pool_tokens: list[torch.Tensor] = []
    pool_scores: list[torch.Tensor] = []
    pool_parents: list[torch.Tensor] = []
    target_argmax: list[torch.Tensor] = []
    pool_offset = 0
    root_teacher_ids: torch.Tensor | None = None
    root_replay_top1_ids: torch.Tensor | None = None
    root_local_topk_hit: torch.Tensor | None = None

    for depth in range(num_depths):
        slot = first_draft_slot + depth
        frontier_width = int(frontier_tokens.shape[-1])
        with torch.no_grad():
            target_probs = functional.softmax(target_sub[:, slot].float(), dim=-1)
            target_top_probs, target_top_ids = target_probs.topk(top_j, dim=-1)

        state_arg = (
            frontier_states
            if frontier_states is not None
            else base_logits.new_empty((0,), dtype=torch.float32)
        )
        segment_args = (
            base_sub[:, slot],
            hidden_sub[:, slot],
            frontier_tokens,
            state_arg,
            target_top_ids,
            target_top_probs,
        )
        segment_kwargs = {
            "markov_head": markov_head,
            "top_k": top_k,
            "margin": margin,
            "hinge_cap": hinge_cap,
        }
        if grad_checkpoint and torch.is_grad_enabled():
            node_logprobs, draft_ids, rank_hinge, new_states = checkpoint(
                _tree_depth_segment,
                *segment_args,
                **segment_kwargs,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            node_logprobs, draft_ids, rank_hinge, new_states = _tree_depth_segment(
                *segment_args, **segment_kwargs
            )

        depth_valid = valid_sub[:, slot].bool() & selected_valid
        num_candidates = frontier_width * top_k
        inputs.node_logprobs.append(node_logprobs)
        inputs.node_valids.append(depth_valid.unsqueeze(-1).expand(-1, num_candidates))
        inputs.depth_valids.append(depth_valid)
        inputs.rank_hinges.append(rank_hinge)
        inputs.rank_weights.append(
            target_top_probs.sum(dim=-1).unsqueeze(-1).expand(-1, frontier_width)
        )

        with torch.no_grad():
            node_mass = target_probs.gather(-1, draft_ids)
            node_mass = node_mass * frontier_mass.repeat_interleave(top_k, dim=-1)
            inputs.node_masses.append(node_mass)
            candidate_scores = frontier_scores.unsqueeze(
                -1
            ) + node_logprobs.detach().view(num_selected, frontier_width, top_k)
            candidate_scores = candidate_scores.flatten(1)
            target_ids = _draft_to_target_ids(
                draft_ids,
                use_draft_vocab=use_draft_vocab,
                d2t=d2t,
            )
            pool_tokens.append(target_ids)
            pool_scores.append(candidate_scores)
            pool_parents.append(frontier_pool.repeat_interleave(top_k, dim=-1))
            target_argmax.append(
                _draft_to_target_ids(
                    target_probs.argmax(dim=-1),
                    use_draft_vocab=use_draft_vocab,
                    d2t=d2t,
                )
            )
            if depth == 0:
                # The root frontier has width one, so the first flattened id is
                # the replay distribution's top-1.  Keep diagnostics in draft
                # vocabulary space; d2t is injective and does not change equality.
                root_teacher_ids = target_probs.argmax(dim=-1)
                root_replay_top1_ids = draft_ids[:, 0]
                root_local_topk_hit = (
                    draft_ids == root_teacher_ids.unsqueeze(-1)
                ).any(dim=-1)

            keep = min(top_k, num_candidates)
            selected_scores, selected_indices = candidate_scores.topk(keep, dim=-1)
            frontier_tokens = target_ids.gather(-1, selected_indices)
            frontier_scores = selected_scores
            frontier_mass = node_mass.gather(-1, selected_indices)
            frontier_pool = pool_offset + selected_indices
            if markov_head is not None and markov_head.head_type == "rnn":
                parent_indices = (
                    (selected_indices // top_k)
                    .unsqueeze(-1)
                    .expand(-1, -1, new_states.shape[-1])
                )
                frontier_states = new_states.detach().gather(1, parent_indices)
            pool_offset += num_candidates

    _pack_and_walk(
        inputs,
        pool_tokens=pool_tokens,
        pool_scores=pool_scores,
        pool_parents=pool_parents,
        target_argmax=target_argmax,
        max_nodes=max_nodes,
    )
    if (
        root_teacher_ids is not None
        and root_replay_top1_ids is not None
        and root_local_topk_hit is not None
    ):
        root_valid = inputs.depth_valids[0]
        root_total = root_valid.to(torch.float32).sum().detach()

        def add_root_diagnostic(name: str, values: torch.Tensor) -> None:
            numerator = (values & root_valid).to(torch.float32).sum().detach()
            inputs.diagnostics[name] = (numerator, root_total.clone())

        replay_top1_correct = root_replay_top1_ids == root_teacher_ids
        add_root_diagnostic(
            "tree_depth0_replay_top1_acc",
            replay_top1_correct,
        )
        add_root_diagnostic(
            "tree_depth0_local_topk_hit",
            root_local_topk_hit,
        )
        add_root_diagnostic(
            "tree_depth0_top1_pack_violation",
            replay_top1_correct & ~inputs.argmax_survived[0],
        )
        if reference_top1_sub is not None:
            main_top1_correct = reference_top1_sub == root_teacher_ids
            add_root_diagnostic(
                "tree_depth0_main_top1_acc",
                main_top1_correct,
            )
            add_root_diagnostic(
                "tree_depth0_main_replay_agreement",
                reference_top1_sub == root_replay_top1_ids,
            )
            add_root_diagnostic(
                "tree_depth0_main_top1_survival_violation",
                main_top1_correct & ~inputs.argmax_survived[0],
            )
    return inputs


def _global_normalized_loss(
    numerator: torch.Tensor, denominator: torch.Tensor
) -> torch.Tensor:
    """Normalize by the global detached weight under averaged DP gradients."""
    global_denominator = denominator.detach().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)
    return numerator * world_size / (global_denominator + _EPS)


@torch.compiler.disable
def compute_tree_training_loss(
    inputs: TreeLossInputs,
    *,
    loss_decay_gamma: float | None,
    survival_alpha: float,
    coverage_alpha: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reduce replay tensors into TreeSpark loss terms and counted metrics."""
    reference = inputs.node_logprobs[0]
    zero = reference.new_zeros((), dtype=torch.float32)
    survival_num = zero.clone()
    survival_den = zero.clone()
    coverage_num = zero.clone()
    coverage_den = zero.clone()
    metrics: dict[str, torch.Tensor] = {}

    for name, (numerator, denominator) in inputs.diagnostics.items():
        metrics[f"{name}_sum"] = numerator
        metrics[f"{name}_total"] = denominator

    for depth, node_logprobs in enumerate(inputs.node_logprobs):
        packed_valid = inputs.node_valids[depth] & inputs.node_packed_masks[depth]
        valid_float = packed_valid.to(torch.float32)
        node_mass = inputs.node_masses[depth]
        survival_num = survival_num - (node_logprobs * node_mass * valid_float).sum()
        survival_den = survival_den + (node_mass * valid_float).sum()

        depth_valid = inputs.depth_valids[depth].to(torch.float32)
        depth_total = depth_valid.sum()
        metrics[f"tree_mass_coverage_depth_{depth}_sum"] = (
            (node_mass * valid_float).sum().detach()
        )
        metrics[f"tree_mass_coverage_depth_{depth}_total"] = depth_total.detach()
        metrics[f"tree_argmax_survival_depth_{depth}_sum"] = (
            inputs.argmax_survived[depth].to(torch.float32) * depth_valid
        ).sum()
        metrics[f"tree_argmax_survival_depth_{depth}_total"] = depth_total.detach()

        depth_weight = depth_valid
        if loss_decay_gamma is not None and loss_decay_gamma > 0:
            depth_weight = depth_weight * math.exp(
                -float(depth) / float(loss_decay_gamma)
            )
        coverage_num = (
            coverage_num
            + (inputs.rank_hinges[depth] * depth_weight.unsqueeze(-1)).sum()
        )
        coverage_den = (
            coverage_den
            + (inputs.rank_weights[depth] * depth_weight.unsqueeze(-1)).sum()
        )

    survival_loss = _global_normalized_loss(survival_num, survival_den)
    coverage_loss = _global_normalized_loss(coverage_num, coverage_den)
    weighted_loss = (
        float(survival_alpha) * survival_loss + float(coverage_alpha) * coverage_loss
    )
    metrics.update(
        tree_survival_loss_sum=survival_num.detach(),
        tree_survival_loss_total=survival_den.detach(),
        tree_coverage_loss_sum=coverage_num.detach(),
        tree_coverage_loss_total=coverage_den.detach(),
        tree_weighted_loss_sum=weighted_loss.detach(),
        tree_weighted_loss_total=weighted_loss.new_ones(()),
    )
    return weighted_loss, metrics


__all__ = [
    "TreeLossInputs",
    "build_tree_loss_inputs",
    "compute_tree_training_loss",
]
