"""Original DSpark backbone, sampled Markov histories, fresh target supervision."""

import time
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from speculators.train.specopd.loss import specopd_loss
from speculators.train.specopd.target import BranchTarget


def select_anchors(input_ids, loss_mask, document_ids, limit, eos_ids):
    valid = loss_mask[0].bool() & (document_ids[0] >= 0)
    for eos in eos_ids:
        valid &= input_ids[0] != eos
    candidates = valid.nonzero().flatten()
    return candidates[torch.randperm(len(candidates), device=candidates.device)[:limit]]


def anchor_backbone(draft, context, input_ids, documents, positions, anchors):  # noqa: PLR0917
    block, device = draft.block_size, input_ids.device
    if positions is None:
        positions = torch.zeros_like(input_ids)
        for doc in documents[0].unique():
            if doc >= 0:
                indices = (documents[0] == doc).nonzero().flatten()
                positions[0, indices] = torch.arange(len(indices), device=device)
    tokens = torch.full(
        (1, len(anchors) * block), draft.mask_token_id, device=device, dtype=torch.long
    )
    tokens[:, ::block] = input_ids[:, anchors]
    hidden = draft.embed_tokens(tokens)
    features = draft.hidden_norm(draft.fc(context))
    query_positions = positions[0, anchors, None] + torch.arange(block, device=device)
    full_positions = torch.cat([positions, query_positions.reshape(1, -1)], -1)
    rotary = draft.rotary_emb(context, full_positions)
    masks = {}
    for sliding in (False, True):
        if draft.uses_sliding_window_attn if sliding else draft.uses_full_attn:
            masks[sliding] = draft._create_attention_mask(  # noqa: SLF001
                document_ids=documents,
                total_seq_len=input_ids.shape[1],
                anchor_positions=anchors,
                device=device,
                sliding_window=draft.sliding_window if sliding else None,
                sliding_window_non_causal=draft.sliding_window_non_causal
                if sliding
                else False,
            )
    for i, layer in enumerate(draft.layers):
        hidden = layer(
            hidden_states=hidden,
            target_hidden=features,
            attention_mask=masks[i in draft.sliding_window_indices],
            position_ids=full_positions,
            use_cache=False,
            position_embeddings=rotary,
        )
    return draft.norm(hidden).reshape(len(anchors), block, -1)


@torch.no_grad()
def sample_branches(draft, hidden, anchors, temperature, eos_ids):
    previous = anchors
    samples, predecessors, masks = [], [], []
    alive = torch.ones_like(anchors, dtype=torch.bool)
    for i in range(hidden.shape[1]):
        predecessors.append(previous)
        masks.append(alive.clone())
        bias = draft.markov_head.block_bias(
            prev_token_ids=previous[:, None], hidden_states=hidden[:, i : i + 1]
        )[:, 0]
        q = ((draft.lm_head(hidden[:, i]) + bias).float() / temperature).softmax(-1)
        token = torch.multinomial(q, 1).squeeze(-1)
        samples.append(token)
        for eos in eos_ids:
            alive &= token != eos
        previous = token
    return torch.stack(samples, 1), torch.stack(predecessors, 1), torch.stack(masks, 1)


def _position_scores(draft, cfg, hidden, previous, teacher_hidden):
    bias = draft.markov_head.block_bias(prev_token_ids=previous, hidden_states=hidden)
    q = ((draft.lm_head(hidden) + bias).float() / cfg["temperature"]).softmax(-1)
    with torch.no_grad():
        teacher_hidden = teacher_hidden.to(device=hidden.device, dtype=hidden.dtype)
        if cfg["target_hidden_norm"] == "pre":
            teacher_hidden = draft.verifier_norm(teacher_hidden)
        p = (
            draft.verifier_lm_head(teacher_hidden).float() / cfg["temperature"]
        ).softmax(-1)
        p = torch.nn.functional.pad(p, (0, 0, 0, hidden.shape[1] - len(p)))
    overlap = torch.minimum(q, p.unsqueeze(0)).sum(-1)
    confidence = hidden.sum(-1) * 0
    if draft.confidence_head is not None:
        features = hidden
        if draft.config.confidence_head_with_markov:
            features = torch.cat(
                [hidden, draft.markov_head.prev_embeddings(previous).to(hidden.dtype)],
                -1,
            )
        confidence = draft.confidence_head(features)
    return overlap, confidence


@torch.compiler.disable
def specopd_forward(  # noqa: C901
    draft,
    *,
    hidden_states,
    input_ids,
    loss_mask,
    document_ids,
    position_ids=None,
    specopd_config,
    **kwargs,
):
    cfg = specopd_config
    if (
        draft.use_draft_vocab
        or draft.markov_head is None
        or draft.markov_head.head_type != "vanilla"
    ):
        raise ValueError(
            "SpecOPD requires original vanilla-Markov full-vocabulary DSpark"
        )
    if draft.config.tree_training_enabled:
        raise ValueError("SpecOPD cannot use TreeSpark training")
    eos = getattr(draft.config.transformer_layer_config, "eos_token_id", None)
    eos_ids = [] if eos is None else eos if isinstance(eos, list) else [eos]
    anchors = select_anchors(
        input_ids, loss_mask, document_ids, cfg["num_anchors"], eos_ids
    )
    width = draft.block_size - int(not draft.config.sample_from_anchor)
    if not len(anchors):
        zero = sum(p.reshape(-1)[0] * 0 for p in draft.parameters() if p.requires_grad)
        loss, metrics = specopd_loss(
            zero.expand(0, width),
            torch.zeros((0, width), device=zero.device, dtype=torch.bool),
        )
        metrics["specopd_target_seconds"] = zero.detach()
        metrics["specopd_target_requests"] = zero.detach()
        return None, loss, metrics
    hidden = anchor_backbone(
        draft, hidden_states, input_ids, document_ids, position_ids, anchors
    )
    hidden = hidden[:, int(not draft.config.sample_from_anchor) :]
    samples, predecessors, valid = sample_branches(
        draft, hidden.detach(), input_ids[0, anchors], cfg["temperature"], eos_ids
    )
    prefixes = []
    for anchor in anchors.tolist():
        doc = document_ids[0, anchor]
        indices = (
            (
                (document_ids[0] == doc)
                & (torch.arange(input_ids.shape[1], device=input_ids.device) <= anchor)
            )
            .nonzero()
            .flatten()
        )
        if len(indices) > 1 and not torch.all(indices[1:] == indices[:-1] + 1):
            raise ValueError("Packed document IDs must be contiguous")
        prefixes.append(input_ids[0, indices].detach().cpu().tolist())
    scorer = kwargs.get("specopd_scorer") or BranchTarget(cfg)
    started = time.perf_counter()
    rows = scorer.score(prefixes, samples, valid)
    elapsed = time.perf_counter() - started
    overlaps, confidences = [], []
    for b in range(len(anchors)):
        branch_overlap, branch_confidence = [], []
        n = int(valid[b].sum().item())
        for start in range(0, width, cfg["loss_chunk_size"]):
            end = min(start + cfg["loss_chunk_size"], width)
            h = hidden[b : b + 1, start:end]
            prev = predecessors[b : b + 1, start:end]
            scores = partial(_position_scores, draft, cfg)
            teacher_hidden = rows[b][start : min(end, n)]
            if torch.is_grad_enabled():
                overlap, confidence = checkpoint(
                    scores, h, prev, teacher_hidden, use_reentrant=False
                )
            else:
                overlap, confidence = scores(h, prev, teacher_hidden)
            branch_overlap.append(overlap)
            if draft.confidence_head is not None:
                branch_confidence.append(confidence)
        overlaps.append(torch.cat(branch_overlap, -1))
        if branch_confidence:
            confidences.append(torch.cat(branch_confidence, -1))
    loss, metrics = specopd_loss(
        torch.cat(overlaps),
        valid,
        torch.cat(confidences) if confidences else None,
        cfg["confidence_alpha"],
    )
    metrics["specopd_target_seconds"] = loss.detach().new_tensor(elapsed)
    metrics["specopd_target_requests"] = loss.detach().new_tensor(len(prefixes))
    return samples, loss, metrics
