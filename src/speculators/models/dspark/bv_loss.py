"""Vocabulary-integrated block-verification loss (arXiv:2609.34832v1).

Only the current token is integrated out. Prefix states follow the stored
target-generated tokens, and must remain differentiable through the drafter.
"""

import math

import torch
from torch.nn.functional import cross_entropy
from torch.utils.checkpoint import checkpoint

LOG_SUMMAND_FLOOR = -80.0
LENGTH_FLOOR = 1e-6
LOGITS_NDIM = 3
SMALL_CENTERED_SCORE = 0.5


def build_bv_labels(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    document_ids: torch.Tensor,
    block_indices: torch.Tensor,
    aligned_loss_mask: torch.Tensor,
    block_size: int,
    sample_from_anchor: bool = True,
    eos_token_ids: tuple[int, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [N, B] actual labels and [N] complete-block validity.

    ``loss_mask[t]`` must describe token t, not the prediction at t. Blocks
    cannot cross document/padding, assistant-mask gaps, or an internal EOS.
    The backbone's mask identifies padded anchors without changing its API.
    """
    indices = block_indices.reshape(-1, block_size)
    start = 0 if sample_from_anchor else 1
    if block_size <= start:
        raise ValueError("BV requires at least one prediction per block")
    label_indices = indices[:, start:] + int(sample_from_anchor)
    in_bounds = (label_indices > 0) & (label_indices < input_ids.shape[1])
    safe_indices = label_indices.clamp(1, input_ids.shape[1] - 1)
    labels = input_ids[0, safe_indices]
    anchor_docs = document_ids[0, indices[:, :1]]
    valid = (
        in_bounds
        & loss_mask[0, safe_indices].bool()
        & (document_ids[0, safe_indices] == anchor_docs)
        & (document_ids[0, safe_indices - 1] == anchor_docs)
        & (anchor_docs >= 0)
    ).all(dim=-1)
    valid = valid & aligned_loss_mask.reshape(-1, block_size).bool().any(-1)
    # EOS may be the last label, but must not precede another proposal.
    preceding = torch.cat([input_ids[0, indices[:, :1]], labels[:, :-1]], -1)
    for eos in eos_token_ids:
        valid = valid & (preceding != eos).all(-1)
    return labels, valid


def integrated_log_scores(
    logits: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    *,
    floor_summands: bool = False,
) -> torch.Tensor:
    """Compute log(bar_alpha_i), [N, B], from corrected logits [N, B, V].

    FP64 is retained for numerical verification; production BF16/FP32 uses
    FP32 throughout. The -80 floor is on vocabulary log summands, not states.
    """
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    log_q = torch.log_softmax(logits.to(dtype), dim=-1)
    log_p = torch.log_softmax(targets.detach().to(dtype), dim=-1)
    log_r = log_q.new_zeros(log_q.shape[0])
    log_a = log_q.new_zeros(log_q.shape[0])
    scores = []
    for i in range(log_q.shape[1]):
        terms = torch.minimum(
            log_a[:, None] + log_p[:, i], log_r[:, None] + log_q[:, i]
        )
        if floor_summands:
            terms = terms.clamp_min(LOG_SUMMAND_FLOOR)
        scores.append(torch.logsumexp(terms, dim=-1))
        token = labels[:, i, None]
        log_r = log_r + log_q[:, i].gather(-1, token).squeeze(-1)
        log_r = log_r - log_p[:, i].gather(-1, token).squeeze(-1)
        log_a = torch.minimum(log_a, log_r)
    return torch.stack(scores, dim=-1)


def aggregate_bv_scores(
    log_scores: torch.Tensor, beta: torch.Tensor, *, annealing: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return backward surrogate and actual power-mean/block-log objective."""
    log_b = math.log(log_scores.shape[-1])
    if not annealing:
        objective = log_b - torch.logsumexp(log_scores, -1).clamp_min(
            math.log(LENGTH_FLOOR)
        )
        return objective, objective
    weights = torch.softmax(beta * log_scores, -1).detach()
    surrogate = -(weights * log_scores).sum(-1)
    # Use expm1/log1p near zero to avoid cancellation, without replacing any
    # positive beta with its geometric-mean limit (scores can have a wide range).
    center = log_scores.mean(-1)
    centered = beta * (log_scores - center[:, None])
    small = centered.abs().amax(-1) <= SMALL_CENTERED_SCORE
    near_zero = torch.log1p(
        torch.expm1(centered.clamp(-SMALL_CENTERED_SCORE, SMALL_CENTERED_SCORE)).mean(
            -1
        )
    )
    log_mean = torch.where(small, near_zero, torch.logsumexp(centered, -1) - log_b)
    safe_beta = torch.where(beta == 0, torch.ones_like(beta), beta)
    objective = -center - log_mean / safe_beta
    return surrogate, objective


def bv_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    valid_blocks: torch.Tensor,
    beta: torch.Tensor,
    *,
    annealing: bool = False,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Local block mean; Trainer scales it for global block-mean gradients.

    Chunked non-reentrant checkpointing recomputes probability intermediates
    in backward. It does not reduce the backbone's pre-existing logit storage.
    Invalid blocks are sanitized before recurrence (not only masked afterward).
    """
    if logits.ndim != LOGITS_NDIM or logits.shape != targets.shape:
        raise ValueError("BV expects matching [blocks, predictions, vocabulary] logits")
    if labels.shape != logits.shape[:2] or valid_blocks.shape != logits.shape[:1]:
        raise ValueError("BV labels/block validity do not match logits")
    if chunk_size < 1 or logits.shape[1] < 1:
        raise ValueError("BV chunk size and prediction length must be positive")

    def compute_chunk(q, p, y, valid):
        q = torch.where(valid[:, None, None], q, 0.0)
        p = torch.where(valid[:, None, None], p, 0.0)
        y = torch.where(valid[:, None], y, 0)
        scores = integrated_log_scores(q, p, y, floor_summands=not annealing)
        backward, objective = aggregate_bv_scores(scores, beta, annealing=annealing)
        return backward, objective.detach(), scores.detach().exp().sum(-1)

    backward_parts, objective_parts, length_parts = [], [], []
    for offset in range(0, logits.shape[0], chunk_size):
        sl = slice(offset, offset + chunk_size)
        args = (logits[sl], targets[sl], labels[sl], valid_blocks[sl])
        if torch.is_grad_enabled() and logits.requires_grad:
            values = checkpoint(
                compute_chunk, *args, use_reentrant=False, preserve_rng_state=False
            )
        else:
            values = compute_chunk(*args)
        backward_parts.append(values[0])
        objective_parts.append(values[1])
        length_parts.append(values[2])
    mask = valid_blocks.to(torch.float32)
    count = mask.sum()
    backward_sum = (torch.cat(backward_parts) * mask).sum()
    objective_sum = (torch.cat(objective_parts) * mask).sum()
    metrics = {
        "bv_loss_sum": objective_sum,
        "bv_loss_total": count,
        "bv_backward_surrogate_sum": backward_sum.detach(),
        "bv_backward_surrogate_total": count.clone(),
        "bv_estimated_draft_accept_len_sum": (torch.cat(length_parts) * mask).sum(),
        "bv_estimated_draft_accept_len_total": count.clone(),
        "bv_valid_blocks": count.clone(),
        "bv_dropped_blocks": (~valid_blocks).sum().to(mask.dtype),
        "bv_beta": beta.detach().clone(),
    }
    return backward_sum / count.clamp_min(1), metrics


def block_tv_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_blocks: torch.Tensor,
    position_weights: torch.Tensor,
    *,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TV token mean on complete BV blocks, then a local block mean.

    Position decay follows the existing token objective. Checkpoint chunks keep
    full-vocabulary probability intermediates bounded, including on Ascend.
    """
    if chunk_size < 1:
        raise ValueError("TV chunk size must be positive")

    def compute_chunk(q, p, valid):
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        q = torch.where(valid[:, None, None], q, 0.0).to(dtype)
        p = torch.where(valid[:, None, None], p.detach(), 0.0).to(dtype)
        per_token = 0.5 * (q.softmax(-1) - p.softmax(-1)).abs().sum(-1)
        return (per_token * position_weights).mean(-1)

    parts = []
    for offset in range(0, logits.shape[0], chunk_size):
        sl = slice(offset, offset + chunk_size)
        args = (logits[sl], targets[sl], valid_blocks[sl])
        if torch.is_grad_enabled() and logits.requires_grad:
            parts.append(
                checkpoint(
                    compute_chunk, *args, use_reentrant=False, preserve_rng_state=False
                )
            )
        else:
            parts.append(compute_chunk(*args))
    count = valid_blocks.sum()
    total = (torch.cat(parts) * valid_blocks).sum()
    return total / count.clamp_min(1), total.detach()


def block_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_blocks: torch.Tensor,
    position_weights: torch.Tensor,
    *,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-argmax CE on complete BV blocks, then a local block mean."""
    if chunk_size < 1:
        raise ValueError("CE chunk size must be positive")

    def compute_chunk(q, p, valid):
        q = torch.where(valid[:, None, None], q, 0.0)
        p = torch.where(valid[:, None, None], p.detach(), 0.0)
        labels = p.argmax(-1)
        per_token = cross_entropy(
            q.reshape(-1, q.shape[-1]), labels.reshape(-1), reduction="none"
        ).reshape(q.shape[:2])
        return (per_token * position_weights).mean(-1)

    parts = []
    for offset in range(0, logits.shape[0], chunk_size):
        sl = slice(offset, offset + chunk_size)
        args = (logits[sl], targets[sl], valid_blocks[sl])
        if torch.is_grad_enabled() and logits.requires_grad:
            parts.append(
                checkpoint(
                    compute_chunk, *args, use_reentrant=False, preserve_rng_state=False
                )
            )
        else:
            parts.append(compute_chunk(*args))
    count = valid_blocks.sum()
    total = (torch.cat(parts) * valid_blocks).sum()
    return total / count.clamp_min(1), total.detach()
