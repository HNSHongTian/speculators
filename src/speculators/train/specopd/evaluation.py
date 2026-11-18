"""Measured multi-round, fixed-width speculative decoding on held-out prompts.

Uses full-prefix teacher queries, without loading a second teacher or measuring
serving throughput. No reference answers are fed back after generation begins.
"""

import hashlib
import json
import time

import torch

from speculators.train.specopd.runtime import anchor_backbone
from speculators.train.specopd.target import BranchTarget


def select_eval_prompts(dataset, count, max_prompt_tokens):
    """Deterministic held-out prompts ending just before first response token."""
    data = getattr(dataset, "data", None)
    if data is None:
        raise ValueError("SpecOPD AL evaluation requires an Arrow validation dataset")
    prompts = []
    for item in data:
        tokens = torch.as_tensor(item["input_ids"]).flatten().tolist()
        mask = torch.as_tensor(item["loss_mask"]).flatten().bool()
        if len(tokens) != len(mask):
            raise ValueError("Evaluation input_ids/loss_mask length mismatch")
        response = mask.nonzero().flatten()
        if not len(response):
            continue
        start = int(response[0])
        # Never truncate prompts or use the reference answer as a fallback.
        if not 0 < start <= max_prompt_tokens:
            continue
        messages = item.get("messages") or []
        if any(not isinstance(m.get("content"), str) for m in messages):
            continue
        prompts.append(tokens[:start])
        if len(prompts) == count:
            break
    if not prompts:
        raise ValueError("No eligible held-out prompts for SpecOPD AL evaluation")
    return prompts


def token_probs(logits, temperature):
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    return (logits.float() / temperature).softmax(-1)


def verify_round(tokens, draft_probs, target_probs, eos_ids, remaining):
    """Exact rejection sampling, including residual correction and bonus token.

    target_probs has W+1 rows: W proposal predictions and one bonus prediction.
    Returns committed tokens and the number of accepted draft tokens. EOS and
    output budget stop commitment immediately, so no phantom bonus is counted.
    """
    committed = []
    accepted = 0
    for i, token in enumerate(tokens):
        p, q = target_probs[i], draft_probs[i]
        ratio = (p[token] / q[token].clamp_min(torch.finfo(q.dtype).tiny)).clamp(max=1)
        if torch.rand((), device=p.device) < ratio:
            committed.append(token)
            accepted += 1
            if token in eos_ids or len(committed) == remaining:
                return committed, accepted
        else:
            residual = (p - q).clamp_min(0)
            if residual.sum() <= 0:
                raise ValueError("Rejected token has no residual probability mass")
            committed.append(int(torch.multinomial(residual, 1)))
            return committed, accepted
    bonus = int(torch.multinomial(target_probs[len(tokens)], 1))
    committed.append(bonus)
    return committed, accepted


def _proposal(draft, states, prefix, temperature, width, eos_ids):  # noqa: PLR0917
    device = next(draft.parameters()).device
    ids = torch.tensor([prefix], device=device, dtype=torch.long)
    context = (
        states[:, :-1].flatten(1).to(device=device, dtype=draft.fc.weight.dtype)[None]
    )
    hidden = anchor_backbone(
        draft,
        context,
        ids,
        torch.zeros_like(ids),
        None,
        torch.tensor([len(prefix) - 1], device=device),
    )[:, int(not draft.config.sample_from_anchor) :]
    previous = ids[:, -1:]
    tokens, distributions = [], []
    for i in range(width):
        bias = draft.markov_head.block_bias(
            prev_token_ids=previous, hidden_states=hidden[:, i : i + 1]
        )
        q = token_probs(draft.lm_head(hidden[:, i : i + 1]) + bias, temperature)[0, 0]
        token = int(torch.multinomial(q, 1))
        tokens.append(token)
        distributions.append(q)
        if token in eos_ids:
            break
        previous = ids.new_tensor([[token]])
    return tokens, torch.stack(distributions)


@torch.no_grad()
def evaluate_acceptance(draft, prompts, config, scorer=None):
    """Run actual decoding with the current draft weights; caller isolates RNG."""
    cfg = config["evaluation"]
    scorer = scorer or BranchTarget(config)
    device = next(draft.parameters()).device
    eos = getattr(draft.config.transformer_layer_config, "eos_token_id", None)
    eos_ids = [] if eos is None else eos if isinstance(eos, list) else [eos]
    width = draft.block_size - int(not draft.config.sample_from_anchor)
    temperature = cfg["temperature"]
    rounds = accepted_total = proposed_total = emitted_total = 0
    request_als = []
    started = time.perf_counter()
    for index, prompt in enumerate(prompts):
        prefix = list(prompt)
        emitted = sample_rounds = sample_accepted = 0
        while emitted < cfg["max_new_tokens"]:
            remaining = cfg["max_new_tokens"] - emitted
            states = scorer.read_states(prefix, index)
            tokens, q = _proposal(
                draft, states, prefix, temperature, min(width, remaining), eos_ids
            )
            # Include the last proposed token to get the bonus distribution.
            rows = scorer.read_states(
                prefix + tokens,
                index,
                slice(len(prefix) - 1, len(prefix) + len(tokens)),
                final_only=True,
            ).to(device=device, dtype=draft.verifier_lm_head.weight.dtype)
            if config["target_hidden_norm"] == "pre":
                rows = draft.verifier_norm(rows)
            p = token_probs(draft.verifier_lm_head(rows), temperature)
            committed, accepted = verify_round(tokens, q, p, eos_ids, remaining)
            prefix.extend(committed)
            emitted += len(committed)
            sample_accepted += accepted
            sample_rounds += 1
            proposed_total += len(tokens)
            if committed[-1] in eos_ids:
                break
        rounds += sample_rounds
        accepted_total += sample_accepted
        emitted_total += emitted
        request_als.append(1 + sample_accepted / sample_rounds)
    if not rounds:
        raise ValueError("SpecOPD AL evaluation produced no verification rounds")
    return {
        # Conventional AL: accepted draft tokens plus one target token per round.
        "specopd_acceptance_length": 1 + accepted_total / rounds,
        "specopd_acceptance_length_request_mean": sum(request_als) / len(request_als),
        "specopd_accepted_draft_length": accepted_total / rounds,
        # Exact emitted progress differs at EOS / max_new_tokens boundaries.
        "specopd_emitted_tokens_per_round": emitted_total / rounds,
        "specopd_acceptance_rate": accepted_total / proposed_total,
        "specopd_verification_rounds": rounds,
        "specopd_accepted_draft_tokens": accepted_total,
        "specopd_proposed_draft_tokens": proposed_total,
        "specopd_output_tokens": emitted_total,
        "specopd_eval_samples": len(prompts),
        "specopd_eval_temperature": temperature,
        "specopd_eval_seconds": time.perf_counter() - started,
        "specopd_eval_prompt_hash": hashlib.sha256(
            json.dumps(prompts).encode()
        ).hexdigest()[:16],
    }
