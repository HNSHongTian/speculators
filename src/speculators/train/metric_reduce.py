"""Validate collective metric schemas before reducing scalar values."""

import hashlib
import json

import torch
import torch.distributed as dist


def reduce_metrics(metrics: dict[str, torch.Tensor], *, device):
    """All ranks use sorted keys and the same fixed-size schema collective.

    Fail on differing key sets/shapes/dtypes instead of silently adding unrelated
    counters or leaving some ranks waiting for additional scalar reductions.
    All participating ranks must run this implementation.
    """
    keys = sorted(metrics)
    schema = [(key, list(metrics[key].shape), str(metrics[key].dtype)) for key in keys]
    digest = hashlib.sha256(json.dumps(schema).encode()).digest()
    signature = torch.tensor(
        [len(keys), int.from_bytes(digest[:7], "big")],
        device=device,
        dtype=torch.int64,
    )
    signatures = [torch.empty_like(signature) for _ in range(dist.get_world_size())]
    dist.all_gather(signatures, signature)
    rows = [value.cpu().tolist() for value in signatures]
    if any(row != rows[0] for row in rows):
        raise RuntimeError(
            f"Distributed metric schema mismatch: rank={dist.get_rank()}, "
            f"local_keys={keys}, signatures_by_rank={rows}. "
            "Check empty-batch branches and deploy identical code on every rank."
        )
    if not keys:
        return {}
    if any(metrics[key].numel() != 1 for key in keys):
        raise ValueError("Counted metrics must be scalar tensors")
    # One value collective replaces a variable-length sequence of collectives.
    packed = torch.stack([metrics[key].detach().float().reshape(()) for key in keys])
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return dict(zip(keys, packed.unbind(), strict=True))
