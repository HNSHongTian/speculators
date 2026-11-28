"""Synchronization shim for dynamically routed custom MoE tensor parallelism."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _broadcast_from_tp_rank_zero(value: Any, device_mesh: Any) -> Any:
    import torch
    import torch.distributed as dist

    if not isinstance(value, torch.Tensor):
        return value

    # The custom Xingchen4 MoE consumes an ordinary replicated tensor. A
    # DTensor here would mean the preceding TP layer exposed the wrong layout.
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = ()  # type: ignore[assignment,misc]
    if isinstance(value, DTensor):
        raise RuntimeError(
            "Xingchen4 MoE received a DTensor instead of a replicated local "
            "tensor; the preceding target TP rule has an incompatible output layout."
        )

    synchronized = value.contiguous()
    dist.broadcast(
        synchronized,
        src=0,
        group=device_mesh.get_group(),
    )
    return synchronized


def install_moe_input_sync_hooks(
    model: Any,
    *,
    device_mesh: Any,
    broadcast_fn: Callable[[Any, Any], Any] | None = None,
) -> int:
    """Make routing and dynamic expert input shapes identical on all TP ranks.

    Xingchen4's Python MoE loops only over experts selected by local Top-K
    routing. Per-expert rowwise TP performs an all-reduce inside ``down_proj``;
    therefore every rank must select exactly the same tokens for every expert.
    Small rank-local differences before Top-K can otherwise produce different
    collective shapes. Synchronizing the fixed-shape MoE input before routing
    preserves both collective ordering and row correspondence.
    """
    existing = getattr(model, "_xingchen_moe_input_sync_hook_handles", None)
    if existing is not None:
        return len(existing)

    synchronize = broadcast_fn or _broadcast_from_tp_rank_zero

    def synchronize_first_input(_module: Any, inputs: tuple[Any, ...]):
        if not inputs:
            return inputs
        first = synchronize(inputs[0], device_mesh)
        return (first, *inputs[1:])

    handles = []
    for _module_name, module in model.named_modules():
        # Xingchen/DeepSeek-style sparse MLP blocks own both the router and the
        # expert collection. Do not attach to individual experts.
        if not hasattr(module, "experts"):
            continue
        if not (hasattr(module, "gate") or hasattr(module, "router")):
            continue
        handles.append(module.register_forward_pre_hook(synchronize_first_input))

    if not handles:
        raise RuntimeError(
            "Could not locate Xingchen4 MoE blocks with experts and a gate/router "
            "for TP input synchronization."
        )

    model._xingchen_moe_input_sync_hook_handles = tuple(handles)
    return len(handles)
