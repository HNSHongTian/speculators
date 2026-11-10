"""Compare actual distributed BV gradients/updates with a concatenated oracle.

Run with torch.distributed.run --standalone --nproc_per_node=2. gloo uses CPU;
nccl/hccl exercise deployment devices. --fsdp tests FSDP2 on accelerators.
"""

# Optional device backends are loaded only after CLI parsing.
# ruff: noqa: PLC0415
import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from speculators.models.dspark.bv_loss import bv_loss
from speculators.models.dspark.metrics import compute_metrics
from speculators.models.metrics import resolve_loss_config
from speculators.train.bv import scale_bv_for_backward

REQUIRED_WORLD_SIZE = 2


def full(tensor):
    return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor


def main():  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["gloo", "nccl", "hccl"], default="gloo")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--bv-weight", type=float, default=1.0)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cpu")
    if args.backend == "hccl":
        import torch_npu  # noqa: F401

        device = torch.device("npu", local_rank)
        torch.npu.set_device(device)
    elif args.backend == "nccl":
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    if args.fsdp and device.type == "cpu":
        parser.error("FSDP verification requires accelerator hardware")
    dist.init_process_group(args.backend)

    def objective(q, p, y, valid, beta, chunk):
        if args.bv_weight == 1.0 and args.tv_weight == 0.0:
            return bv_loss(q, p, y, valid, beta, annealing=True, chunk_size=chunk)
        n, block, vocab = q.shape
        return compute_metrics(
            q.reshape(1, n * block, vocab),
            p.reshape(1, n * block, vocab),
            None,
            torch.ones(1, n * block, device=device),
            block,
            resolve_loss_config("ce"),
            enable_bv_loss=True,
            bv_labels=y,
            bv_valid_blocks=valid,
            bv_beta=beta,
            bv_annealing=True,
            bv_loss_weight=args.bv_weight,
            tv_loss_weight=args.tv_weight,
            bv_chunk_size=chunk,
        )

    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != REQUIRED_WORLD_SIZE:
            raise ValueError("This verifier requires exactly two ranks")
        torch.manual_seed(17)
        module = torch.nn.Linear(4, 7, bias=False).to(device)
        reference = torch.nn.Linear(4, 7, bias=False).to(device)
        reference.load_state_dict(module.state_dict())
        if args.fsdp:
            from torch.distributed.fsdp import fully_shard

            fully_shard(module)
            model = module
        else:
            model = DistributedDataParallel(module)
        x = torch.randn(4, 3, 4, device=device)
        p = torch.randn(4, 3, 7, device=device)
        y = torch.randint(7, (4, 3), device=device)
        beta = torch.tensor(0.4, device=device)
        sl = slice(rank * 2, (rank + 1) * 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        for mask in [
            [True, True, True, False],
            [False, False, True, True],
            [False] * 4,
        ]:
            valid = torch.tensor(mask, device=device)
            opt.zero_grad()
            ref_opt.zero_grad()
            q = model(x[sl])
            local, metrics = objective(q, p[sl], y[sl], valid[sl], beta, 1)
            scaled, update = scale_bv_for_backward(local, metrics["bv_loss_total"])
            scaled.backward()
            expected, _ = objective(reference(x), p, y, valid, beta, 4)
            expected.backward()
            for actual, ref in zip(
                model.parameters(), reference.parameters(), strict=True
            ):
                torch.testing.assert_close(
                    full(actual.grad), ref.grad, atol=2e-6, rtol=2e-5
                )
            before = [full(param).detach().clone() for param in model.parameters()]
            if update:
                opt.step()
                ref_opt.step()
            for actual, ref, old in zip(
                model.parameters(), reference.parameters(), before, strict=True
            ):
                torch.testing.assert_close(full(actual), ref, atol=2e-6, rtol=2e-5)
                if not update:
                    torch.testing.assert_close(full(actual), old, atol=0, rtol=0)
        if rank == 0:
            print(
                f"PASS: {args.backend} {'FSDP2' if args.fsdp else 'DDP'} "
                f"BV={args.bv_weight}, TV={args.tv_weight}; "
                "gradients and AdamW updates; unequal, empty-rank, all-empty blocks"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
