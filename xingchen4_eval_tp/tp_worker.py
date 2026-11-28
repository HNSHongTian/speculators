"""One rank of an isolated torchrun TP replica. Invoked by evaluate_tp.py."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from evaluate_tp import ep, parse_args
from moe_sync import install_moe_input_sync_hooks
from tp_plan import load_or_build_tp_plan, validate_parallel_styles, write_tp_plan


def check_local_shards(target, config, tp):
    """Catch plans accepted by Transformers that miss the checkpoint's real names."""
    attention = experts = 0
    for name, weight in target.named_parameters():
        if name.endswith(".self_attn.o_proj.weight"):
            head_dim = getattr(config, "v_head_dim", None) or config.head_dim
            full_width = config.num_attention_heads * head_dim
            attention += 1
        elif ".mlp.experts." in name and name.endswith((".down_proj.weight", ".down_proj")):
            full_width = config.moe_intermediate_size
            experts += 1
        else:
            continue
        local = weight.to_local() if hasattr(weight, "to_local") else weight
        if local.shape[-1] * tp != full_width:
            raise ValueError(f"{name}: local width {local.shape[-1]}, expected {full_width} / {tp}")
    if not attention or not experts:
        raise ValueError("Could not verify local attention and routed-expert TP shards")


def run(replica, staging, args):
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

    tp = args.tensor_parallel_size
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != tp:
        raise ValueError("Worker WORLD_SIZE must equal tensor-parallel-size")
    device = torch.device("npu", local_rank)
    torch.npu.set_device(device)
    dist.init_process_group("hccl")
    try:
        mesh = init_device_mesh("npu", (tp,), mesh_dim_names=("tp",))
        leader = dist.get_rank() == 0
        logging.basicConfig(level=logging.INFO if leader else logging.WARNING,
                            format=f"[DP{replica}/TP{local_rank}] %(message)s")
        legacy = ep._load_legacy_evaluator()
        legacy.torch, legacy.DynamicCache = torch, DynamicCache
        torch.manual_seed(args.seed)
        torch.npu.manual_seed_all(args.seed)
        args.device = str(device)
        args.worker_shard_index = replica
        args.worker_num_shards = args.data_parallel_size
        args.skip_artifacts = args.skip_artifacts or not leader
        args.no_progress = args.no_progress or not leader

        draft = legacy.load_dspark_draft(args, device)
        ep.install_draft_attention_cat_compatibility(draft)
        load_options = dict(trust_remote_code=args.trust_remote_code, local_files_only=args.local_files_only)
        config = AutoConfig.from_pretrained(args.verifier_model, **load_options)
        if config.num_attention_heads % tp:
            raise ValueError(f"{config.num_attention_heads} attention heads cannot split across TP{tp}")
        plan, _ = load_or_build_tp_plan(model_path=args.verifier_model, tp_plan_path=args.tp_plan)
        validate_parallel_styles(plan)
        target = AutoModelForCausalLM.from_pretrained(
            args.verifier_model, config=config, **load_options,
            torch_dtype="auto" if args.dtype == "auto" else getattr(torch, args.dtype),
            tp_plan=plan, device_mesh=mesh, low_cpu_mem_usage=True,
        ).eval()
        if not getattr(target, "tp_plan", None):
            raise RuntimeError("Verifier loaded without a Transformers tensor-parallel plan")
        check_local_shards(target, config, tp)
        install_moe_input_sync_hooks(target, device_mesh=mesh)
        if leader and replica == 0:
            write_tp_plan(staging / "resolved_tp_plan.json", plan)
        target = ep.LocalOutputTargetAdapter(
            target, list(draft.target_layer_ids), expected_feature_width=int(draft.fc.in_features)
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.verifier_model, **load_options)
        runner = legacy.DSparkOfflineRunner(target, draft, tokenizer, args)
        base = legacy.BaseModelOfflineRunner(target, tokenizer, args) if args.measure_base_speedup else None
        stop_ids = legacy.resolve_stop_token_ids(target, tokenizer)
        for filename in json.loads((staging / "datasets.json").read_text()):
            path = Path(filename)
            dist.barrier()
            row, records = legacy._evaluate_dataset(
                path=path, runner=runner, base_runner=base, args=args, stop_token_ids=stop_ids
            )
            torch.npu.synchronize(device)
            dist.barrier()
            if leader:
                legacy._write_outputs(staging / str(replica) / path.stem, [row],
                                      {} if args.skip_artifacts else {path.stem: records})
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(int(sys.argv[1]), Path(sys.argv[2]), parse_args(sys.argv[3:]))
