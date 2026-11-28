"""Single-node TP × DP launcher for Xingchen4 DSpark offline evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parent))
from xingchen4_eval_ep import evaluate_ep as ep


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--data-parallel-size", type=int, default=2)
    parser.add_argument("--tp-plan", type=Path, help="Optional checkpoint-specific Transformers TP plan")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--block-verification",
        action="store_true",
        help=(
            "Use lossless block verification for linear DSpark. When unset, "
            "keep the default token-by-token verification."
        ),
    )
    parser.add_argument(
        "--tree-top-k",
        type=int,
        default=1,
        help="Per-parent branching factor; values >=2 enable TreeSpark mode.",
    )
    parser.add_argument(
        "--tree-max-nodes",
        type=int,
        default=32,
        help="Maximum non-root tree nodes in each target verification.",
    )
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument(
        "--enable-thinking",
        choices=("false", "true", "default"),
        default="false",
    )
    parser.add_argument(
        "--raw-prompt-mode",
        choices=("auto", "chat_template", "raw"),
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--draft-attn-impl",
        choices=("sdpa", "eager"),
        default="sdpa",
    )
    parser.add_argument("--d2t-path", type=Path, default=None)
    parser.add_argument("--t2d-path", type=Path, default=None)
    parser.add_argument(
        "--sample-from-anchor",
        choices=("true", "false"),
        default=None,
    )
    parser.add_argument("--measure-base-speedup", action="store_true")
    parser.add_argument("--throughput-warmup-samples", type=int, default=1)
    parser.add_argument("--skip-artifacts", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def device_groups(visible, tp, dp):
    if tp < 1 or dp < 1:
        raise ValueError("TP and DP sizes must be positive")
    devices = [item.strip() for item in visible.split(",")]
    if len(devices) != tp * dp or len(set(devices)) != len(devices) or not all(
        item.isdecimal() for item in devices
    ):
        raise ValueError(f"Specify exactly {tp * dp} distinct NPU IDs in ASCEND_RT_VISIBLE_DEVICES")
    return [devices[start:start + tp] for start in range(0, len(devices), tp)]


def launch_workers(groups, staging, argv):
    processes = []
    try:
        for replica, devices in enumerate(groups):
            env = os.environ.copy()
            env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(devices)
            cmd = [
                sys.executable, "-m", "torch.distributed.run", "--standalone",
                f"--nproc-per-node={len(devices)}", str(DIRECTORY / "tp_worker.py"),
                str(replica), str(staging), *argv,
            ]
            print(f"Starting DP replica {replica} on NPUs {env['ASCEND_RT_VISIBLE_DEVICES']}", flush=True)
            processes.append(subprocess.Popen(cmd, env=env))
        while True:
            statuses = [process.poll() for process in processes]
            for replica, status in enumerate(statuses):
                if status not in (None, 0):
                    raise RuntimeError(f"DP replica {replica} exited with status {status}")
            if all(status == 0 for status in statuses):
                return
            time.sleep(0.5)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()


def merge_results(args, staging, legacy, dataset_paths):
    rows, artifacts = [], {}
    for path in dataset_paths:
        name = path.stem
        shards = [staging / str(replica) / name for replica in range(args.data_parallel_size)]
        shard_rows = [legacy._read_worker_row(shard) for shard in shards]
        rows.append(legacy._aggregate_rows(name, shard_rows))
        if not args.skip_artifacts:
            records = []
            for shard, row in zip(shards, shard_rows):
                part = legacy._read_worker_artifacts(shard, name)
                if len(part) != row["num_requests"]:
                    raise RuntimeError(f"Incomplete artifacts for {shard}")
                records.extend(part)
            records.sort(key=lambda record: record["source_index"])
            if any(r["source_index"] != i for i, r in enumerate(records, 1)):
                raise RuntimeError(f"Duplicate or missing source indices in {name}")
            artifacts[name] = records
    legacy._write_outputs(args.output_dir, rows, artifacts)


def main():
    argv = sys.argv[1:]
    args = parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Launch evaluate_tp.py once with python; it starts torchrun for each TP group")
    if args.temperature != 0 or args.tree_top_k < 1 or args.tree_max_nodes < 1:
        raise ValueError("Use temperature=0, tree-top-k >= 1 and tree-max-nodes >= 1")
    if args.tree_top_k > 1 and args.block_verification:
        raise ValueError("Block verification requires tree-top-k=1")
    expected = args.tensor_parallel_size * args.data_parallel_size
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ",".join(map(str, range(expected))))
    groups = device_groups(visible, args.tensor_parallel_size, args.data_parallel_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    legacy = ep._load_legacy_evaluator()
    paths = legacy._discover_datasets(args.datasets_root, legacy._split_csv(args.datasets) or None)
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("Selected datasets must have distinct names for summary and artifact files")
    with tempfile.TemporaryDirectory(dir=args.output_dir, prefix=".tp-shards-") as temp:
        staging = Path(temp)
        (staging / "datasets.json").write_text(json.dumps([str(path) for path in paths]))
        launch_workers(groups, staging, argv)
        merge_results(args, staging, legacy, paths)
        (staging / "resolved_tp_plan.json").replace(args.output_dir / "resolved_tp_plan.json")
    config = dict(vars(args), parallelism="tensor_and_data_parallel", device_groups=groups)
    (args.output_dir / "pytorch_tp_run_config.json").write_text(
        json.dumps(config, default=str, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    # Unwind launch_workers' finally block when a background job is stopped.
    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    main()
