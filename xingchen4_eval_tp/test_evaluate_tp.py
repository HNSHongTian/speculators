"""CPU regression tests for the TP replica launcher and DP metrics."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import evaluate_tp as evaluate
from tp_plan import build_xingchen_tp_plan


class TensorParallelEvaluationTests(unittest.TestCase):
    def test_two_isolated_groups_and_worker_commands(self):
        groups = evaluate.device_groups(",".join(map(str, range(16))), 8, 2)
        self.assertEqual(groups, [list(map(str, range(8))), list(map(str, range(8, 16)))])
        with patch.object(evaluate.subprocess, "Popen", return_value=Mock(poll=lambda: 0)) as start:
            evaluate.launch_workers(groups, Path("/tmp"), ["--seed", "3"])
        for replica, call in enumerate(start.call_args_list):
            cmd = call.args[0]
            self.assertIn("--standalone", cmd)
            self.assertIn("--nproc-per-node=8", cmd)
            self.assertEqual(cmd[-4:], [str(replica), "/tmp", "--seed", "3"])
            self.assertEqual(call.kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"], ",".join(groups[replica]))

    def test_invalid_topology(self):
        for visible in ("0,1", "0,1,1,2"):
            with self.assertRaises(ValueError):
                evaluate.device_groups(visible, 2, 2)

    def test_plan_splits_attention_and_individual_experts(self):
        plan = build_xingchen_tp_plan([
            f"model.layers.0.{name}.weight" for name in
            ("self_attn.q_b_proj", "self_attn.kv_b_proj", "self_attn.o_proj",
             "mlp.experts.0.up_proj", "mlp.experts.0.down_proj")
        ] + ["lm_head.weight"])
        self.assertEqual(plan["model.layers.*.self_attn.kv_b_proj"], "colwise")
        self.assertEqual(plan["model.layers.*.mlp.experts.*.down_proj"], "rowwise")
        self.assertNotIn("ep_router", plan.values())

    def test_merge_uses_counts_and_restores_sample_order(self):
        legacy = evaluate.ep._load_legacy_evaluator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = evaluate.parse_args(["--verifier-model", "v", "--draft-model", "d",
                                        "--datasets-root", tmp, "--output-dir", str(root / "final")])
            for replica, proposals, accepted in [(0, 8, 14), (1, 2, 1)]:
                samples = legacy._shard_records([{}, {}, {}], shard_index=replica, num_shards=2)
                artifacts = [{"source_index": i} for i, _ in samples]
                stats = legacy.EvalStats(elapsed_s=1, num_proposals=proposals,
                                         num_proposed_draft_tokens=3*proposals,
                                         num_accepted_draft_tokens=accepted)
                row = legacy._summary_row("agent", len(samples), stats)
                legacy._write_outputs(root / str(replica) / "agent", [row], {"agent": artifacts})
            evaluate.merge_results(args, root, legacy, [Path("agent.jsonl")])
            summary, = json.loads((args.output_dir / "summary.json").read_text())
            self.assertEqual(summary["num_requests"], 3)
            self.assertEqual(summary["acceptance_length"], 2.5)
            self.assertEqual([r["source_index"] for r in
                              legacy._read_worker_artifacts(args.output_dir, "agent")], [1, 2, 3])
