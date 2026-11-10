"""Integration checks for the migrated DSpark offline evaluator dependency."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import evaluate_ep  # noqa: E402


class OfflineEvaluatorDependencyTests(unittest.TestCase):
    def test_loads_migrated_evaluator_with_required_interface(self) -> None:
        evaluator = evaluate_ep._load_legacy_evaluator()

        required_names = (
            "load_dspark_draft",
            "_parse_bool_override",
            "_load_vocab_mapping_tensors",
            "_ensure_loaded_vocab_mappings",
            "build_dspark_tree_proposal",
            "build_tree_attention_mask",
            "verify_draft_tree",
            "block_verification_probabilities",
            "verify_draft_block",
            "DSparkOfflineRunner",
            "BaseModelOfflineRunner",
            "resolve_stop_token_ids",
            "_discover_datasets",
            "_split_csv",
            "_evaluate_dataset",
            "_write_outputs",
        )
        for name in required_names:
            with self.subTest(name=name):
                self.assertTrue(hasattr(evaluator, name), name)

    def test_block_verification_cli_is_opt_in(self) -> None:
        required = [
            "evaluate_ep.py",
            "--verifier-model",
            "target",
            "--draft-model",
            "draft",
            "--datasets-root",
            "data",
            "--output-dir",
            "output",
            "--expert-parallel-size",
            "2",
        ]
        with patch.object(sys, "argv", required):
            self.assertFalse(evaluate_ep.parse_args().block_verification)
        with patch.object(sys, "argv", [*required, "--block-verification"]):
            self.assertTrue(evaluate_ep.parse_args().block_verification)


if __name__ == "__main__":
    unittest.main()
