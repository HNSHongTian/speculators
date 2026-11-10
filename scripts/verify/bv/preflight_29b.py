"""Read-only deployment checks; never starts training or changes remote data."""

import argparse
import contextlib
import json
from pathlib import Path

# Lazy optional imports keep config-only checks usable without accelerators.
# ruff: noqa: PLC0415
DRAFT_LAYERS = 5
MASK_TOKEN_ID = 139
MAX_TARGET_LAYER = 37
BLOCK_SIZE = 15


def check_files(  # noqa: C901
    model: Path, draft: Path, data: Path, output: Path, anchors: int
) -> dict:
    teacher = json.loads((model / "config.json").read_text())
    teacher = teacher.get("text_config", teacher)
    config = json.loads(draft.read_text())
    if not config.get("model_type"):
        raise ValueError("draft config must declare model_type")
    if config.get("num_hidden_layers") != DRAFT_LAYERS:
        raise ValueError("29B BV recipe requires a five-layer draft config")
    if config.get("hidden_size") != teacher.get("hidden_size"):
        raise ValueError("draft and verifier hidden_size must match")
    vocab = teacher["vocab_size"]
    if config.get("vocab_size", vocab) != vocab:
        raise ValueError("draft config must use the full verifier vocabulary")
    if (
        vocab <= MASK_TOKEN_ID
        or teacher.get("num_hidden_layers", 0) <= MAX_TARGET_LAYER
    ):
        raise ValueError("mask_token_id=139 / target-layer-ids 1,10,19,28,37 invalid")
    if anchors < 1:
        raise ValueError("max anchors must be positive")
    if not data.is_dir():
        raise ValueError(f"Arrow dataset directory does not exist: {data}")
    for folder in (output / "checkpoints").glob("*"):
        if folder.is_dir() and folder.name.isdecimal():
            state_file = folder / "training_state.json"
            if not state_file.exists() or "bv" not in json.loads(
                state_file.read_text()
            ):
                raise ValueError(
                    "Existing non-BV checkpoint: use a new output directory"
                )
    return {
        "vocab_size": vocab,
        "draft_layers": 5,
        "max_anchors": anchors,
        "one_fp32_logits_gb": 4 * anchors * 15 * vocab / 1e9,
    }


def main():  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft-config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-anchors", type=int, default=128)
    parser.add_argument("--require-accelerator", action="store_true")
    args = parser.parse_args()
    result = check_files(
        args.model, args.draft_config, args.data_path, args.output_dir, args.max_anchors
    )
    import torch
    from datasets import load_from_disk

    import speculators
    from speculators.models.dspark.bv_loss import integrated_log_scores

    root = Path(__file__).resolve().parents[3]
    if not Path(speculators.__file__).resolve().is_relative_to(root / "src"):
        raise ValueError("Python imports another speculators checkout")
    if args.require_accelerator:
        # torch_npu installs/registers the Ascend backend in deployment environments.
        with contextlib.suppress(ImportError):
            import torch_npu  # noqa: F401
        accelerator = torch.accelerator.current_accelerator()
        if accelerator is None:
            raise ValueError("No training accelerator is available")
        result["accelerator"] = accelerator.type
    dataset = load_from_disk(str(args.data_path))
    if not {"input_ids", "loss_mask"} <= set(dataset.column_names):
        raise ValueError("Arrow data requires input_ids and token-aligned loss_mask")
    if not len(dataset):
        raise ValueError("Arrow dataset is empty")
    complete = 0
    for row in dataset.select(range(min(32, len(dataset)))):
        ids, mask = row["input_ids"], row["loss_mask"]
        if len(ids) != len(mask) or any(x not in (0, 1, False, True) for x in mask):
            raise ValueError("Invalid token/mask lengths or nonbinary assistant mask")
        if any(x < 0 or x >= result["vocab_size"] for x in ids):
            raise ValueError("Dataset token outside full verifier vocabulary")
        run = 0
        for value in mask[:3072]:
            run = run + 1 if value else 0
            complete += int(
                run >= BLOCK_SIZE + 1
            )  # current anchor selector needs masked anchor too
    if not complete:
        raise ValueError("No complete 15-proposal assistant block in first 32 rows")
    # Tiny CPU oracle proves the requested module is importable, not hardware readiness.
    scores = integrated_log_scores(
        torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), torch.zeros(1, 1, dtype=torch.long)
    )
    torch.testing.assert_close(scores, torch.zeros_like(scores))
    result.update(dataset_rows=len(dataset), sampled_complete_blocks=complete)
    print(json.dumps(result, indent=2))
    print(
        "PASS: local configs/data/import checks. Verify T=1 provenance, token-mask "
        "semantics, endpoints and hidden-state norm/layer conventions separately."
    )


if __name__ == "__main__":
    main()
