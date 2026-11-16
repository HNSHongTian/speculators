#!/usr/bin/env python3
"""Independent TRPO trainer entrypoint.

Reuses ``scripts/train.py`` setup (data, vLLM hidden states, DDP Trainer) but
replaces the draft ``forward`` with the TRPO objective. Existing CE/DSpark
training via ``scripts/train.py`` is unchanged.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import train as train_script  # noqa: E402

from speculators.train.trpo.step import compute_trpo_batch_loss  # noqa: E402

logger = logging.getLogger("speculators")


def _parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--tree-top-k", type=int, default=4)
    pre.add_argument("--tree-max-nodes", type=int, default=32)
    pre.add_argument(
        "--tree-pack-mode",
        type=str,
        default="leaf-up",
        choices=["leaf-up", "score"],
    )
    pre.add_argument("--lambda-ce", type=float, default=0.1)
    pre.add_argument("--max-trpo-anchors", type=int, default=8)
    pre.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="If >0, stop Trainer after this many optimizer steps (smoke trials).",
    )
    known, remaining = pre.parse_known_args()

    # Default backbone to dspark when user did not pass --speculator-type.
    argv0 = sys.argv[0]
    if "--speculator-type" not in remaining:
        remaining = ["--speculator-type", "dspark", *remaining]

    saved = sys.argv
    sys.argv = [argv0, *remaining]
    try:
        args = train_script.parse_args()
    finally:
        sys.argv = saved

    for key, value in vars(known).items():
        setattr(args, key, value)
    return args


def _install_trpo_forward(draft: torch.nn.Module, trpo_cfg: dict[str, Any]) -> None:
    """Replace draft.forward with TRPO loss; keep other methods intact."""

    def forward(  # noqa: ANN001
        self,
        hidden_states,
        input_ids,
        loss_mask,
        verifier_last_hidden_states,
        document_ids,
        position_ids=None,
        **kwargs,
    ):
        # Drop CE/DSpark loss kwargs that Trainer may still pass.
        for key in (
            "loss_config",
            "gamma",
            "confidence_head_alpha",
            "per_position_loss_weight",
            "dpace_alpha",
            "max_anchors",
        ):
            kwargs.pop(key, None)

        loss, metrics = compute_trpo_batch_loss(
            self,
            hidden_states=hidden_states,
            input_ids=input_ids,
            loss_mask=loss_mask,
            verifier_last_hidden_states=verifier_last_hidden_states,
            document_ids=document_ids,
            position_ids=position_ids,
            max_anchors=trpo_cfg["max_anchors"],
            max_trpo_anchors=trpo_cfg["max_trpo_anchors"],
            tree_top_k=trpo_cfg["tree_top_k"],
            tree_max_nodes=trpo_cfg["tree_max_nodes"],
            tree_pack_mode=trpo_cfg["tree_pack_mode"],
            lambda_ce=trpo_cfg["lambda_ce"],
        )
        # Trainer all-reduces metric values, so they must be tensors.
        counted = {}
        device = loss.device
        for key, value in metrics.items():
            counted[f"{key}_sum"] = torch.tensor(float(value), device=device)
            counted[f"{key}_total"] = torch.ones((), device=device)
        counted["loss_sum"] = torch.tensor(
            float(metrics.get("trpo_loss", 0.0)), device=device
        )
        counted["loss_total"] = torch.ones((), device=device)
        return None, loss, counted

    # Bind as unbound function then assign — works for SpeculatorModel instance.
    draft.forward = forward.__get__(draft, draft.__class__)  # type: ignore[method-assign]
    logger.info(
        "Installed TRPO forward (pack=%s top_k=%s max_nodes=%s max_trpo_anchors=%s)",
        trpo_cfg["tree_pack_mode"],
        trpo_cfg["tree_top_k"],
        trpo_cfg["tree_max_nodes"],
        trpo_cfg["max_trpo_anchors"],
    )


def _patch_build_and_trainer(args: argparse.Namespace) -> None:
    trpo_cfg = {
        "max_anchors": int(args.max_anchors),
        "max_trpo_anchors": int(args.max_trpo_anchors),
        "tree_top_k": int(args.tree_top_k),
        "tree_max_nodes": int(args.tree_max_nodes),
        "tree_pack_mode": str(args.tree_pack_mode),
        "lambda_ce": float(args.lambda_ce),
    }
    max_steps = int(getattr(args, "max_steps", 0) or 0)

    orig_build = train_script.build_draft_model

    def build_draft_model(*a, **kw):  # noqa: ANN001
        model = orig_build(*a, **kw)
        _install_trpo_forward(model, trpo_cfg)
        return model

    train_script.build_draft_model = build_draft_model  # type: ignore[assignment]

    # Confidence head is unused by TRPO; allow unused params under DDP.
    from torch.nn.parallel import DistributedDataParallel as _DDP
    from speculators.train import trainer as trainer_mod

    _orig_ddp = _DDP

    def _ddp_unused_ok(module, *a, **kw):  # noqa: ANN001
        kw.setdefault("find_unused_parameters", True)
        return _orig_ddp(module, *a, **kw)

    trainer_mod.DistributedDataParallel = _ddp_unused_ok  # type: ignore[misc,assignment]

    if max_steps > 0:
        from speculators.train import trainer as trainer_mod

        orig_train_epoch = trainer_mod.Trainer.train_epoch

        def train_epoch(self, epoch):  # noqa: ANN001
            # Run original but stop early via monkeypatched loop body.
            return _train_epoch_with_max_steps(self, epoch, max_steps, orig_train_epoch)

        # Simpler: wrap run_training
        orig_run = trainer_mod.Trainer.run_training

        def run_training(self):  # noqa: ANN001
            self._trpo_max_steps = max_steps
            self._trpo_steps_done = 0
            return _run_training_max_steps(self, orig_run)

        trainer_mod.Trainer.run_training = run_training  # type: ignore[assignment]
        del train_epoch  # placeholder kept for clarity


def _run_training_max_steps(trainer, orig_run):  # noqa: ANN001
    """Run a truncated training loop when --max-steps is set."""
    from speculators.train.distributed import get_rank

    max_steps = int(getattr(trainer, "_trpo_max_steps", 0) or 0)
    if max_steps <= 0:
        return orig_run(trainer)

    logger.info("TRPO smoke mode: stopping after %s optimizer steps", max_steps)

    trainer.model.train()
    steps = 0
    device = next(trainer.model.parameters()).device
    for batch in trainer.train_loader:
        gpu_batch = {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        for opt in trainer.optimizers:
            opt.zero_grad(set_to_none=True)
        _tok, loss, metrics = trainer.model(
            **gpu_batch, **trainer.config.train_call_kwargs
        )
        loss.backward()
        for opt in trainer.optimizers:
            opt.step()
        for scheduler in getattr(trainer, "schedulers", []) or []:
            scheduler.step()
        steps += 1
        if get_rank() == 0 and steps % max(1, trainer.config.log_freq) == 0:
            logger.info(
                "trpo_step=%s loss=%s metrics=%s",
                steps,
                float(loss.detach().item()),
                {k: metrics[k] for k in list(metrics)[:8]},
            )
        if steps >= max_steps:
            break
    if get_rank() == 0:
        out = Path(trainer.config.save_path) / "trpo_smoke"
        out.mkdir(parents=True, exist_ok=True)
        raw = (
            trainer.model.module
            if hasattr(trainer.model, "module")
            else trainer.model
        )
        raw.save_pretrained(str(out))
        logger.info("Saved TRPO smoke checkpoint to %s", out)


def main():
    args = _parse_args()
    _patch_build_and_trainer(args)
    logger.info(
        "Launching TRPO via train.main (speculator=%s data=%s)",
        args.speculator_type,
        args.data_path,
    )
    train_script.main(args)


if __name__ == "__main__":
    main()
