"""Step 3: CLI, real Trainer loop, annealing and checkpoint state."""

import json
import runpy
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.train import parse_args
from speculators.models.dspark.bv_loss import bv_loss
from speculators.models.dspark.core import DSparkDraftModel
from speculators.train.bv import bv_progress, scale_bv_for_backward
from speculators.train.trainer import Trainer, TrainerConfig


def parse(monkeypatch, *args):
    monkeypatch.setattr(
        "sys.argv", ["train.py", "--verifier-name-or-path", "test", *args]
    )
    return parse_args()


def test_cli_default_and_enabled_kwargs(monkeypatch):
    old = parse(monkeypatch, "--speculator-type", "dspark")
    assert not old.enable_bv_loss
    assert old.loss_fn == "kl_div"
    train, val = DSparkDraftModel.get_trainer_kwargs(**vars(old))
    assert "enable_bv_loss" not in train
    assert "bv_beta" not in val
    enabled = parse(
        monkeypatch,
        "--speculator-type",
        "dspark",
        "--enable-bv-loss",
        "--loss-fn",
        '{"ce":0.1,"tv":0.9}',
    )
    train, val = DSparkDraftModel.get_trainer_kwargs(**vars(enabled))
    assert train["enable_bv_loss"]
    assert val["enable_bv_loss"]
    assert enabled.bv_anneal_epochs == 3
    assert enabled.bv_chunk_size == 16


@pytest.mark.parametrize(
    "args",
    [
        ["--speculator-type", "dflash"],
        ["--speculator-type", "dspark", "--bv-anneal-epochs", "-1"],
        ["--speculator-type", "dspark", "--bv-anneal-epochs", "nan"],
        ["--speculator-type", "dspark", "--bv-chunk-size", "0"],
        [
            "--speculator-type",
            "dspark",
            "--loss-fn",
            "ce",
            "--per-position-loss-weight",
            "dpace",
        ],
        ["--speculator-type", "dspark", "--no-sample-from-anchor", "--block-size", "1"],
    ],
)
def test_cli_rejects_incompatible_bv(monkeypatch, args):
    with pytest.raises(SystemExit):
        parse(monkeypatch, "--enable-bv-loss", *args)


def test_epoch_clock_and_disabled_anneal():
    assert bv_progress(0, 1, 100, 3) == (0.0, True)
    assert bv_progress(1, 51, 100, 3) == (0.5, True)
    assert bv_progress(2, 100, 100, 3)[0] < 1
    assert bv_progress(3, 1, 100, 3) == (1.0, False)
    assert bv_progress(0, 1, 100, 0) == (1.0, False)


class TinyBVModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.randn(2, 3, 5))
        self.calls = []

    def forward(self, hidden_states, valid, **kwargs):
        self.calls.append(
            (
                float(kwargs["bv_beta"]),
                kwargs["bv_annealing"],
                torch.rand(()).item(),
                kwargs.get("bv_loss_weight", 1.0),
                kwargs.get("ce_loss_weight", 0.0),
                kwargs.get("tv_loss_weight", 0.0),
                kwargs.get("confidence_head_alpha", 1.0),
            )
        )
        loss, metrics = bv_loss(
            self.logits,
            hidden_states,
            torch.ones(2, 3, dtype=torch.long),
            valid,
            kwargs["bv_beta"],
            annealing=kwargs["bv_annealing"],
        )
        metrics["loss_sum"] = metrics["bv_loss_sum"].clone()
        metrics["loss_total"] = metrics["bv_loss_total"].clone()
        return None, loss, metrics


class TinyLoader:
    def __init__(self, valid=True):
        self.dataset = SimpleNamespace(data=SimpleNamespace(_fingerprint="fixed-data"))
        self.batch_sampler = SimpleNamespace(set_epoch=lambda e: None)
        self.batches = [
            {
                "hidden_states": torch.zeros(2, 3, 5),
                "valid": torch.full((2,), valid),
                "document_ids": torch.zeros(1, 8),
            }
            for _ in range(4)
        ]

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def trainer(tmp_path, valid=True):
    t = Trainer.__new__(Trainer)
    t.model = TinyBVModel()
    t.config = TrainerConfig(
        lr=1e-3,
        num_epochs=6,
        save_path=str(tmp_path),
        train_call_kwargs={"enable_bv_loss": True},
        val_call_kwargs={"enable_bv_loss": True},
        bv_recipe={"seed": 42},
    )
    t.local_rank, t.rank, t.device_type = torch.device("cpu"), 0, "cpu"
    t.train_loader, t.val_loader = TinyLoader(valid), TinyLoader(valid)
    t.is_distributed = False
    t.resume_from_checkpoint = False
    t.checkpointer = SimpleNamespace(
        path=tmp_path, previous_epoch=-1, load_best_val_loss=lambda: None
    )
    t.setup_trainer()
    t.optimizers = [torch.optim.AdamW(t.model.parameters(), lr=1e-3)]
    t.schedulers = [torch.optim.lr_scheduler.LambdaLR(t.optimizers[0], lambda _: 1)]
    return t


def test_actual_trainer_loop_and_validation_fixed_objective(tmp_path):
    t = trainer(tmp_path)
    before = t.model.logits.detach().clone()
    t.train_epoch(0)
    assert not torch.equal(before, t.model.logits)
    assert t._bv_optimizer_updates == 4
    assert t.global_step == 4
    assert [c[0] for c in t.model.calls] == pytest.approx([0, 1 / 12, 2 / 12, 3 / 12])
    assert "bv_beta" not in t.config.train_call_kwargs
    state = torch.random.get_rng_state().clone()
    result = t.val_epoch(0)
    torch.testing.assert_close(torch.random.get_rng_state(), state)
    assert all(c[:2] == (1.0, False) for c in t.model.calls[-4:])
    assert result["loss_epoch"] == result["bv_loss_epoch"]
    assert t.val_epoch(5) == result


def test_empty_batch_no_weight_decay_or_scheduler_step(tmp_path):
    t = trainer(tmp_path, valid=False)
    before = t.model.logits.detach().clone()
    last = t.schedulers[0].last_epoch
    t.train_epoch(0)
    torch.testing.assert_close(before, t.model.logits, rtol=0, atol=0)
    assert t.schedulers[0].last_epoch == last
    assert t._bv_optimizer_updates == 0
    with pytest.raises(ValueError, match="no complete valid blocks"):
        t.val_epoch(0)


def test_mid_epoch_resume_clock_rng_and_recipe_validation(tmp_path):
    t = trainer(tmp_path)
    t.global_step, t._bv_optimizer_updates = 2, 2
    t._save_training_state(1, local_step=2)
    saved = json.loads((tmp_path / "1/training_state.json").read_text())
    assert saved["bv"]["optimizer_updates"] == 2
    t.checkpointer.previous_epoch = 1
    t.checkpointer.prev_path = tmp_path / "1"
    t.resume_from_checkpoint = True
    t.setup_trainer()
    assert t.current_epoch == 1
    assert t._resume_local_step == 2
    t.train_epoch(1)
    assert len(t.model.calls) == 2
    assert t.model.calls[0][0] == pytest.approx(0.5)
    fresh = trainer(tmp_path / "fresh")
    fresh.train_epoch(1)
    assert [c[2] for c in t.model.calls] == [c[2] for c in fresh.model.calls[2:]]
    t.config = t.config._replace(bv_anneal_epochs=2)
    with pytest.raises(ValueError, match="recipe"):
        t.setup_trainer()


def test_resume_changed_loader_length_rejected(tmp_path):
    t = trainer(tmp_path)
    t._bv_resume_steps = 5
    with pytest.raises(ValueError, match="length changed"):
        t.train_epoch(0)


def test_single_process_block_normalization():
    x = torch.tensor(2.0, requires_grad=True)
    scaled, update = scale_bv_for_backward(x, torch.tensor(3.0))
    assert scaled == x
    assert update
    scaled, update = scale_bv_for_backward(x, torch.tensor(0.0))
    assert scaled == 0
    assert not update


def test_optimizer_scheduler_resume_restores_next_update(tmp_path):
    source = trainer(tmp_path / "source")
    source.config = source.config._replace(
        scheduler_type="cosine", scheduler_warmup_ratio=0.04
    )
    source.setup_optimizer()
    source.train_epoch(0)
    optim_state = deepcopy(source.optimizers[0].state_dict())
    scheduler_state = deepcopy(source.schedulers[0].state_dict())
    resumed = trainer(tmp_path / "resumed")
    resumed.config = source.config
    resumed.model.load_state_dict(source.model.state_dict())
    resumed.resume_from_checkpoint = True
    resumed.checkpointer.previous_epoch = 0
    resumed.checkpointer.load_optimizer_state_dict = lambda model, opts: opts[
        0
    ].load_state_dict(optim_state)
    resumed.checkpointer.load_scheduler_state_dict = lambda schedulers: schedulers[
        0
    ].load_state_dict(scheduler_state)
    resumed.setup_optimizer()
    assert (
        resumed.optimizers[0].param_groups[0]["lr"]
        == source.optimizers[0].param_groups[0]["lr"]
    )
    source.train_epoch(1)
    resumed.train_epoch(1)
    torch.testing.assert_close(
        source.model.logits, resumed.model.logits, rtol=0, atol=0
    )


def test_save_best_also_persists_bv_training_state(tmp_path):
    t = trainer(tmp_path)
    t.config = t.config._replace(save_best=True)
    t.checkpointer.save_checkpoint = lambda *args: None
    t.checkpointer.save_scheduler_state_dict = lambda *args: None
    t.checkpointer.save_val_metrics = lambda *args: None
    t.checkpointer.update_best_symlink = lambda *args: None
    t.checkpointer.cleanup_keep_only_best = lambda **kwargs: None
    t.maybe_update_best(0, {"loss_epoch": 0.5})
    state = json.loads((tmp_path / "0/training_state.json").read_text())
    assert state["bv"]["recipe"]["anneal_epochs"] == 3
    assert state["local_step"] == 0


def test_disabled_trainer_matches_pre_bv_loop_and_rng(tmp_path):
    legacy = runpy.run_path(
        str(Path(__file__).parents[2] / "fixtures/dspark_trainer_pre_bv.py")
    )["LegacyEpochs"]

    class LegacyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.4))
            self.calls = []

        def forward(self, **kwargs):
            assert "bv_beta" not in kwargs
            draw = torch.rand(())
            self.calls.append(draw.clone())
            loss = (self.weight - draw).square()
            return (
                None,
                loss,
                {"loss_sum": loss.detach(), "loss_total": torch.tensor(1.0)},
            )

    outputs = []
    for old in [True, False]:
        t = trainer(tmp_path / str(old))
        t.config = t.config._replace(
            train_call_kwargs={}, val_call_kwargs={}, bv_recipe=None
        )
        t.model = LegacyModel()
        t.optimizers = [torch.optim.AdamW(t.model.parameters(), lr=1e-3)]
        t.schedulers = [torch.optim.lr_scheduler.LambdaLR(t.optimizers[0], lambda _: 1)]
        torch.manual_seed(10)
        if old:
            legacy.train_epoch(t, 0)
            val = legacy.val_epoch(t, 0)
        else:
            t.train_epoch(0)
            val = t.val_epoch(0)
        t._save_training_state(0, 0)
        state = json.loads((tmp_path / str(old) / "0/training_state.json").read_text())
        assert set(state) == {"epoch", "local_step", "global_step"}
        outputs.append(
            (
                t.model.weight.detach().clone(),
                t.model.calls,
                torch.random.get_rng_state(),
                val,
                state,
            )
        )
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
