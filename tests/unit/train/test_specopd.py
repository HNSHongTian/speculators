"""SpecOPD numerical objective, online histories, gradients and resume contracts."""

import json
import runpy
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from scripts.train import parse_args
from speculators.models.dspark.model_definitions import ConfidenceHead
from speculators.train.checkpointer import SingleGPUCheckpointer
from speculators.train.specopd import target
from speculators.train.specopd.loss import specopd_loss
from speculators.train.specopd.runtime import anchor_backbone, sample_branches
from speculators.train.specopd.target import BranchTarget
from speculators.train.trainer import Trainer, TrainerConfig

ROOT = Path(__file__).resolve().parents[2]


def tiny():
    return runpy.run_path(str(ROOT / "unit/models/test_dspark_tree_forward.py"))[
        "_tiny_model_and_batch"
    ]()


def config(**overrides):
    return dict(
        num_anchors=2,
        temperature=1.0,
        loss_chunk_size=2,
        target_hidden_norm="pre",
        confidence_alpha=1.0,
        **overrides,
    )


class FakeTarget:
    def score(self, prefixes, samples, valid):
        self.prefixes = prefixes
        self.samples = samples.clone()
        return [
            torch.arange(int(n) * 16).reshape(int(n), 16).float() / 100
            for n in valid.sum(-1)
        ]


def test_e2e_numbers_gradient_and_detached_confidence():
    alpha = torch.tensor([[0.9, 0.6, 0.8], [0.8, 0.5, 0.9]], requires_grad=True)
    valid = torch.ones_like(alpha, dtype=torch.bool)
    loss, metrics = specopd_loss(alpha, valid)
    assert loss.item() == pytest.approx(0.428)
    assert metrics["specopd_surrogate_length_sum"].item() == pytest.approx(3.432)
    grad = torch.autograd.grad(loss, alpha)[0]
    assert grad[0, 0].item() == pytest.approx(-(1 + 0.6 + 0.6 * 0.8) / 6)
    confidence = torch.zeros_like(alpha, requires_grad=True)
    loss_conf, _ = specopd_loss(alpha, valid, confidence)
    actual, conf_grad = torch.autograd.grad(loss_conf, (alpha, confidence))
    torch.testing.assert_close(actual, grad)
    torch.testing.assert_close(conf_grad, (0.5 - alpha.detach()) / 6)


def test_masks_and_empty_loss():
    a = torch.tensor([[0.8, 0.4, 0.1]], requires_grad=True)
    loss, m = specopd_loss(a, torch.tensor([[True, True, False]]))
    assert loss.item() == pytest.approx(1 - (0.8 + 0.32) / 2)
    loss.backward()
    assert a.grad[0, 2] == 0
    assert m["specopd_supervised_tokens"] == 2
    with pytest.raises(ValueError, match="prefix closed"):
        specopd_loss(a, torch.tensor([[True, False, True]]))
    loss, _ = specopd_loss(a, torch.zeros_like(a, dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()


@pytest.mark.parametrize("sample_from_anchor", [True, False])
def test_real_dspark_forward_and_gradients(sample_from_anchor):
    model, batch = tiny()
    model.config.sample_from_anchor = sample_from_anchor
    model.config.confidence_head_with_markov = True
    model.confidence_head = ConfidenceHead(20)
    model.eval()
    scorer = FakeTarget()
    samples, loss, metrics = model(
        **batch, specopd_config=config(), specopd_scorer=scorer
    )
    assert samples.shape == (2, 3 if sample_from_anchor else 2)
    assert torch.isfinite(loss)
    loss.backward()
    for param in (
        model.fc.weight,
        model.markov_head.markov_w1.weight,
        model.confidence_head.proj.weight,
    ):
        assert param.grad is not None
        assert param.grad.abs().sum() > 0
    assert model.verifier_lm_head.weight.grad is None
    assert not any("tree" in key or key.startswith("ce_") for key in metrics)


def test_chunking_and_reference_labels_do_not_change_objective():
    model, batch = tiny()
    model.eval()
    torch.manual_seed(21)
    _, first, _ = model(**batch, specopd_config=config(), specopd_scorer=FakeTarget())
    first.backward()
    expected = model.fc.weight.grad.clone()
    model.zero_grad()
    batch["verifier_last_hidden_states"].fill_(float("nan"))
    cfg = config()
    cfg["loss_chunk_size"] = 1
    torch.manual_seed(21)
    _, second, _ = model(**batch, specopd_config=cfg, specopd_scorer=FakeTarget())
    second.backward()
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(expected, model.fc.weight.grad)


def test_backbone_no_future_or_document_leakage():
    model, batch = tiny()
    model.eval()
    batch["document_ids"][0, 8:] = 1
    anchors = torch.tensor([3, 10])

    def forward(data):
        return anchor_backbone(
            model,
            data["hidden_states"],
            data["input_ids"],
            data["document_ids"],
            None,
            anchors,
        )

    expected = forward(batch)
    changed = deepcopy(batch)
    changed["hidden_states"][:, 4:8] += 100
    changed["hidden_states"][:, 11:] += 100
    torch.testing.assert_close(expected, forward(changed))
    changed = deepcopy(batch)
    changed["hidden_states"][:, :8] += 100
    torch.testing.assert_close(expected[1], forward(changed)[1])


def test_eos_masks_and_empty_anchors(monkeypatch):
    model, batch = tiny()
    monkeypatch.setattr(torch, "multinomial", lambda q, n: torch.full((len(q), 1), 31))
    _, _, valid = sample_branches(
        model, torch.randn(2, 3, 16), torch.tensor([1, 2]), 1, [31]
    )
    assert valid.tolist() == [[True, False, False]] * 2
    batch["loss_mask"].zero_()
    _, loss, metrics = model(**batch, specopd_config=config())
    loss.backward()
    assert metrics["specopd_global_positions"] == 0


def test_target_actual_predecessors_and_cleanup(monkeypatch):
    seen = {}

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def generate(client, model, item, **kwargs):
        seen["tokens"] = item["input_ids"]
        return "branch-file"

    def payload(self, handle):
        return {
            "token_ids": torch.tensor(seen["tokens"]),
            "hidden_states": torch.arange(5 * 4).reshape(5, 1, 4).float(),
        }

    monkeypatch.setattr(target.openai, "OpenAI", Client)
    monkeypatch.setattr(target, "generate_hidden_states", generate)
    monkeypatch.setattr(target.FileTransfer, "get_generated", payload)
    monkeypatch.setattr(
        target.FileTransfer, "delete", lambda self, h: seen.update(deleted=h)
    )
    scorer = BranchTarget(
        {
            "endpoints": ["mock"],
            "server_model": None,
            "verifier_name_or_path": "teacher",
            "request_timeout": 10,
            "max_retries": 0,
            "target_hidden_size": 4,
            "concurrency": 1,
        }
    )
    rows = scorer.score(
        [[1, 2, 3]], torch.tensor([[8, 9, 10]]), torch.ones(1, 3, dtype=torch.bool)
    )
    assert seen["tokens"] == [1, 2, 3, 8, 9]
    torch.testing.assert_close(rows[0], torch.arange(8, 20).reshape(3, 4).float())
    assert seen["deleted"] == "branch-file"
    monkeypatch.setattr(target.FileTransfer, "get_generated", lambda *args: None)
    seen.pop("deleted")
    with pytest.raises(ValueError, match="Missing"):
        scorer.score([[1]], torch.tensor([[8]]), torch.ones(1, 1, dtype=torch.bool))
    assert seen["deleted"] == "branch-file"


@pytest.mark.parametrize(
    "extra",
    [
        ["--specopd-temperature", "0"],
        ["--enable-tree-training"],
        ["--enable-bv-loss"],
        ["--noise-std", ".1"],
        ["--fsdp-shard"],
    ],
)
def test_cli_conflicts(monkeypatch, extra):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--verifier-name-or-path",
            "test",
            "--speculator-type",
            "dspark",
            "--train-e2e-loss-specopd",
            *extra,
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_cli_load_weights(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--verifier-name-or-path",
            "test",
            "--speculator-type",
            "dspark",
            "--train-e2e-loss-specopd",
            "--from-pretrained",
            "/model/qwen3_dspark",
        ],
    )
    args = parse_args()
    assert args.from_pretrained == "/model/qwen3_dspark"
    assert args.noise_std == 0
    assert args.specopd_temperature == 1


class TinyOnline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(0.4))

    def forward(self, hidden_states, **kwargs):
        overlap = (self.logit + torch.rand(2, 3) * 0.1).sigmoid()
        loss, metrics = specopd_loss(
            overlap, torch.ones_like(overlap, dtype=torch.bool)
        )
        return None, loss, metrics


def make_trainer(path, resumed=False):
    t = Trainer.__new__(Trainer)
    t.model = TinyOnline()
    t.config = TrainerConfig(
        lr=0.01,
        num_epochs=2,
        save_path=str(path),
        train_call_kwargs={"specopd_config": {"seed": 42}},
        val_call_kwargs={"specopd_config": {"seed": 42}},
        scheduler_warmup_steps=0,
    )
    loader = runpy.run_path(str(ROOT / "unit/train/test_bv_training.py"))["TinyLoader"]
    t.train_loader, t.val_loader = loader(), loader()
    t.local_rank, t.rank, t.device_type = torch.device("cpu"), 0, "cpu"
    t.is_distributed, t.resume_from_checkpoint = False, resumed
    t.checkpointer = SimpleNamespace(
        path=path,
        previous_epoch=0 if resumed else -1,
        prev_path=path / "0",
        load_best_val_loss=lambda: None,
        scheduler_path=lambda epoch: path / "scheduler.pt",
    )
    t.setup_trainer()
    t.optimizers = [torch.optim.AdamW(t.model.parameters(), lr=0.01)]
    t.schedulers = [
        torch.optim.lr_scheduler.LambdaLR(t.optimizers[0], lambda step: 1 - step / 8)
    ]
    return t


def test_mid_epoch_resume_matches_uninterrupted(tmp_path):
    full = make_trainer(tmp_path)
    saved = {}

    def save_middle(epoch, local_step):
        if local_step == 2:
            full._save_training_state(epoch, local_step)
            saved.update(
                model=deepcopy(full.model.state_dict()),
                opt=deepcopy(full.optimizers[0].state_dict()),
                scheduler=deepcopy(full.schedulers[0].state_dict()),
            )

    full.config = full.config._replace(checkpoint_freq=0.5)
    full.maybe_save_checkpoint = save_middle
    full.train_epoch(0)
    (tmp_path / "scheduler.pt").touch()
    resumed = make_trainer(tmp_path, True)
    resumed.model.load_state_dict(saved["model"])
    resumed.optimizers[0].load_state_dict(saved["opt"])
    resumed.schedulers[0].load_state_dict(saved["scheduler"])
    resumed.train_epoch(0)
    torch.testing.assert_close(full.model.logit, resumed.model.logit, rtol=0, atol=0)
    assert resumed.global_step == full.global_step == 4
    assert resumed.val_epoch(0) == resumed.val_epoch(1)
    path = tmp_path / "0/training_state.json"
    state = json.loads(path.read_text())
    state["specopd"]["recipe"]["lr"] = 9
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="recipe mismatch"):
        make_trainer(tmp_path, True)


def test_real_checkpoint_optimizer_and_scheduler_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "speculators.train.checkpointer.get_current_device", lambda: "cpu"
    )
    full = make_trainer(tmp_path)
    full.model, batch = tiny()
    full.setup_optimizer()
    _, loss, _ = full.model(
        **batch, specopd_config=config(), specopd_scorer=FakeTarget()
    )
    loss.backward()
    full._optimizers_step()
    full._schedulers_step()
    full.global_step = 1
    full.checkpointer = SingleGPUCheckpointer(tmp_path)
    full.maybe_save_checkpoint(0, local_step=1)
    resumed = make_trainer(tmp_path)
    resumed.model, _ = tiny()
    resumed.checkpointer = SingleGPUCheckpointer(tmp_path)
    resumed.resume_from_checkpoint = True
    resumed.setup_trainer()
    resumed.setup_model()
    resumed.setup_optimizer()
    assert resumed.global_step == 1
    assert resumed._resume_local_step == 1
    assert (
        resumed.optimizers[0].param_groups[0]["lr"]
        == full.optimizers[0].param_groups[0]["lr"]
    )
    assert resumed.schedulers[0].state_dict() == full.schedulers[0].state_dict()
    for (name, param), (_, loaded) in zip(
        full.model.named_parameters(), resumed.model.named_parameters(), strict=True
    ):
        if param.requires_grad:
            torch.testing.assert_close(param, loaded, rtol=0, atol=0, msg=name)
            for key in ("exp_avg", "exp_avg_sq"):
                if param in full.optimizers[0].state:
                    torch.testing.assert_close(
                        full.optimizers[0].state[param][key],
                        resumed.optimizers[0].state[loaded][key],
                        rtol=0,
                        atol=0,
                    )


def test_uneven_rank_normalization(monkeypatch):
    import torch.distributed as dist  # noqa: PLC0415

    all_alpha = torch.tensor([[0.8, 0.6], [0.7, 0.4], [0.5, 0.9]], requires_grad=True)
    expected, _ = specopd_loss(all_alpha, torch.ones_like(all_alpha, dtype=torch.bool))
    expected.backward()
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "all_reduce", lambda count: count.fill_(6))
    for start, end in ((0, 1), (1, 3)):
        local = all_alpha.detach()[start:end].clone().requires_grad_()
        loss, _ = specopd_loss(local, torch.ones_like(local, dtype=torch.bool))
        loss.backward()
        torch.testing.assert_close(local.grad / 2, all_alpha.grad[start:end])


def test_teacher_file_connector_layout(tmp_path, monkeypatch):
    path = tmp_path / "branch.safetensors"
    tokens = [4, 5, 9]
    states = torch.randn(3, 2, 16)
    save_file({"token_ids": torch.tensor(tokens), "hidden_states": states}, path)
    monkeypatch.setattr(target, "generate_hidden_states", lambda *a, **kw: str(path))
    cfg = {
        "endpoints": ["http://localhost:8000/v1"],
        "server_model": "teacher",
        "verifier_name_or_path": "teacher",
        "request_timeout": 1,
        "max_retries": 0,
        "target_hidden_size": 16,
        "target_layer_ids": [0],
        "concurrency": 1,
    }
    rows = BranchTarget(cfg).score(
        [[4, 5]], torch.tensor([[9, 10]]), torch.ones(1, 2, dtype=torch.bool)
    )
    torch.testing.assert_close(rows[0], states[1:, -1])
    assert not path.exists()


def test_all_empty_step_skips_optimizer_and_scheduler(tmp_path):
    t = make_trainer(tmp_path)

    def empty_forward(**kwargs):
        loss, metrics = specopd_loss(
            t.model.logit.expand(0, 3), torch.zeros(0, 3, dtype=torch.bool)
        )
        return None, loss, metrics

    t.model.forward = empty_forward
    before = t.model.logit.detach().clone()
    before_step = t.schedulers[0].last_epoch
    t.train_epoch(0)
    torch.testing.assert_close(t.model.logit, before, rtol=0, atol=0)
    assert t.schedulers[0].last_epoch == before_step
    with pytest.raises(ValueError, match="no valid positions"):
        t.val_epoch(0)
