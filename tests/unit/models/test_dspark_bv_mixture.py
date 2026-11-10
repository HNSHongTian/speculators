"""Weighted BV/TV CLI, numerical gradients, integration and resume regression."""

import json

import pytest
import torch

from scripts.train import parse_args
from speculators.models.dspark.bv_loss import bv_loss
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark.loss_config import resolve_dspark_loss_config
from speculators.models.dspark.metrics import compute_metrics
from speculators.models.metrics import resolve_loss_config
from speculators.train.bv import validate_bv_resume
from tests.unit.models.test_dspark_bv_forward import tiny_model
from tests.unit.train.test_bv_training import trainer


def cli(monkeypatch, spec, *extra):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--verifier-name-or-path",
            "test",
            "--speculator-type",
            "dspark",
            "--enable-bv-loss",
            "--loss-fn",
            spec,
            *extra,
        ],
    )
    return parse_args()


@pytest.mark.parametrize(
    ("spec", "weights"),
    [
        ('{"bv":0.1,"tv":0.9}', (0.1, 0, 0.9)),
        ('{"bv":0.2,"ce":0.08,"tv":0.72}', (0.2, 0.08, 0.72)),
        ('{"bv":2,"tv":3}', (2, 0, 3)),
        ('{"bv":0.1}', (0.1, 0, 0)),
        ("bv", (1, 0, 0)),
        ('{"bv":0,"ce":0.1,"tv":0.9}', (0, 0.1, 0.9)),
        ("ce", (1, 0, 0)),
        ('{"ce":0.1,"tv":0.9}', (1, 0, 0)),
    ],
)
def test_cli_weights_and_legacy_override(monkeypatch, spec, weights):
    args = cli(monkeypatch, spec)
    train, val = DSparkDraftModel.get_trainer_kwargs(**vars(args))
    for kwargs in (train, val):
        assert (
            kwargs["bv_loss_weight"],
            kwargs["ce_loss_weight"],
            kwargs["tv_loss_weight"],
        ) == weights


@pytest.mark.parametrize(
    "spec",
    [
        '{"bv":-0.1,"tv":0.9}',
        '{"bv":0,"tv":0}',
        '{"bv":NaN}',
        '{"bv":Infinity}',
        '{"bv":true}',
        '{"bv":"0.1"}',
        '{"bv":0.1,"kl_div":0.9}',
        '{"bv":1,"tv":-1}',
    ],
)
def test_cli_rejects_invalid_mix(monkeypatch, spec):
    with pytest.raises(SystemExit):
        cli(monkeypatch, spec)


def test_explicit_bv_requires_switch_and_tv_requires_valid_gamma(monkeypatch):
    with pytest.raises(ValueError):
        resolve_dspark_loss_config('{"bv":0.1,"tv":0.9}')
    with pytest.raises(SystemExit):
        cli(monkeypatch, '{"bv":0.1,"tv":0.9}', "--dflash-decay-gamma", "0")


@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("annealing", [False, True])
@pytest.mark.parametrize(
    "weights",
    [(0.1, 0.0, 0.9), (0.2, 0.3, 0.5), (0.2, 0.0, 0.0), (0.0, 0.1, 0.9)],
)
def test_weighted_loss_logs_and_gradients_match_independent_sum(
    sample, annealing, weights
):
    torch.manual_seed(64)
    n, block, vocab = 3, 4, 7
    start = int(not sample)
    q = torch.randn(1, n * block, vocab, dtype=torch.double, requires_grad=True)
    p = torch.randn_like(q, requires_grad=True)
    labels = torch.randint(vocab, (n, block - start))
    valid = torch.tensor([True, False, True])
    beta = torch.tensor(0.3, dtype=torch.double)
    actual, metrics = compute_metrics(
        q,
        p,
        None,
        torch.ones(1, n * block),
        block,
        resolve_loss_config("ce"),
        sample_from_anchor=sample,
        enable_bv_loss=True,
        bv_labels=labels,
        bv_valid_blocks=valid,
        bv_beta=beta,
        bv_annealing=annealing,
        bv_loss_weight=weights[0],
        ce_loss_weight=weights[1],
        tv_loss_weight=weights[2],
        bv_chunk_size=1,
    )
    qb = q.reshape(n, block, vocab)[:, start:]
    pb = p.detach().reshape(n, block, vocab)[:, start:]
    raw_bv, bv_metrics = bv_loss(qb, pb, labels, valid, beta, annealing=annealing)
    # Independent TV reference: per-position total variation, exponential decay,
    # mean within each block, then average over exactly the valid BV blocks.
    decay = torch.exp(-torch.arange(block - start).float() / 4)
    ce = (
        (
            torch.nn.functional.cross_entropy(
                qb.reshape(-1, vocab),
                pb.argmax(-1).reshape(-1),
                reduction="none",
            ).reshape(n, block - start)
            * decay
        )
        .mean(-1)[valid]
        .mean()
    )
    tv = (
        (0.5 * (qb.softmax(-1) - pb.softmax(-1)).abs().sum(-1) * decay)
        .mean(-1)[valid]
        .mean()
    )
    expected = weights[0] * raw_bv + weights[1] * ce + weights[2] * tv
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, q, retain_graph=True)[0],
        torch.autograd.grad(expected, q)[0],
    )
    assert torch.autograd.grad(actual, p, allow_unused=True)[0] is None
    expected_log = (
        weights[0] * bv_metrics["bv_loss_sum"] / valid.sum()
        + weights[1] * ce.detach()
        + weights[2] * tv.detach()
    )
    torch.testing.assert_close(
        metrics["loss_sum"] / metrics["loss_total"], expected_log
    )
    if weights[1]:
        torch.testing.assert_close(
            metrics["ce_loss_sum"] / metrics["ce_loss_total"], ce
        )
    if weights[2]:
        torch.testing.assert_close(
            metrics["tv_loss_sum"] / metrics["tv_loss_total"], tv
        )


@pytest.mark.parametrize("chunk", [1, 3, 16])
def test_empty_blocks_are_finite_and_have_zero_gradient(chunk):
    q = torch.full((1, 6, 5), float("nan"), requires_grad=True)
    loss, metrics = compute_metrics(
        q,
        q.detach(),
        None,
        torch.ones(1, 6),
        3,
        resolve_loss_config("ce"),
        enable_bv_loss=True,
        bv_labels=torch.zeros(2, 3, dtype=torch.long),
        bv_valid_blocks=torch.zeros(2, dtype=torch.bool),
        bv_loss_weight=0.1,
        ce_loss_weight=0.2,
        tv_loss_weight=0.9,
        bv_chunk_size=chunk,
    )
    assert loss == 0
    assert metrics["loss_sum"] == 0
    loss.backward()
    assert torch.isfinite(q.grad).all()
    assert torch.count_nonzero(q.grad) == 0


def test_real_backbone_gets_mix_and_markov_gradients(monkeypatch):
    model, batch = tiny_model()
    args = cli(
        monkeypatch,
        '{"bv":0.2,"ce":0.1,"tv":0.7}',
        "--max-anchors",
        "2",
    )
    kwargs, _ = DSparkDraftModel.get_trainer_kwargs(**vars(args))
    _, loss, metrics = model(**batch, **kwargs)
    loss.backward()
    torch.testing.assert_close(
        metrics["loss_sum"],
        0.2 * metrics["bv_loss_sum"]
        + 0.1 * metrics["ce_loss_sum"]
        + 0.7 * metrics["tv_loss_sum"],
    )
    assert model.markov_head.markov_w1.weight.grad.abs().sum() > 0
    assert model.verifier_lm_head.weight.grad is None


def test_resume_records_mix_and_tv_decay_without_changing_pure_recipe(tmp_path):
    t = trainer(tmp_path)
    original = t._bv_recipe()
    t.config.train_call_kwargs.update(
        bv_loss_weight=1.0, ce_loss_weight=0.0, tv_loss_weight=0.0
    )
    assert t._bv_recipe() == original
    t.config.train_call_kwargs.update(
        bv_loss_weight=0.1, ce_loss_weight=0.2, tv_loss_weight=0.7, gamma=4.0
    )
    recipe = t._bv_recipe()
    assert recipe["bv_loss_weight"] == 0.1
    assert recipe["ce_loss_weight"] == 0.2
    assert recipe["tv_loss_weight"] == 0.7
    assert recipe["ce_decay_gamma"] == 4
    assert recipe["tv_decay_gamma"] == 4
    saved = {"recipe": json.loads(json.dumps(recipe))}
    validate_bv_resume(saved, recipe)
    t.config.train_call_kwargs["gamma"] = 6.0
    with pytest.raises(ValueError, match="differs"):
        validate_bv_resume(saved, t._bv_recipe())
    with pytest.raises(ValueError, match="differs"):
        validate_bv_resume(saved, original)


def test_compiled_mixture_matches_eager_with_dynamic_beta():
    torch.manual_seed(19)
    q = torch.randn(1, 6, 5)
    p = torch.randn_like(q)
    labels = torch.tensor([[0, 1, 2], [2, 3, 4]])
    valid = torch.tensor([True, True])

    def loss_fn(logits, beta):
        return compute_metrics(
            logits,
            p,
            None,
            torch.ones(1, 6),
            3,
            resolve_loss_config("ce"),
            enable_bv_loss=True,
            bv_labels=labels,
            bv_valid_blocks=valid,
            bv_beta=beta,
            bv_annealing=True,
            bv_chunk_size=1,
            bv_loss_weight=0.2,
            ce_loss_weight=0.1,
            tv_loss_weight=0.7,
        )[0]

    compiled = torch.compile(loss_fn, backend="aot_eager", fullgraph=True)
    for beta in (torch.tensor(0.0), torch.tensor(0.7)):
        eager_q = q.clone().requires_grad_()
        compiled_q = q.clone().requires_grad_()
        eager_loss = loss_fn(eager_q, beta)
        compiled_loss = compiled(compiled_q, beta)
        torch.testing.assert_close(eager_loss, compiled_loss)
        torch.testing.assert_close(
            torch.autograd.grad(eager_loss, eager_q)[0],
            torch.autograd.grad(compiled_loss, compiled_q)[0],
        )
