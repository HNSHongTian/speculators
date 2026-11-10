"""Fixed and epoch-scheduled DSpark BV/CE/TV/confidence weights."""

import json

import pytest

from scripts.train import parse_args
from speculators.models.dspark.loss_config import (
    loss_weights_for_epoch,
    parse_loss_weight_schedule,
)
from tests.unit.train.test_bv_training import trainer

SCHEDULE = json.dumps(
    [
        {"epoch": 0, "bv": 1.0, "ce": 0.0, "tv": 0.0, "confidence": 0.0},
        {"epoch": 3, "bv": 0.8, "ce": 0.02, "tv": 0.18, "confidence": 0.0},
        {"epoch": 5, "bv": 0.4, "ce": 0.06, "tv": 0.54, "confidence": 0.25},
        {"epoch": 7, "bv": 0.2, "ce": 0.08, "tv": 0.72, "confidence": 0.5},
    ]
)


def cli(monkeypatch, *extra):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--verifier-name-or-path",
            "test",
            "--speculator-type",
            "dspark",
            "--enable-bv-loss",
            "--epochs",
            "10",
            *extra,
        ],
    )
    return parse_args()


def test_fixed_weights_are_unchanged_for_every_epoch():
    for epoch in (0, 3, 9):
        assert loss_weights_for_epoch(
            None,
            epoch,
            fixed_bv=0.2,
            fixed_ce=0.08,
            fixed_tv=0.72,
            fixed_confidence=0.4,
        ) == {"bv": 0.2, "ce": 0.08, "tv": 0.72, "confidence": 0.4}


def test_schedule_uses_latest_zero_based_epoch_milestone():
    schedule = parse_loss_weight_schedule(SCHEDULE, num_epochs=10)
    expected = {
        0: (1.0, 0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0, 0.0),
        3: (0.8, 0.02, 0.18, 0.0),
        4: (0.8, 0.02, 0.18, 0.0),
        5: (0.4, 0.06, 0.54, 0.25),
        9: (0.2, 0.08, 0.72, 0.5),
    }
    for epoch, values in expected.items():
        actual = loss_weights_for_epoch(
            schedule,
            epoch,
            fixed_bv=9,
            fixed_ce=9,
            fixed_tv=9,
            fixed_confidence=9,
        )
        actual_values = tuple(
            actual[name] for name in ("bv", "ce", "tv", "confidence")
        )
        assert actual_values == values


@pytest.mark.parametrize(
    "bad",
    [
        "{}",
        "[]",
        '[{"epoch":1,"bv":1,"ce":0,"tv":0}]',
        '[{"epoch":0,"bv":0,"ce":0,"tv":0}]',
        '[{"epoch":0,"bv":1,"ce":0,"tv":0},{"epoch":0,"bv":0,"ce":1,"tv":0}]',
        '[{"epoch":0,"bv":1,"ce":0,"tv":0},{"epoch":10,"bv":0,"ce":1,"tv":0}]',
        '[{"epoch":0,"bv":1,"ce":0,"tv":0,"confidence":0},{"epoch":1,"bv":0,"ce":1,"tv":0}]',
        '[{"epoch":0,"bv":1,"ce":0,"tv":0,"other":1}]',
    ],
)
def test_invalid_schedules_are_rejected(monkeypatch, bad):
    with pytest.raises(SystemExit):
        cli(monkeypatch, "--loss-weight-schedule", bad)


def test_cli_parses_schedule_and_requires_confidence_head(monkeypatch):
    args = cli(monkeypatch, "--loss-fn", "bv", "--loss-weight-schedule", SCHEDULE)
    assert args.loss_weight_schedule[-1] == {
        "epoch": 7.0,
        "bv": 0.2,
        "ce": 0.08,
        "tv": 0.72,
        "confidence": 0.5,
    }
    with pytest.raises(SystemExit):
        cli(
            monkeypatch,
            "--loss-fn",
            "bv",
            "--loss-weight-schedule",
            SCHEDULE,
            "--no-enable-confidence-head",
        )


def test_actual_trainer_dispatches_epoch_weights_and_records_recipe(tmp_path):
    schedule = parse_loss_weight_schedule(SCHEDULE, num_epochs=10)
    t = trainer(tmp_path)
    t.config = t.config._replace(num_epochs=10, loss_weight_schedule=schedule)
    t.train_epoch(0)
    assert all(call[3:] == (1.0, 0.0, 0.0, 0.0) for call in t.model.calls)
    t.model.calls.clear()
    t.train_epoch(5)
    assert all(call[3:] == (0.4, 0.06, 0.54, 0.25) for call in t.model.calls)
    recipe = t._bv_recipe()
    assert recipe["loss_weight_schedule"] == schedule
    assert recipe["token_loss_decay_gamma"] == 4.0
