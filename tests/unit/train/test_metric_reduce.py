"""Regression checks for rank-dependent metric keys/order and empty batches."""

import pytest
import torch

from speculators.train import metric_reduce


def test_sorted_packed_reduction(monkeypatch):
    monkeypatch.setattr(metric_reduce.dist, "get_world_size", lambda: 2)

    def gather(outputs, signature):
        for output in outputs:
            output.copy_(signature)

    def reduce(values, **kwargs):
        torch.testing.assert_close(values, torch.tensor([2.0, 3.0]))
        values.add_(torch.tensor([5.0, 7.0]))

    monkeypatch.setattr(metric_reduce.dist, "all_gather", gather)
    monkeypatch.setattr(metric_reduce.dist, "all_reduce", reduce)
    inputs = {"z": torch.tensor(3.0), "a": torch.tensor(2.0)}
    output = metric_reduce.reduce_metrics(inputs, device="cpu")
    assert list(output) == ["a", "z"]
    assert output["a"].item() == 7
    assert output["z"].item() == 10
    assert inputs["a"].item() == 2


def test_schema_mismatch_fails_before_value_collective(monkeypatch):
    monkeypatch.setattr(metric_reduce.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(metric_reduce.dist, "get_rank", lambda: 0)

    def gather(outputs, signature):
        outputs[0].copy_(signature)
        outputs[1].copy_(signature + 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("must not enter mismatched value collective")

    monkeypatch.setattr(metric_reduce.dist, "all_gather", gather)
    monkeypatch.setattr(metric_reduce.dist, "all_reduce", forbidden)
    with pytest.raises(RuntimeError, match="metric schema mismatch"):
        metric_reduce.reduce_metrics({"loss": torch.tensor(0.0)}, device="cpu")
