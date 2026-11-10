"""Step 4: compare disabled BV with a frozen pre-adaptation implementation."""

import runpy
from pathlib import Path

import pytest
import torch

from speculators.models.dspark.metrics import compute_metrics
from speculators.models.metrics import resolve_loss_config

LEGACY = runpy.run_path(
    str(Path(__file__).parents[2] / "fixtures/dspark_metrics_pre_bv.py")
)["compute_metrics"]


@pytest.mark.parametrize("loss_fn", ["ce", "kl_div", '{"ce":0.1,"tv":0.9}'])
@pytest.mark.parametrize("sample", [True, False])
@pytest.mark.parametrize("confidence", [True, False])
def test_disabled_bv_loss_metrics_gradients_and_update_identical(
    loss_fn, sample, confidence
):
    torch.manual_seed(21)
    q = torch.randn(1, 12, 7)
    p = torch.randn_like(q)
    c = torch.randn(1, 12)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]])
    results = []
    for fn in [LEGACY, compute_metrics]:
        z = q.clone().requires_grad_()
        conf = c.clone().requires_grad_() if confidence else None
        parameters = [z] if conf is None else [z, conf]
        opt = torch.optim.AdamW(parameters, lr=1e-3)
        loss, metrics = fn(
            z, p, conf, mask, 3, resolve_loss_config(loss_fn), sample_from_anchor=sample
        )
        loss.backward()
        grads = [param.grad.clone() for param in parameters]
        opt.step()
        results.append((loss, metrics, grads, [v.detach().clone() for v in parameters]))
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
