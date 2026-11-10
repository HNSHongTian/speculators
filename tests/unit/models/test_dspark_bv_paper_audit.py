"""Paper-to-code audit: independent examples, log placement and diagnostics."""

import itertools
import math

import pytest
import torch

from speculators.models.dspark.bv_loss import (
    aggregate_bv_scores,
    bv_loss,
    integrated_log_scores,
)
from speculators.models.dspark.metrics import compute_metrics
from speculators.models.metrics import resolve_loss_config


@pytest.mark.parametrize("proposal, expected", [([0.3, 0.2], [0.9, 0.83]), ([0.1, 0.2], [0.9, 0.89])])
def test_appendix_b2_same_token_overlap_different_block_acceptance(proposal, expected):
    p = torch.tensor([[[0.2, 0.8], [0.1, 0.9]]], dtype=torch.double)
    q = torch.tensor([[[x, 1-x] for x in proposal]], dtype=torch.double)
    estimate = torch.zeros(2, dtype=torch.double)
    for a, b in itertools.product(range(2), repeat=2):
        scores = integrated_log_scores(q.log(), p.log(), torch.tensor([[a, b]]))
        estimate += p[0, 0, a] * p[0, 1, b] * scores[0].exp()
    torch.testing.assert_close(estimate, torch.tensor(expected, dtype=torch.double))


def test_block_logs_are_averaged_not_log_of_batch_average():
    p = torch.tensor([[[0.9, 0.1]], [[0.9, 0.1]]], dtype=torch.double)
    q = torch.tensor([[[0.1, 0.9]], [[0.9, 0.1]]], dtype=torch.double)
    loss, _ = bv_loss(q.log(), p.log(), torch.tensor([[0], [0]]), torch.tensor([True, True]), torch.tensor(1.0))
    assert loss.item() == pytest.approx(-math.log(0.2) / 2)
    assert loss.item() != pytest.approx(-math.log(0.6))


@pytest.mark.parametrize("beta", [1e-8, 1e-6, 5e-5])
def test_small_positive_beta_logged_objective_is_not_geometric_limit(beta):
    scores = torch.tensor([[-1.0, -1e6]], dtype=torch.double, requires_grad=True)
    surrogate, objective = aggregate_bv_scores(scores, torch.tensor(beta, dtype=torch.double), annealing=True)
    exact = -(torch.logsumexp(beta * scores, -1) - math.log(2)) / beta
    torch.testing.assert_close(objective, exact, rtol=1e-10, atol=1e-5)
    assert not torch.allclose(objective, -scores.mean(-1), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(torch.autograd.grad(surrogate.sum(), scores, retain_graph=True)[0], torch.autograd.grad(exact.sum(), scores)[0])


def test_actual_label_accuracy_distinguished_from_teacher_argmax():
    q = torch.tensor([[[0.0, 5.0], [0.0, 5.0]]])
    _, metrics = compute_metrics(q, q.clone(), None, torch.ones(1, 2), 2, resolve_loss_config("ce"), enable_bv_loss=True, bv_labels=torch.tensor([[0, 0]]), bv_valid_blocks=torch.tensor([True]))
    assert metrics["full_acc_sum"] == 2
    assert metrics["bv_sampled_label_acc_sum"] == 0
    assert metrics["bv_sampled_label_acc_total"] == 2


@pytest.mark.parametrize("annealing", [True, False])
def test_unclamped_equations_match_independent_probability_reference(annealing):
    torch.manual_seed(33)
    q = torch.randn(4, 3, 7, dtype=torch.double, requires_grad=True)
    p = torch.randn_like(q)
    labels = torch.randint(7, (4, 3))
    probability_q, probability_p = q.softmax(-1), p.softmax(-1)
    ratio, prefix = torch.ones(4, dtype=torch.double), torch.ones(4, dtype=torch.double)
    scores = []
    for i in range(3):
        scores.append(torch.minimum(prefix[:, None] * probability_p[:, i], ratio[:, None] * probability_q[:, i]).sum(-1))
        token = labels[:, i, None]
        ratio = ratio * probability_q[:, i].gather(-1, token).squeeze(-1) / probability_p[:, i].gather(-1, token).squeeze(-1)
        prefix = torch.minimum(prefix, ratio)
    scores = torch.stack(scores, -1)
    beta = 0.35
    reference = -(scores.pow(beta).mean(-1).log() / beta).mean() if annealing else -scores.mean(-1).log().mean()
    actual, metrics = bv_loss(q, p, labels, torch.ones(4, dtype=torch.bool), torch.tensor(beta, dtype=torch.double), annealing=annealing, chunk_size=2)
    torch.testing.assert_close(metrics["bv_loss_sum"] / 4, reference)
    torch.testing.assert_close(torch.autograd.grad(actual, q, retain_graph=True)[0], torch.autograd.grad(reference, q)[0])
