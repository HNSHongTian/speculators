"""Step 1: mathematical oracle, annealing gradients and chunked backward."""

import itertools
import math

import pytest
import torch

from speculators.models.dspark.bv_loss import (
    aggregate_bv_scores,
    bv_loss,
    integrated_log_scores,
)


def test_one_token_is_log_overlap_not_linear_tv():
    p = torch.tensor([[[0.7, 0.3]]], dtype=torch.double)
    q = torch.tensor([[[0.2, 0.8]]], dtype=torch.double, requires_grad=True)
    scores = integrated_log_scores(q.log(), p.log(), torch.tensor([[1]]))
    loss, _ = aggregate_bv_scores(scores, torch.tensor(1.0), annealing=False)
    torch.testing.assert_close(loss, torch.tensor([-math.log(0.5)], dtype=torch.double))
    loss.sum().backward()
    assert q.grad is not None
    assert torch.isfinite(q.grad).all()


def test_integrated_estimate_matches_enumerated_conditional_paths():
    # Two-token binary distributions whose second conditional depends on y_1.
    p0, q0 = [0.3, 0.7], [0.6, 0.4]
    p1, q1 = [[0.2, 0.8], [0.9, 0.1]], [[0.7, 0.3], [0.4, 0.6]]
    integrated, sampled, draft_expectation = 0.0, 0.0, 0.0
    for a, b in itertools.product(range(2), repeat=2):
        p = torch.tensor([[p0, p1[a]]], dtype=torch.double)
        q = torch.tensor([[q0, q1[a]]], dtype=torch.double)
        length = (
            integrated_log_scores(q.log(), p.log(), torch.tensor([[a, b]])).exp().sum()
        )
        r1 = q0[a] / p0[a]
        r2 = r1 * q1[a][b] / p1[a][b]
        integrated += p0[a] * p1[a][b] * float(length)
        sampled += p0[a] * p1[a][b] * (min(1, r1) + min(1, r1, r2))
        alpha1 = min(1, p0[a] / q0[a])
        alpha2 = min(1, alpha1 * p1[a][b] / q1[a][b])
        draft_expectation += q0[a] * q1[a][b] * (alpha1 + alpha2)
    assert integrated == pytest.approx(sampled)
    assert integrated == pytest.approx(draft_expectation)


def test_prefix_state_uses_real_label_and_is_not_detached():
    p = torch.tensor([[[0.8, 0.2], [0.4, 0.6]]], dtype=torch.double)
    q = (
        torch.tensor([[[0.2, 0.8], [0.7, 0.3]]], dtype=torch.double)
        .log()
        .requires_grad_()
    )
    a = integrated_log_scores(q, p.log(), torch.tensor([[0, 1]]))
    b = integrated_log_scores(q, p.log(), torch.tensor([[1, 1]]))
    assert a[0, 1] != b[0, 1]
    gradient = torch.autograd.grad(a[0, 1], q)[0]
    assert gradient[0, 0].abs().sum() > 0
    assert torch.autograd.gradcheck(
        lambda z: integrated_log_scores(z, p.log(), torch.tensor([[0, 1]])), (q,)
    )


@pytest.mark.parametrize("beta", [0.0, 0.2, 0.8, 1.0])
def test_annealing_surrogate_has_power_mean_gradient(beta):
    scores = torch.tensor([[-0.4, -2.0, -4.0]], dtype=torch.double, requires_grad=True)
    surrogate, objective = aggregate_bv_scores(
        scores, torch.tensor(beta, dtype=torch.double), annealing=True
    )
    expected = (
        -scores.mean(-1)
        if beta == 0
        else -(torch.logsumexp(beta * scores, -1) - math.log(3)) / beta
    )
    torch.testing.assert_close(objective, expected)
    actual_grad = torch.autograd.grad(surrogate.sum(), scores, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), scores)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    if beta > 0:
        assert not torch.allclose(surrogate, expected)


@pytest.mark.parametrize("annealing", [False, True])
def test_chunking_matches_dense_gradients_and_teacher_is_frozen(annealing):
    torch.manual_seed(8)
    q = torch.randn(5, 3, 7, dtype=torch.double, requires_grad=True)
    p = torch.randn_like(q, requires_grad=True)
    y = torch.randint(7, (5, 3))
    valid = torch.tensor([True, False, True, True, False])
    results = []
    for chunk in [1, 2, 32]:
        loss, metrics = bv_loss(
            q, p, y, valid, torch.tensor(0.3), annealing=annealing, chunk_size=chunk
        )
        gradient, teacher_grad = torch.autograd.grad(loss, (q, p), allow_unused=True)
        assert teacher_grad is None
        assert gradient[~valid].count_nonzero() == 0
        results.append((loss, gradient, metrics))
    for result in results[1:]:
        torch.testing.assert_close(result, results[0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_perfect_match_and_extreme_logits_are_finite(dtype):
    torch.manual_seed(2)
    q = torch.randn(2, 4, 9, dtype=dtype, requires_grad=True)
    y = torch.randint(9, (2, 4))
    valid = torch.ones(2, dtype=torch.bool)
    loss, m = bv_loss(q, q.detach(), y, valid, torch.tensor(1.0))
    assert abs(float(loss.detach())) < 1e-6
    torch.testing.assert_close(
        m["bv_estimated_draft_accept_len_sum"], torch.tensor(8.0)
    )
    for anneal in [True, False]:
        z = torch.tensor([[[1000.0, -1000.0]] * 4], dtype=dtype, requires_grad=True)
        loss, _ = bv_loss(
            z,
            -z.detach(),
            torch.ones(1, 4, dtype=torch.long),
            torch.tensor([True]),
            torch.tensor(0.0),
            annealing=anneal,
        )
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(z.grad).all()


def test_all_invalid_returns_connected_zero():
    q = torch.randn(2, 3, 5, requires_grad=True)
    loss, m = bv_loss(
        q,
        -q.detach(),
        torch.full((2, 3), -100),
        torch.zeros(2, dtype=torch.bool),
        torch.tensor(1.0),
    )
    loss.backward()
    assert loss == 0
    assert q.grad.count_nonzero() == 0
    assert m["bv_loss_total"] == 0


def test_compiled_backward_and_dynamic_tensor_beta():
    torch.manual_seed(11)
    p = torch.randn(2, 3, 5)
    y = torch.ones(2, 3, dtype=torch.long)
    valid = torch.tensor([True, False])

    def objective(q, beta):
        return bv_loss(q, p, y, valid, beta, annealing=True, chunk_size=1)[0]

    compiled = torch.compile(objective, backend="aot_eager", fullgraph=True)
    q = torch.randn_like(p, requires_grad=True)
    for beta in [0.0, 0.2, 0.8]:
        actual = compiled(q, torch.tensor(beta))
        expected = objective(q, torch.tensor(beta))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, q)[0], torch.autograd.grad(expected, q)[0]
        )
