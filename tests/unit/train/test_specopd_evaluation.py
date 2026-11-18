"""Actual verification, rollout history and isolated periodic AL evaluation."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.train import parse_args
from speculators.train.specopd import evaluation


def test_prompt_selection_excludes_answers_and_long_prompts():
    data = SimpleNamespace(
        data=[
            {"input_ids": [1, 2, 3, 4], "loss_mask": [0, 0, 1, 1]},
            {"input_ids": [1, 2, 3, 4], "loss_mask": [0, 0, 0, 1]},
            {"input_ids": [5, 6, 7], "loss_mask": [0, 1, 1]},
        ]
    )
    assert evaluation.select_eval_prompts(data, 2, 2) == [[1, 2], [5]]
    with pytest.raises(ValueError, match="No eligible"):
        evaluation.select_eval_prompts(SimpleNamespace(data=[]), 2, 2)


def test_greedy_rejection_bonus_and_boundaries():
    q = torch.eye(4)[[1, 2]]
    p = torch.eye(4)[[1, 3, 0]]
    assert evaluation.verify_round([1, 2], q, p, [], 9) == ([1, 3], 1)
    p = torch.eye(4)[[1, 2, 3]]
    assert evaluation.verify_round([1, 2], q, p, [], 9) == ([1, 2, 3], 2)
    assert evaluation.verify_round([1, 2], q, p, [1], 9) == ([1], 1)
    assert evaluation.verify_round([1, 2], q, p, [], 2) == ([1, 2], 2)


def test_sampling_uses_residual_on_rejection(monkeypatch):
    q = torch.tensor([[0.8, 0.1, 0.1]])
    p = torch.tensor([[0.2, 0.3, 0.5], [0.3, 0.3, 0.4]])
    monkeypatch.setattr(torch, "rand", lambda *a, **kw: torch.tensor(0.9))
    observed = []

    def sample(probs, n):
        observed.append(probs.clone())
        return torch.tensor([2])

    monkeypatch.setattr(torch, "multinomial", sample)
    assert evaluation.verify_round([0], q, p, [], 4) == ([2], 0)
    torch.testing.assert_close(observed[0], torch.tensor([0.0, 0.2, 0.4]))


def test_rollout_follows_corrected_history(monkeypatch):
    draft = torch.nn.Module()
    draft.verifier_lm_head = torch.nn.Linear(4, 4, bias=False)
    draft.verifier_lm_head.weight.data.copy_(torch.eye(4))
    draft.config = SimpleNamespace(
        sample_from_anchor=True,
        transformer_layer_config=SimpleNamespace(eos_token_id=None),
    )
    draft.block_size = 2
    prefixes = []

    def propose(draft, states, prefix, temperature, width, eos_ids):  # noqa: PLR0917
        prefixes.append(list(prefix))
        return [1, 2], torch.eye(4)[[1, 2]]

    monkeypatch.setattr(evaluation, "_proposal", propose)

    class Teacher:
        def read_states(self, tokens, index=0, positions=None, final_only=False):
            if final_only:
                return torch.eye(4)[[1, 3, 0]]
            return torch.zeros(len(tokens), 2, 4)

    cfg = {
        "target_hidden_norm": "post",
        "evaluation": {"temperature": 0.0, "max_new_tokens": 4},
    }
    metrics = evaluation.evaluate_acceptance(draft, [[7, 8]], cfg, Teacher())
    assert prefixes == [[7, 8], [7, 8, 1, 3]]
    assert metrics["specopd_verification_rounds"] == 2
    assert metrics["specopd_acceptance_length"] == 2
    assert metrics["specopd_accepted_draft_length"] == 1
    assert metrics["specopd_output_tokens"] == 4


@pytest.mark.parametrize("sample_from_anchor", [True, False])
def test_real_dspark_proposal_and_teacher_rounds(sample_from_anchor):
    helpers = runpy.run_path(str(Path(__file__).with_name("test_specopd.py")))
    model, _ = helpers["tiny"]()
    model.config.sample_from_anchor = sample_from_anchor
    model.config.transformer_layer_config.eos_token_id = None
    model.eval()

    class Teacher:
        def read_states(self, tokens, index=0, positions=None, final_only=False):
            states = (
                torch.arange(len(tokens) * 2 * 16).reshape(len(tokens), 2, 16).float()
                / 100
            )
            result = states[:, -1] if final_only else states
            return result if positions is None else result[positions]

    cfg = {
        "target_hidden_norm": "pre",
        "evaluation": {"temperature": 1.0, "max_new_tokens": 8},
    }
    result = evaluation.evaluate_acceptance(model, [[1, 2, 3]], cfg, Teacher())
    width = 3 if sample_from_anchor else 2
    assert result["specopd_output_tokens"] == 8
    assert 1 <= result["specopd_acceptance_length"] <= width + 1
    assert 0 <= result["specopd_acceptance_rate"] <= 1
    assert all(p.grad is None for p in model.parameters())


def test_periodic_hook_keeps_rng_weights_and_resume_recipe(tmp_path, monkeypatch):
    helpers = runpy.run_path(str(Path(__file__).with_name("test_specopd.py")))
    trainer = helpers["make_trainer"](tmp_path)
    before_recipe = trainer._specopd_recipe()
    trainer.config.train_call_kwargs["specopd_config"]["evaluation"] = {
        "interval": 2,
        "samples": 1,
        "max_prompt_tokens": 20,
        "seed": 1234,
    }
    assert trainer._specopd_recipe() == before_recipe
    trainer.val_loader.dataset = SimpleNamespace(
        data=[{"input_ids": [1, 2, 3], "loss_mask": [0, 0, 1]}]
    )
    calls = []

    def evaluate(model, prompts, cfg):
        calls.append((prompts, torch.rand(()).item(), model.training))
        return {"specopd_acceptance_length": 2.5}

    monkeypatch.setattr("speculators.train.trainer.evaluate_acceptance", evaluate)
    rng = torch.random.get_rng_state().clone()
    weight = trainer.model.logit.detach().clone()
    for step in (0, 0, 1, 2):
        trainer.global_step = step
        trainer.maybe_evaluate_specopd()
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert calls[0][2] is False
    assert trainer.model.training
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    torch.testing.assert_close(trainer.model.logit, weight)


@pytest.mark.parametrize(
    "extra",
    [
        ["--specopd-eval-temperature", "-1"],
        ["--specopd-eval-samples", "0"],
        ["--legacy-data"],
    ],
)
def test_eval_cli_validation(monkeypatch, extra):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--verifier-name-or-path",
            "test",
            "--speculator-type",
            "dspark",
            "--train-e2e-loss-specopd",
            "--specopd-eval-interval",
            "100",
            *extra,
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()
