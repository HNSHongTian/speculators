"""Offline regression for custom verifier configs in the --draft-config path."""

import argparse
import json

import pytest
from transformers import Qwen3Config, dynamic_module_utils

from scripts.train import build_draft_model
from speculators.models.utils import get_verifier_config


@pytest.fixture
def local_configs(tmp_path, monkeypatch):
    # Exercise real HF dynamic config loading, without model weights or network.
    monkeypatch.setattr(
        dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path / "cache")
    )
    teacher = tmp_path / "custom_teacher"
    teacher.mkdir()
    (teacher / "configuration_local.py").write_text(
        "from transformers import PretrainedConfig\n"
        "class LocalVerifierConfig(PretrainedConfig):\n"
        "    model_type = 'local_test_verifier'\n"
    )
    (teacher / "config.json").write_text(
        json.dumps(
            {
                "model_type": "local_test_verifier",
                "auto_map": {"AutoConfig": "configuration_local.LocalVerifierConfig"},
                "hidden_size": 64,
                "vocab_size": 200,
                "num_hidden_layers": 40,
            }
        )
    )
    draft = tmp_path / "draft"
    Qwen3Config(hidden_size=64, vocab_size=100, num_hidden_layers=5).save_pretrained(
        draft
    )
    return str(teacher), str(draft)


def test_helper_loads_custom_config_when_trusted(local_configs):
    teacher, _ = local_configs
    config = get_verifier_config(teacher, trust_remote_code=True)
    assert config.model_type == "local_test_verifier"
    assert config.hidden_size == 64


def test_helper_does_not_trust_custom_code_by_default(local_configs):
    teacher, _ = local_configs
    with pytest.raises(ValueError, match="trust_remote_code"):
        get_verifier_config(teacher)


@pytest.mark.parametrize("trust", [False, True])
def test_build_draft_passes_cli_trust_through_to_actual_auto_config(
    local_configs, trust
):
    teacher, draft = local_configs
    args = argparse.Namespace(
        from_pretrained=None,
        speculator_type="dspark",
        draft_config=draft,
        verifier_name_or_path=teacher,
        trust_remote_code=trust,
        mask_token_id=139,
    )

    class ConfigOnlyModel:
        @classmethod
        def from_training_args(cls, verifier_config, **_kwargs):
            return verifier_config

    if not trust:
        with pytest.raises(ValueError, match="trust_remote_code"):
            build_draft_model(args, ConfigOnlyModel, None, None, None)
        return
    config = build_draft_model(args, ConfigOnlyModel, None, None, None)
    assert isinstance(config, Qwen3Config)
    assert config.num_hidden_layers == 5
    assert config.vocab_size == 200
