"""Read-only checks for the original non-BV 29B DSpark launch script."""

# Fixed repository script and arguments, never user-supplied commands.
# ruff: noqa: S603, S607

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/train/train_29b_dspark_online.sh"


def expanded_arguments(rank: int) -> list[str]:
    """Expand launch arguments without running mkdir, nohup, or torchrun."""
    source = SCRIPT.read_text()
    assignments = source.split('LOG_DIR="$OUTPUT_DIR/logs"')[0]
    command = source.split("nohup ", 1)[1].split('> "$LOG_FILE"', 1)[0]
    probe = assignments + "\nprintf '%s\\0' " + command + "\n"
    result = subprocess.run(
        ["bash", "-c", probe, str(SCRIPT), str(rank)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.rstrip("\0").split("\0")


@pytest.mark.parametrize("rank", [0, 1])
def test_original_dspark_uses_legacy_loss_path(rank):
    tokens = expanded_arguments(rank)
    assert tokens[tokens.index("--loss-fn") + 1] == '{"ce": 0.1, "tv": 0.9}'
    assert "--trust-remote-code" in tokens
    assert "--enable-bv-loss" not in tokens
    assert "--loss-weight-schedule" not in tokens
    assert "--bv-anneal-epochs" not in tokens
    assert "--no-enable-confidence-head" not in tokens
    assert "--enable-confidence-head" not in tokens


def test_original_dspark_preserves_supplied_training_parameters():
    tokens = expanded_arguments(0)
    expected = {
        "--epochs": "10",
        "--lr": "3e-4",
        "--total-seq-len": "4096",
        "--block-size": "15",
        "--max-anchors": "512",
        "--markov-rank": "256",
        "--dflash-decay-gamma": "6.0",
        "--scheduler-warmup-ratio": "0.04",
    }
    for flag, value in expected.items():
        assert tokens[tokens.index(flag) + 1] == value
    assert tokens[tokens.index("--draft-config") + 1] == "draft_config_29b_old.json"
    endpoints = tokens[tokens.index("--vllm-endpoint") + 1 :]
    assert endpoints[:4] == [
        "http://10.127.10.106:8000/v1",
        "http://10.127.10.143:8000/v1",
        "http://10.127.10.134:8000/v1",
        "http://10.127.10.110:8000/v1",
    ]


def test_script_uses_repository_source_and_valid_bash():
    source = SCRIPT.read_text()
    assert 'export PYTHONPATH="$CODE_PATH/src${PYTHONPATH:+:$PYTHONPATH}"' in source
    assert "[http://" not in source
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


@pytest.mark.parametrize("rank", [-1, 2])
def test_invalid_rank_is_rejected(rank):
    result = subprocess.run(
        ["bash", str(SCRIPT), str(rank)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "node_rank must be 0 or 1" in result.stderr
