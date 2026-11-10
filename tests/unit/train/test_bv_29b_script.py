"""Step 5: script argument expansion and read-only configuration checks."""

# Fixed repository scripts, never user-supplied commands.
# ruff: noqa: S603, S607, PLC0415
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

from scripts.verify.bv.preflight_29b import check_files

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/train/train_29b_dspark_bv_online.sh"


def expanded_arguments(rank):
    """Expand only assignments and command arguments; never run mkdir/nohup/torchrun."""
    source = SCRIPT.read_text()
    assignments = source.split("LOG_DIR=")[0]
    command = source.split("nohup ", 1)[1].split('> "$LOG_FILE"', 1)[0]
    # printf receives the launch command as data, including env and torchrun.
    probe = assignments + "\nprintf '%s\\0' " + command + "\n"
    result = subprocess.run(
        ["bash", "-c", probe, str(SCRIPT)] + ([] if rank is None else [str(rank)]),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.rstrip("\0").split("\0")


@pytest.mark.parametrize("rank", [None, 0])
def test_script_has_effective_paper_recipe(rank):
    tokens = expanded_arguments(rank)
    for key, value in {
        "--epochs": "10",
        "--lr": "6e-4",
        "--block-size": "15",
        "--total-seq-len": "3072",
        "--bv-anneal-epochs": "3",
        "--seed": "42",
        "--loss-fn": "bv",
        "--max-anchors": "512",
        "--markov-rank": "256",
        "--optimizer": "adamw",
        "--scheduler-type": "cosine",
        "--scheduler-warmup-ratio": "0.04",
        "--noise-std": "0",
    }.items():
        assert tokens[tokens.index(key) + 1] == value
    assert "--enable-bv-loss" in tokens
    assert "--loss-weight-schedule" not in tokens
    assert "--no-enable-confidence-head" in tokens
    assert "--enable-confidence-head" not in tokens
    assert "--confidence-head-alpha" not in tokens
    assert "--dflash-decay-gamma" not in tokens
    assert "--markov-head-type" not in tokens  # Preserve the original default.
    assert "--nnodes=1" in tokens
    assert "--nproc_per_node=16" in tokens
    assert "--node_rank=0" in tokens
    assert "--master_addr=127.0.0.1" in tokens
    assert "--master_port=29500" in tokens
    endpoint_start = tokens.index("--vllm-endpoint") + 1
    assert tokens[endpoint_start : endpoint_start + 6] == [
        f"http://10.127.10.{host}:8000/v1" for host in (130, 135, 106, 143, 141, 131)
    ]
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_supplied_paths_and_launcher_are_preserved():
    tokens = expanded_arguments(0)
    assert tokens[:3] == [
        "env",
        "ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
        "torchrun",
    ]
    assert "/hpfs/huawei-2607/shangxt/speculators/scripts/train.py" in tokens
    assert tokens[tokens.index("--draft-config") + 1] == "draft_config_29b.json"
    assert tokens[tokens.index("--save-path") + 1] == (
        "/hpfs/huawei-2607/shangxt/Telecom-29B/"
        "open_perfect_80w_no_thinking_full_vocab_1014_32_96/checkpoints"
    )
    source = SCRIPT.read_text()
    assert "--dry-run" not in source
    assert "preflight_29b.py" not in source
    assert "COMMAND=(" not in source
    assert "export HCCL_IF_IP=" not in source
    assert (
        "unset HCCL_IF_IP GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME HCCL_SOCKET_IFNAME"
        in source
    )


def test_teacher_endpoints_bypass_proxy_in_worker_environment():
    assignments = SCRIPT.read_text().split("LOG_DIR=")[0]
    result = subprocess.run(
        ["bash", "-c", assignments + '\nprintf "%s\\n%s" "$NO_PROXY" "$no_proxy"'],
        capture_output=True,
        text=True,
        check=True,
    )
    upper, lower = result.stdout.splitlines()
    assert upper == lower
    exclusions = set(upper.split(","))
    tokens = expanded_arguments(0)
    endpoints = [token for token in tokens if token.startswith("http://")]
    assert all(urlparse(endpoint).hostname in exclusions for endpoint in endpoints)
    # Additional teacher address from the user's failed deployment request.
    assert "10.127.10.208" in exclusions


@pytest.mark.parametrize("rank", [1, 2])
def test_script_invalid_rank_fails_without_starting_training(rank):
    result = subprocess.run(
        ["bash", str(SCRIPT), str(rank)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "node_rank must be" in result.stderr


def files(tmp_path):
    model, data = tmp_path / "model", tmp_path / "data"
    model.mkdir()
    data.mkdir()
    (model / "config.json").write_text(
        json.dumps({"hidden_size": 128, "vocab_size": 256, "num_hidden_layers": 40})
    )
    draft = tmp_path / "draft.json"
    draft.write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "hidden_size": 128,
                "vocab_size": 256,
                "num_hidden_layers": 5,
            }
        )
    )
    return model, draft, data, tmp_path / "output"


def test_preflight_is_read_only_and_rejects_bad_layers_and_ce_resume(tmp_path):
    model, draft, data, output = files(tmp_path)
    result = check_files(model, draft, data, output, 128)
    assert result["draft_layers"] == 5
    assert not output.exists()
    config = json.loads(draft.read_text())
    config["num_hidden_layers"] = 1
    draft.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="five-layer"):
        check_files(model, draft, data, output, 128)
    config["num_hidden_layers"] = 5
    draft.write_text(json.dumps(config))
    (output / "checkpoints/0").mkdir(parents=True)
    with pytest.raises(ValueError, match="non-BV"):
        check_files(model, draft, data, output, 128)


def test_full_preflight_with_tiny_arrow_dataset(tmp_path):
    from datasets import Dataset

    model, draft, data, output = files(tmp_path)
    Dataset.from_dict({"input_ids": [[1] * 32], "loss_mask": [[1] * 32]}).save_to_disk(
        str(data)
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify/bv/preflight_29b.py"),
            "--model",
            str(model),
            "--draft-config",
            str(draft),
            "--data-path",
            str(data),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "PASS: local configs/data/import checks" in result.stdout
    assert not output.exists()
