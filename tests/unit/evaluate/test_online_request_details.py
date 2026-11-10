import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    evaluate_dir = Path(__file__).parents[3] / "scripts" / "evaluate"
    path = evaluate_dir / "evaluate.py"
    sys.path.insert(0, str(evaluate_dir))
    spec = importlib.util.spec_from_file_location("online_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_detail_prompts_loads_local_jsonl_and_honors_limit(tmp_path):
    module = _load_module()
    dataset = tmp_path / "samples.jsonl"
    dataset.write_text(
        "\n".join(json.dumps({"prompt": value}) for value in ("one", "two", "three"))
        + "\n"
    )

    prompts = module._detail_prompts(
        str(dataset),
        None,
        '{"text_column":"prompt"}',
        2,
    )

    assert prompts == ["one", "two"]


def test_request_completion_uses_openai_completion_usage(monkeypatch):
    module = _load_module()
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        @staticmethod
        def read():
            return json.dumps(
                {
                    "id": "request-1",
                    "choices": [
                        {
                            "message": {"content": "answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                }
            ).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(module, "urlopen", fake_urlopen)

    result = module._request_completion(
        "http://localhost:8000/v1",
        "target-model",
        "question",
        128,
        '{"temperature":0}',
    )

    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["body"] == {
        "temperature": 0,
        "model": "target-model",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "question"}],
            }
        ],
        "max_tokens": 128,
        "stream": False,
    }
    assert captured["timeout"] == 3600
    assert result == {
        "request_id": "request-1",
        "output": "answer",
        "finish_reason": "stop",
        "prompt_tokens": 3,
        "completion_tokens": 2,
    }


def test_log_request_details_writes_text_lengths_and_acceptance(tmp_path, monkeypatch):
    module = _load_module()
    output_path = tmp_path / "details.jsonl"
    metric_snapshots = iter([["before"], ["after"]])

    monkeypatch.setattr(module, "_fetch_model_name", lambda target: "model")
    monkeypatch.setattr(
        module,
        "_detail_prompts",
        lambda dataset, subset, mapper, limit: ["hello"],
    )
    monkeypatch.setattr(module, "_require_metrics", lambda url: next(metric_snapshots))
    monkeypatch.setattr(
        module,
        "_request_completion",
        lambda *args: {
            "request_id": "request-1",
            "output": "world!",
            "finish_reason": "stop",
            "prompt_tokens": 4,
            "completion_tokens": 3,
        },
    )

    def fake_extract(current, baseline_metrics):
        assert current == ["after"]
        assert baseline_metrics == ["before"]
        return {
            "num_drafts": 2,
            "num_draft_tokens": 8,
            "num_accepted_tokens": 5,
            "acceptance_length": 3.5,
        }

    monkeypatch.setattr(module, "extract_spec_decode_metrics", fake_extract)

    module._log_request_details(
        subset="HumanEval",
        target="http://localhost:8000/v1",
        dataset="dataset-id",
        data_column_mapper='{ "text_column": "prompt" }',
        metrics_url="http://localhost:8000/metrics",
        output_path=output_path,
        max_requests=1,
        max_tokens=128,
        gen_kwargs="",
    )

    record = json.loads(output_path.read_text())
    assert record["prompt"] == "hello"
    assert record["prompt_tokens"] == 4
    assert record["completion_tokens"] == 3
    assert record["prompt_characters"] == 5
    assert record["output_characters"] == 6
    assert record["output"] == "world!"
    assert record["accepted_tokens_per_draft"] == 2.5
    assert record["dspark_acceptance_length"] == 3.5


def test_run_benchmark_does_not_add_detail_pass_when_switch_is_absent(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    args = SimpleNamespace(
        mode="throughput",
        target="http://localhost:8000/v1",
        output_dir=str(tmp_path),
        dataset="dataset-id",
        subsets="HumanEval",
        data_column_mapper='{"text_column":"prompt"}',
        max_concurrency=1,
    )

    monkeypatch.setattr(module, "check_dependencies", lambda: None)
    monkeypatch.setattr(
        module,
        "_run_subset",
        lambda *args, **kwargs: (object(), None, None),
    )

    def unexpected_detail_pass(**kwargs):
        raise AssertionError("detail pass must remain disabled by default")

    monkeypatch.setattr(module, "_log_request_details", unexpected_detail_pass)

    module.run_benchmark(args)
