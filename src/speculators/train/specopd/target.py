"""Concurrent complete-prefix scoring through the existing file connector."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai
import torch

from hs_connectors import FileTransfer
from speculators.data_generation.vllm_client import generate_hidden_states


class BranchTarget:
    def __init__(self, config):
        self.config = config

    def _score_one(self, job):
        index, prefix, samples, n = job
        cfg = self.config
        if not n:
            return torch.empty(0, cfg["target_hidden_size"])
        tokens = prefix + samples[: n - 1]
        start = len(prefix) - 1
        rows = self.read_states(tokens, index, slice(start, start + n), final_only=True)
        if len(rows) != n:
            raise ValueError("Invalid SpecOPD teacher prediction rows")
        return rows

    def read_states(self, tokens, index=0, positions=None, *, final_only=False):
        """Read validated CPU states for training or multi-round verification."""
        cfg = self.config
        endpoint = cfg["endpoints"][index % len(cfg["endpoints"])]
        transfer = FileTransfer(Path())
        # Workers only manipulate CPU tensors; no GPU work in HTTP threads.
        with openai.OpenAI(base_url=endpoint, api_key="EMPTY", max_retries=0) as client:
            handle = generate_hidden_states(
                client,
                cfg["server_model"] or cfg["verifier_name_or_path"],
                {"input_ids": tokens},
                timeout=cfg["request_timeout"],
                max_retries=cfg["max_retries"],
            )
            try:
                payload = transfer.get_generated(handle)
                if payload is None:
                    raise ValueError("Missing SpecOPD teacher hidden states")
                ids = payload["token_ids"].flatten().tolist()
                if ids != tokens:
                    raise ValueError("SpecOPD teacher token history mismatch")
                # File connector layout: [sequence, layers, hidden].
                states = payload["hidden_states"]
                if states.ndim != 3 or states.shape[0] != len(tokens):  # noqa: PLR2004
                    raise ValueError("Unexpected SpecOPD teacher hidden-state layout")
                if (
                    "target_layer_ids" in cfg
                    and states.shape[1] != len(cfg["target_layer_ids"]) + 1
                ):
                    raise ValueError(
                        "Teacher must return configured auxiliary layers "
                        "plus final layer"
                    )
                hidden = states[:, -1]
                if hidden.ndim != 2 or hidden.shape[1] != cfg["target_hidden_size"]:  # noqa: PLR2004
                    raise ValueError("Unexpected SpecOPD teacher hidden-state shape")
                rows = hidden if final_only else states
                if positions is not None:
                    rows = rows[positions]
                rows = rows.clone()
                if not torch.isfinite(rows).all():
                    raise ValueError("Invalid SpecOPD teacher prediction rows")
                return rows
            finally:
                transfer.delete(handle)

    def score(self, prefixes, samples, valid):
        samples = samples.detach().cpu().tolist()
        lengths = valid.sum(-1).cpu().tolist()
        jobs = [
            (i, p, s, n)
            for i, (p, s, n) in enumerate(zip(prefixes, samples, lengths, strict=True))
        ]
        with ThreadPoolExecutor(max_workers=self.config["concurrency"]) as pool:
            return list(pool.map(self._score_one, jobs))
