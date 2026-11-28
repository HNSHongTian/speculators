"""Build a Transformers TP plan for DeepSeek/Xingchen-style MLA MoE models."""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PlanRule:
    suffix: str
    style: str
    prefix_match: bool = False


# This is TP-only. The router is deliberately absent and remains replicated.
# In particular, no ep_router/grouped_gemm EP styles are used.
XINGCHEN_MLA_MOE_RULES = (
    PlanRule("self_attn.q_proj", "colwise"),
    PlanRule("self_attn.q_b_proj", "colwise"),
    PlanRule("self_attn.k_proj", "colwise"),
    PlanRule("self_attn.v_proj", "colwise"),
    PlanRule("self_attn.kv_a_proj_with_mqa", "mla_kv_a_proj"),
    PlanRule("self_attn.kv_b_proj", "colwise"),
    PlanRule("self_attn.o_proj", "rowwise"),
    PlanRule("mlp.experts.gate_up_proj", "packed_colwise"),
    PlanRule("mlp.experts.down_proj", "rowwise"),
    PlanRule("mlp.experts", "moe_tp_experts", prefix_match=True),
    PlanRule("mlp.experts.*.gate_proj", "colwise"),
    PlanRule("mlp.experts.*.up_proj", "colwise"),
    PlanRule("mlp.experts.*.down_proj", "rowwise"),
    PlanRule("mlp.shared_experts.gate_proj", "colwise"),
    PlanRule("mlp.shared_experts.up_proj", "colwise"),
    PlanRule("mlp.shared_experts.down_proj", "rowwise"),
    PlanRule("mlp.shared_expert.gate_proj", "colwise"),
    PlanRule("mlp.shared_expert.up_proj", "colwise"),
    PlanRule("mlp.shared_expert.down_proj", "rowwise"),
    PlanRule("mlp.gate_proj", "colwise"),
    PlanRule("mlp.up_proj", "colwise"),
    PlanRule("mlp.down_proj", "rowwise"),
)


def load_checkpoint_keys(model_path: Path) -> list[str]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as file_obj:
            index = json.load(file_obj)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"{index_path}: missing object field 'weight_map'")
        return list(weight_map)

    single_path = model_path / "model.safetensors"
    if single_path.is_file():
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError(
                "Reading a single-file checkpoint requires safetensors."
            ) from error
        with safe_open(str(single_path), framework="pt", device="cpu") as file_obj:
            return list(file_obj.keys())

    raise FileNotFoundError(
        f"No model.safetensors.index.json or model.safetensors under {model_path}"
    )


def _module_names(parameter_keys: Iterable[str]) -> set[str]:
    modules: set[str] = set()
    for key in parameter_keys:
        parts = key.split(".")
        if parts[-1] in {"weight", "bias", "scale", "zero_point"}:
            parts = parts[:-1]
        for end in range(1, len(parts) + 1):
            modules.add(".".join(parts[:end]))
    return modules


def infer_layer_prefix(parameter_keys: Iterable[str]) -> str:
    candidates: dict[str, int] = {}
    for key in parameter_keys:
        marker = ".layers."
        if marker not in key:
            continue
        prefix = key.split(marker, maxsplit=1)[0]
        candidates[prefix] = candidates.get(prefix, 0) + 1
    if not candidates:
        raise ValueError(
            "Could not infer a transformer layer prefix from checkpoint keys; "
            "expected keys containing '.layers.<index>.'."
        )
    return max(candidates, key=candidates.get)


def _matches(modules: set[str], pattern: str, *, prefix_match: bool) -> bool:
    for module in modules:
        if fnmatch.fnmatchcase(module, pattern):
            return True
        if prefix_match and fnmatch.fnmatchcase(module, pattern + ".*"):
            return True
    return False


def build_xingchen_tp_plan(parameter_keys: Iterable[str]) -> dict[str, str]:
    keys = list(parameter_keys)
    modules = _module_names(keys)
    prefix = infer_layer_prefix(keys)
    layer_prefix = f"{prefix}.layers.*"
    plan: dict[str, str] = {}

    for rule in XINGCHEN_MLA_MOE_RULES:
        pattern = f"{layer_prefix}.{rule.suffix}"
        if rule.suffix == "mlp.experts":
            # The container-level moe_tp_experts style is for the fused expert
            # representation used by DeepSeek/Xingchen. An unfused ModuleList
            # of experts is handled by the per-expert rules below instead.
            fused_children = (
                f"{pattern}.gate_up_proj",
                f"{pattern}.down_proj",
            )
            if not any(
                _matches(modules, child, prefix_match=False)
                for child in fused_children
            ):
                continue
        if _matches(modules, pattern, prefix_match=rule.prefix_match):
            plan[pattern] = rule.style

    output_modules = sorted(
        (
            name
            for name in modules
            if ".layers." not in name
            and (name == "lm_head" or name.endswith(".lm_head"))
        ),
        key=lambda name: (name.count("."), name),
    )
    if not output_modules:
        output_modules = sorted(
            (
                name
                for name in modules
                if ".layers." not in name
                and (name == "output" or name.endswith(".output"))
            ),
            key=lambda name: (name.count("."), name),
        )
    if output_modules:
        plan[output_modules[0]] = "colwise_gather_output"

    has_attention = any("self_attn" in name for name in plan)
    has_moe = any("experts" in name for name in plan)
    if not has_attention:
        raise ValueError(
            "No supported attention projections matched the Xingchen4 checkpoint."
        )
    if not has_moe:
        raise ValueError("No MoE expert modules matched the Xingchen4 checkpoint.")
    if not any(style == "colwise_gather_output" for style in plan.values()):
        raise ValueError("Could not locate lm_head or output projection in checkpoint.")
    return plan


def load_or_build_tp_plan(
    *,
    model_path: Path,
    tp_plan_path: Path | None,
) -> tuple[dict[str, str], list[str]]:
    keys = load_checkpoint_keys(model_path)
    if tp_plan_path is None:
        return build_xingchen_tp_plan(keys), keys

    with tp_plan_path.open(encoding="utf-8") as file_obj:
        plan = json.load(file_obj)
    if not isinstance(plan, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in plan.items()
    ):
        raise ValueError(f"{tp_plan_path}: expected a JSON object of string mappings")
    return plan, keys


def validate_parallel_styles(plan: dict[str, str]) -> None:
    ep_styles = {"ep_router", "grouped_gemm"}.intersection(plan.values())
    if ep_styles:
        raise ValueError(f"Expert-parallel rules are incompatible with the TP evaluator: {sorted(ep_styles)}")
    try:
        from transformers.integrations.tensor_parallel import ALL_PARALLEL_STYLES
    except ImportError as error:
        raise RuntimeError(
            "The installed Transformers build has no PyTorch tensor-parallel "
            "integration. Install a compatible recent Transformers build."
        ) from error

    missing = sorted({style for style in plan.values() if style not in ALL_PARALLEL_STYLES})
    if missing:
        raise RuntimeError(
            "The installed Transformers build lacks required TP styles: "
            + ", ".join(missing)
            + ". Use a Transformers build supporting the resolved checkpoint plan."
        )


def write_tp_plan(path: Path, plan: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(plan, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
