"""DSpark BV/CE/TV mixtures and epoch-based loss-weight schedules."""

import json
import math

from speculators.models.metrics import LossConfig, resolve_loss_config

MAIN_LOSS_NAMES = ("bv", "ce", "tv")
SCHEDULE_LOSS_NAMES = (*MAIN_LOSS_NAMES, "confidence")


def resolve_dspark_loss_config(
    spec: str, *, enable_bv_loss: bool = False
) -> tuple[LossConfig, float, float, float]:
    """Return token config and fixed BV/CE/TV coefficients (no normalization).

    Explicit ``bv``/``{"bv": ..., "ce": ..., "tv": ...}`` requires the BV switch.
    Other existing specs retain the old whole-objective BV override when enabled.
    """
    if not enable_bv_loss:
        return resolve_loss_config(spec), 1.0, 0.0, 0.0
    try:
        parsed = {"bv": 1.0} if spec == "bv" else json.loads(spec)
    except json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict) or "bv" not in parsed:
        return resolve_loss_config(spec), 1.0, 0.0, 0.0
    if set(parsed) - set(MAIN_LOSS_NAMES):
        raise ValueError("Explicit BV mixtures support only 'bv', 'ce', and 'tv' terms")
    for name, weight in parsed.items():
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError(f"Loss weight for '{name}' must be finite and nonnegative")
    bv_weight = float(parsed["bv"])
    ce_weight = float(parsed.get("ce", 0.0))
    tv_weight = float(parsed.get("tv", 0.0))
    if bv_weight == 0 and ce_weight == 0 and tv_weight == 0:
        raise ValueError("At least one BV/CE/TV loss weight must be positive")
    # Kept for the existing forward interface; BV mode selects terms explicitly.
    return resolve_loss_config("ce"), bv_weight, ce_weight, tv_weight


def parse_loss_weight_schedule(  # noqa: C901
    spec: str | None, *, num_epochs: int | None = None
) -> list[dict[str, float]] | None:
    """Parse zero-based epoch milestones for BV/CE/TV/confidence weights.

    Each milestone applies from the start of its epoch through the next milestone.
    BV, CE and TV must be present in every item. Confidence is optional, but must
    either be present in every item or omitted from every item.
    """
    if spec is None:
        return None
    try:
        parsed = json.loads(spec)
    except json.JSONDecodeError as exc:
        raise ValueError("--loss-weight-schedule must be valid JSON") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("--loss-weight-schedule must be a non-empty JSON list")

    schedule: list[dict[str, float]] = []
    confidence_presence = []
    previous_epoch = -1
    required = {"epoch", *MAIN_LOSS_NAMES}
    allowed = {"epoch", *SCHEDULE_LOSS_NAMES}
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Loss schedule item {index} must be a JSON object")
        missing = required - set(item)
        unknown = set(item) - allowed
        if missing:
            raise ValueError(
                f"Loss schedule item {index} is missing: {', '.join(sorted(missing))}"
            )
        if unknown:
            raise ValueError(
                f"Loss schedule item {index} has unknown keys: "
                f"{', '.join(sorted(unknown))}"
            )
        epoch = item["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError(
                f"Loss schedule item {index} epoch must be a nonnegative integer"
            )
        if epoch <= previous_epoch:
            raise ValueError("Loss schedule epochs must be strictly increasing")
        if num_epochs is not None and epoch >= num_epochs:
            raise ValueError(
                f"Loss schedule epoch {epoch} must be smaller than "
                f"--epochs={num_epochs}"
            )
        values: dict[str, float] = {"epoch": epoch}
        for name in SCHEDULE_LOSS_NAMES:
            if name not in item:
                continue
            weight = item[name]
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight < 0
            ):
                raise ValueError(
                    f"Loss schedule item {index} weight '{name}' must be finite "
                    "and nonnegative"
                )
            values[name] = float(weight)
        if not any(values[name] > 0 for name in MAIN_LOSS_NAMES):
            raise ValueError(
                f"Loss schedule item {index} must enable at least one of BV, CE, or TV"
            )
        confidence_presence.append("confidence" in item)
        schedule.append(values)
        previous_epoch = epoch

    if schedule[0]["epoch"] != 0:
        raise ValueError("Loss schedule must start at epoch 0")
    if any(confidence_presence) and not all(confidence_presence):
        raise ValueError("Confidence must be specified in every schedule item or none")
    return schedule


def loss_weights_for_epoch(
    schedule: list[dict[str, float]] | None,
    epoch: int,
    *,
    fixed_bv: float,
    fixed_ce: float,
    fixed_tv: float,
    fixed_confidence: float,
) -> dict[str, float]:
    """Resolve the fixed or most recent milestone weights for ``epoch``."""
    values = {
        "bv": fixed_bv,
        "ce": fixed_ce,
        "tv": fixed_tv,
        "confidence": fixed_confidence,
    }
    if schedule is None:
        return values
    milestone = schedule[0]
    for candidate in schedule[1:]:
        if candidate["epoch"] > epoch:
            break
        milestone = candidate
    values.update({name: milestone[name] for name in MAIN_LOSS_NAMES})
    if "confidence" in milestone:
        values["confidence"] = milestone["confidence"]
    return values
