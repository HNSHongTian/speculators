"""Tree-Relative Policy Optimization (OPPG V2) training utilities."""

from speculators.train.trpo.loss import trpo_loss_from_groups
from speculators.train.trpo.reward import compute_node_rewards, merge_gold_path
from speculators.train.trpo.step import compute_trpo_batch_loss

__all__ = [
    "compute_node_rewards",
    "compute_trpo_batch_loss",
    "merge_gold_path",
    "trpo_loss_from_groups",
]
