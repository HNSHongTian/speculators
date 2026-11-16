# TreeSpark 节点训练的接受指标

`scripts/train_trpo.py` 默认输出，无需新参数。使用选中的训练 anchor、当前 pack mode 和预算，在 `merge_gold_path` **之前**检查草稿树是否包含已有 response 的连续前缀。不会增加骨干或 verifier 前向，不修改 loss 或梯度；CPU 提取和统计仍有额外耗时，现有训练 step 时间包含这部分成本。

令 H 为 anchor 后同一文档内、连续 loss_mask>0 的有效 token 数，L 为原始 packed tree 匹配这些 token 的最长连续前缀长度。H=0 的 anchor 不进入新指标分母。

| 日志指标（前缀 train/） | 定义 |
|---|---|
| tree_accepted_draft_len | L 的 anchor 均值 |
| tree_accept_len | 1+L 的 anchor 均值；+1 是统一长度记账约定，不是额外执行 verifier |
| tree_accept_rate | sum(L)/sum(H)，token 加权比例 |
| tree_valid_depth_mean | H 的 anchor 均值，辅助识别有效长度分布变化 |
| tree_full_accept_rate | L=H 的 anchor 比例，短 horizon 也可能全通过 |
| tree_position_d_accept_rate | 在 H>d 且 L>=d 的 anchor 中，L>d 的比例，即到达该位置后的条件接受率 |
| tree_position_d_survival_rate | 在 H>d 的 anchor 中，L>d 的比例，即连续匹配到该位置的比例 |

位置从0开始，代表第一个草稿 token，与 sample_from_anchor=False 时的保留 slot0 区分。无有效分母时，现有 Trainer 输出0，不表示已观测到全部拒绝。多卡使用原始 sum/total 归约，避免平均各 rank 的比例。空 batch 与非空 batch 保持同样的指标键与顺序。

例如 H=3 的两棵树，匹配长度为2和0：tree_accept_len=2，tree_accepted_draft_len=1，tree_accept_rate=1/3；position_0_accept_rate=0.5，position_1_accept_rate=1，position_1_survival_rate=0.5。

这些指标不是在线验证接受长度，也不同于原 DSpark 的分布重叠 accept_rate/accept_len。response 若由随机采样生成，也不等同于 target argmax 序列。当前训练仍保留原目标的 gold 和 RNN replay 逻辑；仅新增诊断的有效 horizon 做了文档与监督边界限制，不声称修复了目标本身的边界问题。

修改文件：`scripts/train_trpo.py`、`src/speculators/train/trpo/step.py`；新增 `src/speculators/train/trpo/metrics.py`、`tests/unit/train/test_trpo_metrics.py` 和本文档。正在运行的训练进程需重启才能加载修改。
