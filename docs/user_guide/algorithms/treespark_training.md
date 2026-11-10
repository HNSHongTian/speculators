# TreeSpark 训练适配开发说明

本文说明 Speculators 中 DSpark 的 TreeSpark 训练实现、参数、数据对齐、损失归一化和验证方法。实现依据为 [TreeSpark 论文](../../treespark.pdf)及本地 `TreeSpark-main` 参考代码。已有的离线树推理见 [DSpark 离线树投机适配说明](../../TREE_SPECULATIVE_OFFLINE_EVAL.md)。

## 目标与范围

TreeSpark 的“训推一致”指训练时重放部署使用的候选策略：

1. 每个 frontier 父节点展开局部 top-k 子节点。
2. 按联合路径 log-probability 将下一层 frontier 裁回 top-k。
3. 从所有深度的候选池中按全局 top-N 预算选择验证节点。
4. 对最终入树的边计算 survival NLL，对所有展开父节点计算 coverage hinge。

离散 token、frontier 和 packing 选择不传梯度；被选择边的条件概率、coverage logits、DSpark backbone 和 Markov head 保留梯度。Teacher 信息仍复用参考序列上的 verifier logits，因此不会为树的每个替代分支额外运行 verifier。

本次适配只改变训练。离线推理原有的 `--tree-top-k`、`--tree-max-nodes`、tree attention、greedy acceptance walk、KV cache 和 hidden-state gather 保持不变。

## 代码结构

| 文件 | 职责 |
|---|---|
| `src/speculators/models/dspark/tree_loss.py` | 树策略 replay、survival、coverage、训练诊断和分布式归一化 |
| `src/speculators/models/dspark/core.py` | 保存修正前 logits，构造标签有效 mask，将树损失加入 DSpark 总损失 |
| `src/speculators/models/dflash/core.py` | 允许 TreeSpark 传入独立 anchor selection mask，关闭树训练时保持原行为 |
| `src/speculators/models/dspark/config.py` | 将树训练策略写入 checkpoint 的 `config.json` |
| `scripts/train.py` | CLI 参数、合法性校验、日志和从已有 checkpoint 继续训练时的策略覆盖 |

训练数据流如下：

```text
DSpark backbone 一次前向
    ├── reference parent tokens -> 原 Markov 修正 -> 对齐后 base/confidence loss
    └── sampled anchors
          -> model-generated parents
          -> per-parent top-k
          -> frontier top-k
          -> global top-N
          -> survival + coverage
```

树分支使用 backbone 输出的基础 logits，不能使用已经按 reference parent 加过 Markov bias 的 logits，否则会重复修正并把 teacher-forced 父节点混入 replay。

## 候选策略

深度 `d` 的所有父节点共享一次 backbone 预测 `L_d`。父节点 token 为 `t_u` 时：

\[
z_d(u)=L_d+\operatorname{Markov}(t_u),\qquad
q_d(\cdot\mid u)=\operatorname{softmax}(z_d(u)).
\]

每个父节点取 top-k。候选的路径分数是父路径分数加当前条件 log-probability。每一层只保留分数最高的 k 个候选继续展开，全部已产生候选仍留在全局候选池。完成所有深度后，全局保留分数最高的 N 个非根节点。

论文默认 `B=7, k=4, N=32`。完整展开最多产生：

\[
4 + 6\times 4^2 = 100
\]

个候选，最终 verifier 输入最多为一个 anchor 加 32 个候选位置。

训练 packing 与 TreeSpark 参考实现相同，使用浮点 top-N。部署实现对同分候选额外使用浅层优先的稳定规则；连续 logits 下边界精确同分很少见，但若需要逐节点 bitwise 一致，后续应统一两侧 tie-break 和算子精度。

## Survival loss

令参考序列在深度 `d` 的 teacher 分布为 `p_d`。候选节点的参考路径质量递推为：

\[
m(\rho)=1,\qquad m(v)=m(\operatorname{parent}(v))p_d(t_v).
\]

只对全局 top-N 选中的有效节点计算：

\[
L_{\mathrm{surv}}=
-\frac{\sum_{v\in T}m(v)\log q_d(t_v\mid\operatorname{parent}(v))}
{\sum_{v\in T}m(v)+\epsilon}.
\]

路径质量和 packing mask 均 stop-gradient；`log q` 保留梯度。该项没有额外显式深度衰减。

## Coverage loss

Survival 只能给已入树候选正向监督。Coverage 在 packing 之前作用于每个展开父节点，让 teacher 认为重要、但接近局部 top-k 门槛的 token 得到信号。

令 `b_d(u)` 为该父节点第 k 大 draft logit，`J_d` 为 teacher top-J：

\[
h_d(u)=\sum_{i\in J_d}p_d(i)
\min\{c,[\delta-z_d(u)_i+b_d(u)]_+\}.
\]

实现对所有展开父节点聚合，使用 `exp(-d/gamma)` 深度衰减，并用对应 teacher top-J mass 归一化。Coverage 不使用 packing mask，也不乘父路径质量。

最终树训练项为：

\[
L_{\mathrm{tree}}=
\lambda_s L_{\mathrm{surv}}+\lambda_c L_{\mathrm{cov}}.
\]

它与现有 `--loss-fn` 主损失和 confidence loss 相加。

## 训练参数

TreeSpark 默认关闭。仅传 `--enable-tree-training` 才会构树并增加树损失。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--enable-tree-training` | false | 启用训练期树策略 replay |
| `--tree-loss-top-k` | 4 | 每父节点分支数及 frontier 宽度上限 |
| `--tree-loss-num-anchors` | 128 | 每条 packed sequence 最多用于树 replay 的 anchors |
| `--tree-loss-max-nodes` | 32 | 全局打包的非根节点预算 |
| `--tree-rank-margin` | 1.0 | Coverage margin `delta` |
| `--tree-rank-hinge-cap` | 4.0 | Coverage cap `c`；0 表示不截断 |
| `--tree-rank-top-j` | 4 | Coverage 使用的 teacher top-J |
| `--tree-survival-alpha` | 0.1 | `lambda_s` |
| `--tree-rank-alpha` | 0.1 | `lambda_c` |
| `--dflash-decay-gamma` | 4.0 | 原 loss 及 coverage 的深度衰减尺度 |

这里的 0.1/0.1 对齐 `TreeSpark-main` 中 Qwen3-4B recipe；其 Qwen3-8B recipe 使用 1.0/0.5。论文并没有一组对所有模型通用的固定权重，29B 需要根据 base loss 量级和验证接受长度调参。

启用时还会校验：

- `speculator_type` 必须是 `dspark`。
- `top_k >= 2`，节点预算、tree anchors 和 top-J 必须为正数。
- tree anchors 不能超过 `--max-anchors`。
- 两个树损失权重不能同时为零。
- 当前不能与 `--enable-bv-loss` 同时使用。BV Trainer 会按全局有效 block 数缩放整个反向 loss；直接叠加会错误缩放已经按参考质量归一化的树损失。

## 推荐训练命令

下面是在现有 DSpark 命令上新增的 TreeSpark 参数。其余模型、数据、在线 hidden-state 服务和分布式参数沿用当前环境。

```bash
torchrun --standalone --nproc_per_node 8 scripts/train.py \
  --verifier-name-or-path /path/to/verifier \
  --data-path /path/to/training_data \
  --save-path /path/to/output \
  --speculator-type dspark \
  --block-size 7 \
  --sample-from-anchor \
  --max-anchors 512 \
  --num-layers 5 \
  --markov-rank 256 \
  --markov-head-type vanilla \
  --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --dflash-decay-gamma 4 \
  --enable-tree-training \
  --tree-loss-top-k 4 \
  --tree-loss-num-anchors 128 \
  --tree-loss-max-nodes 32 \
  --tree-rank-top-j 4 \
  --tree-rank-margin 1 \
  --tree-rank-hinge-cap 4 \
  --tree-survival-alpha 0.1 \
  --tree-rank-alpha 0.1
```

论文中的 7 个 draft depths 对本仓库默认的 `sample_from_anchor=True` 对应 `--block-size 7`。如果使用 `--no-sample-from-anchor`，slot 0 是保留位，实际树深度为 `block_size - 1`；若仍需要 7 层，应传 `--block-size 8`。RNN Markov head 会先重放保留 slot 以对齐现有树推理状态。

论文基础分布项使用未除以 2 的 L1，本仓库 `tv` 是 `0.5 * L1`，且 CE 标签和归一化也有既有差异。上面的 `0.1 CE + 0.9 TV` 适合与本仓库原 DSpark 做受控对比，但不应声明为 TreeSpark 论文的数值级复现。严格复现还需单独统一 base loss、数据、batch、attention head dimension 和优化配置。

## 词表与标签对齐

树训练支持完整 verifier 词表和已有 draft 裁剪词表：

- top-k、`log q` 和 teacher mass 在 draft 词表中计算。
- 候选成为下一层 Markov 父节点前，通过 `d2t` 转成 verifier token ID。
- Markov `W1` 始终索引 verifier token ID。
- Teacher argmax 的树路径诊断也先转成 verifier token ID。

启用 TreeSpark 时，anchor 选择、DSpark base/confidence loss、position accuracy 和 tree loss 共用同一套预测标签有效性口径。每个 slot 的监督 mask 按它实际预测的 label token 重新对齐，同时检查：

- label 没有越界；
- label 和 prediction context 与 anchor 属于同一 document；
- padding anchor 不参与；
- EOS 可以作为当前深度的合法标签，但 EOS 后续深度无效。
- 路径中任一 label 被 mask 或跨文档后，更深节点不会重新变为有效。

这样 `position_0_acc` 与树指标先共享合法 anchor 口径，避免 padding、文档边界和 EOS 导致的系统偏差。Tree loss 还会受 `--tree-loss-num-anchors` 限制而做二次随机子采样，因此与 `tree_argmax_survival_depth_0` 严格成对的基线是新增的 `tree_depth0_main_top1_acc`，不是全部 base anchors 上的 `position_0_acc`。该变更只在 TreeSpark 配置生效；默认 DSpark 仍使用原 anchor mask 和原统计口径。

## 分布式和显存行为

Survival 和 coverage 分别累计本 rank 的分子与分母。反向时只 all-reduce detached 分母，并乘数据并行 world size，抵消 DDP 的梯度平均，得到按全局监督质量归一化的梯度。

词表 logits 是主要的额外激活。实现采用：

- 最多抽取 `tree_loss_num_anchors` 个 anchors；
- 按深度处理 teacher 分布；
- 对父条件词表 logits 使用 non-reentrant activation checkpointing；
- 候选 ID、父指针、路径分数、参考质量和 packing mask 全部 detached。

如果显存不足，优先降低 `--tree-loss-num-anchors`，这不会改变部署时的 k/N 策略。降低 k 或 N 会改变训练策略，部署参数也必须同步调整。

## Checkpoint 与部署

训练开关和全部树参数都会写入 DSpark checkpoint 的 `config.json`。使用 `--from-pretrained` 继续训练时，本次 CLI 参数覆盖来源 checkpoint 中的旧策略，因此保存的新 checkpoint 反映当前训练。

部署必须匹配训练候选策略：

```bash
python scripts/evaluate/dspark_offline_eval.py \
  ... \
  --temperature 0 \
  --tree-top-k 4 \
  --tree-max-nodes 32
```

树验证当前仅支持 greedy target。训练参数写入 checkpoint 用于审计，离线 evaluator 仍要求显式传部署参数。

## 训练指标

日志新增以下 counted metrics：

| 指标 | 解释 |
|---|---|
| `tree_survival_loss` | 按 packed path mass 归一化的 survival NLL |
| `tree_coverage_loss` | 按 top-J mass 和深度权重归一化的 coverage hinge |
| `tree_weighted_loss` | 加权后实际加入总目标的树 loss |
| `tree_mass_coverage_depth_d` | 深度 d 的 packed reference path mass |
| `tree_argmax_survival_depth_d` | reference argmax 链在 packed tree 中存活到深度 d 的比例 |
| `tree_anchor_eligible_ratio` | 通过 label、document 和 EOS 检查的 anchor 比例 |
| `tree_depth0_main_top1_acc` | 主 DSpark 修正 logits 在树根的 top-1 准确率 |
| `tree_depth0_replay_top1_acc` | Tree replay 重算根分布的 top-1 准确率 |
| `tree_depth0_main_replay_agreement` | 主分支与 replay 根分布的 top-1 一致率 |
| `tree_depth0_local_topk_hit` | teacher top-1 是否进入根节点局部 top-k |
| `tree_depth0_top1_pack_violation` | replay top-1 正确却在全局 packing 后丢失的比例；正常必须为 0 |
| `tree_depth0_main_top1_survival_violation` | 主分支 top-1 正确但树根未存活的比例；正常必须为 0 |

根节点有三个必须满足的关系：`tree_depth0_main_replay_agreement` 应接近 1，两个 `*_violation` 必须为 0。如果 agreement 不为 1，说明主分支与 replay 的 Markov/RNN 状态不一致；如果 pack violation 大于 0，说明候选打包或父子索引有错。在这些项正常时，应在同一子样本上满足 `tree_argmax_survival_depth_0 >= tree_depth0_main_top1_acc`。

诊断使用参考序列条件下的 teacher 分布。它们不是部署时真实 greedy acceptance length；真实接受长度仍应使用离线或在线树验证评测。

## 测试覆盖

相关测试位于：

- `tests/unit/models/test_dspark_tree_loss.py`
- `tests/unit/models/test_dspark_tree_forward.py`
- `tests/unit/train/test_draft_config_init.py`

覆盖内容包括：节点预算、survival/coverage 反向梯度、RNN 保留 slot、裁剪词表映射、全无效本地 anchors、真实 tiny DSpark 前向反向、关闭开关的回归行为、eval 模式不构树、EOS/document mask、根节点主分支/replay 一致性、top-1 不可被 packing 剪掉，以及 CLI 参数和非法组合。
