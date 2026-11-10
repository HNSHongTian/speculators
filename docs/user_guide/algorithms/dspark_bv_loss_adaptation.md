# DSpark online 训练适配 BV loss：设计稿 v1

日期：2026-10-28。本文保留最初的设计稿。后续已按用户要求完成适配；实际实现状态、逐步验证脚本、运行命令和验证边界见 [实现与验证说明](dspark_bv_loss_validation.md)。下文“拟新增”“本轮只新增文档”等描述指最初的文档交付阶段。

实施仓库：`speculators_treespark`（阅读时 HEAD：`a66f57d`）。参考脚本：`speculators_telecom/examples/train/train_29b_dspark_online.sh`。两个仓库角色不同：后者仅提供 29B 环境和启动参数，实际功能与新脚本应落在前者。

## 1. 目标与方案结论

1. 新增 `--enable-bv-loss`，默认关闭。不开启时保留现有 DSpark 的损失计算、标签定义、位置权重、confidence 分支、采样与训练调度行为。
2. 开启时，主分布训练目标切换为**全词表积分、按 block 计算的 log BV surrogate**，前三个 epoch 进行聚合退火。不是把 BV 当作逐 token loss 塞进原来的 `compound_loss()`。
3. 使用 Markov 修正后的 draft logits；保留 backbone 和 Markov head 的梯度。教师保持冻结。
4. 后续新增 `examples/train/train_29b_dspark_bv_online.sh`，使用现有 TeleChat4-29B 路径、在线 hidden-state 服务与 Ascend 多机启动方式，并显式关闭 confidence head，以接近论文设置。
5. BV 训练与部署 block verification 是两项独立工作。后续已在离线评测脚本中新增可选 `--block-verification`；训练开关不会自动开启它，TreeSpark 树验证也不受影响。

**开关的确定语义：** `--enable-bv-loss` 优先于 `--loss-fn`，替换整个主分布损失。参考脚本的 `{"ce": 0.1, "tv": 0.9}` 将切换为权重 1 的 BV，而不是 `0.1×BV+0.9×TV`。这是为了让单个开关即可启用论文目标，避免 CLI 默认 `kl_div` 或旧混合配置继续参与优化。启动时明确打印被覆盖的 `loss_fn`；新脚本删除混合配置，使用 `--loss-fn ce --enable-bv-loss` 表达切换意图。

confidence 是独立辅助项：开关本身不隐式修改模型结构；如用户保留 confidence head，则仍可训练 `BV + confidence_head_alpha × BCE`，但该组合不是本文推荐的论文对齐配置。新脚本显式传 `--no-enable-confidence-head`。

## 2. 论文依据与参数摘录

论文：[BV Loss: Block Verification-Aware Loss for Block Diffusion Speculative Decoding](https://arxiv.org/html/2609.34832v1)。主要依据 §3、附录 A.1–A.4 和 B.4–B.5；本文中的工程接口、边界处理及 29B 迁移建议是结合当前仓库提出的设计。

| 项目 | 论文设置 | 本方案如何使用 |
|---|---|---|
| 教师 / 草稿 | Qwen3-4B、8B / 5 层 DSpark | 教师换为现有 TeleChat4-29B，草稿核验为 5 层 |
| 草稿长度 | 15，不含 anchor | 当前 DSpark 默认 `sample_from_anchor=True`，对应 `--block-size 15` |
| 训练数据 | Open-PerfectBlend 100K，分别由对应教师以 T=1 重生成回答 | 原脚本 80w 数据的生成来源与采样策略需要核验，不能从目录名推断 |
| 优化 | AdamW，峰值 LR 6e-4，6 epochs | 直接采用 |
| LR 调度 | 4% warmup，cosine，seed 42 | 直接采用 |
| 长度 / batch | 最大长度 3072；global 32、local 4 | 长度采用；packing 的 batch 语义差异见 §7 |
| DSpark 分支 | 保留 causal Markov correction，关闭 confidence | 在修正 logits 后算 BV；新脚本关闭 confidence |
| BV 配方 | 全词表积分；前三个 epoch β 从 0 线性升至 1 | 采用；独立于 LR warmup |
| 数值处理 | FP32 log 空间；DSpark 退火期不加 score floor；block-log 阶段 log summand floor=-80、长度 floor=1e-6 | 按阶段实施 |

以上参数见[论文实验及实现附录](https://arxiv.org/html/2609.34832v1)。论文没有给出可直接照抄的 29B 配方；Markov rank/head variant、anchor 数、AdamW betas/eps、weight decay 和梯度裁剪等，不能把本仓库值标成论文值。

## 3. BV 的计算原理与实现选择

### 3.1 从目标路径构造前缀状态

令一个 block 有 B 个预测位置。`y_i` 是**存储训练序列中的实际 token**；`p_i` 为教师在该前缀上的全词表条件分布，`q_i` 为 Markov 修正后的 draft 条件分布。分布温度固定为 1。

定义路径累计比值和保留分数：

\[
r_0=a_0=1,\quad
r_i=r_{i-1}\frac{q_i(y_i)}{p_i(y_i)},\quad
a_i=\min(a_{i-1},r_i)=\min(1,r_1,\ldots,r_i).
\]

这与逐位置 overlap 的累乘不同：累计概率比能够保留前缀概率盈余。实现不得将其替换为 `cumprod(sum(min(p,q)))`。若训练路径确实由这里的 p 采样、q 与实际 proposal 条件分布一致，路径分数之和可估计接受的草稿长度；换用其他来源的回答后，应将目标称为 surrogate，而不能继续宣称该无偏解释成立。[推导来源：§3.2](https://arxiv.org/html/2609.34832v1)

### 3.2 使用全词表积分版本

本方案每个位置使用

\[
\bar a_i=\sum_{v\in V}\min\{a_{i-1}p_i(v),r_{i-1}q_i(v)\},\qquad
\widehat\tau=\sum_{i=1}^{B}\bar a_i.
\]

**先用上一个状态计算当前积分分数，再用当前实际 token 更新 r、a。** 不能用 `bar_a_i` 更新路径状态；只积分当前位置，历史仍沿实际 token 路径递推。[来源：附录 B.4](https://arxiv.org/html/2609.34832v1)

建议用以下等价的 log 空间递推实现；变量 `log_r`、`log_a` 均保留计算图：

```python
# 概念伪代码：只展示有效、完整 block；不代表本轮已实现。
log_p = log_softmax(target_logits.float().detach(), dim=-1)
log_q = log_softmax(corrected_draft_logits.float(), dim=-1)
log_r = zeros(num_blocks, dtype=float32)  # log r_0
log_a = zeros(num_blocks, dtype=float32)  # log a_0
scores = []
for i in range(B):
    log_terms = minimum(log_a[:, None] + log_p[:, i, :],
                        log_r[:, None] + log_q[:, i, :])
    # 退火阶段不 clamp；进入 block-log 阶段后对 log_terms 使用 -80 下限。
    scores.append(logsumexp(log_terms, dim=-1))
    log_r = log_r + gather(log_q[:, i, :], y[:, i]) \
                  - gather(log_p[:, i, :], y[:, i])
    log_a = minimum(log_a, log_r)
log_scores = stack(scores, dim=-1)
```

不在 BF16 中连乘，不对概率先 `softmax` 再直接相除，不给 `log_r/log_a` 加 `detach()`，不把教师 argmax 当成路径 token。全词表归一化必须保留，不能用 top-k 概率代替。

### 3.3 退火目标、反向代理与日志

记 `s_i=log(bar_a_i)`。退火的每 block 标量为

\[
\ell_\beta=-\frac{\operatorname{logsumexp}(\beta s_i)-\log B}{\beta},\quad 0<\beta\le1;
\qquad\ell_0=-\frac1B\sum_i s_i.
\]

β=1 时为 `log(B)-log(tau_hat)`，应先对**每个 block**计算，再对有效 block 平均；不要先跨 batch 平均接受长度再取 log。

推荐按论文实现习惯，退火期间使用

```python
w = softmax(beta * log_scores, dim=-1).detach()
loss_surrogate_per_block = -(w * log_scores).sum(dim=-1)
```

它在未触发数值下限时具有所需梯度，但标量不等于 `ell_beta`。因此分别记录反向用 surrogate 与真正的聚合目标；不能把前者的数值直接用于跨阶段验证比较。β=0 时权重均匀，无需除以 β。

从第 4 个 epoch 起使用 block-log loss。论文只说明 log summand 的下限，没有给出逐行 clamp 源码；v1 将其明确解释为上面词表积分中的 `log_terms`，这是本设计的数值落点，不应声称已经核对作者源码。后续若获得作者实现，应核对截断发生在词表求和前还是前缀求和前，并用极小概率测试确认差异。按本设计的 log 域实现 `log(B) - max(logsumexp(log_scores), log(1e-6))`，并在每个全词表积分的 log summand 上应用 -80 下限。不要误把 -80 截断施加到累计 `log_r` 上。退火阶段与 β=1 阶段的数值处理应独立测试。[依据：附录 A.3–A.4](https://arxiv.org/html/2609.34832v1)

总体 log 目标与逐 block log surrogate 不等价；后者存在 Jensen 差异。本方案实现的是论文实验配方，不声称每个 minibatch 的 log loss 是总体 log 目标的无偏估计。[来源：附录 B.5](https://arxiv.org/html/2609.34832v1)

## 4. 当前仓库关键事实

| 代码位置 | 当前行为 | 对 BV 的影响 |
|---|---|---|
| `scripts/train.py:1057` | `--loss-fn` 默认 `kl_div`，支持混合损失 | 不能把“旧流程”统一视为 CE；默认值保持不动 |
| `src/speculators/models/metrics.py:162` | CE 标签是教师 logits 的 argmax | BV 必须额外拿到实际 token |
| `src/speculators/models/dflash/core.py:321` | 返回 hidden、draft logits、教师 logits、aligned mask、block indices | 现有输出足以定位原始序列；优先在 DSpark 层新增标签构造 |
| `src/speculators/models/dspark/core.py:128` | 先跑 backbone，再加 Markov bias，最后算 metrics/loss | BV 插在 Markov 修正之后 |
| `src/speculators/models/dspark/metrics.py:79` | `compound_loss` 后叠加 confidence BCE | BV 需要独立的 block 聚合分支 |
| `src/speculators/models/dflash/utils.py:22` | anchor 选择检查位置 mask 与序列尾部 | 不保证整个预测 block 同文档、同 assistant 区间 |
| `src/speculators/train/trainer.py:439` | 每个 packed batch 一次更新；保存 epoch/local/global step | 可新增仅在 BV 启用时传递的 β |
| `src/speculators/train/dataloader.py:40` | 按 token budget multipack；没有固定 local batch 参数 | 32 个进程不等于 global 32 条对话 |
| `src/speculators/data_generation/vllm_client.py` | 请求已有 token 序列的 hidden states，`max_tokens=1` | online 不负责按 T=1 重生成整段回答 |

参考 29B 脚本还存在以下配置事实：注释写 Qwen3-4B，但模型路径实际为 TeleChat4-29B；`LOSS_FN` 是 CE/TV 混合；`NUM_LAYERS=5` 没有传给启动命令，实际结构来自 `--draft-config draft_config_29b.json`。本地两个目标目录中未找到该 JSON，因此尚不能验证真实层数、hidden size 和词表配置。

仓库预处理通常将 `loss_mask[t]` 标注为 token t 是否属于 assistant 内容；预分词数据也可以直接提供自己的 mask。29B 远程 Arrow 数据的实际 mask 语义必须核验，不能在已经 shift 的 mask 上重复 shift。

## 5. 拟实施的代码改动

### 5.1 CLI 与兼容性

新增参数：

| 参数 | 默认 | 行为 |
|---|---|---|
| `--enable-bv-loss` | False | 仅适用于 `speculator_type=dspark`，启用独立 BV 主损失 |
| `--bv-anneal-epochs` | 3.0 | β 线性 ramp 时长；0 表示从头采用 β=1 |

第一版固定全词表积分配方，避免同时引入多种 BV 估计器。数值下限作为有测试覆盖的实现常量。

开启 BV 时的校验：

- 非 DSpark 报错；`per-position-loss-weight=dpace` 报错。默认 fixed-exp-decay 参数允许存在，但不作用于 BV 主项，启动日志明确提示。confidence 若启用，其原有权重逻辑保留。
- 要求模型实际使用完整教师词表；检查解析后的 `use_draft_vocab`、映射与词表大小，不能只检查 CLI 是否出现 `--draft-vocab-size`。缩词表归一化会改变 p，本版不支持。
- 校验退火 epoch 非负，预测长度至少 1；遇到数据标签与 logits 词表不兼容立即失败。
- `loss_fn` 即使被覆盖仍做原有语法校验；日志打印 `effective_main_loss=bv_integrated_log`，不悄悄沿用混合权重。

不开启时不创建 BV 标签、不改变 anchor RNG 消耗、不执行 BV 专属 collective、不新增调度数据进入 forward，所有旧 CLI 默认和返回值维持原样。旧 CE/TV 混合脚本也必须完整回归。

### 5.2 标签、mask 与 block 边界

设 anchor 在原始 packed sequence 的索引 p，block slot 为 k。

| 配置 | 当前 slot 对应的教师输出 | 实际标签 | BV 的 B |
|---|---|---|---|
| `sample_from_anchor=True`（推荐） | hidden[p+k] 对下一 token 的分布 | `input_ids[p+k+1]`，k=0…block_size-1 | block_size |
| `sample_from_anchor=False` | 经 shift 后 hidden[p+k-1] 的分布 | `input_ids[p+k]`，只取 k=1…block_size-1 | block_size-1 |

本仓库 DSpark 默认前一种。因此论文的 B=15 应传 **block-size=15，不是 16**；slot 0 输入包含 anchor，并不意味着 slot 0 的输出不参与 loss。后一种配置若使用 B=15，则 block-size=16。

拟在 DSpark 的 BV 专属 helper 中，依据 `anchored_block_indices` 构造 prediction indices、label indices 与 block validity。读取 mask 时采用已确认的 token 语义，并显式对齐到 label 位置。索引越界、padding、跨文档均不得参与。

**第一版采用完整 block 策略：** 只有全部 B 个标签都属于连续有效 assistant 区间、预测位置和标签均与 anchor 同一有效 `document_id`、且 anchor 非填充时才计入 BV。原始 anchor 有效性可利用 backbone 返回的 aligned mask 识别；不得把补零 anchor 当真实样本。EOS 可作为最后一个合法标签，但不能跨过真实 EOS/对话结束边界继续计算。对话模板和 EOS ID 由对应模型配置确认。

暂不引入变长前缀归一化。禁止仅在最后对逐位置 loss 乘 mask：被屏蔽位置会污染后续累计比值。也禁止删除 block 中间无效 token 后拼接两段。记录有效 / 丢弃 block 数，评估完整 block 过滤导致的短回答覆盖率下降；这属于本方案的工程取舍，并非论文明确要求。

为减少对默认训练的影响，先沿用现有 anchor 采样，只在 BV 分支过滤；若有效 block 率过低，再单独评估 BV 专属 anchor 选择，不顺带改旧采样器。

### 5.3 损失模块与分布式归一化

拟新增 `src/speculators/models/dspark/bv_loss.py`：输入修正后 logits、教师 logits、真实 labels、valid-block mask、β/阶段；输出每 block loss、日志量、有效计数。内部显式在 FP32 计算 log-softmax、前缀状态和积分。

在 `dspark/metrics.py` 中通过开关选择新主项或旧 `compound_loss`。不要扩展通用逐 token `LossConfig` 让 BV 假装成逐位置 loss。无开关时保留旧调用路径。

分布式按**全局有效 block 数**归一化，避免不同 rank 的有效 block 数差异改变样本权重。若梯度规约是 W 个数据并行 rank 的平均，rank r 的反向标量为：

\[
L_r = W\,\frac{\sum_{b\in r}\ell_b}{\max(\sum_r N_r,1)}.
\]

只对有效计数做无梯度通信，通信组与实际数据并行组一致；FSDP 的梯度平均语义需通过集成测试核实。日志使用原始 block loss 之和与原始有效计数，不把用于补偿梯度平均的 W 倍标量当作真实均值。

本地零有效 block 时返回连接 draft 计算图的零值，照常参加其他 rank 的 backward/collective，不能单 rank 跳过。全局零有效 block 时统一跳过 optimizer 更新并记录；否则 AdamW 即使零梯度也可能发生 weight decay。epoch 位置时钟与实际 optimizer step 分开，LR scheduler 仅随实际更新推进。该处理仅启用 BV，连续空 batch 应报出数据诊断。

### 5.4 β 调度、验证与恢复

在 `Trainer.train_epoch()` 中，BV 开启时计算：

\[
\beta=\min\left(1,\frac{epoch+(local\_step-1)/steps\_in\_full\_epoch}{bv\_anneal\_epochs}\right).
\]

epoch 为零基；local_step 为完整 epoch 中的一基位置。使用 resume fast-skip 前的完整 epoch 长度。首个 batch β=0，第 4 个 epoch 起 β=1；`anneal_epochs=0` 直接为 1。该时钟按数据进度推进，与 LR warmup 独立。

每步拷贝 train-call kwargs 并注入动态 β/phase，不能原地污染共享 train/val kwargs。可用同设备 FP32 标量 tensor 传 β，避免 `conditional_torch_compile` 因每步 Python 浮点值变化持续重新编译；退火和 block-log 允许独立的有限数量编译图。

验证固定 β=1、固定 block-log 数值处理，`val/loss` 表示跨 epoch 可比较的目标。建议 BV 验证局部固定 anchor 随机种子，并保存/恢复 RNG 状态，避免随机选块噪声和验证扰动训练 RNG；不开启时不改变原验证行为。

BV checkpoint 的 training-state 增补有效 loss recipe、anneal epochs、完整 epoch 长度/进度、实际 optimizer update count 等必要元数据。断点恢复不得重新从 β=0 开始，数据规模/配方变化需显式报错或提示重新训练。旧 CE checkpoint 不作为 BV 的自动续训状态；新脚本用独立输出目录，从头初始化 drafter。

### 5.5 日志命名

- `bv_loss`：按 block 的实际聚合目标；退火时另记 `bv_backward_surrogate`。
- `bv_beta`、`bv_valid_blocks`、`bv_dropped_blocks`、`bv_empty_batches`：调度与覆盖率。
- `bv_estimated_draft_accept_len`：`sum_i bar_a_i` 的 block 平均，不包含 anchor、bonus/correction；它是训练路径估计值，不是推理实测。
- 可记录逐位置 `bar_a_i`、数值 floor 命中比例、每步耗时和峰值显存。

现有 `accept_len` 是逐位置 overlap 累乘再加 1，不能改名后当作 BV 接受长度。若保留，明确标注其旧近似语义；默认分支指标键不变。新增指标遵循现有 `_sum/_total` 聚合约定，真实计数在全局聚合前不要人为 clamp 为 1。

## 6. 29B 新脚本预案

拟新增文件：`speculators_treespark/examples/train/train_29b_dspark_bv_online.sh`。以下仅是脚本设计，新增 CLI 尚未实现，当前不能直接执行。

### 6.1 参数变更表

| 参数 | 参考脚本 | 新脚本默认建议 | 来源 / 说明 |
|---|---|---|---|
| teacher | TeleChat4-29B 既有路径 | 保留 | 项目环境 |
| epoch | 10 | 6 | 论文 |
| learning rate | 3e-4 | 6e-4 | 论文，29B 上仍需监测稳定性 |
| sequence token budget | 4096 | 3072 | 接近论文上限，但本仓库也是 packing budget |
| block size | 7 | 15，显式 `--sample-from-anchor` | 论文 B=15 的当前代码映射 |
| layers | 变量写 5，真正由外部 JSON 决定 | 检查 JSON 确为 5 层 | 不同时传与 JSON 冲突的 `--num-layers` |
| target hidden layers | 1 10 19 28 37 | 保留 | 项目配置；须和服务输出一致 |
| Markov | rank=256，rnn | 保留 | 项目配置，不声称论文使用该变体 |
| main loss | 0.1 CE + 0.9 TV | BV | 新开关 |
| position decay | gamma=6 | 不传，BV 主项不用它 | BV 内部已耦合前缀 |
| confidence | 默认开启 | 显式关闭 | 论文 |
| annealing | 无 | 3 epochs | 论文 |
| optimizer / decay | AdamW / 0.01 | 保留 | AdamW 对齐论文，0.01 继承参考脚本 |
| LR scheduler | cosine / 4% warmup | 保留 | 与论文一致 |
| seed | 默认 42 | 显式 42 | 论文 |
| hidden states dtype | BF16 | 保留；BV 运算 FP32 | 项目及数值要求 |
| hidden-state noise | 隐式默认 0.05 | 显式 `--noise-std 0` | 工程选择：减少训练 q 与推理 q 的差异，论文未明确该值 |
| anchors | 512 | 初始 128，可通过环境变量覆盖 | 29B 全词表显存折中，论文未报告该参数 |
| topology | 2 节点 × 16 NPU | 保留 | 环境约束；不是论文 batch 32 的复现 |
| dataset | 现有 80w Arrow | 接入时默认保留、可覆盖 | 主结果对齐实验建议另备 100K T=1 数据 |
| output | 原训练目录 | 独立 BV 目录 | 避免自动接续旧 CE/TV checkpoint |

若 128 anchors 稳定，测量吞吐和峰值内存后可恢复 512。B 从 7 增至 15 时，同 anchor 数的 draft 位置数约增加 2.14 倍；不能只因为新增 loss 没有参数就认为显存开销不变。

### 6.2 启动命令草案

新脚本保留参考脚本中的 Ascend 环境变量、两机 rank 到 IP 的映射、六个 vLLM endpoint 和 HCCL 网卡配置；清理未使用的 Qwen3 注释及变量。`CODE_PATH` 默认从脚本所在位置反推仓库根目录，避免继续指向另一个 `speculators` 部署。所有输出文件带 node rank，避免共享目录日志与 PID 互相覆盖。

```bash
#!/bin/bash
set -euo pipefail

NODE_RANK="${1:?usage: $0 <node_rank>}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_PATH="${CODE_PATH:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
MASTER_ADDR="${MASTER_ADDR:-10.127.10.100}"
MASTER_PORT="${MASTER_PORT:-8000}"
NIC_NAME="${NIC_NAME:-eth0}"
case "$NODE_RANK" in
  0) LOCAL_IP=10.127.10.100 ;;
  1) LOCAL_IP=10.127.10.206 ;;
  *) echo "node_rank must be 0 or 1" >&2; exit 1 ;;
esac

export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000
export NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1,80.5.5.45,80.5.5.44,80.5.5.54
export no_proxy="$NO_PROXY"
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME" TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"

MODEL="${MODEL:-/hpfs/nlp/yangzhihao/ckpts/TeleChat4-29B/29b_12.05T_wodsa_id20_s3_256k_8_260903_1000_final_hf}"
DATA_PATH="${DATA_PATH:-/hpfs/huawei-2607/shangxt/tmpDataTest/arrow_open_perfected_dehup_80w_no_thinking_full_vocab}"
OUTPUT_DIR="${OUTPUT_DIR:-/hpfs/huawei-2607/shangxt/Telecom-29B/dspark_bv_b15_ep6_lr6e-4}"
DRAFT_CONFIG="${DRAFT_CONFIG:?Set DRAFT_CONFIG to the verified absolute path of draft_config_29b.json}"
MAX_ANCHORS="${MAX_ANCHORS:-128}"
TRAIN_NPUS="${TRAIN_NPUS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
ENDPOINTS=(
  http://10.127.10.130:8000/v1 http://10.127.10.135:8000/v1
  http://10.127.10.106:8000/v1 http://10.127.10.143:8000/v1
  http://10.127.10.141:8000/v1 http://10.127.10.131:8000/v1
)
[[ -f "$DRAFT_CONFIG" ]] || { echo "Missing draft config: $DRAFT_CONFIG" >&2; exit 1; }
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_bv_rank${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/train_rank${NODE_RANK}.pid"

# 确保 editable install / import 指向本仓库；优先使用匹配部署环境的虚拟环境。
export PYTHONPATH="$CODE_PATH/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$CODE_PATH"
nohup env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
  --nnodes=2 --nproc_per_node=16 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  "$CODE_PATH/scripts/train.py" \
  --verifier-name-or-path "$MODEL" --trust-remote-code \
  --data-path "$DATA_PATH" --vllm-endpoint "${ENDPOINTS[@]}" \
  --save-path "$OUTPUT_DIR/checkpoints" \
  --draft-config "$DRAFT_CONFIG" --speculator-type dspark \
  --epochs 6 --lr 6e-4 --total-seq-len 3072 \
  --block-size 15 --sample-from-anchor --max-anchors "$MAX_ANCHORS" \
  --target-layer-ids 1 10 19 28 37 \
  --draft-attn-impl sdpa --mask-token-id 139 \
  --markov-rank 256 --markov-head-type rnn \
  --loss-fn ce --enable-bv-loss --bv-anneal-epochs 3 \
  --no-enable-confidence-head \
  --hidden-states-dtype bfloat16 --noise-std 0 \
  --on-missing generate --on-generate delete \
  --optimizer adamw --weight-decay 0.01 \
  --scheduler-type cosine --scheduler-warmup-ratio 0.04 \
  --seed 42 --num-workers 4 --prefetch-factor 4 --log-freq 1 \
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Log: $LOG_FILE"
echo "PID file: $PID_FILE"
```

最终脚本还应在启动前校验本仓库 import 路径、draft JSON 层数/hidden size、实际完整词表、数据目录以及 BV 功能是否可用；两节点共享相同代码、配置、输出路径。初次训练要求输出目录无旧 checkpoint，后续只允许恢复同配方 BV checkpoint。原脚本保持原样。

## 7. 与论文对齐的范围和限制

### 7.1 online hidden states 不等于 online rollout

沿用的 online 流程是：Arrow 中已有 token 序列 → 请求教师服务补算 hidden states → 本地冻结 norm/lm_head 重建教师 logits → 训练草稿。改变此次 hidden-state 请求的生成温度，不会把已有回答变成论文的 T=1 样本。

如要接近理论采样条件，应另行准备同一 29B 教师、同一 tokenizer/chat template、T=1、完整词表采样的回答，并确认无 top-k/top-p 截断或额外 logit penalty 改变实际分布。固定 stop/EOS 处理并保存生成元数据。若实际用了截断采样，单纯拿原始 logits 的 softmax 作分母就不满足论文的条件。

现有 80w 数据可以用于工程适配实验，但在未核验生成元数据前，只能标为“既有数据上的 BV surrogate 实验”。论文规模对齐组建议单独构建固定 100K 目标生成训练集，并使用独立验证集。当前 `train_data_ratio=0.9` 还会从输入数据切出验证集，不能把输入目录 100K 条等同为实际训练 100K 条。

### 7.2 batch size 无法仅通过当前 CLI 严格复现

当前每个 rank 每步处理一个最大 3072 token 的 packed tensor，其中可能有多条短对话；Trainer 没有梯度累积或固定 local batch=4 的通用参数。

因此保留 32 NPU 意味着每步 32 个 packed tensor，**不等于 32 条样本，更不等于 local batch=4**。也不能通过设 `max_anchors=4` 模拟 local batch，anchor 是单个 packed tensor 内的训练上下文数。

v1 不为了复现 batch 引入全局 dataloader 改造。实际报告每步对话数、有效 tokens、有效 blocks 和 DP size。若后续要求严格 batch 对齐，应单独增加按样本数的 loader/累积方案，例如 8 个 DP worker × 每个 4 条样本；这超出本次最小 BV 适配范围。

### 7.3 29B 显存与算力

BV 的输入规模为 `anchors × B × vocab_size`。FP32 的一份该尺寸张量约为 `4 × anchors × B × vocab_size` 字节。仅作量级示例：若 V=150000，128×15×V 的一份 FP32 张量约 1.15 GB（十进制），512 anchors 则约 4.61 GB；这不是已核实的 29B 词表大小，也不是总显存。

还需计入教师与 draft logits、log probabilities、Markov bias、保存的反向激活、backbone 和优化器。现有 backbone 在切块前还会产生整个 packed sequence 的教师 logits。完整 block 筛选并不自动消除这些既有分配。

第一版以准确的 PyTorch 参考实现为准，避免直接写 fused kernel。通过 anchor chunking、必要的激活重算降低 BV 的额外峰值；仅在 Python 中分块但保留所有计算图不保证反向显存下降。词表分块必须保留全词表 logsumexp 归一化及积分，不能分别对每块词表做 softmax。真实容量与吞吐需在 Ascend 上测量。

### 7.4 推理评估与指标口径

开启 BV 训练不会自动启用 block verification，也不会保证论文在 Qwen3-4B/8B 上的增益迁移到 TeleChat4-29B。`accept_len` 与训练 `bv_estimated_draft_accept_len` 均不能代替真实 decode 指标。

离线评测默认仍使用 token verification：

```bash
python scripts/evaluate/dspark_offline_eval.py \
  --verifier-model /path/to/target \
  --draft-model /path/to/dspark \
  --datasets-root /path/to/jsonl
```

只有显式增加下列开关时，才使用 Sun et al. Algorithm 2 的无损 block verification：

```bash
python scripts/evaluate/dspark_offline_eval.py \
  --verifier-model /path/to/target \
  --draft-model /path/to/dspark \
  --datasets-root /path/to/jsonl \
  --temperature 1 \
  --tree-top-k 1 \
  --block-verification
```

block verifier 计算联合前缀状态，对各长度前缀独立抽样并提交通过的最长前缀；未全接受时从 `(p_tau * target - draft).clamp_min(0)` 归一化后抽取 correction token。它仅支持线性 DSpark，因此与 `--tree-top-k >= 2` 互斥。Ascend 多进程启动会把该开关传给每个 worker。结果中 `acceptance_length` 仍按“已接受 draft token + correction/bonus”计数；block 模式下 `position_accept_prob_means` 记录论文中各前缀的抽样概率 `h_i`。

树验证不能仅因处理多个 token 就当作论文的 block verification。记录 verification 类型、温度、B、EOS 规则、是否包含 bonus/correction、请求数、总生成 token 和同步后的实际耗时，避免和论文不同口径直接对比。

## 8. 验收与测试计划

### 8.1 默认行为兼容

固定 seed、输入 batch 与初始化，对比变更前和不开启 BV 的：forward loss、各参数梯度、一步 optimizer 结果、指标键/值、anchor 抽样以及 checkpoint 内容。覆盖纯 CE、默认 KL、CE/TV 混合、confidence 开关与 sample-from-anchor 两种模式。在相同确定性环境下要求一致，后端非确定性仅使用有依据的数值容差。

不开启时不增加教师请求、不额外做 BV 张量分配或 collective；不改变其它模型算法。使用现有 DSpark metrics/model tests，加必要兼容回归。

### 8.2 数学与梯度测试

1. B=1、β=1：积分 block-log 退化为负 log overlap；线性 `1-overlap` 才是 TV，二者不能混淆。
2. q=p：所有前缀积分分数为 1、估计草稿长度 B、loss 约为 0；另外测试极端 logits 的有限值。
3. 小词表短 block 穷举目标路径，比较 sampled-path 估计与全词表积分估计的期望；包括依赖前缀的条件分布，避免只测试独立 token。
4. 用非 argmax 的真实标签构造路径，检验 cumulative ratio 随真实 y 改变；检查积分 score 使用旧状态、状态由 sampled label 更新。
5. 无 floor、无 min tie 的 FP64 小样本梯度校验；验证 `detach(power weights)` surrogate 与 power-mean 目标梯度一致，但标量一般不同。
6. 梯度到达 Markov 与 backbone，教师无梯度；前缀状态未 detach；位置衰减不影响 BV 主项。

### 8.3 数据与调度测试

覆盖两种 slot/label 偏移、anchor 填充、同文档检查、assistant mask 间断、EOS、序列末尾、缩词表拒绝、空 block、全空 rank。测试末尾有效 token 不发生 roll 到文档开头。

检查 β=0、ramp 中间、第三与第四 epoch 交界、关闭 anneal、epoch 中断恢复、epoch 末恢复。验证固定 β=1，best checkpoint 使用可比较的验证值。检查更改数据长度或配方后的恢复行为。

分布式测试有效 block 不均匀与单 rank 全空，和拼接到单进程计算的全局平均梯度对齐；全局全空时一致跳过更新，不发生 deadlock 或 weight decay。Ascend 测试 FP32 logsumexp/min、autocast、编译图数量与实际数值。

### 8.4 29B 运行验证

先少量固定样本和较小 anchors 跑通 forward/backward、验证与 checkpoint 恢复；再以完整 32 NPU 配置测吞吐、教师服务负载和峰值显存。仅在这些检查通过后运行 6 epochs。

实验至少区分：旧混合损失基线、同新配置下的 CE 对照、BV。旧 CE 本身使用教师 argmax，与论文的存储样本标签 CE 不同，报告应写明；本次不修改该 CE 定义。若要隔离 BV 的实际标签选择贡献，可后续单独增加 sampled-label CE 对照，而非改变默认行为。另可加入 β 固定 1 的 BV 消融。

## 9. 后续文件清单与当前交付边界

| 文件 | 预期工作 |
|---|---|
| `scripts/train.py` | 开关、退火参数、BV 专属校验与配置日志 |
| `src/speculators/models/dspark/core.py` | BV 参数传递、真实 labels/完整 block 构造、Markov 后接入 |
| `src/speculators/models/dspark/bv_loss.py`（新增） | 全词表积分、log 域前缀递推、退火与 block-log、数值处理 |
| `src/speculators/models/dspark/metrics.py` | 旧路径 / BV 路径分支与指标，不改通用 token loss |
| `src/speculators/train/trainer.py` | 仅 BV 的调度、全局有效计数协调、验证及恢复元数据 |
| `tests/unit/models/test_dspark_bv_loss.py` 等（新增） | 数学、梯度、mask、接口测试 |
| trainer / CLI / 分布式测试 | 兼容、调度、恢复与计数规约 |
| `examples/train/train_29b_dspark_bv_online.sh`（新增） | 根据 §6 落地新脚本 |

优先不修改 DFlash 公共 backbone 返回接口；确需暴露原始 anchor validity 时，仅增加兼容的可选元数据路径，并补 DFlash 回归。

正式 29B 训练前还需取得：远程 draft JSON、29B 实际词表与 tokenizer/EOS 配置、Arrow mask 语义、回答生成参数、教师服务 hidden-layer/norm 约定以及硬件容量。它们不阻止先实现与测试 BV 功能，但目前不能声称训练脚本已在该环境验证可运行。

本文档是本轮唯一新增文件。代码实施、`.sh` 文件创建、依赖安装与训练运行均未进行。
