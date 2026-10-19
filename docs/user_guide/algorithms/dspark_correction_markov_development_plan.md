# DSpark Correction Markov 开发报告

> 状态：已完成训练侧功能、单元测试、回归测试和 29B 双机训练脚本。
> 开发分支：`dspark_correction_markov`，基线分支：
> `dspark_new_ar_head_test`。推理引擎适配不在本次范围内。

## 1. 开发目标

从 `dspark_new_ar_head_test` 分支创建
`dspark_correction_markov` 分支，新增独立模型：

```text
dspark_correction_markov
```

该模型同时保留：

1. DSpark/DFlash 原生 backbone、LM Head、训练目标和指标。
2. DSpark 原生 `MarkovHead`，支持 `vanilla`、`gated`和 `rnn`。
3. `dspark_correction` 中已验证的 MMuse 基础 Causal Correction Head。
4. 参考 MMuse 的 Correction-Markov 协作门控。
5. DSpark Confidence Head，可同时使用 Markov 和 Correction 特征。

最终输出不是在 Markov 和 Correction 之间二选一，而是：

```text
final_logits
    = correction_logits
    + collaboration_gate * markov_bias
```

## 2. 隔离原则

新模型必须是第三个独立注册项：

| 模型类型 | 顺序修正结构 |
| --- | --- |
| `dspark` | 仅 Markov |
| `dspark_correction` | 仅 Correction |
| `dspark_correction_markov` | Correction + Markov/RNN |

实现时遵守以下边界：

- 不修改 `src/speculators/models/dspark/` 中的原生实现。
- 不改变 `dspark` 注册名、默认参数和 checkpoint 格式。
- 不修改已存在的 `dspark_correction` 语义。
- 新模型独立注册为 `dspark_correction_markov`。
- 新 checkpoint 使用独立的 `speculators_model_type`和 architecture 名称。
- 共享已验证的 `CausalCorrectionHead`，不复制第二份 Correction 实现。
- 仅参考 MMuse 的 Correction 和 Correction-Markov collaboration，不引入
  Selector、DynamicConv、hidden feedback、MMuse backbone 或其他扩展。

## 3. 与 MMuse 的关系

### 3.1 保持一致的部分

以下协作逻辑与 MMuse 保持一致：

```text
markov_bias = MarkovHead(previous_token, dflash_hidden)

local_gate = sigmoid(
    Linear(correction_causal_state)
)

collaboration_gate = tanh(global_scale) * local_gate

final_logits = correction_logits
             + collaboration_gate * markov_bias
```

其中：

- `local_gate` 为每个 block slot 产生独立标量门值。
- `global_scale` 是整个模型共享的可训练标量。
- `tanh(global_scale)` 将全局缩放约束在 `(-1, 1)`。
- Markov bias 始终生成，是否以及多大程度修正 Correction 结果，
  由 Correction causal state 决定。

### 3.2 针对 RNN 的必要扩展

MMuse 当前的配置验证明确禁止：

```text
Correction + markov_head_type=rnn
```

MMuse 这一限制主要来自顺序 rollout 的 RNN state 管理，不是
Correction-Markov 融合公式本身不能处理 RNN bias。

`dspark_correction_markov` 需要明确扩展为：

- 训练侧允许 `markov_head_type=rnn`。
- 直接复用原生 DSpark `MarkovHead.block_bias()` 的 RNN 分支。
- RNN 仍在每个 draft block 内从左到右维护 state。
- RNN 产生的 `markov_bias` 再经过与 vanilla/gated 完全相同的
  Correction-state collaboration gate。
- 首期仅承诺 teacher-forced 训练路径；推理侧需要单独实现每个
  proposal block 的 RNN state cache。

这是本新模型相对 MMuse 的唯一核心结构扩展。

## 4. 整体模型结构

```text
verifier hidden states
          |
          v
    DFlash/DSpark backbone
          |
          +-----------------------> h_dflash
          |                              |
          |                              +-----------------------+
          |                              |                       |
          v                              v                       v
       LM Head                 Causal Correction Head       MarkovHead
          |                              |                 vanilla/gated/rnn
          v                              |                       |
    base_logits                         |                       v
          |                              |                  markov_bias
          |                              v                       |
          |                    correction_causal_state          |
          |                              |                       |
          |                              +--> collaboration gate-+
          |                              |
          +--------------+---------------+
                         |
                         v
                correction_logits
                         |
                         + collaboration_gate * markov_bias
                         |
                         v
                    final_logits
                         |
              DSpark loss and metrics
```

## 5. Correction 路径

Correction 直接复用 `dspark_correction` 已验证的
`CausalCorrectionHead`。

### 5.1 Correction 输入

对于 block 中第 `i` 个 slot：

```text
x_i = W_h h_i
    + W_t Emb(previous_token_i)
    + Emb(position_i)
```

Logits mode 增加前一位置 target distribution 的低秩编码：

```text
p_(i-1) = softmax(target_logits_(i-1))
r_(i-1) = W_previous_down p_(i-1)
x_i = x_i + W_previous_proj r_(i-1)
```

训练时使用 teacher forcing，所有位置都只能读取当前位置之前的信息。

### 5.2 Hidden mode

```text
delta_hidden
    = W_up(SiLU(W_down(correction_state)))
    * sigmoid(residual_gate(correction_state))

corrected_hidden = h_dflash + delta_hidden
correction_logits = LMHead(corrected_hidden)
```

### 5.3 Logits mode

```text
delta_logits
    = W_up(SiLU(W_down(correction_state)))
    * sigmoid(residual_gate(correction_state))

correction_logits = base_logits + delta_logits
```

## 6. Markov/RNN 路径

### 6.1 Vanilla Markov

```text
previous_embedding_i = MarkovW1(previous_token_i)
markov_bias_i = MarkovW2(previous_embedding_i)
```

### 6.2 Gated Markov

```text
markov_gate_i = sigmoid(
    W_gate([h_dflash_i; previous_embedding_i])
)

markov_bias_i = MarkovW2(
    markov_gate_i * previous_embedding_i
)
```

### 6.3 RNN Markov

RNN 完全保留 DSpark 现有计算：

```text
state_0 = 0

z_i = [state_(i-1); previous_embedding_i; h_dflash_i]

gate_raw_i, candidate_raw_i, output_raw_i
    = split(W_joint(z_i))

rnn_gate_i = sigmoid(gate_raw_i)

state_i
    = rnn_gate_i * state_(i-1)
    + (1 - rnn_gate_i) * tanh(candidate_raw_i)

markov_bias_i = MarkovW2(tanh(output_raw_i))
```

RNN state 只在当前 draft block 内传播，不跨 block 共享。

## 7. Correction-Markov 协作

### 7.1 融合顺序

固定使用：

```text
DFlash -> Correction -> gated Markov residual
```

不采用以下顺序：

```text
DFlash -> Markov -> Correction
```

原因是 MMuse 的 collaboration gate 使用 Correction causal state 判断当前位置
需要多少 Markov 修正，而 Correction 本身不需要感知 Markov 输出。

### 7.2 Markov 使用的 hidden

Markov Head 使用原始 `h_dflash`，不使用 `corrected_hidden`：

```text
markov_bias = MarkovHead(previous_tokens, h_dflash)
```

这与 MMuse 一致，也可以确保 Markov 分支仍保持原生 DSpark 语义。

### 7.3 Collaboration gate

新增两个参数：

```text
correction_markov_gate: Linear(correction_hidden_size, 1)
correction_markov_scale: scalar Parameter
```

每个位置的最终门值：

```text
local_gate_i = sigmoid(
    correction_markov_gate(correction_state_i)
)

collaboration_gate_i
    = tanh(correction_markov_scale) * local_gate_i
```

最终输出：

```text
final_logits_i
    = correction_logits_i
    + collaboration_gate_i * markov_bias_i
```

### 7.4 稳定初始化

参考 MMuse：

```text
correction_up.weight = 0
correction_markov_gate.weight = 0
correction_markov_gate.bias = -2.0
correction_markov_scale = 0
```

因此初始时：

```text
delta_correction = 0
collaboration_gate = 0
final_logits = DFlash base logits
```

这里的“保留 Markov”指 Markov/RNN 结构、参数和训练路径都存在；
为了避免新增两条 residual 在第一步同时扰动 backbone，MMuse 门控使
Markov 对 logits 的实际影响从零开始学习。

## 8. Confidence Head

为了既保留 DSpark Confidence Head 的 Markov 条件，又让它看到
Correction 的块内因果状态，新模型支持两个独立开关：

```text
confidence_head_with_markov
confidence_head_with_correction
```

当两者都开启时：

```text
confidence_features
    = concat(
        h_dflash,
        markov_previous_token_embedding,
        correction_causal_state,
      )
```

注意：原生 DSpark 即使使用 `markov_head_type=rnn`，Confidence Head 使用的
也是 Markov previous-token embedding，而不是 RNN 内部 state。新模型保持这一
语义。

输入维度为：

```text
hidden_size
+ (markov_rank if confidence_head_with_markov else 0)
+ (correction_hidden_size if confidence_head_with_correction else 0)
```

## 9. Loss 与训练指标

主损失继续使用 telecom 仓库原生 DSpark `compute_metrics()`：

- draft token loss。
- per-position loss weighting。
- Confidence Head loss。
- acceptance length 及相关指标。

Hidden Correction mode 可选保留：

```text
SmoothL1(corrected_hidden, verifier_pre_lm_hidden)
```

不从 MMuse 引入 Selector loss、rollout loss 或自适应损失。

为了评估两个顺序分支的协作情况，实现了下列可分布式汇总的
`sum/total` 指标：

```text
correction_markov_gate_sum / correction_markov_gate_total
correction_markov_bias_norm_sum / correction_markov_bias_norm_total
correction_markov_scale_sum / correction_markov_scale_total
```

这些指标不改变训练目标。

## 10. 代码实现方案

### 10.1 新增目录

```text
src/speculators/models/dspark_correction_markov/
├── __init__.py
├── config.py
└── core.py
```

| 文件 | 职责 |
| --- | --- |
| `config.py` | 独立 config、注册名和组合参数验证 |
| `core.py` | DSpark backbone、Correction、Markov/RNN、融合与损失 |
| `__init__.py` | 模型和 config 导出 |

### 10.2 复用策略

```text
DSparkCorrectionMarkovDraftModel
    ├── DFlashDraftModel / DSparkDraftModel 能力
    ├── DSpark MarkovHead
    ├── DSpark ConfidenceHead
    └── dspark_correction.CausalCorrectionHead
```

实际初始化步骤：

1. 继承 `DSparkCorrectionDraftModel`，复用未修改的 DFlash backbone 和
   `CausalCorrectionHead`。
2. 创建原生 DSpark `MarkovHead`，保留其参数名和计算方式。
3. 按 backbone、Markov embedding 和 Correction state 的开关重建新模型的
   Confidence Head。
4. 新增 `correction_markov_gate` 和
   `correction_markov_scale`。

这种做法使 `MarkovHead` 和 Correction 的实现都只有一份，新模型只负责
两条路径的组合。

### 10.3 Forward 步骤

```text
1. 调用原生 _backbone_forward()
2. 恢复 [num_blocks, block_size, ...] block view
3. 构建 previous_token_ids
4. 运行 Correction，得到 correction_logits 和 correction_states
5. 调用原生 MarkovHead.block_bias()
6. 由 correction_states 计算 collaboration gate
7. 融合 correction_logits 和 markov_bias
8. 构建 Confidence Head 输入
9. 调用原生 DSpark compute_metrics()
10. 可选加入 hidden alignment loss
```

## 11. 配置与 CLI 参数

### 11.1 模型选择

| 参数 | 必要值 |
| --- | --- |
| `--speculator-type` | `dspark_correction_markov` |

### 11.2 Markov/RNN 参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--markov-rank` | `256` | Markov 低秩维度，必须大于 0 |
| `--markov-head-type` | `vanilla` | `vanilla`、`gated` 或 `rnn` |

`dspark_correction_markov` 不允许 `--markov-rank 0`，因为该模型的定义就是
Correction 和 Markov 共存。

### 11.3 Correction 参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--correction-output-mode` | `hidden` | `hidden` 或 `logits` |
| `--correction-hidden-size` | `512` | Correction causal state 维度 |
| `--correction-rank` | `256` | Correction 输出低秩维度 |
| `--correction-num-layers` | `1` | Tiny causal Transformer 层数 |
| `--correction-num-heads` | `8` | Attention head 数 |
| `--correction-gate-bias` | `0.0` | Correction residual gate 初始 bias |
| `--correction-hidden-aux-loss` | 关闭 | Hidden alignment loss |
| `--correction-hidden-aux-weight` | `0.1` | Hidden alignment loss 权重 |

### 11.4 Collaboration 参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--correction-markov-gate-bias` | `-2.0` | MMuse 风格的局部门初始 bias |
| `--correction-markov-scale-init` | `0.0` | tanh 前全局 scale 初始值 |

`--correction-with-markov` 不作为额外开关：选择
`dspark_correction_markov` 已经表示必须同时开启两者。

### 11.5 Confidence 参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--enable-confidence-head` | 开启 | 保留 DSpark Confidence Head |
| `--confidence-head-with-markov` | 开启 | 加入 Markov previous-token embedding |
| `--confidence-head-with-correction` | 开启 | 加入 Correction causal state |
| `--confidence-head-alpha` | `1.0` | Confidence loss 权重 |

## 12. 参数验证

启动时必须验证：

```text
markov_rank > 0
correction_hidden_size > 0
correction_rank > 0
correction_num_layers > 0
correction_num_heads > 0
correction_hidden_size % correction_num_heads == 0
```

模式约束：

- `correction_output_mode=logits` 禁止 `correction_hidden_aux_loss=true`。
- `markov_head_type=rnn` 在新模型中是合法值。
- Correction-Markov 参数不能显式传给原生 `dspark`。
- 新模型的专用参数不能被其他 speculator type 静默忽略。

## 13. 训练使用教程

### 13.1 Vanilla Markov + Hidden Correction

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --verifier-name-or-path /path/to/Qwen3-8B \
  --data-path /path/to/processed_data \
  --vllm-endpoint http://localhost:8000/v1 \
  --save-path ./output/dspark_correction_markov_hidden \
  --speculator-type dspark_correction_markov \
  --draft-vocab-size 32000 \
  --block-size 8 \
  --max-anchors 512 \
  --num-layers 5 \
  --draft-attn-impl sdpa \
  --target-layer-ids 1 9 17 25 33 \
  --markov-rank 256 \
  --markov-head-type vanilla \
  --correction-output-mode hidden \
  --correction-hidden-size 512 \
  --correction-rank 256 \
  --correction-num-layers 1 \
  --correction-num-heads 8 \
  --correction-gate-bias 0.0 \
  --correction-markov-gate-bias -2.0 \
  --correction-markov-scale-init 0.0 \
  --enable-confidence-head \
  --confidence-head-with-markov \
  --confidence-head-with-correction \
  --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --confidence-head-alpha 1.0 \
  --epochs 5 \
  --lr 6e-4 \
  --on-missing generate \
  --on-generate delete
```

### 13.2 RNN Markov + Hidden Correction

在上述命令中替换：

```bash
--markov-head-type rnn
```

RNN 使用相同的 `--markov-rank`，不需要额外的 RNN hidden-size 参数。

### 13.3 RNN Markov + Logits Correction

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --verifier-name-or-path /path/to/Qwen3-8B \
  --data-path /path/to/processed_data \
  --vllm-endpoint http://localhost:8000/v1 \
  --save-path ./output/dspark_correction_markov_rnn_logits \
  --speculator-type dspark_correction_markov \
  --draft-vocab-size 32000 \
  --block-size 8 \
  --max-anchors 512 \
  --num-layers 5 \
  --draft-attn-impl sdpa \
  --target-layer-ids 1 9 17 25 33 \
  --markov-rank 256 \
  --markov-head-type rnn \
  --correction-output-mode logits \
  --correction-hidden-size 512 \
  --correction-rank 256 \
  --correction-num-layers 1 \
  --correction-num-heads 8 \
  --correction-gate-bias 0.0 \
  --correction-markov-gate-bias -2.0 \
  --correction-markov-scale-init 0.0 \
  --enable-confidence-head \
  --confidence-head-with-markov \
  --confidence-head-with-correction \
  --no-correction-hidden-aux-loss \
  --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --confidence-head-alpha 1.0 \
  --epochs 5 \
  --lr 6e-4 \
  --on-missing generate \
  --on-generate delete
```

### 13.4 Dry run

正式训练前在命令末尾增加：

```bash
--dry-run
```

生成的 `config.json` 至少应包含：

```json
{
  "speculators_model_type": "dspark_correction_markov",
  "markov_rank": 256,
  "markov_head_type": "rnn",
  "correction_output_mode": "hidden",
  "correction_hidden_size": 512,
  "correction_rank": 256,
  "correction_markov_gate_bias": -2.0,
  "correction_markov_scale_init": 0.0
}
```

### 13.5 TeleChat4-29B 双机在线训练

仓库已新增：

```text
examples/train/train_29b_dspark_correction_markov_online.sh
```

该脚本从原生 `train_29b_dspark_online.sh` 复制并仅增加组合模型所需的
Correction、collaboration 和 Confidence 参数。脚本默认使用
`rnn` Markov head 和 hidden Correction。

启动前需按部署环境检查脚本顶部的 `MODEL`、`DATA_PATH`、
`OUTPUT_DIR`、`CODE_PATH`、`MASTER_ADDR`、两台机器的 `LOCAL_IP` 和
`TRAIN_NPUS`。两台机器使用同一份代码和配置，分别执行：

```bash
# master node
bash examples/train/train_29b_dspark_correction_markov_online.sh 0

# worker node
bash examples/train/train_29b_dspark_correction_markov_online.sh 1
```

脚本通过 `nohup` 后台启动。查看、停止和 TensorBoard 命令会由脚本打印；
也可直接使用：

```bash
tail -f <OUTPUT_DIR>/logs/train_telecom_80w_no_thinking_32_96_dspark_correction_markov_rnn_*.log
kill "$(cat <OUTPUT_DIR>/logs/train.pid)"
tensorboard --logdir <OUTPUT_DIR>/logs --host 0.0.0.0 --port 6006
```

## 14. Checkpoint 兼容方案

### 14.1 新 checkpoint

新 checkpoint 必须保存：

- DFlash/DSpark backbone。
- `markov_head.*`。
- `correction_head.*`。
- `correction_markov_gate.*`。
- `correction_markov_scale`。
- 按新输入维度创建的 `confidence_head.*`。

### 14.2 从 DSpark checkpoint 热启动

后续可增加显式的部分加载能力：

- backbone 权重严格加载。
- `markov_head.*` 严格加载。
- Correction 和 collaboration gate 新建。
- Confidence Head 因输入维度变化，不能直接严格加载；可将原权重拷贝到
  `h_dflash + markov_embedding` 对应的 slice，Correction slice 使用零初始化。

首期实现不应默认忽略 checkpoint key；如果支持热启动，必须使用显式
参数和加载报告。

## 15. 测试与验收结果

### 15.1 新增测试

新增：

```text
tests/unit/models/test_dspark_correction_markov.py
```

已覆盖：

1. `dspark`、`dspark_correction`和 `dspark_correction_markov` 同时注册，
   互不覆盖。
2. 新模型同时存在 `correction_head` 和 `markov_head`。
3. `vanilla`、`gated`、`rnn` 三种 Markov 类型都可构建。
4. Hidden/logits Correction 均可与三种 Markov 类型组合。
5. `sample_from_anchor=True/False` 两种对齐方式通过。
6. Collaboration 公式与手工计算结果完全一致。
7. `correction_markov_scale=0` 时 Markov 分支不改变 Correction logits。
8. 将 scale 设为非零后，Markov/RNN 参数能获得有限梯度。
9. Correction 输出投影、collaboration gate 和 global scale 能获得梯度。
10. RNN state 不跨 block 泄漏。
11. 修改未来 slot 不影响过去 Correction slot，验证因果性。
12. Confidence Head 输入维度在四种开关组合下正确。
13. Hidden auxiliary loss 只能用于 hidden mode。
14. `markov_rank=0` 被拒绝。
15. checkpoint round trip 恢复为新模型类。
16. 原生 DSpark 的 Markov/RNN、metrics 和 CLI 回归测试继续通过。

需要特别注意：MMuse 风格的 `global_scale=0` 初始化会使 Markov 分支在
第一个 backward 中不获得权重梯度，但 `global_scale` 本身可获得梯度。
因此 Markov 梯度测试需要显式将 scale 设为非零，或运行至少两个
optimizer step。

CLI 参数隔离测试位于：

```text
tests/unit/train/test_cli_args.py
```

它验证组合参数可正常解析、`markov_rank=0` 被拒绝，以及 collaboration
参数不能误用于 Correction-only 模型。

### 15.2 专项测试流程

在仓库根目录执行：

```bash
python -m pip install -e '.[dev]'

python -m pytest \
  tests/unit/models/test_dspark_correction_markov.py \
  tests/unit/train/test_cli_args.py \
  -q
```

静态检查和脚本语法检查：

```bash
python -m ruff check \
  scripts/train.py \
  src/speculators/models/dspark_correction_markov \
  tests/unit/models/test_dspark_correction_markov.py \
  tests/unit/train/test_cli_args.py

python -m compileall -q \
  src/speculators/models/dspark_correction_markov \
  scripts/train.py

bash -n examples/train/train_29b_dspark_correction_markov_online.sh
```

### 15.3 DSpark 回归测试流程

```bash
python -m pytest \
  tests/unit/models/test_dspark_correction_markov.py \
  tests/unit/models/test_dspark_correction.py \
  tests/unit/models/test_dspark_model_definitions.py \
  tests/unit/models/test_dspark_metrics.py \
  tests/unit/models/test_dspark_dflare_variants.py \
  tests/unit/train/test_cli_args.py \
  -q
```

实际结果：

```text
92 passed, 14 warnings
```

14 个 warning 均来自 PyTorch `torch.jit.script_method` 弃用提示，与本次模型
实现无关。

也尝试运行仓库全部 `tests/unit`，但原有 Eagle/MTP 用例会从
Hugging Face 下载 `nm-testing/testing-llama3.1.8b-2layer-eagle3` 和
`Qwen/Qwen3.5-0.8B`。当前无网络环境中这些与本功能无关的用例无法完成，
因此验收以上述可完全离线重现的 92 项 DSpark/Correction/CLI 测试为准。

### 15.4 训练验收流程

1. 先执行 15.2 和 15.3 的测试。
2. 用 13.1 的命令加 `--dry-run` 验证真实 verifier 配置和 checkpoint
   保存。
3. 检查 `config.json` 中的组合模型类型、Correction 参数和 Markov
   参数。
4. 使用小数据集和短训练轮数进行 smoke training，检查 loss 有限且
   `correction_markov_gate` / `bias_norm` / `scale` 指标可观测。
5. 最后按 13.5 在两个节点启动完整 29B 训练。

本地已完成模型构建、forward/backward、checkpoint round trip 和上述 92 项
自动化测试。由于当前环境没有 TeleChat4-29B checkpoint、Ascend 集群和在线
vLLM 服务，本次没有实际启动 29B 全量训练。

## 16. 对比实验建议

为了判断增益来自哪个分支，应在相同数据、backbone、block size、
optimizer 和训练 budget 下至少比较：

| 实验 | 模型 | 顺序结构 |
| --- | --- | --- |
| A | `dspark` | vanilla Markov |
| B | `dspark` | RNN Markov |
| C | `dspark_correction` | Correction only |
| D | `dspark_correction_markov` | Correction + vanilla Markov |
| E | `dspark_correction_markov` | Correction + gated Markov |
| F | `dspark_correction_markov` | Correction + RNN Markov |

建议同时报告：

- draft token loss。
- 各位置 acceptance rate。
- 平均 acceptance length。
- Correction residual norm。
- Markov bias norm。
- collaboration gate 均值与分布。
- 训练吞吐和显存占用。

## 17. 推理边界

首期开发仅完成 Speculators 训练侧模型。

实际 speculative decoding 推理还需要：

1. Correction causal attention 的每层 K/V cache。
2. Logits mode 的 previous-logits feedback。
3. Markov previous-token 对齐。
4. RNN Markov 的 block-local recurrent state cache。
5. 每个 slot 的 Correction-Markov collaboration gate。
6. 向 vLLM/SGLang 注册新 checkpoint architecture。

在上述推理适配完成之前，不能将
`dspark_correction_markov` checkpoint 直接当作原生 DSpark checkpoint 部署。

## 18. 已完成的实施项

1. 新增 config 和独立 registry entry。
2. 复用 Correction Head，创建 Markov + Correction 模型。
3. 实现 MMuse 风格 collaboration gate。
4. 先打通 `vanilla + hidden Correction`。
5. 增加 `gated` 和 `rnn` Markov 组合。
6. 打通 logits Correction。
7. 增加 Confidence 双特征输入和 optional hidden loss。
8. 完成 CLI 隔离、checkpoint round trip 和回归测试。
9. 更新开发报告，记录实际文件、测试命令和测试结果。

## 19. 最终验收状态

训练侧实现已满足：

- `dspark_correction_markov` 可通过通用 registry 构建和恢复。
- Correction 和 Markov/RNN 确实同时参与 final logits 计算。
- `markov_head_type=rnn` 完成 forward/backward，不被 MMuse 的原限制拒绝。
- Collaboration 公式与 MMuse 数学定义一致。
- 新模型不引入 MMuse 的其他结构。
- 原生 `dspark` 和 `dspark_correction` 代码与行为不变。
- 所有新增测试和原生 DSpark 回归测试通过。
- 文档中清晰标注训练侧和推理侧的能力边界。
