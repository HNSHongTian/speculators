# DSpark Correction 开发报告

## 1. 开发目标与边界

本次开发在 `dspark_new_ar_head_test` 分支上新增
`dspark_correction` 草稿模型，参考 MMuse 的 Causal Correction 设计，用块内
因果 Correction Head 替代 DSpark 原有的 Markov Head。

设计边界：

- 不修改 `src/speculators/models/dspark/` 的实现、配置和 `dspark` 注册名。
- 新模型独立注册为 `dspark_correction`，checkpoint 不会被识别成原生
  DSpark checkpoint。
- 保留 DFlash backbone、DSpark 的主损失、接受长度指标和 Confidence
  Head。
- Correction 参数显式传给其他模型类型时直接报错，避免用户以为
  原生 DSpark 已启用 Correction。
- 本次实现为训练侧模型。vLLM/SGLang 还没有
  `dspark_correction` 的推理模型适配，训练后的 checkpoint 不能直接作为原生
  DSpark checkpoint 部署。

## 2. 模型结构

### 2.1 原生 DSpark

原生 DSpark 在 DFlash 并行产生的 logits 上增加一阶 Markov 偏置：

```text
target hidden states
        |
        v
  DFlash backbone -----> h_dflash
        |                    |
        v                    v
     LM Head          Markov(prev_token)
        |                    |
        +------ add <--------+
                    |
                    v
             corrected logits
```

Markov Head 主要依赖前一个 token，对更长的块内历史建模能力有限。

### 2.2 DSpark Correction

新结构保留 DFlash backbone，但将顺序修正模块替换为小型因果
Transformer：

```text
previous token embedding ----+
DFlash hidden ---------------+--> feature sum
block position embedding ----+         |
previous logits (logits mode)+         v
                                  causal attention
                                         |
                                      gated MLP
                                         |
                                      RMSNorm
                                         |
                                  low-rank projection
                                         |
                     +-------------------+-------------------+
                     |                                       |
                hidden mode                            logits mode
                     |                                       |
          h_dflash + delta_hidden             base_logits + delta_logits
                     |
                  LM Head
```

Correction 的每个位置可以关注同一 draft block 中的所有早期位置，但
因果 mask 保证它不会看到未来 token，也不会跨 block 传播。

### 2.3 输入特征

对于第 `i` 个 block slot，Correction 的初始状态为：

```text
x_i = W_h h_i + W_t Emb(previous_token_i) + Emb(position_i)
```

`logits` 模式还会增加前一位置 target distribution 的低秩编码：

```text
p_(i-1) = softmax(target_logits_(i-1))
r_(i-1) = W_down p_(i-1)
x_i = x_i + W_logit r_(i-1)
```

训练使用 teacher forcing；前一位置分布来自 verifier target logits，不会把
未经验证的未来信息放入当前位置。第一个有效 slot 使用全零历史
特征或它前面的保留 anchor 分布。

### 2.4 输出模式

#### Hidden mode

```text
delta_hidden = W_up(SiLU(W_down(causal_state))) * sigmoid(gate)
corrected_hidden = h_dflash + delta_hidden
logits = LMHead(corrected_hidden)
```

优点是 Correction 直接改善 pre-LM representation，并可以加入 verifier hidden
alignment loss。缺点是需要对修正后 hidden 做一次完整 LM Head 投影。

#### Logits mode

```text
delta_logits = W_up(SiLU(W_down(causal_state))) * sigmoid(gate)
logits = DFlash_base_logits + delta_logits
```

该模式更接近 DSpark Markov bias 的输出空间，同时通过前一位置的概率分布获得
更强的顺序条件。

### 2.5 稳定初始化

`correction_up.weight` 使用全零初始化，所以新建模型初始时：

```text
delta_hidden = 0  或  delta_logits = 0
```

这使得 Correction 在第一个优化 step 前是 DFlash 输出的 no-op，减少新增
分支对训练初期的扰动。

## 3. 代码实现方案

### 3.1 新增模块

| 文件 | 职责 |
| --- | --- |
| `src/speculators/models/dspark_correction/config.py` | 独立 config schema、参数验证和 model type 注册 |
| `src/speculators/models/dspark_correction/model_definitions.py` | Tiny Causal Attention、Causal Layer 和 Correction Head |
| `src/speculators/models/dspark_correction/core.py` | DFlash backbone、Correction、Confidence、loss 的训练前向 |
| `src/speculators/models/dspark_correction/__init__.py` | 模型和配置导出 |

### 3.2 现有入口变更

- `src/speculators/models/__init__.py`
  - 导入新模块，完成 `dspark_correction` registry 注册。
  - 原 `dspark` registry entry 保持不变。
- `scripts/train.py`
  - 增加 Correction 训练参数。
  - 增加模型类型隔离验证。
  - Correction 参数不允许与原生 `dspark` 一起使用；Markov 参数不允许
    与 `dspark_correction` 一起使用。

### 3.3 Confidence Head

`--confidence-head-with-correction` 打开时，Confidence Head 输入为：

```text
concat(DFlash hidden, Correction causal state)
```

关闭时仅使用 DFlash hidden。这个开关对应原 DSpark 的
`confidence_head_with_markov`，但两者是完全独立的配置字段。

### 3.4 辅助 hidden alignment loss

Hidden mode 可选启用：

```text
SmoothL1(corrected_hidden, verifier_pre_lm_hidden)
```

损失仅在有效 token mask 上求平均，并用
`--correction-hidden-aux-weight` 加权。Logits mode 不会产生 corrected hidden，
因此不允许启用该损失。

## 4. 训练参数

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--speculator-type dspark_correction` | 必填 | 选择新模型，不能写成 `dspark` |
| `--correction-output-mode {hidden,logits}` | `hidden` | 选择 hidden residual 或 logit residual |
| `--correction-hidden-size` | `512` | Correction Transformer hidden width |
| `--correction-rank` | `256` | 输出低秩瓶颈宽度 |
| `--correction-num-layers` | `1` | Correction Transformer 层数 |
| `--correction-num-heads` | `8` | 每层 attention head 数 |
| `--correction-gate-bias` | `0.0` | 残差 sigmoid gate 初始 bias |
| `--confidence-head-with-correction` / `--no-confidence-head-with-correction` | 开启 | Confidence 是否使用 Correction state |
| `--correction-hidden-aux-loss` / `--no-correction-hidden-aux-loss` | 关闭 | Hidden mode 的 verifier hidden 对齐损失 |
| `--correction-hidden-aux-weight` | `0.1` | hidden alignment loss 权重 |

几何约束：

```text
correction_hidden_size % correction_num_heads == 0
```

`dspark_correction` 不使用以下 Markov 参数，启动脚本中应删除：

```text
--markov-rank
--markov-head-type
--confidence-head-with-markov
```

## 5. 使用教程

### 5.1 Hidden mode 推荐起步配置

对现有 DSpark 训练脚本做如下替换：

```diff
--- --speculator-type dspark
+++ --speculator-type dspark_correction
--- --markov-rank 256
--- --markov-head-type vanilla
--- --confidence-head-with-markov
+++ --correction-output-mode hidden
+++ --correction-hidden-size 512
+++ --correction-rank 256
+++ --correction-num-layers 1
+++ --correction-num-heads 8
+++ --correction-gate-bias 0.0
+++ --confidence-head-with-correction
```

完整训练命令示例：

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --verifier-name-or-path /path/to/Qwen3-8B \
  --data-path /path/to/processed_data \
  --vllm-endpoint http://localhost:8000/v1 \
  --save-path ./output/dspark_correction_hidden \
  --speculator-type dspark_correction \
  --draft-vocab-size 32000 \
  --block-size 8 \
  --max-anchors 512 \
  --num-layers 5 \
  --draft-attn-impl sdpa \
  --target-layer-ids 1 9 17 25 33 \
  --correction-output-mode hidden \
  --correction-hidden-size 512 \
  --correction-rank 256 \
  --correction-num-layers 1 \
  --correction-num-heads 8 \
  --correction-gate-bias 0.0 \
  --enable-confidence-head \
  --confidence-head-with-correction \
  --correction-hidden-aux-loss \
  --correction-hidden-aux-weight 0.1 \
  --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --confidence-head-alpha 1.0 \
  --epochs 5 \
  --lr 6e-4 \
  --on-missing generate \
  --on-generate delete
```

### 5.2 Logits mode

将 Correction 参数替换为：

```bash
--correction-output-mode logits \
--correction-hidden-size 512 \
--correction-rank 256 \
--correction-num-layers 1 \
--correction-num-heads 8 \
--confidence-head-with-correction \
--no-correction-hidden-aux-loss
```

Logits mode 禁止搭配 `--correction-hidden-aux-loss`。

### 5.3 正式训练前 dry run

建议在完整命令末尾先加：

```bash
--dry-run
```

dry run 会创建模型并保存初始 checkpoint，但不创建 dataloader、不启动
正式训练。检查生成的 `config.json` 应包含：

```json
{
  "speculators_model_type": "dspark_correction",
  "correction_output_mode": "hidden",
  "correction_hidden_size": 512,
  "correction_rank": 256
}
```

### 5.4 从 checkpoint 继续训练

```bash
python scripts/train.py \
  --verifier-name-or-path /path/to/Qwen3-8B \
  --speculator-type dspark_correction \
  --from-pretrained ./output/dspark_correction_hidden/checkpoint \
  --data-path /path/to/processed_data \
  --save-path ./output/dspark_correction_resume
```

`--from-pretrained` 时 Correction 结构参数从 checkpoint config 恢复，不应再传
`--correction-*` 去修改结构。

## 6. 测试与验收

新增单元测试：

```text
tests/unit/models/test_dspark_correction.py
```

覆盖：

1. `dspark` 和 `dspark_correction` 同时注册，原生 DSpark 没有被替换。
2. Hidden/logits 两种输出 shape 和零初始 no-op。
3. Correction attention 的因果性。
4. `sample_from_anchor=True/False` 下的 hidden/logits 完整 forward/backward。
5. Confidence Head 和 Correction 输出投影获得梯度。
6. Hidden auxiliary loss 产生指标。
7. 非法 head 几何、非法模式组合和 CLI 隔离校验。
8. checkpoint 通过通用 registry 回读为正确模型类。

运行命令：

```bash
python -m pytest tests/unit/models/test_dspark_correction.py -q
python -m pytest tests/unit/train/test_cli_args.py -q
python -m pytest \
  tests/unit/models/test_dspark_model_definitions.py \
  tests/unit/models/test_dspark_metrics.py \
  tests/unit/models/test_dspark_dflare_variants.py \
  -q
```

静态检查：

```bash
python -m ruff check \
  src/speculators/models/dspark_correction \
  src/speculators/models/__init__.py \
  scripts/train.py \
  tests/unit/models/test_dspark_correction.py \
  tests/unit/train/test_cli_args.py
```

## 7. 后续建议

1. 在目标推理引擎中注册 `dspark_correction` 模型，实现单 slot KV cache
   rollout，再进行端到端接受长度评测。
2. 分别对比 hidden/logits mode，保持数据、block size、optimizer 和训练
   budget 不变。
3. 在 Correction 基线收敛后，再单独评估 hidden feedback、Selector 或
   Correction-Markov collaboration，避免首轮实验同时引入过多变量。
