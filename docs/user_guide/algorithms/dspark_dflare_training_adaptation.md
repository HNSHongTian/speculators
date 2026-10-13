# DSpark 训练侧 DFlare 能力适配说明

## 1. 适配范围

本次适配将 DFlare 的两项能力以新模块引入 DSpark，不修改
`src/speculators/models/dspark/` 中的原实现：

| 模型类型 | 用途 | 目标层知识 | Draft 深度 | DSpark 头 |
| --- | --- | --- | --- | --- |
| `dspark` | 原始基线 | 共享 FC 融合 | 由原配置决定 | Markov + confidence |
| `dspark_layerwise` | 单独验证 Scaling target knowledge per layer | 逐 draft layer 自适应融合 + 独立 target K/V | 通常先保持基线深度 | 原样保留 |
| `dspark_deep` | 单独验证 Scaling draft depth | 保持原 DSpark 共享 FC | 至少 6 层，DFlare 对齐建议 7 层 | 原样保留 |

如果最后需要同时打开两项能力，使用 `dspark_layerwise` 并设置
`--num-layers 7` 即可。`dspark_deep` 故意保留原 DSpark 的共享 FC，用于做
只改深度的对照实验。

本次只实现训练侧。没有修改 vLLM、SGLang 或 DSpark 的推理路径；
训练出的 `dspark_layerwise` checkpoint 需要后续增加对应推理模型后才能部署。

## 2. 设计与论文对齐

### 2.1 Scaling target knowledge per layer

DFlare 用可学习矩阵 `W_fuse` 替换 DFlash/DSpark 的单个共享 FC。
对于 `D` 个 draft layer 和 `T` 个 target layer：

```text
W_fuse: [D, T]
alpha_i = softmax(W_fuse[i])
f_i = RMSNorm(sum_j alpha_i[j] * target_hidden[j])
```

实现使用 `einsum("bsth,dt->bsdh", ...)`，产生 `[B, S, D, H]` 的
逐层 target context。每个 draft layer 只取自己的 `[:, :, i, :]`。

同时按 DFlare/AngleSlim 实现将 K/V 投影分开：

```python
k_ctx = self.k_proj_target(target_hidden)
v_ctx = self.v_proj_target(target_hidden)
k_noise = self.k_proj(hidden_states)
v_noise = self.v_proj(hidden_states)
```

这样 target context 和 draft/noise token 不再共享 K/V 参数。初始融合权重也与
AngleSlim 对齐：所有 logit 先置 0，然后为每个 draft layer 按深度映射的
target layer 设置 2.0。

### 2.2 Scaling draft depth

DSpark 本身已经根据
`transformer_layer_config.num_hidden_layers` 构建 `ModuleList`。因此深度适配不需要
复制 backbone；`DSparkDeepDraftModel` 继承原 DSpark，仅在新模型类型下校验深度。
推荐使用 DFlare 主配置的 7 层：

```bash
--speculator-type dspark_deep --num-layers 7
```

## 3. 代码适配清单

### 3.1 逐层 target knowledge 模块

- `src/speculators/models/dspark_layerwise/config.py`
  - 新增 `DSparkLayerwiseSpeculatorConfig`。
  - 注册 `speculators_model_type="dspark_layerwise"`。
  - 保留全部 DSpark 配置字段，新增 `fusion_init_scale`。
- `src/speculators/models/dspark_layerwise/model_definitions.py`
  - 新增 `Qwen3DSparkLayerwiseAttention`。
  - target context 使用 `k_proj_target/v_proj_target`，draft token 继续使用
    `k_proj/v_proj`。
  - 新增 `Qwen3DSparkLayerwiseDecoderLayer`。
- `src/speculators/models/dspark_layerwise/core.py`
  - 新增 `DSparkLayerwiseDraftModel`。
  - 从 `DSparkDraftModel` 继承 Markov head、confidence head、loss 和 metrics。
  - 移除新实例中继承的共享 `fc`，改为 `[D,T]` 融合。
  - 只覆盖 `_backbone_forward()`，不改 DSpark 的 `forward()`。
- `src/speculators/models/dspark_layerwise/__init__.py`
  - 导出 config 和 model。

### 3.2 Draft depth 模块

- `src/speculators/models/dspark_deep/config.py`
  - 新增 `DSparkDeepSpeculatorConfig`。
  - 注册 `speculators_model_type="dspark_deep"`。
  - 新增 `minimum_draft_layers`，默认为 6。
- `src/speculators/models/dspark_deep/core.py`
  - 新增 `DSparkDeepDraftModel`。
  - 继承原 DSpark backbone 和头，只校验深度。
- `src/speculators/models/dspark_deep/__init__.py`
  - 导出 config 和 model。

### 3.3 入口注册

- `src/speculators/models/__init__.py`
  - 导入两个新模块，使 config/model registry 在启动时完成注册。
- `scripts/train.py`
  - 训练参数帮助中增加 `dspark_layerwise` 和 `dspark_deep`。
  - 无需新增特判：现有 registry 和 DFlash-family 训练路径会自动处理新类型。

## 4. 训练步骤

### 4.1 准备 target hidden states

`dspark_layerwise` 的主要变化是 target layer 数量由 5 增加到 9。以 36 层
Qwen3-8B 为例，DFlare/AngleSlim 使用：

```text
1 5 9 13 17 21 25 29 33
```

启动用于在线生成 hidden states 的 vLLM 服务：

```bash
MODEL=/path/to/Qwen3-8B
TARGET_LAYER_IDS="1 5 9 13 17 21 25 29 33"

python scripts/launch_vllm.py "$MODEL" \
  --target-layer-ids $TARGET_LAYER_IDS \
  -- --port 8000
```

服务端和训练端的 `--target-layer-ids` 必须完全相同。如果使用离线缓存，
原 5 层缓存宽度是 `5*H`，新 9 层模型需要 `9*H`，必须重新生成；
不能复用旧 hidden-state 文件。

### 4.2 单独训练 Scaling target knowledge per layer

为了做单变量对照，保持 DSpark 基线的 5 层 draft 深度，只改模型类型和
target layer 集合：

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --verifier-name-or-path /path/to/Qwen3-8B \
  --data-path /path/to/processed_data \
  --vllm-endpoint http://localhost:8000/v1 \
  --save-path ./output/dspark_layerwise_qwen3_8b \
  --speculator-type dspark_layerwise \
  --draft-vocab-size 32000 \
  --block-size 8 \
  --max-anchors 512 \
  --num-layers 5 \
  --target-layer-ids 1 5 9 13 17 21 25 29 33 \
  --markov-rank 256 \
  --markov-head-type vanilla \
  --enable-confidence-head \
  --confidence-head-with-markov \
  --epochs 5 \
  --lr 6e-4 \
  --on-missing generate \
  --on-generate delete
```

如果需要严格对齐 AngleSlim 配置，还应把 block size、数据集、学习率调度、
anchor 数量和 attention layer type 一起对齐。本命令保留现有 DSpark 参数，
便于只比较目标知识注入的影响。

### 4.3 单独训练 Scaling draft depth

深度对照应保持原 DSpark 的 target layer 集合，只将 draft 由 5 层增加到 7 层：

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --verifier-name-or-path /path/to/Qwen3-8B \
  --data-path /path/to/processed_data \
  --vllm-endpoint http://localhost:8000/v1 \
  --save-path ./output/dspark_deep_qwen3_8b \
  --speculator-type dspark_deep \
  --draft-vocab-size 32000 \
  --block-size 8 \
  --max-anchors 512 \
  --num-layers 7 \
  --target-layer-ids 1 9 17 25 33 \
  --markov-rank 256 \
  --markov-head-type vanilla \
  --enable-confidence-head \
  --confidence-head-with-markov \
  --epochs 5 \
  --lr 6e-4 \
  --on-missing generate \
  --on-generate delete
```

此命令需要用 `1 9 17 25 33` 重新启动 vLLM，不能与上一个 9 层 hidden-state
服务混用。

### 4.4 同时开启两项能力

```bash
--speculator-type dspark_layerwise \
--num-layers 7 \
--target-layer-ids 1 5 9 13 17 21 25 29 33
```

这是两项结构改进的合并形态，但不是最干净的单变量实验。

### 4.5 正式训练前的 dry run

可以先在完整参数后增加 `--dry-run`。该命令会构建模型、加载 verifier
权重、保存初始 checkpoint，然后在创建 dataloader 和正式训练前退出。

## 5. 测试代码

新增测试脚本：

```text
tests/unit/models/test_dspark_dflare_variants.py
```

覆盖范围：

1. `dspark`、`dspark_layerwise`、`dspark_deep` 三个 registry key 同时存在，
   新模块没有替换原 DSpark。
2. `[D,T]` 融合输出 shape、softmax 归一、逐层初始偏好和梯度。
3. target hidden-state 宽度与 target layer 数不匹配时给出明确错误。
4. target K/V 和 draft K/V 是独立参数。
5. `dspark_layerwise` 完整训练 forward/backward，并检查融合权重、target K、
   Markov head 和 confidence head 均获得梯度。
6. `dspark_deep` 的 7 层构建、原共享 FC 保留、DSpark 头保留和深度下限校验。
7. `dspark_layerwise` checkpoint 保存后，通过通用 `SpeculatorModel` registry
   自动恢复为正确模型类。

## 6. 运行测试的命令

在仓库根目录运行新增单测：

```bash
python -m pytest tests/unit/models/test_dspark_dflare_variants.py -q
```

运行不需要网络的 DSpark 回归测试：

```bash
python -m pytest \
  tests/unit/models/test_dspark_model_definitions.py \
  tests/unit/models/test_dspark_metrics.py \
  -q
```

运行完整的相关测试（`test_sample_from_anchor.py` 会在本地无缓存时访问
Hugging Face）：

```bash
python -m pytest \
  tests/unit/models/test_dspark_dflare_variants.py \
  tests/unit/models/test_dspark_model_definitions.py \
  tests/unit/models/test_dspark_metrics.py \
  tests/unit/models/test_sample_from_anchor.py \
  -q
```

运行本次变更的静态检查：

```bash
python -m ruff check \
  src/speculators/models/dspark_layerwise \
  src/speculators/models/dspark_deep \
  tests/unit/models/test_dspark_dflare_variants.py \
  scripts/train.py \
  src/speculators/models/__init__.py
```

## 7. 本次验证结果

- 新增 DFlare/DSpark 单测：`8 passed`。
- Ruff 静态检查：通过。
- 原 DSpark 相关回归：`19 passed`；其余 3 项仅因当前隔离环境无法访问
  `huggingface.co` 且本地没有 `Qwen/Qwen2.5-0.5B-Instruct` 配置缓存而失败，
  未进入 DSpark 逻辑断言。

## 8. 推理侧后续工作

先完成训练时不需要修改推理侧。但要将 `dspark_layerwise` checkpoint 用于
真实推理，推理引擎后续至少需要：

1. 识别 `dspark_layerwise` 配置和 checkpoint 权重名。
2. 加载 `layer_fusion_weights`、`k_proj_target` 和 `v_proj_target`。
3. 在每个 draft layer 注入对应的融合 target context。
4. 确保 target hidden-state 捕获层集与 checkpoint 中的
   `aux_hidden_state_layer_ids` 完全相同。
5. 对照训练侧做 logits、融合输出和 acceptance length 一致性测试。

`dspark_deep` 未改变单层计算图，但推理引擎仍必须能从 checkpoint 构建和加载
7 个 draft layer；训练阶段不需要先完成这部分。
