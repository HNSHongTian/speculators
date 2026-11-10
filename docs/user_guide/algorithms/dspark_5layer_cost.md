# 五层 Qwen3 dense DSpark 参数量与训练计算量

依据用户提供配置及当前仓库实现，2026-10-29。参数量通过真实 `DSparkDraftModel` 在 PyTorch meta device 上实例化并统计，同时用独立公式校验；不加载教师权重，不访问网络。

## 1. 配置与统计口径

H=3584，I=18432，V=131072，L=5；辅助教师层数 C=5；Markov rank R=256，类型 vanilla；confidence head 开启并拼接 Markov embedding；block_size B=7；sample_from_anchor=True。

注意 Q 投影维度 Q=32×128=4096，K/V 投影维度 K=8×128=1024。不能用 hidden_size/num_attention_heads=112 替代显式 head_dim=128。

参数数量不随 block_size、序列长度、anchor 数或 BF16/FP32 改变。以下不包含远端教师模型参数。

| 模块 | 参数量 | 是否训练 |
|---|---:|---|
| 5 层 decoder | 1,174,442,240 | 是 |
| 教师 5 层 hidden 拼接投影 fc，17920→3584 | 64,225,280 | 是 |
| vanilla Markov W1 与 W2 | 67,108,864 | 是 |
| 外部 norm 与 hidden_norm | 7,168 | 是 |
| confidence head，3840→1，含 bias | 3,841 | 是 |
| embed_tokens | 469,762,048 | 冻结 |
| 草稿 lm_head | 469,762,048 | 冻结 |
| 本地 verifier_lm_head | 469,762,048 | 冻结 |
| verifier_norm | 3,584 | 冻结 |

可训练参数 **1,305,787,393 ≈ 1.3058B（13.06 亿）**；冻结参数 **1,409,289,728**；训练模型对象总参数 **2,715,077,121 ≈ 2.7151B（27.15 亿）**。

去除训练专用 verifier_lm_head 和 verifier_norm、保留 embedding 与草稿 lm_head 后，为 **2,245,311,489 ≈ 2.2453B**。DFlash 基类也将前两项列入 `_keys_to_ignore_on_save`。推理引擎如何共享/加载权重另受实现影响，此数不代表实际部署显存。

单层 decoder 参数公式：

```text
Q/K/V/O = 2H(Q+K) = 36,700,160
SwiGLU = 3HI      = 198,180,864
RMSNorm = 2H+2d   = 7,424
单层总计         = 234,888,448
```

Markov W1 是 V×R 的 embedding，W2 是 R→V 的无 bias Linear，共 2VR=67,108,864。不能将其按 V×V 的稠密转移矩阵计算。

## 2. 训练矩阵运算量

采用一次乘加=2 FLOPs，TFLOP=10^12 次浮点运算。S 表示每 rank 打包并 padding 后的 total_seq_len，A 表示 max_anchors，N=A×B 为草稿 query 数。每层 query 长度为 N，KV 长度为 S+N；MLP 仅作用于 N 个草稿位置。fc 对全部 S 个教师上下文位置计算；本地 verifier_lm_head 也先对全部 S 个位置计算，再按 anchor 索引选择教师目标。

以下为 attention 的稠密等价矩阵运算量，非 profiler 实测。掩码限制有效 attention 对；如果后端跳过 masked tiles，实际 attention 算术量会变化。也未计入归一化、RoPE、激活函数、softmax、BV/TV/confidence 损失及指标、embedding 梯度散射、优化器、通信、数据传输和额外重计算。

```text
F_layers = L × [4NHQ + 4(S+N)HK + 6NHI + 4N(S+N)Q]

F_forward = 2SCH² + F_layers + 2NHV + 2SHV + 2NRV + 2N(H+R)

F_forward_backward = 4SCH² + 3F_layers + 4NHV + 2SHV + 6NRV + 6N(H+R)
```

最后一项仅在 confidence head 开启且拼接 Markov 时成立。反向估计按常规矩阵乘法梯度推导：

- decoder/Markov/confidence 的投影输入和权重均需梯度，前反向合计约 3×前向；decoder 的首层输入 norm 可训练，因此即使 embedding 冻结，Q/K/V 的输入仍需反向。
- fc 输入为冻结教师特征，不计算输入梯度，合计约 2×前向。
- 草稿 lm_head 权重冻结，但要向草稿 hidden states 传梯度，合计约 2×前向。
- verifier_lm_head 在 no_grad 中，只计一次前向。

以用户先前训练脚本的 **S=4096、A=512** 为例，保持本次提供的 **B=7**，得到 N=3584：

| 模块 | 前向 TFLOP/rank/step | 前向+反向 TFLOP/rank/step |
|---|---:|---:|
| 教师特征融合 fc | 0.5261 | 1.0523 |
| 5 层 Q/K/V/O 与 MLP | 8.7188 | 26.1564 |
| 5 层 attention 的 QKᵀ 与 AV | 2.2549 | 6.7646 |
| 草稿 lm_head | 3.3673 | 6.7345 |
| 本地 verifier_lm_head | 3.8483 | 3.8483 |
| Markov W2 | 0.2405 | 0.7216 |
| confidence Linear | 0.000028 | 0.000083 |
| **合计** | **18.9559** | **45.2776** |

如果 S=3072，A=512、B=7 不变，前向为 **17.4864 TFLOP**，前反向 **42.9251 TFLOP**。这些数不适用于 block_size=15 的另一份脚本，必须按实际配置复算。

无梯度累积且每 rank 都采用上述 S=4096 配置时，16 个数据并行 rank 的全局一步矩阵运算量约 **724.44 TFLOP**，32 个 rank 约 **1.449 PFLOP**；每个 rank 的参数量不会除以 rank 数（DDP 复制模型）。

BV/TV 不新增模型参数，但增加 FP32 概率计算、归约、loss backward 和分块 checkpoint 重计算，所以上述矩阵运算量不等于包含这些操作的完整训练 FLOPs。也不能直接换算训练耗时。当前实现仍按 max_anchors 构造张量，无效 block 的 mask 不自动省掉之前的 backbone 矩阵乘法。

远端 online 29B 教师的上下文前向及生成成本未包含在内。仅凭“29B”和此草稿配置无法精确计算整套 online 系统的计算量，需要教师结构、实际请求长度、缓存命中与服务端执行方式。

## 3. 验证脚本与运行命令

仓库根目录执行，默认配置复现用户提供文本的参数形状与冻结状态：

```bash
export PYTHONPATH="$PWD/src:$PWD/hs_connectors/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/verify/dspark_model_cost.py --seq-len 4096 --max-anchors 512
python scripts/verify/dspark_model_cost.py --seq-len 3072 --max-anchors 512
# 按真实 checkpoint 配置复算（当前公式支持无 attention bias 的 Qwen3、vanilla Markov）
python scripts/verify/dspark_model_cost.py --config /path/to/config.json \
  --seq-len 4096 --max-anchors 512
```

输出包括每个实际模块的总参数/可训练参数、独立参数公式校验、各部分 FLOPs。meta 实例化不分配十几亿参数的真实权重存储。已核对默认配置的两种序列长度；未做真实 NPU FLOPs/吞吐 profiler 测量。

代码依据：`src/speculators/model.py` 的 `_init_vocab`；`src/speculators/models/dflash/core.py` 的构造函数和 `_backbone_forward`；`src/speculators/models/dflash/model_definitions.py` 的 attention 与 decoder；`src/speculators/models/dspark/model_definitions.py` 的 Markov/confidence head；`src/speculators/models/dspark/core.py` 的模型组装。
