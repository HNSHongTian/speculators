# 原始 DSpark 的 SpecOPD 训练

通过 `scripts/train.py --train-e2e-loss-specopd` 开启，默认关闭。不开启时保持原来的训练路径。
要求 full-vocabulary、vanilla Markov 的原始 DSpark；不使用 TreeSpark loss。

## 训练流程与损失

从参考回答的有效区域随机选择 anchors，构造独立 anchor/mask blocks。
共享 DSpark backbone 特征，各分支通过 Markov head 按已采样前驱依次采样。
teacher 在 `参考前缀 + 实际草稿前驱` 上产生新的监督，不能用参考链的旧 logits 替代。
每次更新固定采样 token，重新计算可微草稿概率。

`alpha = sum_v min(p, q)`，`survival_j = product_{i<=j} alpha_i`。
主损失为所有有效位置上 `1-survival_j` 的平均值。
如果模型存在 confidence head，加入以 `alpha.detach()` 为软标签的 BCE，
权重使用 `--confidence-head-alpha`（默认 1；不是已核实的论文 DSpark 权重）。
不叠加旧 token CE、逐位置 TV、BV 或 tree loss；启动日志明确说明旧 `--loss-fn` 不生效。
EOS 本身保留监督，其后的尾部位置屏蔽。空 batch 不更新 optimizer 或 scheduler。
训练时关闭 dropout，保留梯度，保证采样分布与重算分布相同。

参考：SpecOPD https://arxiv.org/html/2611.02846v1 ，式 9、附录 D.1。

## 加载已有 DSpark 权重继续训练

保留已有的数据、teacher 和模型初始化参数，增加：

```bash
--from-pretrained /model/qwen3_dspark \
--train-e2e-loss-specopd \
--save-path /output/qwen3_dspark_specopd
```

使用新的输出目录会加载模型权重，并创建新的 optimizer/scheduler，开始 SpecOPD 阶段。
`/model/qwen3_dspark` 必须含真实模型权重；本仓库对 config-only 目录会初始化新模型。
也可直接沿用仓库从头初始化 DSpark 的参数，不传 `--from-pretrained`。

完整示例（替换 target、数据和服务地址）：

```bash
MODEL=/model/Qwen3-8B \
DATA_PATH=/data/train_arrow \
OUTPUT_DIR=/output/qwen3_dspark_specopd \
VLLM_ENDPOINT=http://127.0.0.1:8000/v1 \
DRAFT_CHECKPOINT=/model/qwen3_dspark \
NPROC_PER_NODE=1 LR=1e-5 EPOCHS=3 \
bash examples/train/dspark_specopd_online.sh
```

这里 LR=1e-5、3 epochs、8 anchors 是示例起点，不是论文完整 DSpark 配方。
实际步数取决于数据、packing 和 rank 数；建议依据独立验证集的实际 acceptance length 选模型。

## 恢复中断的 SpecOPD 运行

对同一 `--save-path` 用相同训练参数重新启动，不添加 `--no-resume-from-checkpoint`。
自动加载最新数字 epoch 目录中的模型、optimizer、scheduler、global step 和 batch 位置。
恢复时保存目录的状态优先于 `--from-pretrained`。
中途 checkpoint 的 sampler 会跳过已经完成的 batch；每个 epoch/step/rank 的随机种子固定。
SpecOPD checkpoint 使用 FP32 保存模型和 optimizer，避免恢复时再次量化状态。

严格检查训练 recipe、数据指纹、world size 和完整 epoch 长度。
修改学习率、epochs、数据或 objective 开始新阶段时，改用
`--from-pretrained /output/old_run/数字epoch` 和新的 `--save-path`。
普通训练输出目录不能直接作为 SpecOPD 的完整状态恢复目录。
端点地址、服务别名、请求并发及重试参数可调整，但必须仍指向同一个 teacher 权重版本。

仓库的异常退出处理还可能写 `interrupted/`：该非数字目录不提供本功能的精确数据位置恢复。
优先从最近的数字 checkpoint 恢复；也可用 `--from-pretrained .../interrupted`
加载权重，在新目录启动新的阶段。

## Teacher 服务要求

沿用仓库的 hidden-state vLLM 服务与 FileTransfer 协议，需要共享文件系统。
普通只返回文本/logprobs 的 OpenAI-compatible endpoint 不够。
teacher 提取的辅助层必须与 checkpoint 的 `target_layer_ids` 一致，
`hidden_states` 最后一层必须是 teacher 最终层（与原 DSpark 数据协议一致）。
target 的 tokenizer、词表和权重必须与草稿引用的 verifier 对齐。
`--specopd-target-hidden-norm pre` 表示返回最终 RMSNorm 之前的状态（默认，与原 DSpark 一致）；
服务返回 norm 后状态时用 `post`，否则会重复归一化。
服务使用自定义模型别名时设置 `--specopd-server-model`。
teacher context limit 需容纳参考前缀 + 分支长度；现有协议额外生成 1 token，该 token 不参与训练。

当前通过并发的独立完整前缀请求取得正确的 branch-conditioned 监督，
尚未实现论文的共享主干 teacher KV/tree executor，不能声称复现其训练加速倍数。
支持单设备与 DDP；暂不支持 FSDP。当前实现用于文本 token 序列。

## 参数与日志

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--specopd-num-anchors` | 8 | 每个 rank 的 packed batch 最大分支数 |
| `--specopd-temperature` | 1 | 采样和 p/q 分布共用温度，必须大于 0 |
| `--specopd-target-concurrency` | 4 | 每个 rank 并发 teacher 请求数 |
| `--specopd-loss-chunk-size` | 4 | 分块计算词表分布的位置数，使用 activation checkpoint |
| `--confidence-head-alpha` | 1 | confidence BCE 权重；0 可关闭其贡献 |
| `--noise-std` | 0（SpecOPD） | SpecOPD 不允许噪声增强，普通模式默认仍为 0.05 |

日志接入现有 logger，包括训练及验证：

- `loss`：主损失加加权 confidence 损失。
- `specopd_e2e_loss`、`specopd_confidence_loss`：主损失与未乘权重的辅助损失。
- `specopd_overlap`：平均 token overlap。
- `specopd_survival_1..W`：各深度平均前缀乘积。
- `specopd_surrogate_length`：每条分支的前缀乘积之和，不含 bonus，不是实测 AL。
- `specopd_supervised_tokens`、`specopd_branches`、`specopd_target_requests`：每 rank 平均计数。
- `specopd_global_positions`：该步全局有效监督位置数。
- `specopd_target_seconds`：每 rank 平均 teacher 请求耗时。
- 保留原有 LR、global step 和训练耗时日志；验证指标带 `_epoch` 后缀。

测试覆盖数值、梯度、真实小 DSpark 前向、Markov 历史、teacher 请求行对齐、
EOS、上下文隔离和断点恢复。仍需在实际 GPU/NPU 与 teacher 服务上完成训练效果评测。

本次验证：SpecOPD、BV、TreeSpark forward、损失权重调度、原始训练脚本及 scheduler
共 67 项 CPU 单元/回归测试通过；DDP 的不均匀样本数归一化经过模拟验证，
尚未完成真实多进程、多 GPU 的训练验证。
