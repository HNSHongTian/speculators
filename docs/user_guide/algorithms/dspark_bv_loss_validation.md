# DSpark BV loss：实现、逐步验证和运行命令

更新：2026-10-28。实现仓库：`speculators_treespark`。设计依据见 [适配设计稿](dspark_bv_loss_adaptation.md)。本文件描述已落地的行为，取代设计稿中“拟新增”的状态。

## 1. 已实现的功能

- `--enable-bv-loss` 默认关闭；关闭时保留原 CE/TV 等训练流程。开启后支持显式 `--loss-fn '{"bv":0.1,"tv":0.9}'` 配置；旧的不含 bv 的配置仍整体被纯 BV 覆盖，详见第 7 节。
- BV 模式支持固定三项权重，例如 `--loss-fn '{"bv":0.2,"ce":0.08,"tv":0.72}'`；不传调度时，三个权重在所有 epoch 保持不变。
- 新增 `--loss-weight-schedule`，按零基 epoch 里程碑覆盖 BV/CE/TV 权重，并可选调度 confidence。每个里程碑持续到下一个里程碑；固定和动态模式都写入 checkpoint recipe，配置变化会拒绝直接 resume。
- BV 使用 Markov 修正后的全词表分布，前缀状态沿实际 `input_ids` 递推；默认前三个 epoch 聚合退火，之后 block-log。验证固定 block-log。
- 新增 `--bv-anneal-epochs`（默认 3，0 表示不退火）和 `--bv-chunk-size`（默认 16）。chunk size 只影响重算粒度，不改变数学目标。
- 只使用完整有效 block，按真实 label 索引检查 assistant mask、document、padding、EOS。支持 `sample_from_anchor` 两种布局；拒绝 BV 与 D-PACE、缩词表的组合。
- 训练按全局有效 block 数归一化；空 rank 仍参与 backward，全局全空时跳过 optimizer 和 LR scheduler，避免 AdamW weight decay 产生伪更新。
- 保存 BV 配方、数据 fingerprint、完整 epoch 长度和实际 optimizer 更新计数；校验断点恢复配置，恢复 β 和学习率。`save_best` 也保存 BV training state。
- BV forward 的 anchor/dropout RNG 由 epoch、完整 local step 和 rank 决定；验证使用固定 RNG 并恢复调用前状态。随机数据增强/外部数据变化不在这项确定性保证内。
- `examples/train/train_29b_dspark_bv_online.sh` 按用户提供文本做最小修改，保留原路径、日志/PID 和 nohup/torchrun 结构；不提供 dry-run 或自动预检。验证通过独立脚本执行。

`--enable-bv-loss` 不隐式关闭 confidence 或改变 hidden-state noise。29B 新脚本显式使用 `--no-enable-confidence-head --noise-std 0`，因此实际目标只包含纯 BV loss，不包含 CE、TV 或 confidence 辅助项。

数值约定：FP32 log 递推；FP64 仅用于 CPU 数值测试。退火期不加 score floor。block-log 阶段对词表积分 log summand 使用 -80 下限，对长度使用 1e-6 下限。这个 clamp 落点是 v1 明确采用的工程解释，尚未核对作者源码；理论与梯度测试在不触发 floor 的区域进行。极接近 β=0 时，日志目标使用连续极限避免相消，反向始终使用对应 β 的 detached power weights。

## 2. 验证环境

所有命令从本仓库根目录执行。`PYTHON_BIN` 必须指向已经安装本仓库及训练依赖的 Python 环境，建议使用部署训练环境本身。

```bash
cd "/Users/gingajamestefan/Downloads/ss/投机推理/speculators_treespark"
export PYTHON_BIN=/tmp/speculators-test-py312/bin/python
```

上述是本次实际测试解释器的本地路径；迁移到服务器后改为服务器环境的 Python，例如 `export PYTHON_BIN=/path/to/train-env/bin/python`，并将 `cd` 改为服务器仓库路径。验证脚本自动把本仓库及 `hs_connectors/src` 放到 `PYTHONPATH`，并为单元测试启用离线模式，不下载模型、不访问在线教师服务。

新环境需要按项目依赖要求安装。已有训练环境中，只补本地包元数据和 pytest 即可，例如：

```bash
"$PYTHON_BIN" -m pip install --no-deps -e ./hs_connectors -e .
"$PYTHON_BIN" -m pip install pytest
```

本地验证环境为 Python 3.12.14、PyTorch 2.12.1、Transformers 5.13.1，计算设备为 CPU。Ascend 环境需使用匹配的 torch/torch_npu，不应照抄本地 CPU 依赖版本覆盖服务器环境。

## 3. 每个适配步骤的脚本、命令和通过标准

### 步骤 1：BV 数学核心与重算

实现：`src/speculators/models/dspark/bv_loss.py`。

验证脚本：`scripts/verify/bv/01_loss.sh`，调用 `tests/unit/models/test_dspark_bv_loss.py`。

```bash
bash scripts/verify/bv/01_loss.sh
```

通过标准：B=1 退化为 log overlap；q=p 时 loss≈0；小词表条件路径穷举与目标路径估计一致；前缀状态没有 detach；β 的反向代理与 power-mean 梯度一致；不同 chunk size 梯度一致；教师无梯度；极端 logits/空 block 保持有限；`torch.compile(..., backend="aot_eager", fullgraph=True)` 的前后向与 eager 一致。

本地结果：**13 passed**。aot_eager 证明图捕获和 AOT backward 兼容，不代表已经验证 CUDA Inductor 或 NPU 编译性能。

### 步骤 2：标签对齐、Markov 和真实 backbone 接入

实现：`src/speculators/models/dspark/core.py`、`metrics.py` 与 `build_bv_labels()`。

验证脚本：`scripts/verify/bv/02_alignment.sh`，调用 `test_dspark_bv_alignment.py` 和 `test_dspark_bv_forward.py`。

```bash
bash scripts/verify/bv/02_alignment.sh
```

通过标准：两种 anchor 布局的真实标签正确；跨文档、padding、mask 间断、尾部越界、内部 EOS 整块排除；最后标签 EOS 可用；vanilla/gated/rnn Markov 均获得梯度；BV 不受旧 CE/TV 系数和位置 gamma 影响；真实小型 DSpark attention/backbone 完成 forward/backward，教师保持冻结。小模型使用随机权重，不依赖 29B 文件或服务。

本地结果：**17 passed**。部署机器可增加真实设备验证：

```bash
BV_TEST_DEVICE=npu:0 bash scripts/verify/bv/02_alignment.sh
# CUDA 机器对应命令：
BV_TEST_DEVICE=cuda:0 bash scripts/verify/bv/02_alignment.sh
```

`BV_TEST_DEVICE` 作用于真实小模型 forward 测试；数学参考测试仍使用 CPU。

### 步骤 3：CLI、Trainer、退火与恢复

实现：`scripts/train.py`、`src/speculators/train/bv.py`、`trainer.py`。

验证脚本：`scripts/verify/bv/03_training.sh`，调用 `test_bv_training.py` 和原有 `test_trainer_scheduler.py`。

```bash
bash scripts/verify/bv/03_training.sh
```

通过标准：非法 CLI 配置被拒绝；β 从 0 正确推进到 1；真正运行 `Trainer.train_epoch()` 更新参数；空 batch 不触发 weight decay/LR 步进；validation 固定目标且不污染 RNG；中途恢复不重置 β；改变配方/epoch 长度会拒绝恢复；scheduler/optimizer 恢复后的下一段更新与连续训练逐项相同；best checkpoint 包含 BV 状态。

本地结果：**22 passed**。其中包含关闭 BV 的 Trainer 对照：与 `a66f57d` 的 train/val loop 快照相比，参数、RNG、日志指标结果和 training-state 基础字段一致。

### 步骤 4：不开开关的兼容性回归

验证脚本：`scripts/verify/bv/04_compatibility.sh`，调用新兼容测试及原有 DSpark metrics/Markov tests。

```bash
bash scripts/verify/bv/04_compatibility.sh
```

通过标准：纯 CE、默认 KL、CE/TV 混合，分别组合 confidence 开/关与两种 anchor 布局；对照 `tests/fixtures/dspark_metrics_pre_bv.py` 的改动前实现，loss、全部 metrics、梯度和一步 AdamW 参数结果要求 `atol=rtol=0`。步骤 2 同时检查关闭 BV 时根本不会构造 BV labels。

本地结果：**27 passed**。两个 `tests/fixtures/*pre_bv.py` 文件仅用于历史行为回归，不属于训练运行路径。

### 步骤 5：29B 脚本与只读预检

实现：`examples/train/train_29b_dspark_bv_online.sh`、`scripts/verify/bv/preflight_29b.py`。

验证脚本：`scripts/verify/bv/05_script.sh`。

```bash
bash scripts/verify/bv/05_script.sh
bash -n examples/train/train_29b_dspark_bv_online.sh
```

通过标准：bash 语法正确；独立验证器仅展开变量和启动参数，不执行 mkdir/nohup/torchrun；不传 rank 和显式传 rank=0 时的参数均包含纯 BV、10 epochs、LR=6e-4、B=15、3072 token budget、3-epoch BV 退火、关闭 confidence 和 noise=0，且不包含 loss weight schedule。检查原路径、torchrun、512 anchors 和六个教师地址保持不变；当前为单机 16 卡、127.0.0.1:29500 rendezvous，rank=1/2 被拒绝，清除旧多机网卡/IP 绑定。

本地结果（2026-10-31 单机版）：**8 passed**。原有只读预检工具的独立测试仍保留，但训练启动脚本不再调用它。

如需在服务器手动执行只读预检，可独立运行：

```bash
python3 scripts/verify/bv/preflight_29b.py \
  --model /absolute/path/to/teacher \
  --draft-config /absolute/path/to/draft_config_29b.json \
  --data-path /absolute/path/to/arrow_dataset \
  --output-dir /absolute/path/to/output \
  --max-anchors 512 --require-accelerator
```

预检读取模型/草稿 JSON、Arrow 前 32 条数据，核验维度、5 层、完整词表、token ID 范围、mask 长度/二值性、至少存在一个完整 15-token assistant block、本仓库 import 和加速器可用性。它不能从二值 mask 推断是否被预先 shift，也不能证明回答由 T=1 生成；这两项必须结合数据生成记录核实。预检不访问服务 endpoint、不加载 29B 权重，不证明显存足够。

### 步骤 6：实际两进程分布式验证

验证脚本：`scripts/verify/bv/06_distributed.sh`，调用 `distributed.py`。会启动两个本机进程并使用回环通信，不访问外部服务。

```bash
bash scripts/verify/bv/06_distributed.sh --backend gloo
```

通过标准：真实 DDP backward 和 AdamW 更新与拼接全量 block 的单进程参考一致，覆盖不均匀有效 block、单 rank 全空、全部 rank 全空。最后输出 `PASS: gloo DDP gradients and AdamW updates...`。

本地结果：**PASS**。此测试未 mock collective。最初本机 `localhost` rendezvous 解析失败，脚本已使用 `127.0.0.1:0` 自动分配端口，重试通过。

Ascend 两张空闲训练卡的命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 bash scripts/verify/bv/06_distributed.sh --backend hccl
# 仅在部署采用 FSDP2 时执行：
ASCEND_RT_VISIBLE_DEVICES=0,1 bash scripts/verify/bv/06_distributed.sh --backend hccl --fsdp
```

CUDA 两张卡的对应命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/verify/bv/06_distributed.sh --backend nccl
CUDA_VISIBLE_DEVICES=0,1 bash scripts/verify/bv/06_distributed.sh --backend nccl --fsdp
```

HCCL/NCCL/FSDP2 命令已提供，**本地尚未执行**，不得把 Gloo DDP 通过写成这几种后端均已通过。

### 步骤 10：固定 BV/CE/TV 与 epoch 动态权重

实现：`dspark/loss_config.py`、`bv_loss.py`、`metrics.py`、`core.py`、`trainer.py` 和 `scripts/train.py`。

验证脚本：`scripts/verify/bv/10_loss_schedule.sh`。

```bash
bash scripts/verify/bv/10_loss_schedule.sh
```

通过标准：固定 BV/CE/TV 的前向、梯度和日志等于三个独立目标的加权和；epoch 里程碑在指定 epoch 生效并保持到下一里程碑；非法 JSON、负数、重复/越界 epoch、缺失主损失、confidence 配置不完整被拒绝；真实 Trainer 在不同 epoch 向模型传入正确权重；schedule 和 token position gamma 写入断点配方；10-epoch 29B 脚本参数可只读展开且 bash 语法正确。

固定模式示例：

```bash
--enable-bv-loss \
--loss-fn '{"bv":0.2,"ce":0.08,"tv":0.72}'
```

动态模式示例：

```bash
--enable-bv-loss \
--loss-fn bv \
--loss-weight-schedule '[{"epoch":0,"bv":1,"ce":0,"tv":0},{"epoch":3,"bv":0.8,"ce":0.02,"tv":0.18},{"epoch":7,"bv":0.2,"ce":0.08,"tv":0.72}]'
```

如需动态 confidence，必须从训练开始创建 confidence head，并在每个里程碑都填写 `confidence`。例如最后一个里程碑可写 `"confidence":0.5`；使用 `--no-enable-confidence-head` 时任何正 confidence 调度都会在 CLI 阶段报错。

## 4. 一键验证和静态检查

```bash
# 一键执行步骤 1—5、论文公式审计 07、混合损失 09 和权重调度 10；不含分布式测试。
bash scripts/verify/bv/run_all.sh
# 分布式另行执行：
bash scripts/verify/bv/06_distributed.sh --backend gloo
```

每次改动后以脚本的 pytest 最终输出为准；pytest 的 torch.jit/tqdm deprecation/experimental warnings 不属于失败。

静态验证：

```bash
bash -n examples/train/train_29b_dspark_bv_online.sh
git diff --check
"$PYTHON_BIN" -m ruff check --ignore PLR0917 \
  scripts/train.py scripts/verify/bv/*.py \
  src/speculators/models/dspark/bv_loss.py \
  src/speculators/models/dspark/core.py src/speculators/models/dspark/metrics.py \
  src/speculators/train/bv.py src/speculators/train/trainer.py \
  tests/unit/models/test_dspark_bv*.py tests/unit/train/test_bv*.py \
  tests/unit/train/test_dspark_loss_weight_schedule.py
```

上述 targeted lint 通过；`PLR0917` 是仓库既有多位置参数接口的告警，本次未为消除它而重构原接口。若环境没有 ruff，可单独安装 ruff；它不属于训练运行依赖。

## 5. 正式 29B 训练命令与参数

当前改为单机 16 卡，只在一台训练节点启动；当前目录必须能找到原脚本引用的 `draft_config_29b.json`，该 JSON 应配置 5 层。CODE_PATH 需与实际部署仓库一致：

```bash
cd /hpfs/huawei-2607/shangxt/speculators
# 不传参数默认 node_rank=0；也兼容显式传 0
bash examples/train/train_29b_dspark_bv_online.sh
```

默认配置：TeleChat4-29B、5 层配置文件、B=15、10 epochs、LR=6e-4、AdamW、weight_decay=0.01、cosine+4% warmup、seed=42、max sequence budget=3072、BF16 hidden states、512 anchors、BV chunk=16、前三个 epoch BV 退火、关闭 confidence、noise=0。Markov rank=256/vanilla、target layers=1/10/19/28/37、mask ID=139、六个教师服务地址继承用户提供的脚本；训练拓扑为单节点 16 NPU。512 anchors 和 Markov 具体配置不是论文确认的超参数。单独声明 NUM_LAYERS 变量不会改变模型结构，实际层数由 draft_config_29b.json 决定。

当前单机脚本使用 `LOSS_FN='bv'` 和 `--enable-bv-loss`，不传 `--loss-weight-schedule`，并显式使用 `--no-enable-confidence-head`。因此 10 个 epoch 始终只优化 BV loss：前 3 个 epoch 将 β 从 0 线性增加到 1，后 7 个 epoch 使用β=1 的 blockwise log BV。

训练继续使用现有 Arrow 数据与 online hidden-state 请求。代码不会自动重生成 T=1 回答，不会把 80w 数据自动裁成论文 100K，也不会将 token packing 自动变成论文 local batch=4/global batch=32。完整词表 BV 的显存成本明显高于单 token CE；先用小型数据验证真实 29B backward/服务负载，再进行正式 10-epoch 运行。

代码目录保留 `/hpfs/huawei-2607/shangxt/speculators`，必须部署本次 BV 适配代码；draft config 保留原来的相对路径 `draft_config_29b.json`。输出目录保留 `/hpfs/huawei-2607/shangxt/Telecom-29B/open_perfect_80w_no_thinking_full_vocab_1014_32_96`，日志/PID 命名也保持原样。若该目录已有旧非 BV checkpoint，不能作为当前 BV 配方恢复点。参数在脚本中直接修改，不再提供环境变量覆盖或 dry-run/preflight-only 模式。

观察 `bv_loss`、`bv_backward_surrogate`、`bv_beta`、`bv_valid_blocks`、`bv_dropped_blocks`、`bv_empty_batches` 和 `bv_estimated_draft_accept_len`。计数型裸指标经过现有 logger 规约后为各 rank 的平均数；`*_sum/*_total` 比值按真实有效计数聚合。旧 `full_acc` 仍与教师 argmax 比较；旧 `accept_len` 仍是 token-overlap 累乘近似，不是 BV 实测接受长度。

本轮已完成实现及本地验证，**未启动远端 29B 训练**。本地没有 `/hpfs/...` 模型/Arrow/config，也没有 Ascend 设备或对应在线教师服务，实际服务兼容性、NPU 数值/显存、FSDP 和训练收敛仍需按上述命令在部署环境验证。

## 6. 自定义教师配置 trust_remote_code 报错修复

2026-10-29 的服务器日志显示 `--draft-config` 路径在加载教师配置时没有传递 `--trust-remote-code`，同时工具函数来自 site-packages。修复将开关从 `build_draft_model` 经 `load_draft_transformer_layer_config`、`get_verifier_config` 传给 `AutoConfig.from_pretrained`；默认仍为 False。启动脚本仅增加一行 `export PYTHONPATH="$CODE_PATH/src${PYTHONPATH:+:$PYTHONPATH}"`，确保导入对应源码。

同步 `scripts/train.py`、`src/speculators/models/utils.py` 和启动脚本到两台服务器；启动脚本的 CODE_PATH 必须指向同步后的实际仓库。日志中的实际路径为 `/hpfs/huawei-2607/xiaoyf/speculators_shangxt/speculators`，如继续使用此位置，应将部署脚本中的 CODE_PATH 设置为此路径。

验证脚本：`scripts/verify/bv/08_trust_remote_config.sh`。本地结果：**71 passed**，包含真实本地自定义配置加载、未授权时拒绝加载、CLI 配置构造链路，以及现有 draft-config/启动脚本回归。没有加载 29B 权重，也没有访问教师服务。

```bash
# 在同步后的仓库根目录执行
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import speculators.models.utils as m; print(m.__file__)'
# 输出应指向当前仓库的 src/speculators/models/utils.py
PYTHON_BIN=python3 bash scripts/verify/bv/08_trust_remote_config.sh
```


## 7. 可配置 BV 权重及 BV+TV 混合损失

### 参数用法

保留 `--enable-bv-loss` 作为开关，使用现有 `--loss-fn` JSON 形式选择系数：

```bash
# 0.1 × BV + 0.9 × TV；没有额外 CE 项
--enable-bv-loss --loss-fn '{"bv":0.1,"tv":0.9}'
# 只使用 0.1 × BV
--enable-bv-loss --loss-fn '{"bv":0.1}'
# 纯 BV，权重 1
--enable-bv-loss --loss-fn bv
# 原 CE+TV：不加 --enable-bv-loss
--loss-fn '{"ce":0.1,"tv":0.9}'
```

权重直接相乘后求和，不自动归一化，不要求总和为 1。必须为有限非负数，至少一项大于 0；显式 BV 配置仅支持 bv 和 tv，不支持同时混入 ce 等其他项。`{"bv":0,"tv":0.9}` 允许使用 BV 有效 block 筛选路径上的 TV；不等同于旧 token-mask TV 数据选择方式。没有 BV 开关时，bv 不是合法的通用 token loss 名称。

兼容旧行为：`--enable-bv-loss --loss-fn ce`，以及任何旧的不含 bv 的合法 loss 配置，仍被覆盖为权重 1 的纯 BV。要启用 TV 混合，必须显式写 `{"bv":...,"tv":...}`，不能继续使用旧的 ce/tv JSON。BV+TV 是本次新增实验配置，不是论文原来的纯 BV 实验配方。

### 数学和日志约定

对 Markov 修正后的 draft 分布计算两项损失。两项使用相同的完整有效 block，教师分布不参与反向传播：

- BV 保持原积分公式、三轮聚合退火和后续 block-log。权重作用于整个 BV 反向代理，不改变 β 调度；日志记录对应的实际 BV 目标。
- TV 每个位置为 `0.5 * sum_v |q(v)-p(v)|`。沿用原 TV 的指数位置衰减 `exp(-j/gamma)`（j 从第一个预测位置的 0 开始，默认 gamma=4），然后对 B 个预测位置取平均，再对有效 block 取平均。分母为 B，不是位置权重之和。
- 总损失为 `w_bv * L_bv + w_ce * L_ce + w_tv * L_tv + 1.0 * L_confidence`，其中前三项按 epoch 调度，confidence 权重在当前 29B 脚本中固定为 1.0。
- DDP 按全局有效 block 数归一化两项梯度。全局无有效 block 时不更新 optimizer/scheduler。
- TV 同样按 bv_chunk_size 分块并在 backward 重算，避免整批保存全词表概率中间结果。

日志 `bv_loss` 和 `tv_loss` 是未乘系数的项，TV 已包含上述位置衰减；`bv_weighted_loss`、`tv_weighted_loss` 是加权后项。`main_loss` 是两项实际目标的和，`loss` 另外包含可选 confidence 辅助项。退火期 `bv_backward_surrogate` 是未乘 BV 系数的反向代理，其标量不要求等于 bv_loss。纯 BV 权重 1 的旧日志结构保持不变。

混合系数和 TV gamma 写入 checkpoint 配方。修改系数或有效 TV gamma 后不能直接作为同一训练任务恢复；请使用新输出目录。旧纯 BV 配方在权重仍为 1、TV 为 0 时保持兼容。

### 逐项验证与命令

1. 配置解析、非法权重、旧开关行为：`test_cli_weights_and_legacy_override` 等测试。
2. 独立 BV/TV 加权公式、梯度与实际目标日志、两种 anchor 布局、教师冻结：`test_weighted_loss_logs_and_gradients_match_independent_sum`。
3. 空 block、真实小型 backbone/Markov 梯度、恢复配方、动态 β 的 AOT 编译：同文件的剩余测试。

以上三步统一使用验证脚本 `scripts/verify/bv/09_mixture.sh`，可分别筛选执行：

```bash
export PYTHON_BIN=/path/to/train-env/bin/python
bash scripts/verify/bv/09_mixture.sh -k 'cli or explicit_bv'
bash scripts/verify/bv/09_mixture.sh -k weighted_loss
bash scripts/verify/bv/09_mixture.sh -k 'empty_blocks or backbone or resume or compiled'
# 全部混合损失检查
bash scripts/verify/bv/09_mixture.sh
# 旧路径及集成回归
bash scripts/verify/bv/run_all.sh
```

本地混合损失测试 **34 passed**；全部上述单元/集成回归累计 **129 passed**。AOT 测试使用 aot_eager，不代表 NPU 编译后端已验证。

4. 真实两进程 DDP 的不均匀 block、单 rank 空、全空场景：

```bash
bash scripts/verify/bv/06_distributed.sh --backend gloo --bv-weight 0.1 --tv-weight 0.9
# 部署设备验证：仅在一台机器的两张空闲 NPU 上运行
ASCEND_RT_VISIBLE_DEVICES=0,1 bash scripts/verify/bv/06_distributed.sh \
  --backend hccl --bv-weight 0.1 --tv-weight 0.9
```

CPU/Gloo **PASS**，梯度与一步 AdamW 更新均与拼接所有 block 的单进程参考一致。首次沙箱禁止本机 socket，获准后重试通过。HCCL 和真实 29B 混合训练尚未运行。


## 8. online 教师请求返回 Squid 错误页

17:15 的部署日志显示，rank 10 的 DataLoader worker 在 GET `http://10.127.10.208:8000/v1/models` 时收到 Squid `ERR_READ_ERROR / (104) Connection reset by peer`。其后其他 rank 的 SIGTERM/SIGKILL 是 torchrun 结束任务的结果，不能据此认定 OOM。失败发生在获取教师模型列表阶段，尚未进入该 batch 的 BV 前后向。

最小脚本修复：将现有六个教师地址和日志中的 10.127.10.208 加入 NO_PROXY，并继续同步小写 no_proxy。没有修改教师 endpoint 列表、损失、训练参数，也没有禁用全部代理。部署时其他新增教师 IP 也应加入排除列表。

验证：`test_teacher_endpoints_bypass_proxy_in_worker_environment` 检查脚本实际导出的大小写变量一致，并覆盖所有配置教师地址和本次报错地址。

```bash
PYTHON_BIN=python3 bash scripts/verify/bv/05_script.sh
# 在实际训练机检查报错服务是否可直接访问（不使用代理）
curl --noproxy '*' --connect-timeout 5 --max-time 15 -i \
  http://10.127.10.208:8000/v1/models
```

本地脚本验证 **8 passed**。未访问服务器；直连检查应返回 HTTP 200 及模型列表 JSON。若直连也失败，需要检查对应教师服务日志/监听端口及链路，NO_PROXY 修复不能保证服务本身健康。必须修改实际启动脚本中的导出语句；只在终端设置 NO_PROXY 会被脚本原有赋值覆盖。
