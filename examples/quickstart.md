# Qwen3-4B DSpark 训练 / 评测 Quickstart

## 路径

| 用途 | 路径 |
|------|------|
| 代码 | `/home/z50062551/speculators` |
| 目标模型 | `/home/z50062551/weights/Qwen3-4B` |
| 草稿（DeepSpec 原版） | `/home/z50062551/weights/dspark_qwen3_4b_block7` |
| 草稿（Speculators 格式） | `/home/z50062551/weights/dspark_qwen3_4b_block7_speculators` |
| 评测 JSONL | `/home/z50062551/eval` |
| 原始 5k JSONL | `/home/z50062551/datasets/open-perfectblend-qwen3-4b-regen-5k.jsonl` |
| 处理后 Arrow | `/home/z50062551/datasets/perfectblend_qwen3_4b_5k_processed` |

## 1. 起容器

镜像：`vllm-ascend:v0300rc2-ds4`，容器名：`dspark-qwen34b-train`（已创建可直接用）。

```bash
# 已存在则跳过创建
docker start dspark-qwen34b-train

# 进入
docker exec -it dspark-qwen34b-train bash
cd /home/z50062551/speculators
```

新建容器（仅首次需要）：

```bash
docker run -d --name dspark-qwen34b-train --privileged --shm-size=64g --network host \
  --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /home:/home -w /home/z50062551/speculators \
  vllm-ascend:v0300rc2-ds4 bash -c 'sleep infinity'

# 容器内安装（仅首次）
docker exec -it dspark-qwen34b-train bash -lc '
  cd /home/z50062551/speculators &&
  pip install -e ./hs_connectors --no-deps -q &&
  pip install -e . --no-deps -q &&
  pip install "datasets>=4.0.0,<=5.0.0" accelerate "numpy>=2.0,<2.3" "tqdm<=4.68.4" -q
'
```

## 2. 数据预处理（已做过可跳过）

在容器内执行：

```bash
cd /home/z50062551/speculators
python scripts/prepare_data.py \
  --model /home/z50062551/weights/Qwen3-4B \
  --data /home/z50062551/datasets/open-perfectblend-qwen3-4b-regen-5k.jsonl \
  --output /home/z50062551/datasets/perfectblend_qwen3_4b_5k_processed \
  --seq-length 4096
```

## 3. 起训练（TRPO，2+4）

NPU 0 留给已有进程。vLLM 用 **1、2**，TRPO 训练用 **3、4、5、6**。两个终端都先 `docker exec -it dspark-qwen34b-train bash`。

```bash
# 终端 1：vLLM（NPU 1,2）
cd /home/z50062551/speculators
VLLM_NPUS=1,2 DP_SIZE=2 bash examples/train/dspark_qwen3_4b_server.sh
```

等服务就绪（返回 `200`）后再开训练：

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/v1/models
```

```bash
# 终端 2：TRPO（NPU 3-6，MAX_STEPS=0 跑完 1 个 epoch）
cd /home/z50062551/speculators
TRAIN_NPUS=3,4,5,6 NUM_TRAIN_NPUS=4 MAX_STEPS=0 \
  bash examples/train/trpo_qwen3_4b_trial.sh
```

```bash
# 看日志
tail -f /home/z50062551/speculators/output/trpo_qwen3_4b_perfectblend_5k_trial/logs/trpo_*.log

# 停训练
kill $(cat /home/z50062551/speculators/output/trpo_qwen3_4b_perfectblend_5k_trial/logs/trpo.pid)
```

checkpoint 输出：`/home/z50062551/speculators/output/trpo_qwen3_4b_perfectblend_5k_trial/checkpoints`

## 4. 离线评测（DSpark）

用 `scripts/evaluate/dspark_offline_eval.py`，读本地 `/home/z50062551/eval/*.jsonl`。  
草稿必须用 Speculators 格式：`dspark_qwen3_4b_block7_speculators`（不要传 DeepSpec 原版目录）。

默认 `--tree-top-k 1` 为线性 DSpark；`--tree-top-k >= 2` 才是树投机。

多卡是**数据并行**：每张卡各加载一份完整模型，按样本轮询切分后汇总。父进程不要设 `ASCEND_RT_VISIBLE_DEVICES`，交给 `--ascend-devices`。

已知数据集默认样本上限（与 DeepSpec 对齐，可用 `--max-samples` 覆盖）：gsm8k/math500/alpaca/arena-hard-v2/livecodebench=500，humaneval=164，mbpp=256，mt-bench=80，aime25=30。

### 4.1 单卡

```bash
docker exec -it dspark-qwen34b-train bash
cd /home/z50062551/speculators

ASCEND_RT_VISIBLE_DEVICES=3 \
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
python scripts/evaluate/dspark_offline_eval.py \
  --verifier-model /home/z50062551/weights/Qwen3-4B \
  --draft-model /home/z50062551/weights/dspark_qwen3_4b_block7_speculators \
  --datasets-root /home/z50062551/eval \
  --datasets gsm8k \
  --output-dir /home/z50062551/speculators/eval_results/qwen3_4b_dspark_gsm8k \
  --max-new-tokens 2048 \
  --temperature 0 \
  --dtype bfloat16 \
  --device npu \
  --enable-thinking false \
  --raw-prompt-mode auto \
  --draft-attn-impl sdpa
```

### 4.2 多卡（推荐）

```bash
docker exec -it dspark-qwen34b-train bash
cd /home/z50062551/speculators

PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
python scripts/evaluate/dspark_offline_eval.py \
  --verifier-model /home/z50062551/weights/Qwen3-4B \
  --draft-model /home/z50062551/weights/dspark_qwen3_4b_block7_speculators \
  --datasets-root /home/z50062551/eval \
  --datasets gsm8k,math500,aime25,humaneval,mbpp,livecodebench,mt-bench,alpaca,arena-hard-v2 \
  --output-dir /home/z50062551/speculators/eval_results/qwen3_4b_dspark_official \
  --ascend-devices 3,4,5,6 \
  --max-new-tokens 2048 \
  --temperature 0 \
  --dtype bfloat16 \
  --device npu \
  --enable-thinking false \
  --raw-prompt-mode auto \
  --draft-attn-impl sdpa
```

结果：`summary.json` / CSV 在 `--output-dir`；分片在 `_shards/`。
