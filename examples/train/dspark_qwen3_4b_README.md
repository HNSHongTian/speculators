# Qwen3-4B DSpark Ascend training env

Container: `dspark-qwen34b-train`  
Image: `vllm-ascend:v0300rc2-ds4`

## Local paths

| Role | Path |
|------|------|
| Target (verifier) | `/home/z50062551/weights/Qwen3-4B` |
| Draft checkpoint (DeepSpec ref) | `/home/z50062551/weights/dspark_qwen3_4b_block7` |
| Raw 5k JSONL | `/home/z50062551/datasets/open-perfectblend-qwen3-4b-regen-5k.jsonl` |
| Processed Arrow | `/home/z50062551/datasets/perfectblend_qwen3_4b_5k_processed` |
| Code | `/home/z50062551/speculators` |

## Enter container

```bash
docker exec -it dspark-qwen34b-train bash
cd /home/z50062551/speculators
```

## Run training (two terminals / sessions)

```bash
# 1) Start vLLM (default NPUs 1,2)
bash examples/train/dspark_qwen3_4b_server.sh

# 2) After server is ready, start trainer (default NPUs 3-6)
bash examples/train/dspark_qwen3_4b_trainer.sh
```

Hyperparameters match `dspark_qwen3_4b_block7`: `block_size=7`, `num_layers=5`, `target_layer_ids=1 9 17 25 33`.

Note: the local draft under `weights/dspark_qwen3_4b_block7` is DeepSpec format (`Qwen3DSparkModel`). Speculators trains its own checkpoint format under `output/.../checkpoints`. Do not pass that DeepSpec dir to `--from-pretrained` unless converted.
