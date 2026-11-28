# Xingchen4 DSpark 离线测评（TP × DP）

在仓库根目录执行。使用和 `xingchen4_eval_ep` 相同的 Ascend PyTorch、torch_npu、主模型自定义代码及已安装的 Speculators。Transformers 还需支持 `tp_plan`、`device_mesh` 和按 checkpoint 解析出的 TP styles。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python xingchen4_eval_tp/evaluate_tp.py \
  --verifier-model "$VERIFIER_MODEL" \
  --draft-model "$DRAFT_MODEL" \
  --datasets-root "$DATASETS_ROOT" \
  --output-dir ./eval_results/xingchen4_tp8_dp2 \
  --tensor-parallel-size 8 --data-parallel-size 2 \
  --temperature 0 --max-new-tokens 2048 \
  --tree-top-k 1 --dtype bfloat16 --draft-attn-impl sdpa
```

`DATASETS_ROOT` 可以是单个 JSONL 或数据集目录。`--enable-thinking`、`--raw-prompt-mode` 等参数与 EP 入口使用相同的默认值。用相同参数对比 EP 和 TP；若要测 thinking 场景，在两个入口都显式指定 `--enable-thinking true`。首次运行可以加 `--max-samples 2 --max-new-tokens 32`，两组各评一条。

启动器在单机上为每个 DP 副本启动独立的 torchrun。每组有自己的 rendezvous、HCCL 集合通信和组内 rank 0。TP8 时主模型的注意力及 MLP/专家投影按张量切分，各卡保留全部专家的本地权重分片；DSpark 草稿模型在组内复制。实际使用的 TP 规则见输出目录的 `resolved_tp_plan.json`。

两个副本先按相同 seed 和样本上限选取同一组候选，再按索引交错分片。`--max-samples` 是每个数据集所有副本合计的上限。每个副本只由组内 rank 0 提交结果。最终输出 `summary.csv`、`summary.json` 和按样本顺序恢复的 `artifacts/<dataset>.jsonl`。沿用原离线测评的多卡汇总函数，接受长度由所有副本的累计接受 token 数和提案数计算，耗时取各副本报告的最大值。不同副本独立加载模型，因此该耗时不含启动和加载，也不计入副本间启动时间差。

线性 DSpark 默认逐 token 验证；`--block-verification` 可启用线性 block 验证，`--tree-top-k 4 --tree-max-nodes 32` 可启用树验证。参数组合限制、mHC 隐状态转换与 NPU 草稿注意力兼容处理沿用 EP 流程。
