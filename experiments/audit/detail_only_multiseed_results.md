# Detail-only 多 seed 10k 结果（统一 840 评价）

日期：2026-09-22（训练完成于 2026-09-14 16:26–16:27，因会话中断延迟至本日完成统一评价与归档）

---

## 1. 单元定义

**D-only = Base-B 架构 + 第四路 detail supervision，完全无 AAM。**

- 模型文件：`new/model_dsmo_rs50_detailloss_only.py`（SHA256 `47351819…`）
- 与 Base-B（`new/model_dsmo_rs50.py`）差异：**仅 3 行**（启用 `seg_head_detailloss = SegHead(128,64,1)` 与第 4 路训练输出），其余计算图逐项一致（静态审计：`detail_only_model_audit.md`）
- 保留：标准 `SELayer(128,128)`、DAPPM、SqueezeBodyEdge、UAFM arm1/arm2、`head_seg1 = out`（56×56 detail-semantic fused 主输出）
- 不含：eSE、AdaptiveChannelWeight、任何 AAM 相关参数（state_dict 实测仅 `se.*` 与 A2 不同）
- detail 监督：与 A2 完全相同的 `detail_aggregate_loss`、相同 detail target 生成、相同权重 3

## 2. 协议（与 Base-B/A1/A2 完全一致）

NEU-Seg：train 3630 / test 840（SHA256 `498b038f…` / `2c45b319…`），4 类含 background，200×200；train transform = Compose([Transforms_PIL(200,200), ToTensor])，test = TestRescale+ToTensor，无 ImageNet Normalize；batch 16；Adam lr 1e-4 / wd 2e-6；cosine_annealing T_max=48000 eta_min=1e-6；10000 iters；from scratch；顶层 `seed` 键生效（iter-1 梯度指纹三 seed 互异）；val_interval=10000（**仅终点评价，best==last by design，无 test 选择**）。

loss_weights `[10, 1, 3, 3]`，loss 列表 [ohem_ce, bce_with_logits, ohem_ce, detail_aggregate_loss]——与 A2 逐项相同。

## 3. 统一 840 终点评价结果

评价：`unified_eval_840_ms.py`，全部 840 张、batch1、`pred=outputs[0]`、4 类 IoU 含 background。使用 iter10000 checkpoint（best==last by design）。

| Seed | mIoU | background | crazing | inclusion | patches |
|---|---:|---:|---:|---:|---:|
| 1337 | 0.855020 | 0.975329 | 0.744403 | 0.884103 | 0.816246 |
| 2026 | 0.852310 | 0.975353 | 0.742859 | 0.885297 | 0.805733 |
| 3407 | 0.855968 | 0.975591 | 0.750921 | 0.882386 | 0.814974 |
| **mean ± SD** | **0.854433 ± 0.001898** | 0.975424 ± 0.000143 | 0.746061 ± 0.004261 | 0.883929 ± 0.001469 | 0.812318 ± 0.005806 |

monitor（832 张 batch16 drop_last，训练内终点读数，仅供交叉核对）：1337 = 0.854903，2026 = 0.851998，3407 = 0.856412。

## 4. ΔD_noA = D − Base-B（本轮新增最重要指标）

| Seed | Base-B | D-only | ΔD_noA |
|---|---:|---:|---:|
| 1337 | 0.854558 | 0.855020 | +0.000462 |
| 2026 | 0.847473 | 0.852310 | +0.004838 |
| 3407 | 0.854493 | 0.855968 | +0.001475 |

**mean = +0.002258 ± 0.002291，正向 3/3 seeds。**

方向完全一致，但幅度小：均值 +0.23 个百分点，约为 Base-B 自身 seed 间波动（SD 0.41 pt）的一半。

## 5. 运行现场与可复现性

| Seed | run 目录 | manifest |
|---|---|---|
| 1337 | `repro_runs/multiseed_10k/seed_1337/detail_only/` | `experiments/audit/detail_only_seed1337_manifest.json` |
| 2026 | `repro_runs/multiseed_10k/seed_2026/detail_only/` | `experiments/audit/detail_only_seed2026_manifest.json` |
| 3407 | `repro_runs/multiseed_10k/seed_3407/detail_only/` | `experiments/audit/detail_only_seed3407_manifest.json` |

每个 run 保存 `code/`（自包含代码副本，模型与训练脚本 SHA256 见 manifest）、`config/`、`logs/`、`checkpoints/`、`analysis/`（pre_run_manifest、stdout/stderr、train_summary、unified_eval_840.json）。三个 manifest 均含 requested_seed / effective_seed / seed_config_key / seed_verification 字段并通过 `python -m json.tool` 校验。

环境：Python 3.11.13，torch 2.7.1+cu126，单张 A100-PCIE-40GB（三个 run 并行，与无关 nnUNet 任务共享 GPU）。训练时长约 71 分钟/run（15:16→16:27）。

训练命令（每个 run 目录内）：

```bash
cd code && PYTHONPATH=. python3 train_detail_only_10k.py \
  --config ../config/config.yml --run-id detail_only_s{seed}_10k_codex1
```

## 6. 备注

- 训练于 2026-09-14 完成（16:26–16:27 落盘 checkpoint 与 summary），会话中断后于 2026-09-22 完成统一 840 评价与归档；checkpoint、日志、配置全程未改动。
- D-only 单元补齐后，2×2 factorial 四格齐备（B / A / D / AD），完整效应分解见 `a2ms_2x2_factorial_analysis.md`，类别级分析见 `a2ms_2x2_classwise_analysis.md`。
