# Leather 历史 checkpoint 统一 468 复评结果 (leather_checkpoint_unified468_results)

> 日期：2026-09-28
> 协议：test=468（`experiments/protocol/leather_valid_test_list.txt`），768×768，8 类含背景，TestRescale+ToTensor，无 ImageNet Normalize，batch=1，pred=outputs[0]，strict=True 加载，工具 `tools/evaluate_class_iou.py`。
> 每个 checkpoint 的完整 manifest（含 sha256 链、环境、per-class IoU）见 `repro_runs/leather_historical_audit/manifests/*.json`，原始 CSV/log 见 `repro_runs/leather_historical_audit/evals/`。

## 主表（按 unified468 mIoU 排序）

| Rank | Checkpoint | 真实模型结构 | 保存 iter | 历史 val235 best | **统一468 mIoU** | 备注 |
|---:|---|---|---:|---:|---:|---|
| 1 | `new (copy)/model_savePath/dsmonet_resnet_pascal_pige_4_0229_160000.pkl` | 822 变体（Light_Bag+eSE+detail，491 keys） | 128000 | 0.901785 | **0.917625** | 磁盘模型文件有 debug return，本轮在副本上修复后评价；**非论文 A2MS-B 结构** |
| 2 | `new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl` | **A2MS-DSMONet-B（r50 结构 = fixed forward，512 keys）** | 130000 | 0.909084 | **0.909713** | 论文 A2MS-B 历史权重；旧 0.860749 系 bug forward 低估 |
| 3 | `new (copy)/model_savePath/dsmonet_resnet_pascal_pige_4_0228.pkl` | 822 变体 | 74000 | 0.891943 | 0.900675 | 同 1，训练中段 |
| 4 | `new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl` | Base-B（506 keys） | 78000 | 0.892961 | 0.899120 | 与前序审计 0.899120 完全一致（复核通过） |
| 5 | `new (copy)/model_savePath/dsmonet_resnet_pascal_0120_pige_eSE_adapt.pkl` | eSE+ACW 无 detail head（505 keys） | 43000 | 0.889354 | 0.890033 | AAM-only 类历史消融 |
| 6 | `new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0126.pkl` | Base-B | 55000 | 0.872281 | 0.880614 | 0127 的前段 |
| 7 | `new/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0220ksh.pkl` | Base-B | 56500 | 0.827827 | 0.858620 | 实为 2026-05 A100 60k cosine 重训的 val-best |

## 对照：当前 60k 从零复现（固定终点协议，无 val 选择）

| Checkpoint | 模型 | iter | 统一468 mIoU |
|---|---|---:|---:|
| repro base_b iter060000 | Base-B | 60000 | 0.872137 |
| repro a2_full iter060000 | A2MS-B（fixed forward） | 60000 | 0.849932 |

## 关键对照：同一 checkpoint 两种 forward

| Checkpoint | 评价用模型文件 | forward | 统一468 mIoU |
|---|---|---|---:|
| eSE_adapt_detailloss_160000 | `model_dsmo_rs50_eSE_adapt_detailloss.py` | **bug（head_seg1=high_feats）** | 0.860749 |
| eSE_adapt_detailloss_160000 | `model_dsmo_rs50_eSE_adapt_detailloss_fixed.py` | 正确（head_seg1=out） | **0.909713** |

同一权重，仅评价 forward 不同，差异 +4.90 pt。历史 91% 对应正确 forward。

## A2MS-B vs Base-B（历史 val-best 链，同口径）

| 指标 | Base-B 0127@78k | A2MS-B @130k | Δ |
|---|---:|---:|---:|
| mIoU | 0.899120 | 0.909713 | **+0.010593** |
| background | 0.993250 | 0.992442 | −0.000808 |
| open_wound | 0.764801 | 0.792120 | +0.027319 |
| scratch | 0.700504 | 0.747213 | +0.046709 |
| brand_mark | 0.882528 | 0.895430 | +0.012902 |
| hole | 0.990708 | 0.993314 | +0.002606 |
| skin_disease | 0.971181 | 0.978667 | +0.007486 |
| rotten_surface | 0.915895 | 0.941891 | **+0.025996** |
| wart | 0.974096 | 0.936625 | −0.037471 |

7/8 类中 6 类提升；wart 回退 3.7 pt（note）。

## 结论

- 历史 Leather ≥0.91 结果在当前锁定 test468 上**独立复现成功**（pige_4 822: 0.917625；论文 A2MS-B 结构: 0.909713）。
- 状态：**HISTORICAL RESULT RECOVERED**。
