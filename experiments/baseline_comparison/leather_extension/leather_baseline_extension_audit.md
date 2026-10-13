# Leather Baseline Extension 审计报告（U-Net / DeepLabv3+ / FDSNet）

> 日期：2026-10-13  
> 目的：为 CJIG 论文 Leather 跨材质验证补充 Classical + Industrial 基线，回答“Leather 是否仅展示本文模型”的潜在审稿问题。  
> 本实验完全独立于既有冻结结果，未覆盖任何已有 checkpoint / JSON / audit 文件。

## 一、协议声明（与 repro_runs/leather_60k_seed1337 逐项一致）

| 项 | 值 |
|---|---|
| 训练集 | pige train1638（`new (copy)/dataset/pige/train.txt`） |
| 测试集 | 锁定 test468（`experiments/protocol/leather_valid_test_list.txt`，sha256 f9ac23b3…） |
| 输入 | 768×768 原生（768=16×48，U-Net 无需 208 padding——该 padding 是 NEU-200 专用规避） |
| 类别 | 8（含 background），mIoU 含背景 |
| 训练 | from scratch，seed 1337，Adam lr=1e-4 wd=2e-6，CosineAnnealingLR(T_max=48000, eta_min=1e-6) 按 iter，batch 8，60000 iters |
| 选模纪律 | 训练中无 val/test loader、无任何模型选择、固定终点 iter60000 |
| checkpoint | 预注册 {10k,30k,60k} + 每 5000 原子 recovery（未发生故障恢复） |
| 评价 | batch1，TestRescale 等价 + ToTensor，无 ImageNet Normalize，pred=outputs[0] |

协议保真证据：本批三模型在 iter 59800/59900/60000 的 LR（1.50/1.53/1.55e-05）与原 leather_60k 链 base_b 逐位一致。

## 二、模型定义与损失（与 NEU 240k baseline 相同来源）

| 模型 | 类别 | 定义文件 | 入口 | 损失 |
|---|---|---|---|---|
| U-Net | Classical | `new/model_unet.py` | `unet(feature_scale=1, n_classes=8, deconv, BN)` | plain CE |
| DeepLabv3+ | Classical | `new (copy)/model/deeplabv3.py` | `DeepLab(num_classes=8, resnet50, os16, pretrained=False)` | plain CE |
| FDSNet | Industrial defect | `third_party/FDSNet/core/models/fdsnet.py` @ ced4d0d | `FDSNet(num_classes=8, aux=True)` | OHEM-CE + 0.5·BCE(edge) + 0.5·BCE(se) |

## 三、结果（test468，mIoU 含背景）

| 模型 | 类别 | mIoU | background | open_wound | scratch | brand_mark | hole | skin_disease | rotten_surface | wart | ckpt sha256(前16) | 训练耗时 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| U-Net | Classical | **82.42** | 98.96 | 57.57 | 58.76 | 84.08 | 98.12 | 92.62 | 75.62 | 93.67 | eb4bba2e608e2104 | 9.21h |
| DeepLabv3+ | Classical | **87.60** | 99.26 | 70.48 | 62.99 | 87.53 | 98.79 | 95.63 | 88.83 | 97.31 | 7f6c5e7ad1d9b713 | 6.23h |
| FDSNet | Industrial defect | **78.19** | 98.29 | 65.14 | 34.41 | 81.67 | 98.10 | 88.48 | 73.31 | 86.13 | c2cd214f1ebe3e23 | 3.38h |

同口径参照（60k 固定终点存档值，非论文值）：DSMONet-B 87.21 / A2MS-DSMONet-B 84.99（`repro_runs/leather_60k_seed1337/leather_results.json`）。

### 效率（768² FP32 batch1，沿用 Leather 既有效率协议：warmup 50 / measure 200 / FPS=1000/mean）

| 模型 | Params/M | 延迟 mean/ms | 延迟 p95/ms | FPS |
|---|---:|---:|---:|---:|
| U-Net | 31.04 | 35.38 | 50.40 | 28.26 |
| DeepLabv3+ | 59.34 | 19.00 | 28.89 | 52.63 |
| FDSNet | 2.60 | 5.67 | 6.87 | 176.52 |

测量与评价使用完全相同的原生 768 推理路径（`scripts/bench_efficiency_768.py`，构建器共享 `eval_leather_ext.py`）。
测量开始时 GPU 有外部空闲进程持有 9348 MiB 显存（共享机常态）；p95/mean∈[1.21,1.52]，与历史干净基线（1.42）同区间，未见共享污染型尖峰。

## 四、评价工具与交叉验证

- DeepLabv3+、FDSNet：官方 `tools/evaluate_class_iou.py`（原生 768 构建器）直接评价。
- U-Net：官方工具注册的 U-Net 构建器硬编码 NEU 专用 200→208 pad/crop，768 下负 padding 会把输入错误裁剪为 208；
  故使用 `scripts/eval_leather_ext.py`——**逐行复用官方工具**的 SegmentationListDataset / confusion_matrix / compute_metrics / load_state，仅替换模型构建为原生 768。
- 驱动交叉验证：同一 checkpoint 下 driver vs 官方工具 — DeepLabv3+ |ΔmIoU|=2.7e-07，FDSNet |ΔmIoU|=4.2e-07（六位小数舍入量级），**通过**。

## 五、与论文 Leather 主表的口径关系（必须如实标注）

- 论文 Leather 主表（DSMONet-B 89.91 / A2MS 90.97 / STDC 88.24 / PIDNet 86.20 / DDRNet 84.21）来自**历史 val235 监控 + val-best 选择**的 checkpoint 统一 test468 复评；
- 本批三模型为 **60k 固定终点、无验证监控**协议；两协议的选择口径不同；
- 因此：本批数字与同协议 60k 存档值（87.21/84.99）直接可比；与 val-best 主表并列时必须在表注中说明协议差异，不得静默混排。

## 六、观察（不作论文结论，仅记录事实）

- DeepLabv3+（87.60）高于同协议 DSMONet-B 存档值（87.21）0.39 pt——单种子、单终点，不构成排名结论；
- FDSNet（78.19）在 Leather 上明显低于其在 NEU-Seg 的相对表现； wart 0.861 / open_wound 类较低；
- U-Net（82.42）居中；三模型与本文模型的差距在 open_wound/scratch 两类最集中。

## 七、文件清单

```
repro_runs/leather_baseline_extension/
├── README.md                        协议与结构说明
├── configs/                         3 正式 + 3 smoke 配置
├── scripts/                         3 训练脚本 + eval_leather_ext.py
├── {unet,deeplabv3p,fdsnet}/        checkpoints(10k/30k/60k+recovery) logs analysis
└── results/                         官方+驱动 csv/log/json、本审计、汇总 json
```
