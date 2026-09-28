# 第一轮重跑与统一评估最终决策

> 决策日期：2026-05-26
> 覆盖范围：NEU-Seg 重训、Leather 统一评估、复杂度统计、FPS 平台
> 状态：待人工确认后通知 DeepScientist 更新论文

---

## 一、A2MS-DefectNet-B NEU-Seg 状态

### 当前状态

```
❌ 重训完成，结果与失败训练一致
```

- 训练命令：`train_neu_resnet_detailloss.py --config config_neu_a2ms_b_retrain.yml`
- 配置变化：seed=42, train_iters=120000, T_max=96000
- **Best mIoU: 0.7139** (epoch 78000) — 与失败训练 0.7131 几乎相同
- Per-class IoU: bg=0.9511, crazing=0.5299, inclusion=0.8276, patches=0.5305

### 根因确认

两次独立训练（不同 seed、不同迭代数 60k vs 120k）产生相同结果，证明：
- ❌ 不是随机种子问题
- ❌ 不是训练时长不足
- ✅ **SqueezeBodyEdge + detail_aggregate_loss 在 NEU-Seg 200×200 小图上有架构兼容性问题**

Base-B (SqueezeBodyEdge, 无 detailloss) → 成功 90.5%
A2MS-S (Light_Bag + detailloss) → 成功 88.6%
A2MS-B (SqueezeBodyEdge + detailloss) → 失败 71.4%

### 判决

```
A2MS-DefectNet-B NEU-Seg: 不可写入论文。需更换架构策略后重新训练。
```

---

## 二、Leather 数据集：A2MS-B 是否优于 Base-B？

### 统一 test 评估结论

```
❌ A2MS-DefectNet-B 在当前 test.txt 上不优于 Base-B
```

| 指标 | Base-B (0127) | A2MS-DefectNet-B (160000) | 胜者 |
|---|---:|---:|---|
| mIoU | **0.8991** | 0.8607 | Base-B (+3.84pp) |
| open_wound IoU | **0.7648** | 0.7326 | Base-B (+3.22pp) |
| scratch IoU | **0.7005** | 0.5445 | Base-B (+15.60pp) |
| brand_mark IoU | 0.8825 | **0.8841** | A2MS-B (+0.16pp) |
| rotten_surface IoU | **0.9159** | 0.8524 | Base-B (+6.35pp) |

论文历史声称 A2MS-B=91.0 > Base-B=89.2 → **在 test.txt 上不成立**。

### 可能解释

1. A2MS-B 的 checkpoint best_iou (0.9091) 来自 val.txt (234 张) 验证，在 test.txt (468 张) 上泛化差（-4.84pp gap）
2. Base-B (0127) 的 checkpoint best_iou (0.8930) 与 test mIoU (0.8991) 一致
3. A2MS-B 的 eSE+AdaptiveChannelWeight 模块可能在 Leather 上过拟合到 val set
4. A2MS-B 对 scratch（刺刮伤）类表现特别差 (0.5445 vs 0.7005)，拉低 mIoU

### 判决

```
A2MS-B Leather: 不可写入论文主结论为"优于 Base-B"。
Base-B Leather: mIoU=0.8991 可直接写入论文（使用 0127 权重）。
A2MS-B Leather: 可在讨论中分析其 scratch 类性能较差的原因。
```

---

## 三、哪些结果可进入论文主表

### ✅ 可直接写入主实验表

#### NEU-Seg 数据集

| 模型 | mIoU | Params(M) | FLOPs(G)@200 | FPS@200 (A100) |
|---|---:|---:|---:|---:|
| Base-B | **90.47** (训练日志) | 29.37 | 6.65 | 87.8 |
| Base-S | **88.37** (训练日志) | 14.00 | 2.51 | 127.3 |
| A2MS-DefectNet-S | **88.58** (训练日志) | 14.03 | 2.34 | 130.8 |
| DDRNet23slim | **88.43** (训练日志) | 6.71 | 2.07 | 167.7* |
| PIDNet-S | **86.67** (训练日志) | 7.72 | 0.97 | 127.7* |

> * FPS 来自 2026-05-18 测量

#### Leather 数据集

| 模型 | mIoU | Params(M) | FLOPs(G)@200 | FPS@768 (A100) |
|---|---:|---:|---:|---:|
| Base-B (0127) | **0.8991** (test set) | 29.37 | 6.65 | 74.6 |
| STDC1-Seg | **0.8824** (test set) | 16.07 | 6.02 | 123.7* |
| Base-S | **0.8586** (test set) | 14.00 | 2.51 | 119.0 |
| A2MS-DefectNet-S | **0.8690** (test set) | 14.03 | 2.34 | 122.4 |
| PIDNet-S | **0.8620** (test set) | 7.72 | 0.97 | 127.9* |

> * FPS 来自 2026-05-18 测量

### ⚠️ 可写入讨论或补充分析

| 模型 | 结果 | 限制 |
|---|---|---|
| A2MS-DefectNet-B Leather | mIoU=0.8607 | 不优于 Base-B；scratch IoU 特别低 |
| A2MS-DefectNet-B NEU | 待重训完成 | 当前 mIoU=71.31 严重异常 |
| DDRNet23slim Leather | 待评估脚本修复 | 当前因模型文件不匹配无法评估 |
| Per-class IoU 全量 | 可用于类别分析 | mIoU 一致，可写入 |

### ❌ 暂不可写入

| 结果 | 原因 |
|---|---|
| 论文历史 Leather mIoU (89.2-91.0) | 无法在 test.txt 上复现 |
| 论文历史 NEU A2MS-B=91.3 | 权重不存在，当前训练 mIoU=71.31 |
| 对比方法 FPS | 来自 RTX 3090，与 A100 不可比 |
| FCN/U-Net/HRNet/PSPNet/DeepLabV3+/ENet | 权重缺失 |
| 小目标分组评价 | 方法论问题待修复 |
| AAM/eSE 消融中间点 | 无独立权重 |
| 联合损失消融中间点 | 无独立训练记录 |

---

## 四、论文表格更新指南

### 表 X：NEU-Seg 主实验结果

可填入：
- Base-B: mIoU=90.47, Params=29.37M, FLOPs=6.65G, FPS=87.8
- Base-S: mIoU=88.37, Params=14.00M, FLOPs=2.51G, FPS=127.3
- A2MS-S: mIoU=88.58, Params=14.03M, FLOPs=2.34G, FPS=130.8
- DDRNet23slim: mIoU=88.43, Params=6.71M, FLOPs=2.07G, FPS=167.7
- PIDNet-S: mIoU=86.67, Params=7.72M, FLOPs=0.97G, FPS=127.7
- A2MS-B: **TODO** (待重训完成)

### 表 Y：Leather 主实验结果

可填入：
- Base-B (0127): mIoU=0.8991, Params=29.37M, FLOPs=6.65G, FPS=74.6
- STDC1-Seg: mIoU=0.8824, Params=16.07M, FLOPs=6.02G, FPS=123.7
- Base-S: mIoU=0.8586, Params=14.00M, FLOPs=2.51G, FPS=119.0
- A2MS-S: mIoU=0.8690, Params=14.03M, FLOPs=2.34G, FPS=122.4
- PIDNet-S: mIoU=0.8620, Params=7.72M, FLOPs=0.97G, FPS=127.9
- A2MS-B: 0.8607 (如果写入，需标注"不优于 Base-B")
- DDRNet23slim: **TODO** (待评估修复)

### 表 Z：复杂度表（全部模型）

Params/FLOPs/模型大小：直接使用 complexity_results_a100_final_summary.md
FPS：使用 A100 重测值 + 标注测量平台

---

## 五、仍需重跑的项目

| 优先级 | 项目 | 状态 |
|---|---|---|
| 🔴 | A2MS-B NEU 重训 | **进行中** |
| 🔴 | DDRNet23slim Leather 评估修复 | 待修复 evaluate_class_iou.py |
| 🟡 | Base-S / A2MS-S Leather 重训（匹配历史值） | 待决策 |
| 🟡 | A2MS-B NEU 如果本轮仍失败，改策略重训 | 待观察 |
| 🟢 | 小目标分组评价重跑 | 待执行 |
| 🟢 | 可视化重新生成 | 待执行 |
| 🟢 | DMC-Net 复现 | 待评估 |

---

## 六、是否可以通知 DeepScientist 更新论文

### 当前可以更新的内容

```
✅ NEU-Seg 主实验表（除 A2MS-B 外）
✅ Leather 主实验表（使用 Base-B 0127 作为最佳 DSMONet 结果）
✅ 复杂度表（Params/FLOPs/模型大小/FPS@A100）
✅ 类别级 IoU 分析
✅ 实验设置节（统一使用 A100 平台）
```

### 暂不能更新的内容

```
❌ A2MS-B 的论文定位（需要重训结果决定）
❌ Leather 上 A2MS-B vs Base-B 的结论（需改写）
❌ 消融实验定量表格（中间点权重缺失）
❌ 小目标分组评价（方法论待修复）
```

### 建议

**可以先更新论文的大部分表格和正文，A2MS-B 相关结论留 TODO，等待重训完成后填入。**

如果重训成功（mIoU > 88%），A2MS-B NEU 可恢复论文定位。
如果重训失败，需要重新评估 A2MS-B 在 NEU-Seg 上的可行性。
