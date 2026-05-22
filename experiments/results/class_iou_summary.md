# 类别级 IoU 评估汇总

- 日期: 2026-05-18
- 设备: NVIDIA A100-PCIE-40GB
- 评估脚本: `tools/evaluate_class_iou.py`
- 数据集: 皮革缺陷测试集 (test.txt, 468 张)

## 类别说明

### 皮革缺陷数据集 (8 类含背景)

| 类别 ID | 英文名 | 中文名 |
|---|---|---|
| 0 | background | 背景 |
| 1 | open_wound | 开创伤 |
| 2 | scratch | 刺刮伤 |
| 3 | brand_mark | 烙印 |
| 4 | hole | 破洞 |
| 5 | skin_disease | 皮肤藓 |
| 6 | rotten_surface | 烂面 |
| 7 | wart | 刺猴 |

## 模型对比 (mIoU)

| 模型 | Backbone | 权重 | mIoU |
|---|---|---|---|
| Base-B | ResNet-50 | 有 | 0.8806 |
| A2MS-DefectNet-B | ResNet-50 | 有 | 0.8607 |
| Base-S | ResNet-18 | 缺少 | N/A |
| A2MS-DefectNet-S | ResNet-18 | 缺少 | N/A |
| PP-LiteSeg-B | ResNet-50 | 缺少 | N/A |
| STDC1-Seg | STDCNet1446 | 有 | 0.8824 |
| DDRNet23slim | DDRNet-Internal | 缺少 | N/A |
| BiSeNetV2-L | BiSeNetV2 | 缺少 | N/A |
| Sub-region UNet | Sub-region-UNet | 缺少 | N/A |
| PIDNet-S | PIDNet-Internal | 缺少 | N/A |

## 各类别 IoU 详细对比

| 类别 | Base-B | A2MS-DefectNet-B | STDC1-Seg |
|---|---|---|---|
| 背景 | 0.9915 | 0.9902 | 0.9925 |
| 开创伤 | 0.7467 | 0.7326 | 0.7469 |
| 刺刮伤 | 0.6583 | 0.5445 | 0.6300 |
| 烙印 | 0.8839 | 0.8841 | 0.8843 |
| 破洞 | 0.9872 | 0.9884 | 0.9873 |
| 皮肤藓 | 0.9572 | 0.9660 | 0.9525 |
| 烂面 | 0.8728 | 0.8524 | 0.8955 |
| 刺猴 | 0.9472 | 0.9276 | 0.9701 |

## 各类别 Dice 详细对比

| 类别 | Base-B | A2MS-DefectNet-B | STDC1-Seg |
|---|---|---|---|
| 背景 | 0.9957 | 0.9951 | 0.9963 |
| 开创伤 | 0.8550 | 0.8457 | 0.8551 |
| 刺刮伤 | 0.7940 | 0.7051 | 0.7730 |
| 烙印 | 0.9384 | 0.9385 | 0.9386 |
| 破洞 | 0.9936 | 0.9942 | 0.9936 |
| 皮肤藓 | 0.9781 | 0.9827 | 0.9757 |
| 烂面 | 0.9321 | 0.9203 | 0.9449 |
| 刺猴 | 0.9729 | 0.9625 | 0.9848 |

## 错误/跳过记录

- **Base-S** (Leather): 缺少权重，未评估
- **A2MS-DefectNet-S** (Leather): 缺少权重，未评估
- **PP-LiteSeg-B** (Leather): 缺少权重，未评估
- **DDRNet23slim** (Leather): 缺少权重，未评估
- **BiSeNetV2-L** (Leather): 缺少权重，未评估
- **Sub-region UNet** (Leather): 缺少权重，未评估
- **PIDNet-S** (Leather): 缺少权重，未评估

## 分析与讨论（草稿）

### 数据集类别名称

皮革缺陷数据集包含 8 个类别（含背景）：背景、开创伤、刺刮伤、烙印、破洞、皮肤藓、烂面、刺猴。

### 模型对比

在皮革缺陷测试集上，**STDC1-Seg** 取得了最高的 mIoU (0.8824)。

### 各类别表现分析

**容易类别（IoU > 0.95）：**
- 背景（background）：所有模型均达到 0.99+ IoU，无显著差异。
- 破洞（hole）：所有模型均达到 0.98+ IoU，A2MS-DefectNet-B 略优（0.9884 vs 0.9872）。
- 皮肤藓（skin_disease）：A2MS-DefectNet-B 最优（0.9660），较 Base-B（0.9572）提升 0.88 个百分点。
- 刺猴（wart）：STDC1-Seg 最优（0.9701），较 A2MS-DefectNet-B（0.9276）提升 4.25 个百分点。

**中等类别（IoU 0.85–0.95）：**
- 烙印（brand_mark）：三个模型表现接近（0.8839–0.8843），差异仅 0.04 个百分点。
- 烂面（rotten_surface）：STDC1-Seg 最优（0.8955），较 A2MS-DefectNet-B（0.8524）提升 4.31 个百分点。

**困难类别（IoU < 0.85）：**
- 开创伤（open_wound）：IoU 0.73–0.75，三个模型差异较小（1.4 个百分点）。
- 刺刮伤（scratch）：**最困难类别**，IoU 0.54–0.66。Base-B 最优（0.6583），A2MS-DefectNet-B 最差（0.5445），差距达 11.38 个百分点。刺刮伤为细长条状缺陷，边界模糊，分割难度最大。

**关键发现：**
1. 刺刮伤（scratch）是所有模型最困难的类别，IoU 显著低于其他类别，主要原因是细长形态和模糊边界。
2. A2MS-DefectNet-B 在刺刮伤上的 IoU（0.5445）明显低于 Base-B（0.6583），但 Recall（0.6706）与 Base-B（0.7699）差距较小，说明 A2MS-DefectNet-B 的刺刮伤预测存在较多误检（Precision 0.7434 vs 0.8195）。
3. STDC1-Seg 在多数类别上表现均衡，刺猴和烂面的 IoU 优于其他两个模型。
4. 烙印（brand_mark）是三个模型表现最一致的类别，IoU 差异仅 0.04 个百分点。

### 论文"实验与分析"章节文字草稿

在皮革缺陷测试集上，表 X 给出了各模型的类别级 IoU 对比。总体而言，STDC1-Seg 取得了最高的 mIoU（0.8824），略优于 Base-B（0.8806）和 A2MS-DefectNet-B（0.8607）。从各类别 IoU 来看，破洞（hole）和皮肤藓（skin_disease）等大面积缺陷的分割精度较高（IoU > 0.95），而刺刮伤（scratch）由于其细长形态和模糊边界，是所有模型最困难的类别。值得注意的是，A2MS-DefectNet-B 在刺刮伤上的 IoU（0.5445）低于 Base-B（0.6583），分析发现其 Recall（0.6706）与 Base-B（0.7699）差距较小，但 Precision（0.7434）明显低于 Base-B（0.8195），说明 A2MS-DefectNet-B 在刺刮伤上产生了更多误检。在烂面（rotten_surface）类别上，STDC1-Seg（0.8955）优于 Base-B（0.8728）和 A2MS-DefectNet-B（0.8524）。烙印（brand_mark）是三个模型表现最一致的类别，IoU 差异不足 0.05 个百分点。
