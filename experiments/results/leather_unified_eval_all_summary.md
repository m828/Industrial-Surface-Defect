# Leather 统一 test 评估完整结果

> 评估日期：2026-05-26
> 评估协议：eval_protocol_lock.md (Leather: test.txt, 468张, 768×768, 8类)
> 脚本：tools/evaluate_class_iou.py
> GPU：NVIDIA A100-PCIE-40GB

## 评估结果总表

| 排名 | 模型 | 权重 | mIoU | bg | open_wound | scratch | brand_mark | hole | skin_disease | rotten_surface | wart |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **Base-B (0127)** | dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl | **0.8991** | 0.9933 | 0.7648 | 0.7005 | 0.8825 | 0.9907 | 0.9712 | 0.9159 | 0.9741 |
| 2 | STDC1-Seg | sdtdcnet_pige_stdc2_pige.pkl | 0.8824 | 0.9925 | 0.7469 | 0.6300 | 0.8843 | 0.9873 | 0.9525 | 0.8955 | 0.9701 |
| 3 | Base-B (0126) | dsmonet_resnet_pascal_pige_dsmor50_0126.pkl | 0.8806 | 0.9915 | 0.7467 | 0.6583 | 0.8839 | 0.9872 | 0.9572 | 0.8728 | 0.9472 |
| 4 | **A2MS-DefectNet-S** | dsmonet_resnet_pascal_pige_a2ms_s.pkl | **0.8690** | 0.9918 | 0.7097 | 0.6084 | 0.8751 | 0.9884 | 0.9499 | 0.8602 | 0.9685 |
| 5 | PIDNet-S | fcn_pascal_pige_pid_s.pkl | 0.8620 | 0.9894 | 0.7100 | 0.5971 | 0.8643 | 0.9852 | 0.9436 | 0.8817 | 0.9248 |
| 6 | A2MS-DefectNet-B | dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl | 0.8607 | 0.9902 | 0.7326 | 0.5445 | 0.8841 | 0.9884 | 0.9660 | 0.8524 | 0.9276 |
| 7 | Base-S | dsmonet_resnet_pascal_pige_dsmor18_0228_80000.pkl | 0.8586 | 0.9907 | 0.6953 | 0.5916 | 0.8555 | 0.9865 | 0.9429 | 0.8474 | 0.9590 |
| — | PP-LiteSeg-B | (缺权重) | — | — | — | — | — | — | — | — | — |
| — | BiSeNetV2-L | (缺代码+权重) | — | — | — | — | — | — | — | — | — |
| — | Sub-region UNet | (缺Leather权重) | — | — | — | — | — | — | — | — | — |

## 关键发现

### 1. Base-B (0127) 是最佳模型 (mIoU=0.8991)
- 第二名 STDC1-Seg (0.8824)，差距 1.67pp
- Base-B (0127) 在所有类别上表现均衡，open_wound IoU=0.7648（所有模型最高），scratch IoU=0.7005（所有模型最高）
- Base-B (0127) 在 rotten_surface 上达到 0.9159，远超所有其他模型

### 2. A2MS-DefectNet-B 不是 Leather 最佳模型
- A2MS-B (0.8607) 排第 5，落后 Base-B 3.84pp
- A2MS-S (0.8690) 排第 3，表现优于 A2MS-B
- A2MS-B 的 scratch IoU=0.5445 是所有模型最低，这是主要短板
- 论文中声称 A2MS-B=91.0 > Base-B=89.2 的结论**在当前 test.txt 上不成立**

### 3. 与历史声称值严重不一致

| 模型 | 历史声称 | 当前 test mIoU | 差距 | 定性 |
|---|---:|---:|---:|---|
| Base-B | 89.2 | **89.91** (0127) | +0.71 | ✅ 一致（使用 0127 权重） |
| A2MS-B | 91.0 | **86.07** | -4.93 | ❌ 严重高估 |
| Base-S | 88.1 | **85.86** | -2.24 | ⚠️ 略高 |
| A2MS-S | 89.7 | **86.90** | -2.80 | ⚠️ 略高 |

### 4. 各模型的最难/最易类别

| 模型 | 最难类别 (IoU) | 最易类别 (IoU) |
|---|---|---|
| Base-B (0127) | scratch (0.7005) | hole (0.9907) |
| STDC1-Seg | scratch (0.6300) | background (0.9925) |
| A2MS-S | scratch (0.6084) | background (0.9918) |
| A2MS-B | scratch (0.5445) | background (0.9902) |
| Base-S | scratch (0.5916) | background (0.9907) |
| PIDNet-S | scratch (0.5971) | background (0.9894) |

所有模型中 scratch（刺刮伤）都是最难类别，hole（破洞）是最易类别之一。

### 5. 缺失评估

- **DDRNet23slim**：评估脚本使用错误的模型文件（model_dsmo_ddr_s.py 而非 model_ddr.py），需修复后重新评估
- **PP-LiteSeg-B**：缺少 Leather 权重
- **BiSeNetV2-L**：缺少代码
- **Sub-region UNet**：缺少 Leather 权重

## 输出文件

| 文件 | 路径 |
|---|---|
| 完整 CSV | `experiments/results/class_iou_results.csv` (rows 2-91) |
| 本摘要 | `experiments/results/leather_unified_eval_all_summary.md` |
