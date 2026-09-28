# Leather 统一评估测试结果

> 评估日期：2026-05-26
> 评估协议：eval_protocol_lock.md
> 测试集：`new (copy)/dataset/pige/test.txt` (468 张, 768×768, 8 类)
> 脚本：`Industrial-Surface-Defect/tools/evaluate_class_iou.py`
> GPU：NVIDIA A100-PCIE-40GB

## 评估结果

### Base-B (权重: dsmonet_resnet_pascal_pige_dsmor50_0126.pkl)

| 指标 | 值 |
|---|---|
| **mIoU** | **0.8806** |
| background | 0.9915 |
| open_wound | 0.7467 |
| scratch | 0.6583 |
| brand_mark | 0.8839 |
| hole | 0.9872 |
| skin_disease | 0.9572 |
| rotten_surface | 0.8728 |
| wart | 0.9472 |

- Checkpoint best_iou: 0.8723
- Test mIoU vs ckpt: +0.0083 ✅ 一致

### A2MS-DefectNet-B (权重: dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl)

| 指标 | 值 |
|---|---|
| **mIoU** | **0.8607** |
| background | 0.9902 |
| open_wound | 0.7326 |
| scratch | 0.5445 |
| brand_mark | 0.8841 |
| hole | 0.9884 |
| skin_disease | 0.9660 |
| rotten_surface | 0.8524 |
| wart | 0.9276 |

- Checkpoint best_iou: 0.9091
- Test mIoU vs ckpt: **-0.0484** ❌ 显著差异

### STDC1-Seg (权重: sdtdcnet_pige_stdc2_pige.pkl)

| 指标 | 值 |
|---|---|
| **mIoU** | **0.8824** |
| background | 0.9925 |
| open_wound | 0.7469 |
| scratch | 0.6300 |
| brand_mark | 0.8843 |
| hole | 0.9873 |
| skin_disease | 0.9525 |
| rotten_surface | 0.8955 |
| wart | 0.9701 |

## 关键发现

### 1. A2MS-B test mIoU (0.8607) vs checkpoint best_iou (0.9091)

差距 -4.84pp。可能原因：
- 训练时验证集 (val.txt 234 张) 与测试集 (test.txt 468 张) 分布不同
- checkpoint best_iou 是在 val.txt 上的 best mIoU，而 test.txt 可能包含更难样本
- 过拟合到 val set

**A2MS-B 的 test mIoU 0.8607 远低于论文声称的 91.0。**

### 2. Base-B 一致性良好

Test mIoU 0.8806 vs ckpt best_iou 0.8723，差距 +0.0083，在正常范围内。

### 3. 与历史声称值对比

| 模型 | 历史声称 | 当前 test mIoU | 差距 |
|---|---:|---:|---|
| Base-B | 89.2 | 88.06 | -1.14pp |
| A2MS-B | 91.0 | 86.07 | **-4.93pp** |

A2MS-B 差异最大。Base-B 差异在可接受范围内。

### 4. 需要评估的剩余模型

| 模型 | 权重 | 需要操作 |
|---|---|---|
| Base-B (0127) | `new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl` | 更新 evaluate_class_iou.py MODEL_CONFIGS |
| Base-S | `new/model_savePath/dsmonet_resnet_pascal_pige_dsmor18_0228_80000.pkl` | 更新 evaluate_class_iou.py MODEL_CONFIGS |
| A2MS-S | `new/model_savePath/dsmonet_resnet_pascal_pige_a2ms_s.pkl` | 更新 evaluate_class_iou.py MODEL_CONFIGS |
| DDRNet23slim | `new/model_savePath/ddr_pascal_pige_ddr23s.pkl` | 更新 evaluate_class_iou.py MODEL_CONFIGS |
| PIDNet-S | `new/model_savePath/fcn_pascal_pige_pid_s.pkl` | 更新 evaluate_class_iou.py MODEL_CONFIGS |
