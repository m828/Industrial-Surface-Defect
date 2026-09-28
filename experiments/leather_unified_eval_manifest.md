# Leather 统一评估清单 (Leather Unified Evaluation Manifest)

> 生成日期：2026-05-22
> 评估协议：eval_protocol_lock.md (Leather: test.txt, 467张, 768×768, 8类)
> 权重来源说明：不同权重来自不同训练运行，best_iou_in_ckpt 是该训练运行验证集上的最佳 mIoU

---

## 一、统一评估清单

### 必须评估的 7 个模型

| # | 论文模型名 | 模型定义文件 | 模型入口类 | 最佳候选权重路径 | 权重来源 | 可加载 | best_iou_in_ckpt | 备注 |
|---|---:|---|---:|---|---|---:|---|
| 1 | Base-B | `new/model_dsmo_rs50.py` | DSMONet(num_classes=8, backbone=resnet50()) | `new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl` | RTX 3090 旧训练 | ✅ | 0.8930 | **接近历史 89.2，推荐作为论文权重** |
| 2 | Base-B (备选) | same | same | `new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0126.pkl` | RTX 3090 旧训练 | ✅ | 0.8723 | 旧评估 (evaluate_class_iou.py) mIoU=0.8806 |
| 3 | Base-B (新训) | same | same | `new/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0220ksh.pkl` | A100 新训练 | ✅ | 0.8278 | 新训练结果较低 |
| 4 | Base-S | `new/model_dsmo_rs18.py` | DSMONet(num_classes=8, backbone=resnet18()) | `new/model_savePath/dsmonet_resnet_pascal_pige_dsmor18_0228_80000.pkl` | A100 新训练 | ✅ | 0.8397 (训练日志) | 新训练，需评估脚本验证 |
| 5 | A2MS-DefectNet-B | `new/model_dsmo_rs50_eSE_adapt_detailloss.py` | DSMONet(num_classes=8, backbone=resnet50()) | `new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl` | RTX 3090 旧训练 | ✅ | 0.9091 | **接近历史 91.0，推荐作为论文权重** |
| 6 | A2MS-DefectNet-B (备选) | `new/model_dsmo822.py` | DSMONet(num_classes=8, backbone=resnet50()) | `new/model_savePath/dsmonet_resnet_pascal_augnew_51500.pkl` | A100 新训练 | ✅ | 0.8560 | 不同模型架构 (model_dsmo822)，非 A2MS-B |
| 7 | A2MS-DefectNet-S | `new/model_dsmo_rs18_eSE_adapt_detailloss_822.py` | DSMONet(num_classes=8, backbone=resnet18()) | `new/model_savePath/dsmonet_resnet_pascal_pige_a2ms_s.pkl` | A100 新训练 | ✅ | 0.8472 (训练日志) | 新训练，需评估脚本验证 |
| 8 | DDRNet23slim | `new/model_ddr.py` | DualResNet_imagenet(num_classes=8) | `new/model_savePath/ddr_pascal_pige_ddr23s.pkl` | A100 新训练 | ✅ | 0.8317 (训练日志) | 新训练 |
| 9 | PIDNet-S | `new/pid.py` | PIDNet | `new/model_savePath/fcn_pascal_pige_pid_s.pkl` | A100 新训练 | ✅ | 0.8406 (训练日志) | 新训练 |
| 10 | STDC1-Seg | `new/stdcnet.py` | STDC1 | `new (copy)/model_savePath/sdtdcnet_pige_stdc2_pige.pkl` | 旧训练 | ⚠️ 待确认 | — | 文件名含 stdc2 但标记为 STDC1-Seg；旧评估 mIoU=0.8824 |

---

## 二、权重架构兼容性矩阵

| 权重文件 | model_dsmo_rs50 (Base-B) | model_dsmo_rs50_eSE_adapt_detailloss (A2MS-B) | model_dsmo822 (DSMONet822) | model_dsmo_rs50_eSE_adapt (无 detailloss) |
|---|---:|---:|---:|---:|
| 0126 (336MB) | ✅ | ❌ | N/A | N/A |
| 0127 (336MB) | ✅ | ❌ | N/A | N/A |
| 0220ksh (337MB) | ✅ | ❌ | N/A | N/A |
| 160000 (338MB) | N/A | ✅ | N/A | N/A |
| augnew_* 系列 (336MB) | N/A | ❌ (squeeze_body_edge mismatch) | ✅ | N/A |
| 0120_eSE_adapt (337MB) | N/A | ❌ (seg_head_detailloss missing) | N/A | ✅ |
| 0228/0229 (337MB) | N/A | ❌ | ❌ (seg_head_detailloss unexpected) | N/A |

**关键发现**：
- `new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl` 的 best_iou_in_ckpt = 0.9091，非常接近历史声称的 91.0
- `new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl` 的 best_iou_in_ckpt = 0.8930，接近历史 89.2
- augnew 系列是 **model_dsmo822.py**（不同架构），不是论文定义的 A2MS-DefectNet-B

---

## 三、Base-B 和 A2MS-DefectNet-B 优先评估

### 评估计划

先评估以下两个模型，验证评估脚本和协议：

1. **Base-B** → 权重 `new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl`, dest_iou=0.8930
2. **A2MS-DefectNet-B** → 权重 `new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl`, dest_iou=0.9091

### 评估命令模板

```bash
cd "/workspace/Industrial Surface Defect/Industrial-Surface-Defect"
source /opt/conda/etc/profile.d/conda.sh && conda activate base

# Base-B Leather test evaluation
python tools/evaluate_class_iou.py \
    --model_type dsmors50 \
    --weight_path "/workspace/Industrial Surface Defect/new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl" \
    --test_txt "/workspace/Industrial Surface Defect/new (copy)/dataset/pige/test.txt" \
    --input_size 768 768 \
    --num_classes 8 \
    --output_dir "experiments/results/leather_unified_eval/base_b_0127"

# A2MS-B Leather test evaluation
python tools/evaluate_class_iou.py \
    --model_type dsmors50_eSE_adapt_detailloss \
    --weight_path "/workspace/Industrial Surface Defect/new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl" \
    --test_txt "/workspace/Industrial Surface Defect/new (copy)/dataset/pige/test.txt" \
    --input_size 768 768 \
    --num_classes 8 \
    --output_dir "experiments/results/leather_unified_eval/a2ms_b_160000"
```

[需人工确认] — `tools/evaluate_class_iou.py` 的确切参数名和 `model_type` 的合法取值。

---

## 四、其他候选权重（非标准模型，仅供参考）

以下权重使用不同的模型架构，不纳入论文主实验表，但可作为诊断参考：

| 权重文件 | 模型 | mIoU | 备注 |
|---|---|---|---|
| `new (copy)/model_savePath/dsmonet_resnet_pascal_0120_pige_eSE_adapt.pkl` | eSE+Adapt (无 detailloss) | 0.8894 (ckpt) | AAM 消融中间点 |
| `new/model_savePath/dsmonet_resnet_pascal_augnew_51500.pkl` | DSMONet822 | 0.8560 (ckpt) | 不同架构，非 A2MS-B |
| `new/model_savePath/dsmonet_resnet_pascal_augnew_60000.pkl` | DSMONet822 | 0.8371 (ckpt) | 不同架构 |
