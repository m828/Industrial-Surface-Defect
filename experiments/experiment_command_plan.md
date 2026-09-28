# 实验重跑命令计划

> 生成日期：2026-05-22
> 状态：命令模板，部分需人工确认后执行
> 注意：标注 `[需人工确认]` 的命令不可直接运行

---

## 通用前提

### 工作目录
```bash
# 所有命令基于以下工作目录：
WORK_DIR="/workspace/Industrial Surface Defect/new"
cd "$WORK_DIR"
```

### 数据集路径
```bash
# NEU-Seg
NEU_TRAIN="./dataset/train_neu.txt"        # symlink → new (copy)/dataset/train_neu.txt
NEU_TEST="./dataset/test_neu.txt"          # symlink → new (copy)/dataset/test_neu.txt

# Leather/Pige
PIGE_TRAIN="./dataset/pige/train.txt"      # symlink → new (copy)/dataset/pige/train.txt
PIGE_VAL="./dataset/pige/val.txt"          # symlink → new (copy)/dataset/pige/val.txt
PIGE_TEST="./dataset/pige/test.txt"        # symlink → new (copy)/dataset/pige/test.txt
```

### 权重基础路径
```bash
WEIGHT_DIR="$WORK_DIR/model_savePath"
```

---

## R1. A2MS-DefectNet-B NEU-Seg 重新训练

### 背景
当前权重 `dsmonet_resnet_pascal_0109_neu_eSE_adapt_datailloss.pkl` 的 mIoU=71.31，严重异常。

### 方案 A：使用现有脚本重新训练
```bash
cd "/workspace/Industrial Surface Defect/new"
python train_neu_resnet_detailloss.py \
    --config dsmonet_resnet_detailloss.yml \
    2>&1 | tee runs_other/a2ms_b_neu_retrain.log
```
**注意**：[需人工确认] 此命令使用的配置和脚本与异常训练相同，需先排查配置问题后再运行。

### 方案 B：排查异常原因后修改配置重新训练
```bash
# 步骤 1：对比异常训练日志和正常训练日志
cd "/workspace/Industrial Surface Defect/new"
diff <(grep "Loss:" runs_other/train_all/a2ms_b_neu.log | tail -20) \
     <(grep "Loss:" runs_other/train_all/base_b_neu.log | tail -20)

# 步骤 2：检查 detailloss 配置
cat dsmonet_resnet_detailloss.yml | grep -A 20 "loss"

# 步骤 3：[需人工确认] 根据排查结果调整配置后重新训练
# python train_neu_resnet_detailloss.py --config <修改后的配置>
```

### 方案 C：使用旧脚本和新种子重新训练
```bash
# [需人工确认] 需要确认旧脚本路径和参数
cd "/workspace/Industrial Surface Defect/new (copy)"
# python train_neu_resnet_detailloss.py --config ... 
```

---

## R2. Leather 数据集统一评估

### 背景
所有 Leather 模型需在同一测试集 (test.txt, 467 张) 上使用统一预处理评估。

### 模型评估命令模板

```bash
cd "/workspace/Industrial Surface Defect/Industrial-Surface-Defect"

# --- Base-B (使用旧权重 0126, 历史评估 mIoU=88.06) ---
python tools/evaluate_class_iou.py \
    --model_type dsmors50 \
    --weight_path "/workspace/Industrial Surface Defect/new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0126.pkl" \
    --test_txt "/workspace/Industrial Surface Defect/new (copy)/dataset/pige/test.txt" \
    --input_size 768 768 \
    --num_classes 8 \
    --output_dir "experiments/results/eval/base_b_leather_0126"

# --- Base-B (使用新权重 0220ksh, 新训练 mIoU=82.78) ---
python tools/evaluate_class_iou.py \
    --model_type dsmors50 \
    --weight_path "/workspace/Industrial Surface Defect/new/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0220ksh.pkl" \
    --test_txt "/workspace/Industrial Surface Defect/new (copy)/dataset/pige/test.txt" \
    --input_size 768 768 \
    --num_classes 8 \
    --output_dir "experiments/results/eval/base_b_leather_0220ksh"

# --- A2MS-B (使用旧权重 160000, 大小 338MB) ---
python tools/evaluate_class_iou.py \
    --model_type dsmors50_eSE_adapt_detailloss \
    --weight_path "/workspace/Industrial Surface Defect/new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl" \
    --test_txt "/workspace/Industrial Surface Defect/new (copy)/dataset/pige/test.txt" \
    --input_size 768 768 \
    --num_classes 8 \
    --output_dir "experiments/results/eval/a2ms_b_leather_160000"

# --- A2MS-B (使用新权重 augnew, 新训练 mIoU=85.60) ---
python tools/evaluate_class_iou.py \
    --model_type dsmors50_eSE_adapt_detailloss \
    --weight_path "/workspace/Industrial Surface Defect/new/model_savePath/dsmonet_resnet_pascal_augnew_60000.pkl" \
    --test_txt "/workspace/Industrial Surface Defect/new (copy)/dataset/pige/test.txt" \
    --input_size 768 768 \
    --num_classes 8 \
    --output_dir "experiments/results/eval/a2ms_b_leather_augnew"

# --- Base-S ---
python tools/evaluate_class_iou.py \
    --model_type dsmors18 \
    --weight_path "/workspace/Industrial Surface Defect/new/model_savePath/dsmonet_resnet_pascal_pige_dsmor18_0228_80000.pkl" \
    --test_txt "/workspace/Industrial Surface Defect/new (copy)/dataset/pige/test.txt" \
    --input_size 768 768 \
    --num_classes 8 \
    --output_dir "experiments/results/eval/base_s_leather"

# --- A2MS-S ---
python tools/evaluate_class_iou.py \
    --model_type dsmors18_eSE_adapt_detailloss \
    --weight_path "/workspace/Industrial Surface Defect/new/model_savePath/dsmonet_resnet_pascal_pige_a2ms_s.pkl" \
    --test_txt "/workspace/Industrial Surface Defect/new (copy)/dataset/pige/test.txt" \
    --input_size 768 768 \
    --num_classes 8 \
    --output_dir "experiments/results/eval/a2ms_s_leather"

# --- DDRNet23slim ---
python tools/evaluate_class_iou.py \
    --model_type ddr \
    --weight_path "/workspace/Industrial Surface Defect/new/model_savePath/ddr_pascal_pige_ddr23s.pkl" \
    --test_txt "/workspace/Industrial Surface Defect/new (copy)/dataset/pige/test.txt" \
    --input_size 768 768 \
    --num_classes 8 \
    --output_dir "experiments/results/eval/ddr_leather"

# --- PIDNet-S ---
python tools/evaluate_class_iou.py \
    --model_type pid \
    --weight_path "/workspace/Industrial Surface Defect/new/model_savePath/fcn_pascal_pige_pid_s.pkl" \
    --test_txt "/workspace/Industrial Surface Defect/new (copy)/dataset/pige/test.txt" \
    --input_size 768 768 \
    --num_classes 8 \
    --output_dir "experiments/results/eval/pid_s_leather"
```

**⚠️ 重要**：以上命令中的 `--model_type` 参数名称和取值需要人工确认。`evaluate_class_iou.py` 的实际参数可能与模板不同。请先运行 `python tools/evaluate_class_iou.py --help` 确认参数。

### 简化方案：从训练日志手动提取最佳 checkpoint 的 per-class IoU

```bash
# 每个模型的训练日志中已有 per-class IoU，可直接提取最佳 checkpoint 的值
cd "/workspace/Industrial Surface Defect/new"

# 示例：提取 Base-B Leather 的 best checkpoint per-class IoU
grep -A 20 "Mean IoU" runs_other/train_all/base_b_pige.log | \
    grep -E "Cls|IoU" | \
    sort -t: -k2 -rn | head -20
```

---

## R3. 所有模型 Per-Class IoU 批量重跑

```bash
cd "/workspace/Industrial Surface Defect/Industrial-Surface-Defect"

# [需人工确认] 如果 evaluate_class_iou.py 支持批量模式：
# python tools/evaluate_class_iou.py --batch --config experiments/configs/batch_eval_leather.yml

# 否则逐个评估（参考 R2 中的命令模板）
```

---

## R4. FPS 在 RTX 3090 上重新测量

### 方案 A：在 RTX 3090 机器上运行现有脚本
```bash
# [需人工确认] 需要在 RTX 3090 机器上执行
cd "/workspace/Industrial Surface Defect/Industrial-Surface-Defect"
python experiments/scripts/complexity_stats.py \
    --device cuda:0 \
    --warmup 50 \
    --runs 200 \
    --batch_size 1 \
    --output experiments/results/complexity_results_rtx3090.csv
```

### 方案 B：修改论文标注为 A100
不需要重新测量，但需更新 `fixed_existing_results.md` 和论文正文中所有 FPS 的平台标注。

---

## R5. 小目标/细长缺陷分组评价重跑

```bash
cd "/workspace/Industrial Surface Defect/Industrial-Surface-Defect"

# [需人工确认参数名称]
python tools/evaluate_small_object_groups.py \
    --test_txt "/workspace/Industrial Surface Defect/new (copy)/dataset/pige/test.txt" \
    --num_classes 8 \
    --input_size 768 768 \
    --models \
        "Base-B:/workspace/Industrial Surface Defect/new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0126.pkl:dsmors50" \
        "Base-S:/workspace/Industrial Surface Defect/new/model_savePath/dsmonet_resnet_pascal_pige_dsmor18_0228_80000.pkl:dsmors18" \
        "A2MS-B:/workspace/Industrial Surface Defect/new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl:dsmors50_eSE_adapt_detailloss" \
        "A2MS-S:/workspace/Industrial Surface Defect/new/model_savePath/dsmonet_resnet_pascal_pige_a2ms_s.pkl:dsmors18_eSE_adapt_detailloss" \
        "DDRNet23slim:/workspace/Industrial Surface Defect/new/model_savePath/ddr_pascal_pige_ddr23s.pkl:ddr" \
        "PIDNet-S:/workspace/Industrial Surface Defect/new/model_savePath/fcn_pascal_pige_pid_s.pkl:pid" \
        "STDC1-Seg:/workspace/Industrial Surface Defect/new (copy)/model_savePath/sdtdcnet_pige_stdc2_pige.pkl:stdc" \
    --output experiments/results/small_object_group_results_v2.csv \
    --save_sample_ids
```

---

## R6. 可视化结果重新生成

```bash
cd "/workspace/Industrial Surface Defect/Industrial-Surface-Defect"

# [需人工确认] 可视化生成脚本路径和参数
# python tools/generate_qualitative_figures.py \
#     --test_txt "/workspace/Industrial Surface Defect/new (copy)/dataset/pige/test.txt" \
#     --model_a "Base-B" --weight_a "..." \
#     --model_b "A2MS-B" --weight_b "..." \
#     --model_c "STDC1-Seg" --weight_c "..." \
#     --scenarios small_target elongated multi_defect texture_confusion base_miss base_boundary \
#     --num_per_scenario 5 \
#     --include_failure_cases \
#     --output_dir "figures/qualitative_results_v2/"
```

---

## R7. 对比方法基线重建

### FCN
```bash
cd "/workspace/Industrial Surface Defect/new"
# [需人工确认] 需要确认配置文件名和参数
python train_pige_fcn.py --config <fcn_config>.yml
```

### U-Net
```bash
cd "/workspace/Industrial Surface Defect/new"
python train_pige_unet.py --config unet_pascal.yml
```

### DeepLabV3+
```bash
cd "/workspace/Industrial Surface Defect/new"
# [需人工确认] 需要确认配置文件名
python train_pige_deeplabv3.py --config <deeplab_config>.yml
```

### PSPNet
```bash
cd "/workspace/Industrial Surface Defect/new"
# [需人工确认] 模型使用 new (copy)/model/pspnet.py
python train_pige_psp.py --config <psp_config>.yml
```

### ENet
```bash
cd "/workspace/Industrial Surface Defect/new"
# [需人工确认] 模型使用 new (copy)/model/enet.py
python train_pige_enet.py --config <enet_config>.yml
```

### STDC-Seg
```bash
cd "/workspace/Industrial Surface Defect/new"
python train_pige_stdc.py --config <stdc_config>.yml
# [需人工确认] 需要确认 STDC1 和 STDC2 的配置区别
```

---

## R8-R10. 消融实验中间点训练

### eSE×1 (AAM 消融)
```bash
cd "/workspace/Industrial Surface Defect/new"
# [需人工确认] 使用 model_dsmo_rs50_eSE.py, 需要专门配置文件
# python train_neu_resnet_detailloss.py \
#     --config config_neu_eSE_x1.yml \
#     2>&1 | tee runs_other/ablation/eSE_x1_neu.log
```

### AAM×1 (AAM 消融)
```bash
cd "/workspace/Industrial Surface Defect/new"
# [需人工确认] 使用 model_dsmo_rs50_eSE_adapt.py, 需要专门配置文件
# python train_neu_resnet_detailloss.py \
#     --config config_neu_AAM_x1.yml \
#     2>&1 | tee runs_other/ablation/AAM_x1_neu.log
```

### 联合损失消融 (l0+l1+l2 / +ledge / +lmask)
```bash
cd "/workspace/Industrial Surface Defect/new"
# [需人工确认] 需要为每个消融点创建专门配置文件
# 禁用特定损失模块: 修改 loss 配置 list 或设置权重为 0
# 例如 l0+l1+l2 只有三项损失:
# python train_neu_resnet_detailloss.py \
#     --config config_neu_l012_only.yml \
#     2>&1 | tee runs_other/ablation/l012_neu.log
```

---

## 一键批量评估脚本模板

```bash
#!/bin/bash
# 保存为 experiments/scripts/batch_evaluate_leather.sh
# [需人工确认] 所有参数和路径

set -e

WORK_DIR="/workspace/Industrial Surface Defect/Industrial-Surface-Defect"
TEST_TXT="/workspace/Industrial Surface Defect/new (copy)/dataset/pige/test.txt"
OUTPUT_DIR="experiments/results/batch_eval_$(date +%Y%m%d)"
NUM_CLASSES=8
INPUT_H=768
INPUT_W=768

cd "$WORK_DIR"
mkdir -p "$OUTPUT_DIR"

declare -A MODELS
MODELS=(
    ["Base-B_0126"]="dsmors50:/workspace/Industrial Surface Defect/new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0126.pkl"
    ["Base-S"]="dsmors18:/workspace/Industrial Surface Defect/new/model_savePath/dsmonet_resnet_pascal_pige_dsmor18_0228_80000.pkl"
    ["A2MS-B_160000"]="dsmors50_eSE_adapt_detailloss:/workspace/Industrial Surface Defect/new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl"
    ["A2MS-S"]="dsmors18_eSE_adapt_detailloss:/workspace/Industrial Surface Defect/new/model_savePath/dsmonet_resnet_pascal_pige_a2ms_s.pkl"
    ["DDRNet23slim"]="ddr:/workspace/Industrial Surface Defect/new/model_savePath/ddr_pascal_pige_ddr23s.pkl"
    ["PIDNet-S"]="pid:/workspace/Industrial Surface Defect/new/model_savePath/fcn_pascal_pige_pid_s.pkl"
    ["STDC1-Seg"]="stdc:/workspace/Industrial Surface Defect/new (copy)/model_savePath/sdtdcnet_pige_stdc2_pige.pkl"
)

for model_name in "${!MODELS[@]}"; do
    IFS=':' read -r model_type weight_path <<< "${MODELS[$model_name]}"
    echo "=== Evaluating $model_name ==="
    python tools/evaluate_class_iou.py \
        --model_type "$model_type" \
        --weight_path "$weight_path" \
        --test_txt "$TEST_TXT" \
        --input_size "$INPUT_H" "$INPUT_W" \
        --num_classes "$NUM_CLASSES" \
        --output_dir "$OUTPUT_DIR/$model_name" \
        2>&1 | tee "$OUTPUT_DIR/${model_name}.log"
    echo "=== $model_name done ==="
done

echo "=== ALL DONE ==="
```

---

## 重要提示

1. **所有标注 `[需人工确认]` 的命令不可直接运行**，需要：
   - 检查脚本实际参数名称和取值
   - 确认模型加载方式（`--model_type` 参数映射到哪个模型文件）
   - 确认数据路径是否可访问

2. **批量评估前先测试单个模型**：
   ```bash
   # 先用 Base-B 测试评估脚本是否正常工作
   cd "/workspace/Industrial Surface Defect/Industrial-Surface-Defect"
   python tools/evaluate_class_iou.py --help
   ```

3. **训练前检查 GPU 显存**：
   ```bash
   nvidia-smi
   # Leather 768×768 batch=8 需要约 16-20GB 显存
   # NEU-Seg 200×200 batch=16 需要约 8-12GB 显存
   ```

4. **数据集路径需要确认**：
   - `new/` 下的 `dataset/` 是 symlink 到 `new (copy)/dataset/`
   - 如果 symlink 断裂，需修复或使用绝对路径

5. **结果保存规范**：
   - 评估结果统一保存到 `Industrial-Surface-Defect/experiments/results/` 下
   - 文件名包含模型名、数据集、评估日期
   - 每个评估输出: `.csv` (per-class IoU) + `.json` (完整指标) + `.log` (运行日志)
