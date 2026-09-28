# evaluate_class_iou.py 命令示例

> 生成日期：2026-06-15  
> 状态：新统一单模型 CLI。旧命令计划中的 `--weight_path`、`--test_txt`、`--input_size`、`--num_classes` 已作为兼容别名保留；推荐新命令统一使用短横线参数。

## Base-B Leather Smoke Test

```bash
cd "/workspace/Industrial Surface Defect/Industrial-Surface-Defect"
python tools/evaluate_class_iou.py \
  --dataset leather \
  --test-list experiments/protocol/leather_valid_test_list.txt \
  --model-name Base-B \
  --model-file "/workspace/Industrial Surface Defect/new/model_dsmo_rs50.py" \
  --weight "/workspace/Industrial Surface Defect/new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl" \
  --num-classes 8 \
  --input-size 768 768 \
  --output-csv experiments/results/smoke_base_b_leather_class_iou.csv \
  --save-log experiments/logs/smoke_base_b_leather.log
```

## A2MS-DefectNet-B Leather

```bash
python tools/evaluate_class_iou.py \
  --dataset leather \
  --test-list experiments/protocol/leather_valid_test_list.txt \
  --model-name A2MS-DefectNet-B \
  --model-file "/workspace/Industrial Surface Defect/new/model_dsmo_rs50_eSE_adapt_detailloss.py" \
  --weight "/workspace/Industrial Surface Defect/new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl" \
  --num-classes 8 \
  --input-size 768 768 \
  --output-csv experiments/results/a2ms_b_leather_class_iou.csv \
  --save-log experiments/logs/a2ms_b_leather.log
```

## STDC1-Seg Leather

```bash
python tools/evaluate_class_iou.py \
  --dataset leather \
  --test-list experiments/protocol/leather_valid_test_list.txt \
  --model-name STDC1-Seg \
  --model-file "/workspace/Industrial Surface Defect/new (copy)/stdc.py" \
  --weight "/workspace/Industrial Surface Defect/new (copy)/model_savePath/sdtdcnet_pige_stdc2_pige.pkl" \
  --num-classes 8 \
  --input-size 768 768 \
  --output-csv experiments/results/stdc1_leather_class_iou.csv \
  --save-log experiments/logs/stdc1_leather.log
```

## DDRNet23slim Leather

```bash
python tools/evaluate_class_iou.py \
  --dataset leather \
  --test-list experiments/protocol/leather_valid_test_list.txt \
  --model-name DDRNet23slim \
  --model-file "/workspace/Industrial Surface Defect/new/model_ddr.py" \
  --weight "/workspace/Industrial Surface Defect/new/model_savePath/ddr_pascal_pige_ddr23s.pkl" \
  --num-classes 8 \
  --input-size 768 768 \
  --output-csv experiments/results/ddrnet_leather_class_iou.csv \
  --save-log experiments/logs/ddrnet_leather.log
```

说明：DDRNet loader 已改为 `model_ddr.py::DualResNet_imagenet`。如权重仍加载失败，应记录为权重/结构需单独适配，不影响 Base-B、A2MS-B、STDC1、PIDNet 的评估。

## PIDNet-S Leather

```bash
python tools/evaluate_class_iou.py \
  --dataset leather \
  --test-list experiments/protocol/leather_valid_test_list.txt \
  --model-name PIDNet-S \
  --model-file "/workspace/Industrial Surface Defect/new/pid.py" \
  --weight "/workspace/Industrial Surface Defect/new/model_savePath/fcn_pascal_pige_pid_s.pkl" \
  --num-classes 8 \
  --input-size 768 768 \
  --output-csv experiments/results/pidnet_s_leather_class_iou.csv \
  --save-log experiments/logs/pidnet_s_leather.log
```

## NEU-Seg 示例

NEU-Seg 的 test list 中每行顺序为 `label image`，脚本会在 `--dataset neu` 下自动按该顺序解析。

```bash
python tools/evaluate_class_iou.py \
  --dataset neu \
  --test-list "/workspace/Industrial Surface Defect/new (copy)/dataset/test_neu.txt" \
  --model-name Base-B \
  --model-file "/workspace/Industrial Surface Defect/new/model_dsmo_rs50.py" \
  --weight "/workspace/Industrial Surface Defect/new/model_savePath/dsmonet_resnet_pascal_0110_neu_120000.pkl" \
  --num-classes 4 \
  --input-size 200 200 \
  --output-csv experiments/results/base_b_neu_class_iou.csv \
  --save-log experiments/logs/base_b_neu.log
```

## 兼容旧命令参数

以下旧参数仍可用：

- `--model_type` 等价于 `--model-name`
- `--weight_path` 等价于 `--weight`
- `--test_txt` 等价于 `--test-list`
- `--input_size` 等价于 `--input-size`
- `--num_classes` 等价于 `--num-classes`
- `--output_dir` 会在目录下写入 `class_iou.csv`
