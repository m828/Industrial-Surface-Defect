# DDRNet23slim Leather Smoke Check

> Date: 2026-06-15  
> Scope: DDRNet23slim load/forward feasibility check, followed by full Leather/Pige smoke test because the check passed.  
> Restrictions followed: no training, no paper edits, no weight overwrite, no update to `fixed_existing_results.md`.

## 1. Model and Weight

| Item | Value |
|---|---|
| Model name | DDRNet23slim |
| Model file | `/workspace/Industrial Surface Defect/new/model_ddr.py` |
| Model entry | `DualResNet_imagenet(num_classes=8)` |
| Training script evidence | `/workspace/Industrial Surface Defect/new/train_pige_ddr.py` uses `DualResNet_imagenet(num_classes=n_classes)` |
| Weight | `/workspace/Industrial Surface Defect/new/model_savePath/ddr_pascal_pige_ddr23s.pkl` |
| Checkpoint keys | `best_iou`, `epoch`, `model_state`, `optimizer_state`, `scheduler_state` |

## 2. Load/Forward Check

| Check | Result |
|---|---|
| Strict checkpoint load | Success |
| Dummy input | `[1, 3, 768, 768]` |
| Forward | Success |
| Forward output shape | `[1, 8, 768, 768]` |

DDRNet23slim is compatible with the current unified evaluator.

## 3. Full Smoke Test

Because load/forward passed, a full Leather/Pige 468-sample smoke test was run.

| Item | Value |
|---|---|
| Dataset | Leather/Pige |
| Test list | `experiments/protocol/leather_valid_test_list.txt` |
| Effective samples | 468 |
| Input size | 768 x 768 |
| Classes | 8, including background |
| mIoU including background | **0.842052** |
| Output CSV | `experiments/results/smoke_ddrnet_leather_class_iou.csv` |
| Log | `experiments/logs/smoke_ddrnet_leather.log` |

## 4. Evaluation Command

```bash
cd "/workspace/Industrial Surface Defect/Industrial-Surface-Defect"
python3 tools/evaluate_class_iou.py --dataset leather --test-list experiments/protocol/leather_valid_test_list.txt --model-name DDRNet23slim --model-file '/workspace/Industrial Surface Defect/new/model_ddr.py' --weight '/workspace/Industrial Surface Defect/new/model_savePath/ddr_pascal_pige_ddr23s.pkl' --num-classes 8 --input-size 768 768 --output-csv experiments/results/smoke_ddrnet_leather_class_iou.csv --save-log experiments/logs/smoke_ddrnet_leather.log
```

## 5. Status

DDRNet23slim can enter the next Leather full-model unified evaluation round. The current smoke-test value is traceable, but should not be used as a final paper result until the full comparison set is locked.

