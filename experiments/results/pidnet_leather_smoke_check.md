# PIDNet-S Leather Smoke Check

> Date: 2026-06-15  
> Scope: PIDNet-S model entry, checkpoint load, and forward feasibility check.  
> Restrictions followed: no training, no paper edits, no weight overwrite, no full PIDNet evaluation in this round.

## 1. Model and Weight

| Item | Value |
|---|---|
| Model name | PIDNet-S |
| Model file | `/workspace/Industrial Surface Defect/new/pid.py` |
| Weight | `/workspace/Industrial Surface Defect/new/model_savePath/fcn_pascal_pige_pid_s.pkl` |
| Training script evidence | `/workspace/Industrial Surface Defect/new/train_pige_pid.py` |
| Training-time entry | `PIDNet(m=2, n=3, num_classes=8, planes=32, ppm_planes=96, head_planes=128, augment=True)` |
| Checkpoint keys | `best_iou`, `epoch`, `model_state`, `optimizer_state`, `scheduler_state` |

## 2. Load/Forward Check

| Check | Result |
|---|---|
| Strict load with training-matched `augment=True` entry | Success |
| Dummy input | `[1, 3, 768, 768]` |
| Forward in eval mode | Success |
| Forward output shape | `[1, 8, 768, 768]` |
| Strict load with `augment=False` prediction entry | Failed, expected because auxiliary-head parameters differ |

## 3. Evaluation Readiness

PIDNet-S can enter the next Leather/Pige unified evaluation round using:

```bash
python3 tools/evaluate_class_iou.py --dataset leather --test-list experiments/protocol/leather_valid_test_list.txt --model-name PIDNet-S --model-file '/workspace/Industrial Surface Defect/new/pid.py' --weight '/workspace/Industrial Surface Defect/new/model_savePath/fcn_pascal_pige_pid_s.pkl' --num-classes 8 --input-size 768 768 --output-csv experiments/results/smoke_pidnet_leather_class_iou.csv --save-log experiments/logs/smoke_pidnet_leather.log
```

The current unified evaluator already uses the training-matched `augment=True` entry and calls `model.eval()`, so forward returns the main segmentation output only.

## 4. Status

PIDNet-S is **ready for next-round full Leather/Pige evaluation**. No full PIDNet evaluation was run in this round because the task requested only feasibility unless already required.

