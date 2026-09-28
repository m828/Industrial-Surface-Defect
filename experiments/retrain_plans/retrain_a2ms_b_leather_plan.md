# Retrain A2MS-B Leather Plan

> Date: 2026-06-16  
> This is a preparation plan only. Do not start training without explicit approval.

## 1. Recommendation

Retraining is recommended if Leather/Pige is intended to support the A2MS-DefectNet-B main-result claim. The current A2MS-B Leather checkpoint has locked-test mIoU=0.860749, below Base-B 0127 mIoU=0.899120.

## 2. Splits To Use

Use the locked dataset split:

- train: `/workspace/Industrial Surface Defect/new (copy)/dataset/pige/train.txt`
- val: `/workspace/Industrial Surface Defect/new (copy)/dataset/pige/val.txt`
- final test: `experiments/protocol/leather_valid_test_list.txt` (468 samples)

## 3. Validation vs Test Rule

Validation `best_iou` is only for checkpoint selection. It must not be reported as the paper test result.

The final result must be obtained by running `tools/evaluate_class_iou.py` on the locked 468-sample test list.

## 4. Candidate Training Script and Config

Primary candidate:

```text
/workspace/Industrial Surface Defect/new/train_pige_a2ms_b.py
/workspace/Industrial Surface Defect/new/config_pige_a2ms_b.yml
```

Also audit legacy variants before choosing:

```text
/workspace/Industrial Surface Defect/new/train_pige_resnet_loss904.py
/workspace/Industrial Surface Defect/new/dsmonet_resnet_pige_loss.yml
```

## 5. Checkpoint and Log Saving

Recommended new directory:

```text
/workspace/Industrial Surface Defect/Industrial-Surface-Defect/experiments/retrain_a2ms_b_leather_20260616/
```

Save:

- `checkpoints/a2ms_b_leather_20260616_best_val.pkl`
- `checkpoints/a2ms_b_leather_20260616_last.pkl`
- `logs/train.log`
- `manifest/train_command.txt`
- `manifest/config_used.yml`
- `results/final_test_class_iou.csv`
- `results/final_test_summary.md`

## 6. Avoiding Overfitting

- Monitor train/val mIoU gap.
- Keep validation-only checkpoint selection and never tune on test results.
- Preserve augmentations exactly as configured unless a new experiment branch is declared.
- Compare per-class IoU, especially `scratch`, where current A2MS-B is weak.
- Avoid repeatedly selecting checkpoints based on the locked test set.

## 7. Recommended Command Draft

Do not run yet:

```bash
cd "/workspace/Industrial Surface Defect/new"
python3 train_pige_a2ms_b.py --config config_pige_a2ms_b.yml
```

Before running, redirect checkpoint and log output to the new experiment directory and verify no historical `.pkl` path will be overwritten.

## 8. Final Test Command Draft

After training completes and best-val checkpoint is selected:

```bash
cd "/workspace/Industrial Surface Defect/Industrial-Surface-Defect"
python3 tools/evaluate_class_iou.py --dataset leather --test-list experiments/protocol/leather_valid_test_list.txt --model-name A2MS-DefectNet-B --model-file '/workspace/Industrial Surface Defect/new/model_dsmo_rs50_eSE_adapt_detailloss.py' --weight '<new_best_val_checkpoint>' --num-classes 8 --input-size 768 768 --output-csv experiments/retrain_a2ms_b_leather_20260616/results/final_test_class_iou.csv --save-log experiments/retrain_a2ms_b_leather_20260616/logs/final_test.log
```

## 9. Success Criteria

A retrained Leather A2MS-B checkpoint is useful only if:

- it strictly loads into the declared model entry;
- final test mIoU is traceable on the 468-sample locked test set;
- it improves over the current A2MS-B checkpoint;
- ideally it also exceeds Base-B 0127, or the paper explicitly discusses why it does not.
