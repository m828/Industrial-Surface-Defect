# Evaluation Protocol Lock After Smoke Test

> Date: 2026-06-15  
> Scope: Experiment protocol synchronization after Leather/Pige split audit and Base-B Leather smoke test.  
> Restrictions followed: no training, no paper edits, no weight overwrite, no update to `fixed_existing_results.md`.

## 1. Leather/Pige Test Split Decision

Leather/Pige test set is officially locked as **468 valid samples** for subsequent unified evaluation.

Evidence:

- `/workspace/Industrial Surface Defect/new (copy)/dataset/pige/test.txt` contains 468 total lines.
- All 468 lines are non-empty.
- No duplicate full rows, duplicate image paths, duplicate label paths, or malformed rows were found.
- Every listed image and label exists on disk.
- `/workspace/Industrial Surface Defect/new/dataset/pige/test.txt` and `/workspace/Industrial Surface Defect/new (copy)/dataset/pige/test.txt` are content-identical. The `new/dataset` path resolves to the same dataset tree through a symlink.
- The audited valid list is stored at `experiments/protocol/leather_valid_test_list.txt`.

## 2. Locked Leather/Pige Evaluation Protocol

| Item | Locked Value |
|---|---|
| Dataset | Leather/Pige |
| Test list | `experiments/protocol/leather_valid_test_list.txt` |
| Effective test samples | 468 |
| Number of classes | 8 |
| mIoU includes background | Yes |
| Input size | 768 x 768 |
| Preprocessing | TestRescale-equivalent resize + ToTensor(/255) |
| ImageNet Normalize | Not used |
| Label remapping | Not used |
| Main evaluation script | `tools/evaluate_class_iou.py` |

## 3. Current Verified Smoke Baseline

Base-B Leather with the 0127 checkpoint has passed the unified smoke test under this protocol:

- Weight: `/workspace/Industrial Surface Defect/new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl`
- mIoU including background: **0.899120**
- Test samples: 468
- CSV: `experiments/results/smoke_base_b_leather_class_iou.csv`
- Log: `experiments/logs/smoke_base_b_leather.log`

This is a verified candidate result, not a final paper-table result.

## 4. Use Rules

- All later Leather/Pige model comparisons must use `experiments/protocol/leather_valid_test_list.txt`.
- Do not mix Base-B 0126 and 0127 checkpoint results.
- Do not treat val/best_iou as a test result.
- Do not write smoke-test results into `fixed_existing_results.md`.
- Results should enter the paper only after full same-protocol comparison and manual approval.

