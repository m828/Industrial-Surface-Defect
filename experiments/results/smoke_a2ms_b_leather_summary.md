# A2MS-DefectNet-B Leather Smoke Test Summary

> Date: 2026-06-15  
> Scope: single-model smoke test under the locked Leather/Pige 468-sample protocol.  
> Restrictions followed: no training, no paper edits, no weight overwrite, no update to `fixed_existing_results.md`.

## 1. Evaluation Setup

| Item | Value |
|---|---|
| Dataset | Leather/Pige |
| Test list | `experiments/protocol/leather_valid_test_list.txt` |
| Effective test samples | 468 |
| Model | A2MS-DefectNet-B |
| Model file | `/workspace/Industrial Surface Defect/new/model_dsmo_rs50_eSE_adapt_detailloss.py` |
| Model entry | `DSMONet(num_classes=8, backbone=resnet50())` |
| Weight | `/workspace/Industrial Surface Defect/new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl` |
| Input size | 768 x 768 |
| Classes | 8, including background |
| Preprocessing | TestRescale-equivalent resize + ToTensor(/255), no ImageNet Normalize |
| Device | NVIDIA A100-PCIE-40GB |
| Output CSV | `experiments/results/smoke_a2ms_b_leather_class_iou.csv` |
| Log | `experiments/logs/smoke_a2ms_b_leather.log` |

## 2. Evaluation Command

```bash
cd "/workspace/Industrial Surface Defect/Industrial-Surface-Defect"
python3 tools/evaluate_class_iou.py --dataset leather --test-list experiments/protocol/leather_valid_test_list.txt --model-name A2MS-DefectNet-B --model-file '/workspace/Industrial Surface Defect/new/model_dsmo_rs50_eSE_adapt_detailloss.py' --weight '/workspace/Industrial Surface Defect/new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl' --num-classes 8 --input-size 768 768 --output-csv experiments/results/smoke_a2ms_b_leather_class_iou.csv --save-log experiments/logs/smoke_a2ms_b_leather.log
```

## 3. Result

| Metric | Value |
|---|---:|
| mIoU including background | **0.860749** |
| Effective test samples | 468 |
| Weight load | Success |
| Smoke test | Passed |

## 4. Per-Class Metrics

| Class ID | Class | IoU | Dice | Precision | Recall |
|---:|---|---:|---:|---:|---:|
| 0 | background | 0.990210 | 0.995081 | 0.996883 | 0.993286 |
| 1 | open_wound | 0.732640 | 0.845692 | 0.815331 | 0.878401 |
| 2 | scratch | 0.544550 | 0.705124 | 0.743374 | 0.670618 |
| 3 | brand_mark | 0.884139 | 0.938507 | 0.929675 | 0.947508 |
| 4 | hole | 0.988418 | 0.994175 | 0.990202 | 0.998181 |
| 5 | skin_disease | 0.965973 | 0.982692 | 0.976193 | 0.989278 |
| 6 | rotten_surface | 0.852424 | 0.920333 | 0.869350 | 0.977669 |
| 7 | wart | 0.927637 | 0.962460 | 0.935920 | 0.990550 |

## 5. Required Comparison Answers

1. A2MS-B Leather current mIoU is **0.860749**.
2. It is **not higher** than Base-B Leather 0127 mIoU=0.899120. The gap is **-0.038371**.
3. It is **not close enough to treat as checkpoint best_iou=0.9091**. The gap to 0.9091 is **-0.048351**. The checkpoint `best_iou` should still be treated as validation-time checkpoint selection metadata, not as a test result.
4. Possible reasons for being lower than Base-B:
   - the checkpoint may correspond to validation best_iou rather than the locked test split;
   - the locked test distribution may be harder than the validation split;
   - model entry or evaluation-time output branch may differ from the historical evaluation path;
   - class order or label mapping should be rechecked against training code and historical scripts;
   - this checkpoint may not exactly match the paper-defined A2MS-DefectNet-B variant;
   - the checkpoint may reflect under-training or overfitting.
5. Recommendation: **retrain or at least fully audit A2MS-B Leather before using it as a paper main result**. Under the current locked test protocol, this checkpoint should not be used to claim an improvement over Base-B.

## 6. Paper Safety Decision

This result is a traceable smoke-test result only. It must not be written into `fixed_existing_results.md`, the manuscript abstract, the conclusion, or the final paper table without later same-protocol model comparison and manual approval.

