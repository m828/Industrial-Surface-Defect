# Leather/Pige Unified Evaluation All Final Summary

> Date: 2026-06-16  
> Protocol: `experiments/protocol/leather_valid_test_list.txt`, 468 samples, 768 x 768, 8 classes, mIoU includes background, TestRescale + ToTensor, no ImageNet Normalize.  
> Restrictions followed: no training, no paper edits, no weight overwrite, no update to `fixed_existing_results.md`.

## 1. mIoU Ranking

| Rank | Model | mIoU | Background | Open wound | Scratch | Brand mark | Hole | Skin disease | Rotten surface | Wart |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Base-B | 0.899120 | 0.993250 | 0.764801 | 0.700504 | 0.882528 | 0.990708 | 0.971181 | 0.915895 | 0.974096 |
| 2 | STDC-Seg (STDCNet1446) | 0.882401 | 0.992544 | 0.746916 | 0.629960 | 0.884325 | 0.987338 | 0.952547 | 0.895466 | 0.970113 |
| 3 | A2MS-DefectNet-S | 0.868997 | 0.991825 | 0.709699 | 0.608361 | 0.875100 | 0.988379 | 0.949903 | 0.860228 | 0.968483 |
| 4 | PIDNet-S | 0.862009 | 0.989440 | 0.709984 | 0.597094 | 0.864323 | 0.985162 | 0.943552 | 0.881693 | 0.924821 |
| 5 | A2MS-DefectNet-B | 0.860749 | 0.990210 | 0.732640 | 0.544550 | 0.884139 | 0.988418 | 0.965973 | 0.852424 | 0.927637 |
| 6 | Base-S | 0.858613 | 0.990719 | 0.695297 | 0.591564 | 0.855523 | 0.986517 | 0.942885 | 0.847433 | 0.958964 |
| 7 | DDRNet23slim | 0.842052 | 0.989058 | 0.732579 | 0.395457 | 0.852755 | 0.982948 | 0.939142 | 0.889241 | 0.955232 |

## 2. Base-B vs A2MS-B

| Item | mIoU |
|---|---:|
| Base-B 0127 | 0.899120 |
| A2MS-DefectNet-B | 0.860749 |
| A2MS-B - Base-B | -0.038371 |

A2MS-DefectNet-B is lower than Base-B on the locked Leather/Pige test set. The current A2MS-B Leather result should not be used as a paper main claim of improvement.

## 3. Baseline Relative Performance

- STDC-Seg (STDCNet1446): mIoU=0.882401; lower than Base-B by -0.016719, higher than A2MS-B by 0.021652.
- PIDNet-S: mIoU=0.862009; lower than Base-B by -0.037111, slightly higher than A2MS-B by 0.001260.
- DDRNet23slim: mIoU=0.842052; lower than both Base-B and A2MS-B.
- A2MS-DefectNet-S: mIoU=0.868997; higher than Base-S by 0.010384, but lower than Base-B.

## 4. Paper-Table Safety

Can be considered as traceable same-protocol candidates after manual paper-level approval:

- Base-B 0127: strongest current Leather result and traceable.
- Base-S: traceable base version result.
- A2MS-DefectNet-S: traceable, modestly above Base-S.
- STDC-Seg (STDCNet1446), PIDNet-S, DDRNet23slim: traceable baseline comparisons.

Should not be used as a main A2MS-B improvement result:

- A2MS-DefectNet-B: traceable but underperforms Base-B by 0.038371.

Failed/not evaluated:

- PP-LiteSeg-B: no matching Leather checkpoint found in `new/` or `new (copy)/` `.pkl` scan.
- BiSeNetV1-L: no matching Leather checkpoint found; code identity remains manually ambiguous.

## 5. Output Artifacts

- Unified final CSV: `experiments/results/leather_unified_eval_all_final.csv`
- Base-B CSV/log: `experiments/results/smoke_base_b_leather_class_iou.csv`, `experiments/logs/smoke_base_b_leather.log`
- A2MS-B CSV/log: `experiments/results/smoke_a2ms_b_leather_class_iou.csv`, `experiments/logs/smoke_a2ms_b_leather.log`
- DDRNet CSV/log: `experiments/results/smoke_ddrnet_leather_class_iou.csv`, `experiments/logs/smoke_ddrnet_leather.log`
- PIDNet-S CSV/log: `experiments/results/smoke_pidnet_leather_class_iou.csv`, `experiments/logs/smoke_pidnet_leather.log`
- STDC-Seg CSV/log: `experiments/results/smoke_stdc_leather_class_iou.csv`, `experiments/logs/smoke_stdc_leather.log`
- Base-S CSV/log: `experiments/results/smoke_base_s_leather_class_iou.csv`, `experiments/logs/smoke_base_s_leather.log`
- A2MS-S CSV/log: `experiments/results/smoke_a2ms_s_leather_class_iou.csv`, `experiments/logs/smoke_a2ms_s_leather.log`

## 6. Commands Newly Run This Round

```bash
python3 tools/evaluate_class_iou.py --dataset leather --test-list experiments/protocol/leather_valid_test_list.txt --model-name PIDNet-S --model-file '/workspace/Industrial Surface Defect/new/pid.py' --weight '/workspace/Industrial Surface Defect/new/model_savePath/fcn_pascal_pige_pid_s.pkl' --num-classes 8 --input-size 768 768 --output-csv experiments/results/smoke_pidnet_leather_class_iou.csv --save-log experiments/logs/smoke_pidnet_leather.log
```

```bash
python3 tools/evaluate_class_iou.py --dataset leather --test-list experiments/protocol/leather_valid_test_list.txt --model-name STDC-Seg --model-file '/workspace/Industrial Surface Defect/new (copy)/stdc.py' --weight '/workspace/Industrial Surface Defect/new (copy)/model_savePath/sdtdcnet_pige_stdc2_pige.pkl' --num-classes 8 --input-size 768 768 --output-csv experiments/results/smoke_stdc_leather_class_iou.csv --save-log experiments/logs/smoke_stdc_leather.log
```

```bash
python3 tools/evaluate_class_iou.py --dataset leather --test-list experiments/protocol/leather_valid_test_list.txt --model-name Base-S --model-file '/workspace/Industrial Surface Defect/new/model_dsmo_rs18.py' --weight '/workspace/Industrial Surface Defect/new/model_savePath/dsmonet_resnet_pascal_pige_dsmor18_0228_80000.pkl' --num-classes 8 --input-size 768 768 --output-csv experiments/results/smoke_base_s_leather_class_iou.csv --save-log experiments/logs/smoke_base_s_leather.log
```

```bash
python3 tools/evaluate_class_iou.py --dataset leather --test-list experiments/protocol/leather_valid_test_list.txt --model-name A2MS-DefectNet-S --model-file '/workspace/Industrial Surface Defect/new/model_dsmo_rs18_eSE_adapt_detailloss_822.py' --weight '/workspace/Industrial Surface Defect/new/model_savePath/dsmonet_resnet_pascal_pige_a2ms_s.pkl' --num-classes 8 --input-size 768 768 --output-csv experiments/results/smoke_a2ms_s_leather_class_iou.csv --save-log experiments/logs/smoke_a2ms_s_leather.log
```
