# Base-B Leather 0127 Verified Candidate

> Date: 2026-06-15  
> Status: verified candidate under the locked Leather/Pige protocol.  
> Restrictions followed: no training, no paper edits, no weight overwrite, no update to `fixed_existing_results.md`.

## Candidate Result

| Item | Value |
|---|---|
| Model | Base-B Leather |
| Weight | `/workspace/Industrial Surface Defect/new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl` |
| Dataset | Leather/Pige |
| Test list | `experiments/protocol/leather_valid_test_list.txt` |
| Effective test samples | 468 |
| Input size | 768 x 768 |
| Number of classes | 8 |
| mIoU includes background | Yes |
| mIoU | **0.899120** |
| Output CSV | `experiments/results/smoke_base_b_leather_class_iou.csv` |
| Log | `experiments/logs/smoke_base_b_leather.log` |

## Traceability

The result is traceable to:

- audited 468-sample test list;
- model entry `DSMONet(num_classes=8, backbone=resnet50())`;
- model file `/workspace/Industrial Surface Defect/new/model_dsmo_rs50.py`;
- checkpoint `dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl`;
- unified evaluator `tools/evaluate_class_iou.py`;
- saved CSV and log listed above.

## Use Decision

This result can be treated as the current Base-B Leather main-result candidate. It must not be written to `fixed_existing_results.md` at this stage.

Before entering a paper table, it still needs same-protocol comparison with:

- A2MS-DefectNet-B Leather;
- STDC-Seg/STDCNet variant;
- DDRNet23slim;
- PIDNet-S;
- any other final baseline set chosen for the paper.

Do not mix this 0127 result with the earlier Base-B 0126 value around 0.8806.

