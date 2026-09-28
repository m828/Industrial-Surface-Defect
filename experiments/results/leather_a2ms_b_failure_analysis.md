# A2MS-B Leather Failure Analysis

> Date: 2026-06-16  
> Scope: objective analysis of why A2MS-DefectNet-B Leather underperforms Base-B under the locked 468-sample test protocol.  
> This document does not modify the manuscript and does not turn validation `best_iou` into a test result.

## 1. Observed Result

| Model | Weight | Test mIoU including background |
|---|---|---:|
| Base-B | `/workspace/Industrial Surface Defect/new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl` | 0.899120 |
| A2MS-DefectNet-B | `/workspace/Industrial Surface Defect/new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl` | 0.860749 |
| Gap | A2MS-B - Base-B | -0.038371 |

A2MS-B is therefore a verified underperforming checkpoint on Leather/Pige test=468, not a current paper main-result checkpoint.

## 2. Plausible Causes To Audit

- Checkpoint `best_iou` may correspond to validation rather than the locked 468-sample test set.
- The locked test=468 split may be harder or differently distributed than validation.
- A2MS-B may be overfit to the validation split or to training-time augmentations.
- The evaluation model entry may not exactly match the paper-defined A2MS-DefectNet-B variant.
- ELMM/AAM/DetailLoss need to be verified in the active model file and training script, not inferred from filename.
- Category order and label mapping should be rechecked against the training `DataGenerator`, label PNG values, and historical scripts.
- The checkpoint may be under-trained, trained with a different protocol, or selected from an unstable training run.

## 3. Current Evidence

- Model entry used in evaluation: `DSMONet(num_classes=N, backbone=resnet50())`.
- Model file: `/workspace/Industrial Surface Defect/new/model_dsmo_rs50_eSE_adapt_detailloss.py`.
- Weight loaded strictly and evaluation completed successfully.
- Per-class weakness is most visible on `scratch` IoU=0.544550, which is lower than Base-B scratch IoU=0.700504.
- The result is below checkpoint metadata `best_iou=0.9091` reported in earlier notes, reinforcing that validation metadata cannot be treated as test result.

## 4. Retraining Recommendation

A2MS-B Leather should be retrained or deeply audited before any paper main-table use. If retraining is chosen, preserve:

- exact Git commit / script hash;
- train/val/test split paths;
- model file path and constructor arguments;
- all CLI arguments and config files;
- random seed and CUDA/device information;
- best-val checkpoint, last checkpoint, and final locked-test CSV/log;
- validation curves and final test confusion matrix.

## 5. Safety Decision

Do not write current A2MS-B Leather mIoU=0.860749 into the manuscript as an improvement over Base-B. It may only be used as a failure-analysis or checkpoint-audit result until retraining or a corrected checkpoint changes the conclusion.
