# Leather Smoke Round 2 Decision

> Date: 2026-06-15  
> Scope: Leather/Pige protocol synchronization, Base-B candidate fixation, A2MS-B smoke test, and baseline feasibility checks.  
> Restrictions followed: no training, no paper edits, no weight overwrite, no update to `fixed_existing_results.md`, no small-object/elongated-defect evaluation.

## 1. Is Leather test=468 formally locked?

**Yes.**

The locked Leather/Pige test list is:

```text
experiments/protocol/leather_valid_test_list.txt
```

Final audit conclusion:

- `new (copy)/dataset/pige/test.txt` and `new/dataset/pige/test.txt` are content-identical.
- Total rows: 468.
- Non-empty rows: 468.
- No duplicate rows.
- No malformed rows.
- All image/label files exist.
- Evaluation uses 768 x 768 input, TestRescale + ToTensor, no ImageNet Normalize.
- mIoU includes background.

Protocol documents generated/updated this round:

- `experiments/protocol/eval_protocol_lock_after_smoke.md`
- `experiments/protocol/leather_protocol_final_decision.md`

## 2. Can Base-B 0127 be treated as the Leather main-result candidate?

**Yes, as a verified candidate only.**

| Item | Value |
|---|---|
| Model | Base-B |
| Weight | `/workspace/Industrial Surface Defect/new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl` |
| Test samples | 468 |
| mIoU including background | **0.899120** |
| Status | Verified candidate, not final paper-table result |

Record:

```text
experiments/results/base_b_leather_0127_verified_candidate.md
```

Do not mix this 0127 result with the earlier 0126 result around 0.8806.

## 3. Did A2MS-DefectNet-B Leather pass smoke test?

**Yes, the smoke test passed.**

| Item | Value |
|---|---|
| Model | A2MS-DefectNet-B |
| Weight | `/workspace/Industrial Surface Defect/new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl` |
| Test samples | 468 |
| mIoU including background | **0.860749** |
| CSV | `experiments/results/smoke_a2ms_b_leather_class_iou.csv` |
| Log | `experiments/logs/smoke_a2ms_b_leather.log` |

## 4. Is A2MS-DefectNet-B better than Base-B on Leather?

**No.**

| Comparison | mIoU |
|---|---:|
| Base-B 0127 | 0.899120 |
| A2MS-DefectNet-B | 0.860749 |
| Gap | -0.038371 |

A2MS-B is also below the checkpoint metadata `best_iou=0.9091` by about -0.048351. That `best_iou` should still be treated as validation/checkpoint-selection metadata, not as a test result.

## 5. If A2MS-B is not better than Base-B, is retraining recommended?

**Yes, retraining or a deeper checkpoint/model-entry audit is recommended before using A2MS-B Leather as a main paper result.**

Possible causes remain:

- checkpoint `best_iou` corresponds to validation rather than the locked test split;
- locked test distribution is harder;
- evaluation-time model branch or model entry differs from historical scripts;
- class order or label mapping needs another audit against training code;
- checkpoint may not exactly match the paper-defined A2MS-DefectNet-B variant;
- model may be under-trained or overfit.

## 6. How should STDC be named?

Recommended current name:

```text
STDC-Seg (STDCNet1446)
```

Reason:

- historical test script loads the checked weight into `BiSeNet('STDCNet1446', 8)`;
- strict load and 768 dummy forward succeeded for `BiSeNet('STDCNet1446', 8)`;
- strict load failed for `BiSeNet('STDCNet813', 8)`;
- checkpoint filename contains `stdc2`, but the actual checked model entry is STDCNet1446.

Do not force the label `STDC1-Seg` or `STDC2-Seg` until the external STDC naming convention is manually confirmed. If a paper table needs a conservative label now, use **STDC-Seg (STDCNet1446)**.

## 7. Can DDRNet be evaluated?

**Yes.**

DDRNet23slim passed strict checkpoint load, 768 dummy forward, and a full Leather/Pige smoke test.

| Item | Value |
|---|---|
| Model entry | `DualResNet_imagenet(num_classes=8)` |
| Weight | `/workspace/Industrial Surface Defect/new/model_savePath/ddr_pascal_pige_ddr23s.pkl` |
| Smoke mIoU including background | **0.842052** |
| CSV | `experiments/results/smoke_ddrnet_leather_class_iou.csv` |
| Log | `experiments/logs/smoke_ddrnet_leather.log` |

DDRNet can enter the next full-model unified evaluation stage.

## 8. Can PIDNet be evaluated?

**Yes, ready for next-round full evaluation.**

PIDNet-S passed strict checkpoint load and 768 dummy forward with the training-matched entry:

```text
PIDNet(m=2, n=3, num_classes=8, planes=32, ppm_planes=96, head_planes=128, augment=True)
```

The `augment=False` prediction entry does not strictly load this checkpoint, so the unified evaluator should continue using the `augment=True` entry with `model.eval()`.

## 9. Can we enter Leather full-model unified evaluation?

**Yes, with one naming caution.**

Ready/partly completed:

- Base-B 0127: evaluated and traceable.
- A2MS-DefectNet-B: evaluated and traceable, but not better than Base-B.
- DDRNet23slim: evaluated and traceable.
- PIDNet-S: load/forward ready, full evaluation can run next.
- STDC-Seg (STDCNet1446): load/forward ready; full evaluation can run next, but naming should stay conservative.

Suggested next Leather evaluation order:

1. STDC-Seg (STDCNet1446) full evaluation.
2. PIDNet-S full evaluation.
3. Base-S and A2MS-S Leather under the same 468-sample protocol.
4. Re-run/verify any already completed model only if model entry or weight identity changes.

## 10. Can A2MS-B NEU retraining start?

**Not in this round.**

Technically, the project can move toward A2MS-B NEU retraining preparation after this smoke round, but training should start only after manual confirmation of:

- no recoverable historical 91.3 NEU checkpoint exists;
- exact training script/config/seed/iteration count;
- new non-overwriting checkpoint and log output paths;
- evaluation command for the new checkpoint;
- whether Base-B NEU should be re-smoke-tested before retraining A2MS-B.

Current recommendation: finish Leather full-model unified evaluation first, then present the A2MS-B NEU retraining command and output paths for approval.

