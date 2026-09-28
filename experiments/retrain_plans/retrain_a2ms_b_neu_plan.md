# Retrain A2MS-B NEU Plan

> Date: 2026-06-16  
> This is a preparation plan only. Do not start training without explicit approval.

## 1. Recommended Work Directory

Use a new non-overwriting directory:

```text
/workspace/Industrial Surface Defect/Industrial-Surface-Defect/experiments/retrain_a2ms_b_neu_20260616/
```

## 2. Training Script

Primary candidate:

```text
/workspace/Industrial Surface Defect/new/train_neu_a2ms_b_lightbag.py
```

Fallback/legacy candidate to audit before use:

```text
/workspace/Industrial Surface Defect/new/train_neu_resnet_detailloss.py
```

## 3. Config File

Primary candidate:

```text
/workspace/Industrial Surface Defect/new/config_neu_a2ms_b_lightbag.yml
```

Also compare against:

```text
/workspace/Industrial Surface Defect/new/config_neu_a2ms_b.yml
/workspace/Industrial Surface Defect/new/config_neu_a2ms_b_retrain.yml
```

## 4. Model File

Primary candidate:

```text
/workspace/Industrial Surface Defect/new/model_dsmo_rs50_eSE_adapt_detailloss_lightbag.py
```

Also audit the previous A2MS-B file:

```text
/workspace/Industrial Surface Defect/new/model_dsmo_rs50_eSE_adapt_detailloss.py
```

## 5. Output Checkpoint Directory

Recommended:

```text
/workspace/Industrial Surface Defect/Industrial-Surface-Defect/experiments/retrain_a2ms_b_neu_20260616/checkpoints/
```

Do not save into `/workspace/Industrial Surface Defect/new/model_savePath/` unless explicitly approved.

## 6. Log Directory

Recommended:

```text
/workspace/Industrial Surface Defect/Industrial-Surface-Defect/experiments/retrain_a2ms_b_neu_20260616/logs/
```

Also save copied config and command into:

```text
/workspace/Industrial Surface Defect/Industrial-Surface-Defect/experiments/retrain_a2ms_b_neu_20260616/manifest/
```

## 7. Avoiding Overwrites

- Use timestamped checkpoint names, e.g. `a2ms_b_neu_20260616_best_val.pkl` and `a2ms_b_neu_20260616_last.pkl`.
- Copy, do not edit, source config into the experiment folder.
- Log the exact training command before launch.
- Refuse to start if target checkpoint path already exists.

## 8. Recommended Command

Dry-run command draft only:

```bash
cd "/workspace/Industrial Surface Defect/new"
python3 train_neu_a2ms_b_lightbag.py --config config_neu_a2ms_b_lightbag.yml
```

Before running, patch or configure the script so checkpoints and logs go to the new experiment directory above.

## 9. Metrics To Monitor

- validation mIoU including background;
- per-class IoU for background/crazing/inclusion/patches;
- training loss and validation loss;
- best-val iteration;
- final locked-test mIoU on `test_neu.txt` using `tools/evaluate_class_iou.py`;
- confusion matrix and per-class Dice/Precision/Recall.

## 10. Stop Criteria

Stop when one of these is reached:

- configured maximum iterations complete;
- validation mIoU plateaus for a pre-defined patience window;
- loss diverges or NaN appears;
- checkpoint output path or logging path is wrong.

## 11. Success Criteria

Training is successful only if:

- strict checkpoint load succeeds;
- final evaluation uses `test_neu.txt`, 840 samples, 200 x 200, 4 classes, background included;
- mIoU approaches or exceeds the historical A2MS-B NEU target around 0.913, or clearly improves over verified Base-B NEU under the same protocol;
- all logs, configs, weights, and final CSV are traceable.

## 12. Configuration-Anomaly Criteria

Treat the run as anomalous if:

- mIoU is around the previously observed abnormal 0.7131;
- class IoU collapses for one or more non-background classes;
- validation mIoU is high but locked-test mIoU is sharply lower;
- model file or constructor does not match the intended A2MS-B definition;
- output checkpoint accidentally lands in a historical weight directory.
