# NEU mask identity verification

Audit time UTC: 2026-10-19T10:34:46.857022+00:00
Test list: `/workspace/Industrial Surface Defect/repro_runs/long240k_seed1337/base_b/code/dataset/test_neu.txt`
Test-list SHA-256: `2c45b319e04d2307eb0977720de90b98a4867863bce8eca6f8b682295f928f06`
Cached GT vs source annotations: 840/840 pixel-equal
Manifest order equals test-list order: True
Directory sets equal test set: {'gt': True, 'base': True, 'a2': True}
All masks 200x200: True
All labels within 0..3: True

## Independent mIoU reproduction
- base: mIoU=0.912408374468041, expected=0.912408374468041, absolute error=0
- a2: mIoU=0.914544217767114, expected=0.914544217767114, absolute error=0

The prediction cache script uses `model(img)[0].data.max(1)[1]` and writes uint8 PNG masks directly; no post-processing branch was observed in that script.
