# Paired Case-Level Analysis — Base-B vs A2 Full @240k (seed 1337)

All 840 test images evaluated identically for both models from the frozen prediction cache (mIoU reproduction verified exactly). Paired per-(image,class) deltas; tie = |Δ| ≤ 1e-6. Bootstrap = 5000 paired resamples over case indices. **CI caveat: reflects uncertainty across the fixed 840 test cases for seed-1337 predictions, not variability across training seeds.**

## Per-image 4-class mIoU (includes background)

Δ mean **+0.00031**, median 0.0, 95% CI **[−0.00176, +0.00251]** — includes zero. Cases: 437 improved / 2 tie / 401 worsened. The endpoint aggregate advantage (+0.21 pt) is not reflected as a consistent per-image win; it is carried by a subset of larger-defect images.

## Per-class paired counts (IoU / BF1@2px / BIoU@2px)

| Class | n valid | ΔIoU mean | pos/tie/neg (IoU) | ΔBF1@2px mean | pos/tie/neg (BF1) | ΔBIoU@2px mean | pos/tie/neg (BIoU) |
| ----- | ------: | --------: | ----------------: | ------------: | ----------------: | -------------: | -----------------: |
| crazing | 368 | −0.00714 | 174/7/187 | −0.00596 | 79/159/130 | −0.00453 | 172/5/191 |
| inclusion | 361 | −0.00003 | 198/1/162 | +0.00036 | 103/161/97 | +0.00318 | 202/2/157 |
| patches | 353 | +0.00393 | 177/1/175 | +0.00297 | 95/173/85 | +0.00537 | 181/1/171 |

Improved-share of decided (non-tie) cases: crazing IoU 48.2% / BF1 37.8%; inclusion IoU 55.0% / BF1 51.5%; patches IoU 50.3% / BF1 52.8%.

## Bootstrap 95% CIs for mean Δ (paired)

| Class | ΔIoU CI | ΔBIoU@2px CI | ΔBF1@2px CI |
| ----- | ------- | ------------ | ----------- |
| crazing | [−0.01144, −0.00304] | [−0.01071, −0.00241] | [−0.01301, −0.00452] |
| inclusion | [−0.00331, +0.00335] | [−0.00150, +0.00755] | [−0.00603, +0.00628] |
| patches | [+0.00007, +0.00859] | [+0.00126, +0.00964] | [−0.00102, +0.00691] |

Reading: per-case crazing deltas are **significantly negative** (all CIs exclude zero); patches region/BIoU **significantly positive**; inclusion null. Note the inversion vs the pixel-aggregate headline (crazing aggregate IoU +0.32 pt): A2 wins crazing on a few large-region images while losing slightly on the typical case.

## Case selection & visualization (frozen rule)

Selection: rank GT-present cases by ΔBF1@2px → top-3 improved / bottom-3 worsened / 3 nearest-zero, per class (crazing, patches). No hand-picking. 18 panels in `repro_runs/mechanism_eval_240k/visualizations/` (original | GT | Base pred | A2 pred | Base err | A2 err; TP green / FP red / FN blue), manifest `viz_manifest.json`, selection `case_selection.json`.

Qualitative reading of the extremes (illustrative only, not statistics): top-improved crazing cases (e.g. 000653, ΔBF1 +0.1332) show A2 **filling interior holes** that Base-B leaves as misclassified speckles inside large crack regions; bottom-worsened cases (e.g. 000722, ΔBF1 −0.4818) show A2 **over-growing crazing as an FP halo around a strong horizontal inclusion band while partially missing the thin upper crack** — consistent with the aggregate-level pattern (interior-fill gain on big regions, precision loss on thin/fine structures).

Machine-readable: `paired_case_metrics.csv` (2520 rows = 840×3 classes; per-case region IoU/Dice, BIoU/BF-precision/recall/F1 @1/2/3px, deltas, GT area/ratio, crazing complexity), `paired_case_analysis.json`.
