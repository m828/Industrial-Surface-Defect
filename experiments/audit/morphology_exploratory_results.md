# Morphology Exploratory Results — crazing complexity (EXPLORATORY)

Status: **completed** (sample-level proxy analysis). Component-level analysis: **skipped with reason** — per the frozen protocol, sample-level quantile analysis was prioritized; NEU per-image masks frequently merge adjacent defect regions, making connected-component perimeter/elongation annotation-dependent, so component-level cuts were deferred rather than hard-done.

Proxy (frozen in protocol): `complexity = perimeter / sqrt(area)`, perimeter = |contour(GT)| (3×3 inner contour), per image with crazing present (n=364). Tertile split at computed edges [9.7564, 11.9291] → low / mid / high.

## Results (Δ = A2 − Base, macro over group)

| Complexity | n | ΔIoU | ΔBIoU@2px | ΔBF1@2px | IoU pos/tie/neg |
| ---------- | -: | ---: | --------: | -------: | --------------: |
| low | 121 | +0.00004 | −0.00212 | −0.00287 | 62/2/57 |
| mid | 122 | −0.00431 | −0.00399 | −0.00805 | 63/0/59 |
| high | 121 | **−0.01741** | **−0.01364** | **−0.01479** | 49/1/71 |

## Verdict on "越复杂，A2收益越大"

**不支持 — 且方向相反。** The hypothesized trend (A2 helps more on elongated/complex cracks) is inverted in the data: the high-complexity tertile shows the *largest negative* deltas on all three metrics (ΔIoU −1.74 pt, ΔBF1 −1.48 pt), while the low-complexity tertile is a wash. A2's crazing behavior — filling interiors of large/simple regions at the cost of fine boundary precision — is exactly what this proxy would predict to hurt most when boundaries dominate (high perimeter²/area).

Exploratory status: single proxy, single class, seed-1337 predictions; cannot serve as sole primary evidence. But it is consistent with, and mechanistically explanatory for, the primary boundary and scale results.

Machine-readable: `morphology_exploratory_results.json`.
