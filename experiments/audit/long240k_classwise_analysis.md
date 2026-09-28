# long240k Classwise Analysis (seed 1337, unified-840 endpoint)

Scope: endpoint (frozen) classwise comparison + trajectory-level class consistency. Single matched seed; no significance claims.

## Endpoint per-class IoU (240k, frozen)

| class     |   Base-B |  A2 Full |   Δ (A2−Base) |
| --------- | -------: | -------: | ------------: |
| background| 0.986201 | 0.986369 |     +0.000168 |
| crazing   | 0.841464 | 0.844681 |     +0.003217 |
| inclusion | 0.934863 | 0.935494 |     +0.000631 |
| patches   | 0.887106 | 0.891633 |     +0.004527 |

A2 is numerically ahead on all four classes; the margin concentrates in the two detail-sensitive defect classes (crazing +0.32 pt, patches +0.45 pt) rather than background/inclusion.

## Consistency across the trajectory

| class     | points with A2>Base (of 6) | pattern                    |
| --------- | :------------------------: | -------------------------- |
| crazing   |           **6/6**          | + at 10k/30k/60k/120k/180k/240k |
| inclusion |            3/6             | sign alternates, |Δ|≤0.29 pt after 10k |
| patches   |            3/6             | − at 60k–180k, + at 10k/30k/240k |

The clearest class-level signal is **crazing**: A2 leads at every evaluated point from 10k through 240k (Δ +0.32 to +1.58 pt). This is directionally consistent with the 10k 2×2 finding (Full's crazing gain was 3/3 seeds positive) and with the defect morphology (thin, elongated crack-like regions where boundary/detail information should matter most).

patches shows no stable mid-training ordering (A2 behind at 60k/120k/180k, ahead at both endpoints and 30k) — the endpoint gain (+0.45 pt) is real for this seed but the trajectory does not show persistent classwise dominance.

inclusion is essentially a tie after 10k (|Δ| ≤ 0.29 pt).

## vs the 10k matched 3-seed picture

- 10k (3 seeds): Full − Base mean +0.53 pt, 2/3 positive; crazing + at 3/3 seeds; patches + at 3/3 seeds.
- 240k (1 seed): Full − Base +0.21 pt; crazing + at 6/6 trajectory points; patches + at endpoint but 3/6 trajectory points.

The long-training picture neither amplifies nor reverses the 10k signal: the mIoU gap stays in the same small positive range (+0.2 ~ +0.8 pt across the trajectory), and the crazing class advantage persists. Under the frozen protocol's single-seed conclusion limits, the correct statement is: **AAM + detail supervision (A2) achieved a numerically higher endpoint mIoU for seed 1337, with the advantage concentrated in crazing; stability across seeds is unverified at 240k.**

## Endpoint Dice / Precision / Recall

Available in full in `long240k_endpoint_results.json` (per-class dice/precision/recall + 4×4 confusion matrices for both models). Headline Dice: A2 ≥ Base on all four classes (bg 0.993138 vs 0.993053; crazing 0.915802 vs 0.913908; inclusion 0.966672 vs 0.966335; patches 0.942712 vs 0.940176).
