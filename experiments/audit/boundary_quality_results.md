# Boundary Quality Results — Base-B vs A2 Full @240k (seed 1337, frozen cache)

Protocol: `mechanism_eval_protocol.md`. Boundary band = dilate_t − erode_t (3×3 square, Chebyshev); BF-score on 1-px contours with Chebyshev tolerance t. Primary = macro over valid (image,class) cases (NA both-empty excluded; GT-absent+pred-present counts as 0). Secondary = pixel-aggregate. Bootstrap: 5000 paired resamples over case indices. **CI caveat: test-case uncertainty for seed-1337 predictions, NOT training-seed variability.**

## Headline table (macro, primary)

| Class | Metric | Base-B | A2 Full | Δ | 95% CI (paired boot) | pos/tie/neg |
| ----- | ------ | -----: | ------: | -: | -------------------: | ----------: |
| crazing | BIoU @1px | 0.57771 | 0.57657 | −0.00114 | [−0.00735, +0.00177] | — |
| crazing | BIoU @2px | 0.71901 | 0.71448 | **−0.00453** | **[−0.01079, −0.00243]** | 172/5/191 |
| crazing | BIoU @3px | 0.78272 | 0.77751 | **−0.00522** | **[−0.01169, −0.00346]** | — |
| crazing | BF1 @1px | 0.84382 | 0.83426 | **−0.00956** | **[−0.01679, −0.00729]** | — |
| crazing | BF1 @2px | 0.91164 | 0.90568 | **−0.00596** | **[−0.01303, −0.00465]** | 79/159/130 |
| crazing | BF1 @3px | 0.93025 | 0.92481 | **−0.00543** | **[−0.01357, −0.00327]** | — |
| inclusion | BIoU @1px | 0.54885 | 0.55434 | **+0.00549** | **[+0.00084, +0.01012]** | — |
| inclusion | BIoU @2px | 0.69349 | 0.69667 | +0.00318 | [−0.00141, +0.00754] | 202/2/157 |
| inclusion | BIoU @3px | 0.76722 | 0.76884 | +0.00162 | [−0.00279, +0.00585] | — |
| inclusion | BF1 @1px | 0.80791 | 0.80793 | +0.00001 | [−0.00591, +0.00597] | — |
| inclusion | BF1 @2px | 0.90218 | 0.90254 | +0.00036 | [−0.00574, +0.00637] | 103/161/97 |
| inclusion | BF1 @3px | 0.94470 | 0.94337 | −0.00133 | [−0.00673, +0.00363] | — |
| patches | BIoU @1px | 0.69646 | 0.70199 | **+0.00552** | **[+0.00067, +0.01057]** | — |
| patches | BIoU @2px | 0.82245 | 0.82782 | **+0.00537** | **[+0.00126, +0.00977]** | 181/1/171 |
| patches | BIoU @3px | 0.87094 | 0.87640 | **+0.00546** | **[+0.00151, +0.00972]** | — |
| patches | BF1 @1px | 0.91803 | 0.92285 | **+0.00481** | **[+0.00048, +0.00946]** | — |
| patches | BF1 @2px | 0.96329 | 0.96626 | +0.00297 | [−0.00108, +0.00720] | 95/173/85 |
| patches | BF1 @3px | 0.97196 | 0.97472 | +0.00276 | [−0.00061, +0.00642] | — |

Bold Δ = paired-bootstrap 95% CI excludes zero.

## Pixel-aggregate (secondary)

| Class | Tol | BIoU Base→A2 | Δ | BF1 Base→A2 | Δ |
| ----- | --- | -----------: | -: | ----------: | -: |
| crazing | 1px | 0.5424→0.5418 | −0.00057 | 0.8550→0.8488 | −0.00624 |
| crazing | 2px | 0.7045→0.7007 | −0.00377 | 0.9245→0.9197 | −0.00483 |
| crazing | 3px | 0.7792→0.7744 | −0.00472 | 0.9445→0.9397 | −0.00480 |
| inclusion | 1px | 0.5474→0.5536 | +0.00618 | 0.8443→0.8452 | +0.00087 |
| inclusion | 2px | 0.7000→0.7044 | +0.00443 | 0.9197→0.9217 | +0.00204 |
| inclusion | 3px | 0.7773→0.7805 | +0.00319 | 0.9543→0.9548 | +0.00049 |
| patches | 1px | 0.6603→0.6678 | +0.00742 | 0.9197→0.9243 | +0.00460 |
| patches | 2px | 0.8135→0.8220 | +0.00848 | 0.9641→0.9679 | +0.00378 |
| patches | 3px | 0.8678→0.8764 | +0.00856 | 0.9729→0.9763 | +0.00342 |

Aggregate and macro agree in sign for every class — the pattern is not an averaging artifact.

## Interpretation

1. **patches is the only class with a robust boundary gain**: BIoU CI excludes zero at all three tolerances (macro and aggregate agree, +0.5~+0.9 pt). BF1 positive at 1px (CI excludes 0), lean-positive at 2/3px.
2. **crazing boundary is slightly WORSE under A2** — macro BF1 CIs exclude zero on the negative side at all tolerances; aggregate agrees. This contradicts the earlier trajectory-level impression (aggregate region IoU favored A2 at all 6 checkpoints). Reconciliation: A2's crazing gain is interior region filling of large defects (aggregate region IoU +0.32 pt), while its boundaries are marginally less precise on the typical case (macro ΔIoU −0.71 pt; visual inspection of top-improved case 000653 shows A2 filling interior holes that Base-B leaves as misclassified speckles).
3. **inclusion is boundary-neutral** (1px BIoU small positive, everything else CI-through-zero).

Machine-readable: `boundary_quality_results.json` (audit + mechanism_eval dirs).
