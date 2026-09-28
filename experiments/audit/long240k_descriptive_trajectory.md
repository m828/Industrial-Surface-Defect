# long240k Descriptive Trajectory (post-freeze review)

**Status of these numbers**: all intermediate checkpoints (10k–180k) were evaluated ONLY AFTER the 240k endpoint results were frozen in `long240k_endpoint_results.json`. They are **descriptive only** — never used for checkpoint selection, early stopping, or any training decision (the training scripts constructed no test loader; checkpoints were saved purely by pre-registered iteration).

Eval protocol identical to endpoint: unified 840, batch=1, TestRescale+ToTensor, no Normalize, `pred=outputs[0]`, 4-class IoU incl. background.

## Trajectory table

|  Iter |   Base-B |  A2 Full |   Δ (A2−Base) |
| ----: | -------: | -------: | ------------: |
|  10k | 0.850011 | 0.857877 |     +0.007866 |
|  30k | 0.883436 | 0.887451 |     +0.004014 |
|  60k | 0.900702 | 0.899401 |     −0.001300 |
| 120k | 0.906391 | 0.907390 |     +0.000999 |
| 180k | 0.911850 | 0.910999 |     −0.000850 |
| 240k | 0.912408 | 0.914544 |     +0.002136 |  ← endpoint (frozen)

Δ sign: 4/6 positive; Δ mean over the six points ≈ +0.0030; |Δ| ≤ 0.008 at every point — the two models track each other closely throughout training.

## Per-class IoU trajectories

crazing (A2 ahead at **all 6** evaluated points):

|  Iter |   Base-B |  A2 Full |        Δ |
| ----: | -------: | -------: | -------: |
|  10k | 0.737173 | 0.752932 | +0.015758 |
|  30k | 0.794670 | 0.806975 | +0.012304 |
|  60k | 0.825547 | 0.826258 | +0.000710 |
| 120k | 0.828094 | 0.834294 | +0.006200 |
| 180k | 0.841603 | 0.845489 | +0.003885 |
| 240k | 0.841464 | 0.844681 | +0.003217 |

inclusion (mixed: + − − + − +):

|  Iter |   Base-B |  A2 Full |        Δ |
| ----: | -------: | -------: | -------: |
|  10k | 0.872209 | 0.884210 | +0.012002 |
|  30k | 0.910383 | 0.909792 | −0.000590 |
|  60k | 0.923925 | 0.921207 | −0.002717 |
| 120k | 0.928120 | 0.928794 | +0.000674 |
| 180k | 0.931910 | 0.929003 | −0.002908 |
| 240k | 0.934863 | 0.935494 | +0.000631 |

patches (mixed: + + − − − +):

|  Iter |   Base-B |  A2 Full |        Δ |
| ----: | -------: | -------: | -------: |
|  10k | 0.816275 | 0.818385 | +0.002110 |
|  30k | 0.848002 | 0.851167 | +0.003164 |
|  60k | 0.869135 | 0.866222 | −0.002913 |
| 120k | 0.884404 | 0.881256 | −0.003148 |
| 180k | 0.887811 | 0.883733 | −0.004078 |
| 240k | 0.887106 | 0.891633 | +0.004527 |

## Is either model still rising at 240k?

Late-stage mIoU increments:

- Base-B: 120k→180k **+0.005459**, 180k→240k **+0.000558**
- A2 Full: 120k→180k **+0.003609**, 180k→240k **+0.003545**

Base-B's last-60k gain (+0.06 pt) is an order of magnitude smaller than its 120k→180k gain and within the Δ-noise band observed between the two models — effectively **plateaued**. A2's 180k→240k gain (+0.35 pt) is not diminishing relative to its previous 60k segment, so A2 shows a **slight residual upward trend**, but one more 60k segment of this size would only add ~+0.3 pt. Verdict per the frozen protocol wording: **Base-B 已明显平台；A2 仍有上升趋势(幅度小)** — single seed, descriptive only; no training extension performed or recommended by this document.

## Cross-protocol historical context (NOT same-yardstick)

Historical 832-monitor / test-driven-best trajectory (old environment): 10k ≈ 0.857, 30k ≈ 0.882, 60k ≈ 0.90, 112k ≈ 0.9118, 165k–240k ≈ 0.913–0.916.
Present unified-840 fixed-endpoint trajectory: Base-B 0.850 / 0.883 / 0.901 / 0.906 / 0.912 / 0.9124; A2 0.858 / 0.887 / 0.899 / 0.907 / 0.911 / 0.9145.
The shapes match closely; A2's endpoint (0.914544) falls inside the historical 0.913–0.916 band, Base-B (0.912408) marginally below its low edge. Historical numbers benefitted from test-driven best-checkpoint selection; present numbers are fixed endpoints — see `long240k_final_decision.md` §口径.
