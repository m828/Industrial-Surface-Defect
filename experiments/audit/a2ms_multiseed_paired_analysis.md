# A2MS Multiseed Paired Analysis

> Date: 2026-09-08. Data: `a2ms_multiseed_10k_results.md`. Design: 3 matched seeds × {B0, A1, A2}, endpoint-only eval, unified 840.

## 1. ΔAAM (A1 − B0): does AAM help?

- Per-seed: +0.31 pt (1337), +1.34 pt (2026), −0.22 pt (3407) → mean **+0.48 pt ± 0.79 pt**, positive 2/3.
- The sign flips across seeds and the SD (0.79 pt) exceeds the mean (0.48 pt); B0's own run variability is 0.41–0.71 pt, the same order as the effect.
- **Class level**: crazing ΔAAM is positive in **3/3** seeds (+0.13/+1.48/+0.65 pt, mean +0.75 pt) — the only class with a consistent AAM direction; inclusion flips sign at 3407 (−1.02 pt); patches 2/3.
- **Verdict: 暂不支持 (no stable mIoU benefit across seeds)** — with a documented, consistent-but-small positive signal on crazing only. Not "明显负向": mean is positive and 2/3 seeds are positive.

## 2. ΔDetail (A2 − A1): does detail supervision help (on fixed forward + AAM)?

- Per-seed: +0.50 pt (1337), −0.31 pt (2026), −0.04 pt (3407) → mean **+0.05 pt ± 0.41 pt**, positive 1/3.
- Direction is unstable; mean ≈ 0; magnitude within run variability.
- Class level: crazing 1/3 (+1.04/−0.18/−0.50), patches 2/3 (+0.62/−0.87/+0.68), inclusion 1/3. The single-seed 1337 crazing gain (+1.04 pt) from last round does NOT replicate at 2026/3407 — confirming it was seed luck, not an effect.
- **Verdict: 暂不支持 (no stable benefit; effect indistinguishable from zero at 10k)**. Not "明显负向" either: mean ≈ 0, no systematic class regression.

## 3. ΔFull (A2 − B0): does the full method beat Base-B?

- Per-seed: +0.82 pt (1337), +1.03 pt (2026), −0.26 pt (3407) → mean **+0.53 pt ± 0.69 pt**, positive 2/3.
- A2 mean mIoU (0.857451) vs B0 (0.852175): +0.53 pt, within ~1.3× of B0's own SD.
- Class level is more consistent than the headline: crazing ΔFull positive **3/3** (+1.17/+1.30/+0.16, mean +0.88 pt), patches positive **3/3** (+0.97/+1.98/+0.34, mean +1.10 pt); background/inclusion flip sign once.
- **Verdict: 与 Base-B 基本相当** — a positive-leaning but unstable mIoU delta; the consistent (though small) per-class gains on the two defect classes with the finest structures (crazing, patches) are the only stable signal in the entire matrix.

## 4. Stability analysis

| claim candidate | seeds consistent? | size vs B0 variability (SD 0.41 pt) | supported? |
|---|---|---|---|
| AAM improves mIoU | 2/3 | mean 0.48 pt ≈ 1.2×SD | no (unstable) |
| AAM improves crazing | 3/3 | mean 0.75 pt | weakly, class-level only |
| detail improves mIoU | 1/3 | mean 0.05 pt | no |
| detail improves crazing/patches | 1/3, 2/3 | sign flips | no |
| Full improves mIoU | 2/3 | mean 0.53 pt ≈ 1.3×SD | no (unstable, positive-leaning) |
| Full improves crazing & patches | 3/3, 3/3 | 0.88 / 1.10 pt | yes (class-level, small) |

Interpretation: at 10k, the components' effects are at/below the run-variability scale. The mIoU (dominated by background + inclusion pixel counts) washes out the defect-class signal; where the signal survives (crazing/patches for the full method), it is consistent but small (≤ ~1 pt).

## 5. What this rules out

- The 1337 single-seed story ("AAM +0.31 pt, detail +0.50 pt") is **not reproducible** as a stable claim — it was one favorable draw.
- Historical 0.913–0.916 remains unusable as method evidence (model has neither AAM nor executed detail loss).
- Nothing here justifies long-training spend (20k+) on the current cells, nor does it prove the components worthless: effects are unproven at 10k, with a consistent class-level hint favoring the full method on defect classes.
