# Mechanism Evaluation Protocol (PRE-REGISTERED)

**Frozen before any Base-vs-A2 model comparison was computed.** Signed-off inputs: frozen 240k checkpoints only; no training, no threshold tuning, no post-processing, no TTA.

- Date: 2026-09-24
- Models: Base-B `new/model_dsmo_rs50.py` @ iter240000 (ckpt sha256 `46c5b83c…`) vs A2 Full `new/model_dsmo_rs50_eSE_adapt_detailloss_fixed.py` @ iter240000 (ckpt sha256 `5beaaa1c…`)
- Data: locked NEU test split, 840 images, 200×200, 4 classes incl. background; transform TestRescale+ToTensor, no Normalize; prediction = argmax of `outputs[0]`; no CRF/morphology/TTA
- Predictions cached once (`repro_runs/mechanism_eval_240k/{base_predictions,a2_predictions,gt}`); every downstream metric computed from this single frozen cache. Cache validity gate: recomputed 4-class mIoU must reproduce Base-B 0.912408 / A2 0.914544 within float tolerance.

## 1. Boundary definitions (fixed in advance)

All morphology uses a **3×3 square structuring element** (Chebyshev metric), iterated `t` times for tolerance `t` pixels.

- **Contour** of binary mask m: `contour(m) = m − erode_1(m)` (1-px inner contour).
- **Boundary band** @t: `band_t(m) = dilate_t(m) − erode_t(m)` (symmetric band of total width ≈ 2t+1 px straddling the edge).
- **Boundary IoU @t** per (image, class): `|band_t(pred) ∩ band_t(gt)| / |band_t(pred) ∪ band_t(gt)|`.
- **BF-score @t** per (image, class): precision = fraction of `contour(pred)` pixels within Chebyshev distance t of `contour(gt)` (via distance transform of gt contour); recall = symmetric; F1 = harmonic mean. Precision=recall=F1=0 when denominator empty unless NA rule applies.
- Boundary Dice reported only as supplementary (`2|A∩B|/(|A|+|B|)` on bands) — not a headline metric.

Tolerances (primary, fixed): **1 px, 2 px, 3 px** at 200×200. Ratio-based widths (d=0.005/0.01/0.02 of image size = 1/2/4 px at 200px) are noted as equivalent to ~1/2/4 px and not separately reported to avoid scale ambiguity.

### Empty-class rules (per image, per class)

| GT | pred | handling |
| -- | ---- | -------- |
| absent | absent | **NA** — excluded from that class's averages and case counts |
| present | absent | all metrics = 0 (recall failure) |
| absent | present | BIoU=0, BF precision=0 → BF1=0; counts as false-positive case |

### Averaging

- **Primary**: macro-average over valid (image, class) pairs (NA excluded). Reported with n = valid cases.
- **Secondary**: pixel-aggregate (pool boundary bands across the whole split) as a sensitivity number.

## 2. Paired case analysis

Per (image, class) valid case: `Δ_metric = A2 − Base`. Report mean, median, SD, IQR, and positive / tie / negative counts (tie: |Δ| < 1e-6). Classes: crazing, inclusion, patches.

Bootstrap: **5000 paired resamples over case indices** (Base and A2 always sampled jointly) → 95% percentile CI for mean Δ of: per-image 4-class mIoU, per-class IoU, BIoU@2px, BF1@2px.
Mandatory caveat in all reporting: *CI reflects uncertainty across the fixed 840 test cases for seed-1337 predictions, not variability across training seeds.*

## 3. Scale stratification (bins frozen from GT distribution BEFORE seeing model results)

GT area-ratio distribution on the 840 test images (`gt_area_distribution.json`):

| class | n images | min | P33 | median | P67 | max | <1% | 1–5% | >5% |
| ----- | -------: | --: | --: | -----: | --: | --: | --: | ---: | --: |
| crazing | 364 | 0.0021 | 0.02732 | 0.0417 | 0.06186 | 0.3506 | 7.4% | 50.3% | 42.3% |
| inclusion | 360 | 0.0202 | 0.10642 | 0.1349 | 0.18496 | 0.6059 | **0.0%** | 8.9% | 91.1% |
| patches | 353 | 0.0054 | 0.03738 | 0.0576 | 0.08505 | 0.1750 | 2.8% | 41.6% | 55.5% |

- **Primary binning (per-class quantiles)**: Small = area_ratio ≤ class P33; Medium = (P33, P67]; Large = > P67. ~121/120/123 images per class tier.
- **Supplementary absolute binning**: <1% / 1–5% / >5%. **Pre-registered caveat**: the `<1%` bin is empty for inclusion and `>5%` holds 91.1% of inclusion cases → absolute bins are uninformative for inclusion and reported only with this caveat; if a bin has n < 20 it is reported but flagged underpowered.
- Metrics per (class × size group): IoU, Dice, BIoU@2px, BF1@2px (primary tolerance 2px; 1px/3px in machine-readable output), macro-averaged over images in the group (GT present by construction; pred empty → 0).

## 4. Morphology complexity (EXPLORATORY, crazing only)

Complexity proxy per image (GT present): `perimeter / sqrt(area)`, perimeter = |contour(gt)| pixel count. Tertile split (low/mid/high). Compare IoU and BF1@2px deltas across tertiles. Marked exploratory; cannot serve as sole primary evidence.
Connected-component-level analysis: **skipped with reason** — sample-level quantile analysis prioritized per brief; component-level perimeter/elongation deferred to avoid multiplying exploratory cuts (NEU per-image masks often merge adjacent regions, making component counts annotation-dependent).

## 5. Efficiency benchmark (separate chain from accuracy eval)

- Same GPU (A100-PCIE-40GB), same process/tool for both models; record GPU sharing state at benchmark time.
- Params: total / trainable, in M; Δ and relative %.
- MACs: `thop` @ input 1×3×200×200, reported as **tool-reported MACs**; FLOPs = 2×MACs stated explicitly as derived.
- Latency: batch=1, FP32, `model.eval()`, `torch.no_grad()`, full forward (all outputs returned, identical call pattern for both models); warmup 100 iters, measure 1000 iters, `torch.cuda.synchronize()` per iteration; report mean / median / P95 ms and FPS = 1000/mean_ms. Supplementary: batch=8 (warmup 50, measure 200).
- Peak GPU memory: batch=1 FP32 via `torch.cuda.reset_peak_memory_stats()` + `max_memory_allocated()`, in MB.
- Efficiency forward passes never feed accuracy metrics.

## 6. Case visualization (selection rule frozen)

For crazing and patches separately: rank valid cases by ΔBF1@2px → select top-3 improved, bottom-3 worsened, 3 closest-to-zero (|Δ| smallest). No hand-picking. Per case: original / GT / Base pred / A2 pred / Base error map / A2 error map (TP green, FP red, FN blue, other dark). Analysis-grade figures only, not final paper figures.

## 7. Decision framing (from brief §39, unchanged)

M1 boundary/detail-driven gain; M2 small-defect-driven gain; M3 broad gains; M4 essentially equivalent; M5 mixed/unstable evidence. Judged against: boundary deltas (all 3 tolerances, classwise), scale-group deltas, complexity trend, paired case counts, bootstrap CIs — with the sample-level ≠ seed-level caveat attached to every inferential statement.
