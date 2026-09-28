# A2MS Multiseed 10k Results (matched seeds 1337 / 2026 / 3407)

> Date: 2026-09-08. Protocol: `a2ms_multiseed_10k_protocol.md` (pre-registered before any new run).
> All numbers = **unified 840-image eval** of the iter-10000 endpoint checkpoint (batch 1, TestRescale+ToTensor, no Normalize, `pred = outputs[0]`, 4 classes incl. background). No test-based selection (val_interval=10000, best==last by design).
> seed 1337 = reused existing runs (consistency checklist in protocol §5); seeds 2026/3407 = new runs (`repro_runs/multiseed_10k/seed_*/`; manifests: `multiseed_*_s2026_manifest.json`, `multiseed_*_s3407_manifest.json`, all `python -m json.tool`-validated).

## 1. Main matched table (mIoU, unified 840)

| Seed | Base-B | A1 (fixed+AAM) | A2 (fixed+AAM+detail) | ΔAAM = A1−B0 | ΔDetail = A2−A1 | ΔFull = A2−B0 |
|---:|---:|---:|---:|---:|---:|---:|
| 1337 | 0.854558 | 0.857703 | 0.862727 | +0.003145 | +0.005024 | +0.008169 |
| 2026 | 0.847473 | 0.860870 | 0.857755 | +0.013397 | −0.003115 | +0.010282 |
| 3407 | 0.854493 | 0.852281 | 0.851870 | −0.002212 | −0.000411 | −0.002623 |
| **mean ± SD** | **0.852175 ± 0.004072** | **0.856951 ± 0.004344** | **0.857451 ± 0.005435** | **+0.004777 ± 0.007931** | **+0.000499 ± 0.004145** | **+0.005276 ± 0.006922** |
| positive seeds | | | | 2/3 | 1/3 | 2/3 |

Per-run training-time monitor (832/840) for reference — 1337: B0 0.854517, A1 0.857491, A2 0.862492; 2026: B0 0.847977, A1 0.860649, A2 0.857568; 3407: B0 0.854363, A1 0.852110, A2 0.851640.

## 2. Per-class IoU (unified 840)

| Seed | Model | background | crazing | inclusion | patches |
|---:|---|---:|---:|---:|---:|
| 1337 | B0 | 0.975691 | 0.745787 | 0.881747 | 0.815005 |
| 1337 | A1 | 0.975628 | 0.747054 | 0.889642 | 0.818488 |
| 1337 | A2 | 0.976608 | 0.757477 | 0.892093 | 0.824731 |
| 2026 | B0 | 0.974027 | 0.737988 | 0.878929 | 0.798947 |
| 2026 | A1 | 0.976266 | 0.752779 | 0.886911 | 0.827523 |
| 2026 | A2 | 0.975708 | 0.750970 | 0.885565 | 0.818779 |
| 3407 | B0 | 0.976060 | 0.742172 | 0.889088 | 0.810653 |
| 3407 | A1 | 0.974265 | 0.748694 | 0.878912 | 0.807254 |
| 3407 | A2 | 0.973899 | 0.743723 | 0.875805 | 0.814055 |

## 3. Per-class paired deltas (percentage points)

| class | ΔAAM per seed | pos | ΔDetail per seed | pos | ΔFull per seed | pos |
|---|---|---|---|---|---|---|
| background | −0.01 / +0.22 / −0.18 | 1/3 | +0.10 / −0.06 / −0.04 | 1/3 | +0.09 / +0.17 / −0.22 | 2/3 |
| **crazing** | **+0.13 / +1.48 / +0.65** | **3/3** | +1.04 / −0.18 / −0.50 | 1/3 | **+1.17 / +1.30 / +0.16** | **3/3** |
| inclusion | +0.79 / +0.80 / −1.02 | 2/3 | +0.25 / −0.14 / −0.31 | 1/3 | +1.04 / +0.66 / −1.33 | 2/3 |
| patches | +0.35 / +2.86 / −0.34 | 2/3 | +0.62 / −0.87 / +0.68 | 2/3 | **+0.97 / +1.98 / +0.34** | **3/3** |

## 4. Run variability (from B0's own matched seeds, as mandated)

B0 across seeds: 0.854558 / 0.847473 / 0.854493 → range 0.71 pt, SD 0.41 pt. Any component claim must exceed this scale to be considered real at 10k.

## 5. Exploratory paired statistics (n=3, reference only — NOT decision criteria)

- ΔAAM: mean +0.48 pt, paired t ≈ 1.04 (p ≈ 0.41)
- ΔDetail: mean +0.05 pt, paired t ≈ 0.21 (p ≈ 0.85)
- ΔFull: mean +0.53 pt, paired t ≈ 1.32 (p ≈ 0.32)

Per task rules: p-values are exploratory only; conclusions rest on effect direction + size + consistency.
