# A2MS-B Fixed-Forward 10k Controlled Results

> Date: 2026-09-08
> Experiments: **A1** = fixed forward + AAM (eSE + AdaptiveChannelWeight), detail supervision OFF (`loss_weights [10,1,3,0]`); **A2** = fixed forward + AAM + detail supervision ON (`loss_weights [10,1,3,3]`).
> Both: fixed model `model_dsmo_rs50_eSE_adapt_detailloss_fixed.py` (single change `head_seg1 = high_feats → out`), from scratch, seed 1337, 10k iters, batch 16, Adam 1e-4, wd 2e-6, cosine_annealing T_max 48000, val_interval 10000 — identical to Base-B 10k control and A2MS-B 10k diagnostics.
> All numbers below are **unified 840-image evals** (batch 1, TestRescale+ToTensor, no Normalize, `pred = outputs[0]`, 4 classes incl. background). Training-time monitor numbers (832/840) are in the manifests and were: A1 0.857491, A2 0.862492.

## Main result table

| Model | Correct fused head | eSE + AdaptiveChannelWeight | Detail supervision | background | crazing | inclusion | patches | mIoU |
|---|:-:|:-:|:-:|---:|---:|---:|---:|---:|
| Base-B (control) | Yes | No | No | 0.975691 | 0.745787 | 0.881747 | 0.815005 | 0.854558 |
| Historical rs50 (reference) | Yes | No | No | 0.975534 | 0.753761 | 0.885399 | 0.816719 | 0.857853 |
| Broken A2MS-B (control) | **No (7×7)** | Yes | Yes | 0.944719 | 0.506517 | 0.806921 | 0.499565 | 0.689430 |
| **A1 fixed+AAM/no-detail** | Yes | Yes | No | 0.975628 | 0.747054 | 0.889642 | 0.818488 | **0.857703** |
| **A2 fixed+AAM+detail** | Yes | Yes | Yes | 0.976608 | 0.757477 | 0.892093 | 0.824731 | **0.862727** |

Per-class Dice / precision / recall and confusion matrices: `a2ms_b_fixed_a1_manifest.json`, `a2ms_b_fixed_a2_manifest.json`.

## Deltas (percentage points)

| comparison | mIoU Δ | bg Δ | crazing Δ | inclusion Δ | patches Δ |
|---|---:|---:|---:|---:|---:|
| A1 − Broken A2MS-B | **+16.827** | +3.091 | +24.054 | +8.272 | +31.892 |
| A1 − Base-B | +0.315 | −0.006 | +0.127 | +0.790 | +0.348 |
| A2 − A1 (detail on) | +0.502 | +0.098 | +1.042 | +0.245 | +0.624 |
| A2 − Base-B | +0.817 | +0.092 | +1.169 | +1.035 | +0.973 |
| Historical rs50 − Base-B (same-model noise gauge) | +0.330 | — | — | — | — |

## Supplementary diagnostics

- GradAudit (parameter grad L2) at iter 1/100/1000 — A1: arm2 30.01/16.66/20.90; eSE 2.9e-4/8.2e-3/0.368; ACW 1.3e-4/1.8e-3/0.184; detail_head 0.0/0.0/0.0 (off by design). A2 identical except detail_head 4.54/2.03/3.19 (active, no explosion).
- At iter 1, A1/A2 gradients over shared modules are bitwise-similar (arm2 30.011051 vs 30.011047) — same seed, same init; the only divergence source is the detail term.
- Training losses @10k: A1 8.34 (L1 4.09 / L2 4.25), A2 10.68 (L1 4.08 / L2 6.60, incl. detail term); Base-B control ended at 10.27.

## Reproducibility anchors

- Run dirs: `repro_runs/neu_a2ms_b_fixed_aam_nodetail_10k_20260908_121000/`, `repro_runs/neu_a2ms_b_fixed_aam_detail_10k_20260908_121000/`
- Best checkpoints (sha256): A1 `65784a7b…bcf8`, A2 `eb97d7d2…1ea55` (full hashes in manifests)
- Manifests: `experiments/audit/a2ms_b_fixed_a1_manifest.json`, `experiments/audit/a2ms_b_fixed_a2_manifest.json` (validated with `python -m json.tool`)
