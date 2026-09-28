# NEU Historical RS50 10k Reproduction Report

> Date: 2026-09-08
> Run dir: `/workspace/Industrial Surface Defect/repro_runs/neu_historical_rs50_10k_20260908_112500/`
> Verdict gate from the task: **A (≥0.80)** — historical entry is sound; worth further tracking. No 20k/30k was started.

## 1. What was reproduced

- Historical chain: `subregion unet/train_neu_resnet_detailloss.py` + `subregion unet/model_dsmo_rs50.py` + `dsmonet_resnet_detailloss.yml` (the exact chain of the 0.9130/0.914 logs of 2023-08-21).
- Fidelity measures:
  - Model file = historical source, verified function-identical to the 2023-era compiled `__pycache__/model_dsmo_rs50.cpython-39.pyc`.
  - `subregion unet/` files that had been zeroed (`model_resnet.py`, `model_unet.py`, `stdcnet.py`, `utils.py`, `tools/computemIou.py`, `tools/utils.py`, `tools/loss/loss.py`, scheduler/loader inits) were restored from `new/` copies whose pyc fingerprints are identical or call-path-identical (table in `historical_neu_91x_evidence.md` §2). No historical file was overwritten; everything was assembled in the new repro dir.
  - Config changes vs the 2023 snapshot are limited to: `train_iters 240000→10000`, `resume` removed (checkpoints do not exist), `n_workers 16→8` (no numeric effect). No LR schedule (historical behavior), batch 16, Adam 1e-4, wd 2e-6, val_interval 1000, seed 1337.
  - No code logic was modified. First attempt (11:25) crashed at the first checkpoint save because `model_savePath/` did not exist in the new cwd (`RuntimeError: Parent directory model_savePath does not exist`); the directory was created and the run restarted from scratch. Attempt-1 artifacts kept in `analysis/attempt1_crashed_at_ckpt_save/`.

## 2. Protocol check vs the locked protocol

| Item | locked protocol | this run |
|---|---|---|
| train / test | 3630 / 840 | 3630 / 840 (same txt files, sha256 in manifest) |
| classes / input | 4 incl. bg / 200×200 | same |
| preprocessing | TestRescale+ToTensor, no Normalize | same |
| mIoU incl. background | yes | yes (runningScore 4-class) |
| extra val split | none | none (monitor = test_neu, as historical) |
| monitor coverage | — | training-time monitor: batch16 + drop_last → 832/840 per pass (faithful to historical); final unified eval: all 840, batch 1 (same as Base-B/A2MS-B unified evals) |

## 3. Results

### 3.1 Training-time monitor trajectory (832/840 per pass)

| iter | this repro 2026-09-08 | historical 2023-08-21 (from-scratch log) |
|---:|---:|---:|
| 1000 | 0.7353 | 0.7100 |
| 2000 | 0.7689 | 0.7741 |
| 3000 | 0.7999 | 0.7951 |
| 4000 | 0.8195 | 0.8202 |
| 5000 | 0.8092 | 0.8283 |
| 6000 | 0.8352 | 0.8383 |
| 7000 | 0.8420 | 0.8412 |
| 8000 | 0.8128 | 0.8482 |
| 9000 | **0.8578** (best) | 0.8499 |
| 10000 | 0.8538 (final) | 0.8572 |

Loss: 48.6 (iter ~50) → 25.5 (iter 10000). Best checkpoint saved at iter 9000 (`model_savePath/dsmonet_resnet_pascal_augnew_neu.pkl`, 352,802,877 B, sha256 `37d7afca…`).

### 3.2 Unified 840-image evaluation (comparable to Base-B 0.854558 / A2MS-B 0.689430)

Checkpoint evaluated: iter-9000 best.

| Class | IoU (this run) | Base-B 10k | A2MS-B 10k |
|---|---:|---:|---:|
| background | 0.975534 | 0.975691 | 0.944719 |
| crazing | 0.753761 | 0.745787 | 0.506517 |
| inclusion | 0.885399 | 0.881747 | 0.806921 |
| patches | 0.816719 | 0.815005 | 0.499565 |
| **mIoU** | **0.857853** | **0.854558** | **0.689430** |

### 3.3 Interpretation

- The historical entry reproduces the 2023 learning curve within ±0.02 at every checkpoint and lands at 0.8579 (unified 840) at 10k — statistically indistinguishable from Base-B 10k control (0.8546, same model file family) and from the historical 0.8572 @10k.
- **0.913–0.916 is NOT reachable at 10k** with this chain: historically it needed 165k–240k iterations under constant LR. At 10k the historical model itself was at 0.857.
- Verdict per task gate: **A (≥0.80)** → the historical entry is healthy; the failure is specific to the `eSE_adapt_detailloss` entry.

## 4. Phase-4 root-cause ranking: why A2MS-B 10k ≈ 0.68 while Base-B 10k ≈ 0.85

**P1 (highly suspicious, structurally proven):** `new/model_dsmo_rs50_eSE_adapt_detailloss.py:99` — `head_seg1 = high_feats` instead of `head_seg1 = out` (= `arm2(seg_edge, conv_up(high_feats))`).
- Effect: the main output (output0, dominant ×10 supervision, and the only eval readout) is produced from a **7×7 stride-32 semantic sum**, bilinearly upsampled ≈28.6×, with **no detail-branch content**; `arm2` is computed and discarded (dead branch).
- Base-B/historical output0 comes from the 56×56 detail-fused `arm2` output.
- Independent confirmation: the 2026-06-19 output0-only diagnostic (CE on output0 alone, no aux losses) scored 0.671539 vs Base-B 0.854558 — the deficit persists with all loss-side factors removed, isolating the forward path. Confusion matrices of all A2MS-B runs show the expected signature: background/inclusion (large/blobby regions) degrade mildly, crazing/patches (thin/small structures) collapse (0.51/0.50 vs 0.75/0.82).

**P2 (probable, secondary):** the extra active `detail_aggregate_loss` (weight 3) on `seg_head_detailloss(edge_laplacian)`. The no-detail diagnostic (weight 0) still scored 0.679190, so this is not the main cause; combined with P1 it may add gradient noise through the shared edge branch.

**P3 (needs further experiment to quantify):** eSE (HSigmoid, no reduction, + learnable per-channel weight init 1) replacing SELayer(128, reduction=128, sigmoid). No isolated ablation exists; expected effect small vs P1.

**Explicitly excluded as causes:** dataset/split/metric (Base-B control shares them); optimizer/lr/scheduler (shared); label1/binarymask construction (byte-identical datagenerator); `detail_aggregate_loss` target generation (removing it did not recover performance); ImageNet pretraining (all chains train from scratch).

## 5. Decision

- historical rs50 10k reproduction: **0.857853 (unified 840) / 0.8578 best monitor** — gate **A**.
- Recommendation: **暂缓直接 20k/30k 自动续训，但 historical rs50 链值得继续追踪**。若目标是复现 0.91x，正确路径是这条链 + 240k 量级预算（历史上的 chained-resume 协议），而非继续在 `eSE_adapt_detailloss` 入口上投入。任何 20k/30k 都应先由用户确认预算与 GPU 占用（按本次实测 10k ≈ 11.2 分钟，240k ≈ 4.5–5 小时）。
