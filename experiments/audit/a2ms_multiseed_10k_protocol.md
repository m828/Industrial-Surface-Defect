# A2MS Multiseed 10k Protocol (pre-registered)

> Date: 2026-09-08 (registered BEFORE any new run started)
> Goal: matched-seed paired comparison of Base-B / A1 / A2 at 10k under the locked NEU-Seg protocol, to judge whether AAM and detail supervision gains are stable across seeds.

## 1. Pre-registered seed list (frozen, no additions/removals based on results)

```text
seeds = [1337, 2026, 3407]
```

## 2. Experiment definitions (frozen)

| id | model file | forward | AAM | detail supervision | loss list | loss_weights |
|---|---|---|---|---|---|---|
| **B0** Base-B | `new/model_dsmo_rs50.py` | correct (head_seg1=arm2 out) | no (SELayer) | no (3 outputs) | [ohem_ce, bce, ohem_ce] | [10,1,3] |
| **A1** | `new/model_dsmo_rs50_eSE_adapt_detailloss_fixed.py` | correct (head_seg1=out) | yes (eSE+ACW) | off (weight 0 on output3; head kept) | [ohem_ce, bce, ohem_ce, detail_aggregate] | [10,1,3,0] |
| **A2** | same fixed file | correct | yes | on (weight 3) | same as A1 | [10,1,3,3] |

Δ_AAM = A1 − B0; Δ_Detail = A2 − A1; Δ_Full = A2 − B0 (all paired per seed).

## 3. Locked training protocol (identical across all 9 runs)

- data: NEU-Seg `new (copy)/dataset/train_neu.txt` (3630) / `test_neu.txt` (840); sha256 recorded per run
- train transform: `Transforms_PIL(200,200)` (random rotate/flip/noise/blur) + `ToTensor`; test: `TestRescale(200,200)+ToTensor`; no ImageNet Normalize anywhere
- batch 16; Adam lr 1e-4, wd 2e-6; scheduler cosine_annealing T_max=48000, eta_min=1e-6; train_iters 10000; from scratch (resume null)
- **seed mechanism (verified)**: scripts seed via `cfg.get("seed", 1337)` = TOP-LEVEL yaml key. Nested `training.seed` is ignored by the scripts. Therefore every new-seed config carries a top-level `seed: 2026` / `seed: 3407`. Existing configs without a top-level key ran 1337 by default (this is why all prior runs are seed 1337).
- matched-seed note: within a triplet, A1/A2 share identical model+script (init RNG bitwise-matched, verified iter-1 gradient identity last round); B0 differs only by architecture-driven RNG consumption (no eSE/ACW/detail-head modules) — inherent and unavoidable; pairing is at seed level.

## 4. Test-endpoint rule (no test-based selection)

- `val_interval = 10000` → exactly ONE monitor evaluation, at the 10k endpoint. No test-informed early stop, no best-checkpoint hunting, no extension. `best` in checkpoint filenames = last by design (single endpoint eval).
- Final reporting metric = unified 840 eval of the iter-10000 checkpoint: batch 1, all 840, TestRescale+ToTensor, no Normalize, `pred = outputs[0]`, 4-class IoU incl. background, plus Dice/Precision/Recall/confusion matrix.

## 5. Seed-1337 triplet: REUSED (not rerun)

Consistency check vs this protocol (§8 of the task):

| item | B0 (base_b_10k_control) | A1 (last round) | A2 (last round) |
|---|---|---|---|
| train/test lists | same files ✓ | same ✓ | same ✓ |
| scheduler | cosine T_max 48000 ✓ | ✓ | ✓ |
| optimizer | Adam 1e-4 / 2e-6 ✓ | ✓ | ✓ |
| train_iters / batch | 10000 / 16 ✓ | ✓ | ✓ |
| seed | 1337 (default) ✓ | 1337 (default) ✓ | 1337 (default) ✓ |
| endpoint rule | val_interval=10000 ✓ | ✓ | ✓ |
| unified 840 eval | project unified eval, batch1/840/TestRescale+ToTensor/no-Normalize/outputs[0] ✓ | same protocol ✓ | same protocol ✓ |
| loss_weights | [10,1,3] ✓ | [10,1,3,0] ✓ | [10,1,3,3] ✓ |

→ all items match; **seed 1337 = reused existing runs** (B0: mIoU 0.854558; A1: 0.857703; A2: 0.862727).

## 6. Exclusions (per task)

- no Detail-only (D) cell this round (no 2×2 factorial yet)
- no 20k/30k/60k/240k runs; no hyperparameter changes; no multi-seed paper edits; historical rs50 0.857853 NOT used as a noise estimate (different scheduler/environment, non-matched) — run variability is estimated from B0's own matched seeds only
- broken `model_dsmo_rs50_eSE_adapt_detailloss.py` untouched; no paper files touched

## 7. Run layout

```text
repro_runs/multiseed_10k/
    seed_2026/{base_b,a1,a2}/{code,config,logs,checkpoints,analysis,manifest.json}
    seed_3407/{base_b,a1,a2}/{code,config,logs,checkpoints,analysis,manifest.json}
```

Seed-1337 artifacts stay at their original locations (base_b_10k_control/, neu_a2ms_b_fixed_aam_{nodetail,detail}_10k_20260908_121000/) and are referenced, not copied.
