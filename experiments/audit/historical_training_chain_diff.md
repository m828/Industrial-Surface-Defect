# Training-Chain Diff Audit: historical `subregion unet` vs current `new/`

> Date: 2026-09-08
> Chains compared:
> **HIST** = `subregion unet/train_neu_resnet_detailloss.py` + `dsmonet_resnet_detailloss.yml` (0.913 chain; the 0.9159 chain differs only in the 3-loss config)
> **A2MS-B** = `new/train_neu_resnet_detailloss.py` / `train_neu_resnet_detailloss_retrain_a2ms_b_originalsplit.py` + `config_neu_a2ms_b_retrain_clean_originalsplit.yml` (0.68 chain)
> **Base-B** = `new/train_neu_resnet.py` / `train_neu_base_b_control_originalsplit.py` + `config_neu_base_b.yml` (0.8546 chain)

## 1. Data

| Item | HIST | A2MS-B | Base-B |
|---|---|---|---|
| train list | `./dataset/train_neu.txt` (3630) | same file | same file |
| monitor list | `./dataset/test_neu.txt` (840) | same file | same file |
| train transform | `Transforms_PIL(200)` = random choice of {rotate90/180/270, h/v-flip, gaussian noise σ=10, gaussian blur 3×3}, each followed by resize 200² (img bilinear / label NEAREST); then ToTensor (/255) | identical | identical |
| monitor transform | `TestRescale(200)+ToTensor`, no Normalize | identical | identical |
| batch / shuffle / drop_last | 16 / True / True (train); monitor loader also batch16, drop_last=True → 832/840 per pass | 16 / True / True; monitor drop_last=True (training-time monitor); final unified eval = separate script, all 840 | same as A2MS-B |
| label remap | none (PNG values 0..3) | same | same |
| augmentation drift risk | `subregion unet/dataAug_new.py` has 2 debug prints in `ToTensor` (spam only, no numerics); `new/dataAug_new.py` additionally defines RandomScale/RandomCrop which are **not** reachable from `Transforms_PIL` (aug list = methods of `Augmentations_PIL` only) → numerics identical | | |

## 2. `label1` (binarymask) full trace — identical in all three chains

`datagenerator_neu.py` (byte-identical in `subregion unet/` and `new/`):

1. dataset: mask PNG → `convert('L')` → numpy → `mask0[mask0 != 0] = 1` → foreground=1, background=0 (binary, derived from the same GT mask)
2. dataloader: returned as third item; PIL mode L
3. transforms: NEAREST-resized alongside labels (rotate/flip applied geometrically to all three)
4. `ToTensor`: `torch.from_numpy` → `.long()`
5. training loop: `label1.squeeze(1).to(dtype=torch.int64)` → shape (N,200,200), values {0,1}
6. loss: used only in the binary terms `loss_fn[j](torch.stack([output_f, output_d],1), label1)` where `output_f = output[:,0]`, `output_d = output[:,1:].sum(1)` (foreground logit = sum of the 3 defect-class logits — note: plain sum, not logsumexp)

No version drift detected in label1 construction.

## 3. Detail/boundary target

- HIST: `detail_aggregate_loss` (from `tools/loss/loss.py`) builds the boundary target **inside the loss** from the 4-class `labels`: Laplacian conv (kernel [-1,-1,-1;-1,8,-1]) at strides 1/2/4/8, clamp>0.1→1, fused with fixed weights [0.6,0.3,0.1], then BCE+Dice against the 1-ch detail output. **But historically it never executed** (model returns 3 outputs; loop zips to len(outputs)=3; `loss_fn[3]` untouched). The 9130 checkpoint therefore contains **no detail-loss training** despite the run name.
- A2MS-B: model returns a real 4th output (`seg_head_detailloss(edge_laplacian)`), so `detail_aggregate_loss(output3, labels) * 3` **does execute** — target generation identical to the above.
- Base-B: no detail loss (3-loss config).

## 4. Loss formulas (as executed)

HIST (per iteration, weights `[10,1,3,1]` truncated to 3 outputs):

```
L = 10·OHEM(out0, y)          + 10·OHEM(bin(out0), y_bin)
  +  1·BCE4(out1, onehot(y))  +  1·BCE(bin(out1), y_bin)
  +  3·OHEM(out2, y)          +  3·OHEM(bin(out2), y_bin)
```

A2MS-B clean retrain (weights `[10,1,3,3]`, 4 outputs):

```
L = 10·OHEM(out0, y) + 10·OHEM(bin(out0), y_bin)
  +  1·BCE4(out1, onehot(y)) + 1·BCE(bin(out1), y_bin)
  +  3·OHEM(out2, y) + 3·OHEM(bin(out2), y_bin)
  +  3·detail_aggregate_loss(out3, y)        # ACTIVE here
```

Base-B: same as HIST (weights `[10,1,3]`, 3 outputs, no `j<3` guard needed).

where `bin(o) = stack([o[:,0], o[:,1:].sum(1)], dim=1)`; OHEM = `ohem_cross_entropy` (thres 0.7, min_kept 100000); BCE4 = `bce_with_logits_loss` with one-hot expansion.

Executed-loss differences HIST→A2MS-B: (a) output0 now comes from a 7×7 feature (see model diff), so the dominant ×10 terms train a different tensor; (b) +`3·detail_loss` on the laplacian-edge head; (c) supervision targets/masks themselves unchanged.

## 5. Optimizer / scheduler / loop mechanics

| Item | HIST | A2MS-B (current retrain) | Base-B (10k control) |
|---|---|---|---|
| optimizer | Adam lr 1e-4, wd 2e-6 | same | same |
| scheduler | **none** (ConstantLR; log: "Using No LR Scheduling") | cosine_annealing T_max 48000, eta_min 1e-6 | cosine_annealing T_max 48000 |
| total iters | 240000 (chained resumes) | 10k diagnostics (0.67–0.69 runs) | 10k control |
| val interval | 1000 | 500 | 500 |
| seed | 1337 (script default) | 1337 | 1337 |
| resume | `torch.load` (py3.9-era), model+opt+sched, strict default True | patched `weights_only=False` for torch 2.7; same strict semantics | same |
| AMP | none | none | none |
| BN | standard train/eval | same | same |
| best-ckpt rule | monitor mIoU improvement → save | same (+ iter-tagged names, overwrite guards) | same |

Note the scheduler discrepancy: HIST ran constant LR for 240k iters; current 10k runs use cosine over T_max=48000 → at iter 10k LR ≈ 0.93e-4 (mild difference, cannot explain a 0.85→0.68 gap; Base-B control shares the cosine schedule and reaches 0.8546).

## 6. Summary of chain-level differences that are real

1. **Model file** (decisive): HIST/Base-B use `model_dsmo_rs50.py`; A2MS-B uses `model_dsmo_rs50_eSE_adapt_detailloss.py` with `head_seg1 = high_feats` (7×7) instead of `arm2(...)` (56×56) — see `subregion_vs_new_rs50_diff.md`.
2. **Active 4th detail loss** in A2MS-B only (weight 3 on a 1-ch boundary head driven by the Laplacian edge feature).
3. Iteration budget: 240k (historical) vs 10k (current diagnostics) — explains why historical absolute numbers are higher, not why A2MS-B is *below Base-B at equal 10k*.
4. Scheduler: none (hist) vs cosine (current) — shared by Base-B control, so controlled for.
