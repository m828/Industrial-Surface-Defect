# A2MS-B Fixed-Forward Static Audit

> Date: 2026-09-08
> Subject: `new/model_dsmo_rs50_eSE_adapt_detailloss_fixed.py` (new file; the broken `model_dsmo_rs50_eSE_adapt_detailloss.py` is preserved untouched as failure evidence).

## 0. The bug and the fix

Broken (`new/model_dsmo_rs50_eSE_adapt_detailloss.py:96-99`):

```python
high_feats = high_feat + seg_body + high_feat_se   # 7x7, stride-32
high_feat  = self.conv_up(high_feats)              # 56x56
out        = self.arm2(seg_edge, high_feat)        # 56x56 detail-semantic fusion
head_seg1  = high_feats                            # ◀ arm2 DISCARDED; main output = 7x7 raw semantics
```

Fixed (only change; `diff` vs broken file = 1 logical line + comments):

```python
head_seg1 = out                                    # 56x56 fused feature, same as Base-B/historical
```

SHA256:

| file | sha256 |
|---|---|
| broken `model_dsmo_rs50_eSE_adapt_detailloss.py` | `2d2f6c43aff9cc5e98bb4d7c3188afbe82bdd87e26a741dce30053c86a715605` |
| fixed `model_dsmo_rs50_eSE_adapt_detailloss_fixed.py` | `a0e6fd52b00162ebd16805c473a59983cbd34c2308da115cd170ba1710bb376a` |

Audit script: `repro_runs/static_audit_fixed_forward.py`; raw JSON: `repro_runs/static_audit_fixed_forward.json`.

## 1. Main-output path (fixed model, verified by forward hooks)

```text
x (2,3,200,200)
→ backbone x5 (2,2048,7,7)
→ DAPPM cm            → (2,128,7,7)
→ UAFM arm1(x5, cm)   → high_feat (2,128,7,7)
→ SqueezeBodyEdge     → seg_body (2,128,7,7), seg_edge (2,128,7,7)
   backbone x1 (2,64,100,100) → bot_fine → (2,128,50,50) → Laplacian → (2,128,50,50)
   seg_edge  → conv_up(×8)   → edge_deep (2,128,56,56)
   edge_laplacian → bilinear → (2,128,56,56)
→ eSE(high_feat) → (2,128,7,7) → conv_up → feat_se (2,128,56,56)
→ edge_fusion(cat[edge_deep, edge_laplacian, feat_se]) → seg_edge (2,128,56,56)
→ high_feats = high_feat + seg_body + high_feat_se     → (2,128,7,7)
→ high_feat  = conv_up(high_feats)                     → (2,128,56,56)
→ out        = UAFM arm2(seg_edge, high_feat)          → (2,128,56,56)
→ head_seg1  = out                                     → (2,128,56,56)   ✔ fused feature
→ seg_heads(head_seg1) → (2,4,56,56) → bilinear 200×200 → output0 (2,4,200,200)
```

`output0` therefore comes from the **56×56 detail-semantic fused feature**, not the 7×7 raw semantic sum. (Contrast: in the broken model the first `seg_heads` call fires on a **7×7** input — hook record `seg_heads#0 = (2,4,7,7)`.)

Auxiliary heads (training mode): output1 = seg_heads(seg_edge 56×56)→200²; output2 = seg_heads(high_feat 7×7)→200²; output3 = seg_head_detailloss(edge_laplacian 56×56)→(2,1,56,56)→200². Eval mode returns 1 output (2,4,200,200).

## 2. arm2 is no longer a dead branch (gradient proof)

Backward through the exact training loss (weights [10,1,3,3], ohem/bce/ohem/detail), random batch, seed 1337:

| module | fixed model grad L2 | broken model grad L2 |
|---|---:|---:|
| **arm2** | **5.162626** (12/18 param tensors) | **0.000000 (0/18 — fully dead)** |
| arm1 | 55.97 | 220.25 |
| eSE | 1.74e-4 | 5.66e-2 |
| AdaptiveChannelWeight | 9.4e-5 | 3.06e-2 |
| edge_fusion | 2.885 | 0.392 |
| seg_heads | 18.30 | 22.16 |
| detail_head | 3.142 | 3.142 |
| DAPPM | 17.12 | 68.04 |
| backbone | 3181.9 | 12720.7 |

- Fixed model: arm2 receives real gradients → participates in `loss(output0)`. The 6/18 arm2 param tensors without grad are `conv_xy_atten1` (channel-attention branch), unused because `arm2 = UAFM(128,128,tokens=False)` is spatial-only — identical to Base-B/historical by design.
- Broken model: all 18 arm2 param tensors have `.grad is None` — formally confirming the dead branch.
- eSE and AdaptiveChannelWeight receive nonzero (small at init) gradients in the fixed model → AAM is genuinely in the optimization path (via feat_se→edge_fusion→arm2 and high_feat_se→high_feats→arm2).

## 3. Tensor-shape record (fixed model, batch 2, training mode)

| tensor | shape |
|---|---|
| input | (2,3,200,200) |
| DAPPM out / arm1 out / seg_body / seg_edge / eSE out | (2,128,7,7) |
| bot_fine / laplacian | (2,128,50,50) |
| edge_deep / feat_se / seg_edge(fused) / conv_up(high_feats) / **arm2 out = head_seg1** | (2,128,56,56) |
| seg_heads#0 (output0 pre-upsample) | (2,4,56,56) |
| seg_heads#2 (output2 pre-upsample) | (2,4,7,7) |
| seg_head_detailloss (output3 pre-upsample) | (2,1,56,56) |
| output0..2 / output3 | (2,4,200,200) / (2,1,200,200) |
| eval outputs | 1 × (2,4,200,200) |

Matches the expected `7×7 → conv_up → 56×56 → arm2 → head_seg1 → seg_heads → 200×200` profile.

## 4. Conclusion of static audit

- Fix is exactly one line; no other module, loss, or hyperparameter touched.
- output0 path restored to Base-B/historical semantics (56×56 detail-semantic fusion).
- arm2 gradient flow proven; AAM (eSE + AdaptiveChannelWeight) gradient flow proven; detail head gradient flow proven.
- Cleared to start controlled 10k experiments A1/A2.
