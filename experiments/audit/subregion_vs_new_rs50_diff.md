# Model Diff Audit: `subregion unet/model_dsmo_rs50.py` vs `new/model_dsmo_rs50_eSE_adapt_detailloss.py` (reference: `new/model_dsmo_rs50.py` = Base-B)

> Date: 2026-09-08. Method: full-source reading of the three files + pyc-level verification that the historical compiled model matches the current `subregion unet` source (see `historical_neu_91x_evidence.md` §2).

## 0. Headline

`subregion unet/model_dsmo_rs50.py` (historical 0.913–0.916 entry) and `new/model_dsmo_rs50.py` (Base-B, 10k mIoU 0.8546) are **computationally identical**. `new/model_dsmo_rs50_eSE_adapt_detailloss.py` (the failing "A2MS-B" entry, 10k mIoU ≈ 0.67–0.69) differs from both in **three** places, one of which is a severe forward-path regression.

## A. Backbone — no differences

| Item | historical / Base-B | A2MS-B |
|---|---|---|
| Source | `model_resnet.py::resnet50()` (torchvision-derived, returns `x1,x2,x3,x4,x5` = strides 2/4/8/16/32, 64/256/512/1024/2048 ch) | same |
| pretrained | `False` (from scratch; kaiming init; no ImageNet weights) | same |
| `backbone_indices` | `[0,1,2,3,4]`; used: `[0]` (64ch, s2) and `[-1]` (2048ch, s32) | same |
| dilation / stem / freeze | none / standard 7×7 s2 + maxpool / none | same |

Identical in all three files. The backbone is not the cause.

## B. Detail–semantic interaction — data flow

Historical/Base-B forward (both files byte-equivalent in logic):

```
x5 (s32,2048) ──DAPPM──▶ high_feats(128) ──UAFM_arm1(x5,·)──▶ high_feat(128 @7×7)
high_feat ──SqueezeBodyEdge──▶ seg_body(7×7), seg_edge(7×7)
x1 (s2,64) ──bot_fine(s2)──▶ edge_shallow(128 @50×50) ──Laplacian──▶ edge_laplacian(50×50)
seg_edge ──conv_up(×8, ConvTranspose)──▶ edge_deep(56×56)
high_feat ──SELayer(128,red=128,sigmoid)──▶ high_feat_se ──conv_up──▶ feat_se(56×56)
seg_edge56 = edge_fusion( cat[edge_deep, edge_laplacian→56, feat_se] )   # 1×1 convs 384→128
high_sum  = high_feat + seg_body + high_feat_se        # @7×7
high_up   = conv_up(high_sum)                          # @56×56
out       = UAFM_arm2(seg_edge56, high_up)             # spatial-attention fusion @56×56
head_seg1 = out            # ◀◀◀ main output feature
output0   = seg_heads(head_seg1) → bilinear → 200×200  # used at eval AND as supervised output0
```

A2MS-B (`new/model_dsmo_rs50_eSE_adapt_detailloss.py:96-99`):

```python
high_feats = high_feat + seg_body + high_feat_se
high_feat  = self.conv_up(high_feats)
out        = self.arm2(seg_edge, high_feat)     # computed, then DISCARDED
head_seg1  = high_feats                          # ◀◀◀ 7×7 (stride-32) semantic sum
```

| Path | historical / Base-B | A2MS-B |
|---|---|---|
| main output `head_seg1` | `arm2(seg_edge, conv_up(high_sum))` @ **56×56**, detail-guided | `high_sum` @ **7×7**, no detail content, no arm2 |
| `arm2` module | in output path | dead branch (forward-computed, unused, no gradient) |
| detail→semantic fusion into output0 | yes (seg_edge + laplacian shallow edge via edge_fusion→arm2) | **none** — output0 sees only DAPPM/arm1/SE semantics at 7×7 |
| semantic→detail | seg_edge from SqueezeBodyEdge feeds edge branch | same (but its product never reaches output0) |

Consequence: A2MS-B's primary prediction is a 7×7 map bilinearly upsampled ≈28.6× to 200×200. Supervision on output0 (weight 10, the dominant term) trains exactly this low-resolution path, and eval reads the same path.

## C. Upsampling / spatial reconstruction

| Item | historical / Base-B | A2MS-B |
|---|---|---|
| `conv_up` | 3 × ConvTranspose2d(k2,s2): 128→256→128→128, ×8 | identical module |
| final upsample | `F.interpolate(·, 200×200, bilinear, align_corners=False)` | same |
| PixelShuffle / subregion / rearrange | **none anywhere** (despite the directory name "subregion unet", no subregion mechanism exists in this file) | none |
| effective output0 feature grid | 56×56 | **7×7** |

No spatial-reconstruction logic was "lost" from `subregion unet/` — the current `new/model_dsmo_rs50.py` preserves it exactly. What changed is only in the eSE_adapt_detailloss variant.

## D. Attention

| Item | historical / Base-B | A2MS-B |
|---|---|---|
| module | `SELayer(128, reduction=128)` → 128→1→128, **Sigmoid** | `eSE(128)`: AdaptiveAvgPool → Conv2d(128,128,1×1) → **HSigmoid** → × learnable `AdaptiveChannelWeight` (init 1) |
| position | `se(high_feat)` @7×7, then conv_up | same position |

**Answer to the mandated question:** the historical `subregion unet/model_dsmo_rs50.py` does **NOT** contain eSE / AdaptiveChannelWeight / any "adaptive attention". It has a plain SE block (with an aggressive reduction=128). The paper's "eSE / adaptive attention" component exists only in the `new/` eSE_adapt variant. (Verified at forward-graph level; the 2023-era pyc confirms this was already true historically.)

## E. Output heads

| | historical / Base-B | A2MS-B |
|---|---|---|
| # outputs (training) | **3** | **4** |
| output0 | seg_heads(head_seg1 @56×56) → 200², 4ch — main seg | seg_heads(high_sum @7×7) → 200², 4ch — main seg |
| output1 | seg_heads(seg_edge56 @56×56) → 200², 4ch — detail-branch aux seg | same (seg_edge path) |
| output2 | seg_heads(high_feat @7×7) → 200², 4ch — semantic aux | same |
| output3 | — (commented out) | `seg_head_detailloss(edge_laplacian @50×50)` → 200², **1ch** — boundary/detail |
| seg head sharing | one shared `SegHead(128,64,4)` for all outputs | shared `seg_heads` + separate `seg_head_detailloss(128,64,1)` |
| eval output | `[seg_heads(head_seg1)]` | `[seg_heads(head_seg1)]` — **but head_seg1 is the 7×7 tensor** |

So the historical "detailloss" run's forward is **three-output**, not four: in `train_neu_resnet_detailloss.py` the loop `zip(range(len(outputs)), outputs, weights=[10,1,3,1])` stops at j=2, and **`loss_fn[3]` (`detail_aggregate_loss`) is never executed** in the historical 9130 chain (verified: model pyc has no `seg_head_detailloss`; script logs merely print the configured list). The current "四路联合监督" understanding applies only to the `new/` eSE variant, where a real 4th output exists and `detail_aggregate_loss` actually runs with weight 3.

## F. Consolidated comparison table

| Aspect | historical 91x (`subregion unet/model_dsmo_rs50.py`) | Base-B (`new/model_dsmo_rs50.py`) | A2MS-B (`new/model_dsmo_rs50_eSE_adapt_detailloss.py`) |
|---|---|---|---|
| vs Base-B | identical (comments/whitespace + dead pre-compute at L100-102) | — | 3 real changes |
| main output resolution | 56×56 (arm2-fused) | 56×56 (arm2-fused) | **7×7 (raw semantic sum)** |
| arm2 used | yes | yes | **no (dead)** |
| SE block | SELayer(128, red=128), sigmoid | same | eSE(128), HSigmoid + learnable channel weight |
| # training outputs | 3 | 3 | 4 (incl. 1-ch detail head on edge_laplacian) |
| params | 29.37 M | 29.37 M | 29.37 M + eSE/detail-head overhead |
