# Scale-Stratified Results — Base-B vs A2 Full @240k (seed 1337, frozen cache)

Bins frozen in `mechanism_eval_protocol.md` from GT distribution BEFORE any model comparison: per-class P33/P67 on GT area ratio (Small ≤ P33 < Medium ≤ P67 < Large). Metrics macro-averaged over GT-present images in each group. Primary tolerance 2px.

## Primary quantile table (Base / A2, Δ = A2 − Base)

| Class | Size | n | Base IoU | A2 IoU | ΔIoU | Base BF1@2px | A2 BF1@2px | ΔBF1 |
| ----- | ---- | -: | -------: | -----: | ---: | -----------: | ---------: | ---: |
| crazing | Small | 120 | 0.7072 | 0.6948 | **−0.01240** | 0.9180 | 0.9061 | −0.01197 |
| crazing | Medium | 124 | 0.8191 | 0.8141 | −0.00495 | 0.9341 | 0.9276 | −0.00652 |
| crazing | Large | 120 | 0.8827 | 0.8783 | −0.00438 | 0.9352 | 0.9279 | −0.00728 |
| inclusion | Small | 119 | 0.8734 | 0.8762 | +0.00282 | 0.8761 | 0.8820 | +0.00593 |
| inclusion | Medium | 122 | 0.9089 | 0.9065 | −0.00237 | 0.8905 | 0.8863 | −0.00414 |
| inclusion | Large | 119 | 0.9647 | 0.9642 | −0.00047 | 0.9555 | 0.9549 | −0.00058 |
| patches | Small | 117 | 0.7773 | 0.7846 | **+0.00733** | 0.9612 | 0.9621 | +0.00095 |
| patches | Medium | 119 | 0.8435 | 0.8431 | −0.00036 | 0.9541 | 0.9580 | +0.00393 |
| patches | Large | 117 | 0.9397 | 0.9446 | **+0.00490** | 0.9748 | 0.9788 | +0.00402 |

Per-group paired counts and bootstrap CIs: `scale_stratified_results.json`.

## Findings

1. **"Small 组增益最大"跨类别不成立**：crazing Small 是 A2 最差的组（ΔIoU −1.24 pt，ΔBF1 −1.20 pt），而 patches Small（+0.73 pt）与 inclusion Small（+0.28 pt IoU / +0.59 pt BF1）为正。尺度效应类别依赖，无统一"小缺陷受益"规律。
2. crazing 三个尺度组全负，且越小越负（Small −1.24 < Medium −0.50 < Large −0.44 pt）——与 boundary 结果同向：典型（尤其小面积）crazing 病例 A2 略差。
3. patches 正收益在 Small（+0.73 pt）与 Large（+0.49 pt），Medium 持平；幅度均 <1 pt。
4. 全部 9 个组别 |ΔIoU| ≤ 1.24 pt，无任何大幅度增益。

## Supplementary absolute bins (<1% / 1–5% / >5% area)

Pre-registered caveat honored: inclusion `<1%` bin empty (0.0% share), `>5%` holds 91.1% of inclusion → absolute binning uninformative for inclusion; n<20 bins flagged underpowered.

| Class | Bin | n | ΔIoU | ΔBF1@2px | flag |
| ----- | --- | -: | ---: | -------: | ---- |
| crazing | <1% | 27 | −0.01032 | −0.02367 | ok |
| crazing | 1–5% | 183 | −0.00896 | −0.00851 | ok |
| crazing | >5% | 154 | −0.00460 | −0.00598 | ok |
| inclusion | <1% | 0 | — | — | empty (pre-registered) |
| inclusion | 1–5% | 32 | −0.00274 | −0.00888 | ok |
| inclusion | >5% | 328 | +0.00024 | +0.00127 | ok |
| patches | <1% | 10 | +0.06053 | +0.02916 | **underpowered (n<20), not interpretable** |
| patches | 1–5% | 147 | +0.00236 | +0.00051 | ok |
| patches | >5% | 196 | +0.00223 | +0.00348 | ok |

Absolute-bin directions agree with the quantile analysis (crazing negative everywhere; patches positive; inclusion neutral). The patches `<1%` +6.05 pt figure rests on 10 images and is flagged non-interpretable.

Machine-readable: `scale_stratified_results.json`.
