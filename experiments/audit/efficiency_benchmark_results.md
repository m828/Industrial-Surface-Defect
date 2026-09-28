# Efficiency Benchmark Results — Base-B vs A2 Full @240k

Same GPU (A100-PCIE-40GB), same process/tool (`thop` 0.1.1), FP32, input 1×3×200×200, batch=1 primary.

**Protocol amendment (documented)**: persistent GPU sharing during benchmarking (3 foreign compute processes, ~18.5 GB, ~85–100% util) made v1 sequential measurement unfair (A2 implausibly "faster" than Base). v2 uses **interleaved 10×100-iteration segments** so contention affects both models equally; absolute latency/FPS below are therefore shared-GPU numbers (inflated vs idle), while the **Δ between models is fair**. Params/MACs/peak-memory are contention-independent.

**Inference-graph note**: in `eval()` mode both models return only `outputs[0]` (main seg head); the A2 detail head (`seg_head_detailloss`) and auxiliary heads x2/x3 are **training-only**. A2's inference-time additions over Base-B are exactly eSE (1×1 channel conv at 1×1 spatial) + AdaptiveChannelWeight — hence the near-zero MACs delta.

## Headline table

| Metric | Base-B | A2 Full | Δ (abs) | Δ (rel) |
| ------ | -----: | ------: | ------: | ------: |
| Params total (M) | 29.371 | 29.461 | +90,175 | **+0.31%** |
| Params trainable (M) | 29.371 | 29.461 | +90,175 | +0.31% |
| MACs (thop, tool-reported) | 6.64819 G | 6.64821 G | +16,128 | **+0.00024%** |
| FLOPs (derived = 2×MACs) | 13.296 G | 13.296 G | +32,256 | +0.00024% |
| Latency b1 mean (ms) | 26.205 | 26.759 | +0.554 | +2.11% |
| Latency b1 median (ms) | 22.781 | 22.903 | +0.122 | +0.54% |
| Latency b1 P95 (ms) | 42.424 | 45.428 | +3.004 | +7.08% |
| FPS (b1, = 1000/mean) | 38.16 | 37.37 | −0.79 | −2.07% |
| Peak GPU mem b1 (MB, median of 3) | 270.3 | 270.3 | +0.0 | +0.00% |
| Supplementary: b8 throughput (img/s) | 224.0 | 214.6 | −9.4 | −4.20% |

## Verdict

A2 Full's inference overhead is real but very small: +0.31% parameters, +0.0002% MACs, ≈+2% mean latency, zero peak-memory difference. On this hardware both models run at ~37–38 FPS (batch=1, shared GPU; faster on an idle GPU). This supports describing A2 as keeping the **realtime / efficient** property with a favorable accuracy-efficiency trade-off — though per the mechanism analyses, the accuracy side of that trade-off is itself small and class-dependent. Per the brief, we do not write "negligible overhead" as an unqualified claim; we report the measured +2.1% latency / +0.31% params.

Machine-readable: `efficiency_benchmark_results.json` (includes GPU state, both benchmark versions' provenance; v1 sequential numbers superseded by v2 interleaved and not used).
