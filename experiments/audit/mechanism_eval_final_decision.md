# Mechanism Evaluation — Final Decision

**Question (brief §49): with overall mIoU only +0.21 pt over Base-B, what does A2 actually improve?**

## Evidence summary (all from frozen 240k checkpoints, locked 840 test split, seed 1337)

| Axis | Result | Direction |
| ---- | ------ | --------- |
| Overall per-image 4-class mIoU Δ | +0.00031, CI [−0.00176, +0.00251], 437/2/401 pos/tie/neg | **null** |
| crazing boundary (BIoU/BF1 @1/2/3px, macro + aggregate) | Δ ≈ −0.4 ~ −1.0 pt, CIs exclude zero (negative) | **negative** |
| crazing region IoU | aggregate +0.32 pt (endpoint); per-case mean −0.71 pt, CI negative | mixed → typical case negative |
| inclusion boundary & region | CIs all include zero | null |
| patches boundary (BIoU @1/2/3px) | +0.55 ~ +0.86 pt, **CIs exclude zero (macro & aggregate)** | **positive, robust across tolerances** |
| patches BF1 | +0.28 ~ +0.48 pt, CI excludes zero @1px only | weak positive |
| Scale effect | class-dependent: crazing worst in Small (−1.24 pt), patches best in Small (+0.73 pt) | no uniform small-defect law |
| crazing complexity trend | low ≈ 0, mid −0.43 pt, high **−1.74 pt** | **inverse of hypothesis** |
| Efficiency | params +0.31%, MACs +0.0002%, latency +2.1%, peak mem +0.0% | overhead negligible-to-small |

## Mechanism verdict: **M5 — 证据混合，无法形成稳定机制解释**

Why not the others:
- **M1 (boundary/detail-driven)**: rejected as a general claim — the most boundary-sensitive class (crazing) gets *worse* under A2 (all tolerances, both averaging modes, CI-backed), and the complexity trend is inverse. Only patches shows robust boundary gains.
- **M2 (small-defect-driven)**: rejected — Small helps patches (+0.73 pt) but hurts crazing (−1.24 pt); no cross-class consistency.
- **M3 (broad gains)**: rejected — inclusion is null, per-image mIoU CI includes zero, 401/840 images worsened.
- **M4 (essentially equivalent)**: close, but too strong — the patches BIoU gain is reproducible across all three tolerances and both averaging modes with CIs excluding zero; a real, if small, class-specific effect exists.

The defensible mechanism statement is narrow: **A2's modules yield a small, reproducible boundary+region improvement on patches and interior-region filling on large crazing regions, at the cost of slightly reduced boundary precision on thin/complex crazing — net effect ≈ +0.2 pt mIoU at 240k, with near-zero inference overhead.** The originally hoped-for narrative ("AAM + detail supervision systematically improve boundary/detail-sensitive defects") is **not supported**.

## Next-step recommendation: **P3 — 需要简化 AAM/detail 贡献**

Rationale:
1. The accuracy evidence does not justify P1 (enter paper-results restructuring now): single-seed +0.21 pt with mixed mechanism is not a results story.
2. P2 (multi-seed 240k, ~6 GPU·days) is not the bottleneck: even if the +0.21 pt proved seed-stable, the mechanism analyses show the gain is not the claimed mechanism — more seeds would only firm up a tiny, mechanistically-mixed effect.
3. P4 (redesign the method) is overreaction: A2 is not worse overall and keeps realtime cost; nothing here invalidates the base architecture.
4. **P3**: rewrite the contribution around what is verifiable — (a) the fixed detail-semantic DSMONet pipeline reaching 0.9124/0.9145 under a clean fixed-endpoint protocol; (b) a small class-specific boundary gain on patches; (c) near-zero inference overhead (+0.31% params, +2.1% latency) — and drop/soften claims that AAM/detail supervision generally improve boundary or detail-sensitive defect quality. If the paper needs a stronger method story, the honest path is redesign informed by these failure modes (thin-crack boundary erosion), which is a new research iteration, not a reporting one.

## Compliance record

- No training, no new checkpoints/seeds, no threshold/post-processing/TTA, no test-set or metric-definition changes after freezing (`mechanism_eval_protocol.md` signed before any model comparison; GT-distribution stats computed pre-comparison to freeze bins).
- Single prediction cache (`repro_runs/mechanism_eval_240k/`, `cache_manifest.json`); frozen endpoint mIoU reproduced exactly (0.912408 / 0.914544) before any analysis proceeded.
- Efficiency chain separated from accuracy chain; v1 contention-biased latency superseded by interleaved v2 (documented amendment).
- No edits to paper or `fixed_existing_results.md`.
- INC-001 (GPU outage) delayed evaluation only; zero training impact.
