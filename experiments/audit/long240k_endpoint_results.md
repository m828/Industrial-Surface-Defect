# long240k Endpoint Results (FROZEN)

- Protocol: `long240k_protocol_frozen.md` (Protocol L1, historical-like)
- Seed: 1337 (canonical; rationale in manifests)
- Endpoint rule: fixed 240000-iter budget, **best = last by design**, no test-based checkpoint selection at any point (training scripts construct no test loader)
- Eval: unified 840-image NEU test, batch=1, TestRescale+ToTensor, no ImageNet Normalize, `pred = outputs[0]`, 4-class IoU **including background**
- Frozen artifact: `long240k_endpoint_results.json` (sha256 `e55a346537c1824e1057496990c70792163d7990f2bc9a64f632f4c47bc6fdb5`), written before any intermediate-checkpoint eval
- 91% criterion (pre-registered): raw endpoint mIoU ≥ 0.910000

## Core endpoint table

| Model   | Seed |  Iter | Scheduler     |        bg |   crazing |  inclusion |   patches |     mIoU |
| ------- | ---: | ----: | ------------- | --------: | --------: | ---------: | --------: | -------: |
| Base-B  | 1337 | 240k | constant 1e-4 | 0.986201 | 0.841464 | 0.934863 | 0.887106 | **0.912408** |
| A2 Full | 1337 | 240k | constant 1e-4 | 0.986369 | 0.844681 | 0.935494 | 0.891633 | **0.914544** |

```
ΔLong = A2_240k − Base_240k = +0.002136 (+0.21 pt)
```

Per-class Δ (A2 − Base): background +0.000168, crazing +0.003217, inclusion +0.000631, patches +0.004527 — all four classes numerically positive.

## 91% verdicts

| Model   | endpoint mIoU | ≥ 0.910000? |
| ------- | ------------: | :---------: |
| Base-B  |      0.912408 |    **YES**  |
| A2 Full |      0.914544 |    **YES**  |

→ **Scenario B** (both reach 91%): the historical 91.x capability of the Base-like DSMONet family is reproduced under the current unified-840 fixed-endpoint protocol, and the fixed A2 Full also reaches 91% under identical conditions.

## Additional endpoint metrics

Dice: Base-B {bg 0.993053, crazing 0.913908, inclusion 0.966335, patches 0.940176}; A2 {bg 0.993138, crazing 0.915802, inclusion 0.966672, patches 0.942712}.
Full precision / recall / confusion matrices: see `long240k_endpoint_results.json`.

## Conclusion limits (single matched seed)

- A2's +0.21 pt endpoint advantage is a **numerical** advantage for seed 1337 only. Per the frozen protocol this must NOT be reported as "A2 consistently outperforms Base-B"; "consistent" requires multiple long-training seeds.
- Comparison to historical 0.913–0.916 is cross-protocol: historical numbers came from repeated 832/840 test-split monitoring with test-driven best-checkpoint saving under an older software stack; the present numbers are single fixed-endpoint evaluations with no test feedback. See `long240k_final_decision.md`.

## Provenance

- Checkpoints: Base-B `…/base_b/checkpoints/base_b_s1337_long240k_codex1_iter240000.pkl` (sha256 `46c5b83c…`), A2 `…/a2_full/checkpoints/a2_full_s1337_long240k_codex1_iter240000.pkl` (sha256 `5beaaa1c…`)
- Training completed: Base-B 2026-09-23 11:36 (wall 75221 s), A2 2026-09-23 20:36 (wall 32418 s); zero test evals during training
- INC-001: GPU/NVML unavailable 2026-09-23 ~23:57 → 2026-09-24 ~ (post-training); delayed evaluation only, no training impact; see `repro_runs/long240k_seed1337/INCIDENTS.md`
