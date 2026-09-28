# A2MS-B Fixed-Forward Ablation Analysis

> Date: 2026-09-08. Evidence: `a2ms_b_fixed_forward_audit.md`, `a2ms_b_fixed_10k_results.md`, manifests A1/A2. All comparisons under the locked NEU protocol (3630/840, 200×200, 4 classes incl. background, unified 840 eval, seed 1337, 10k, from scratch).

## 1. Was `head_seg1 = high_feats` the main cause of the 0.689 collapse?

**Answer: 是（确认）.**

- Only the forward line differs between Broken A2MS-B (0.689430) and the fixed model; data, losses, weights, optimizer, scheduler, seed, and eval protocol are identical.
- The fix restores performance to 0.857703 (A1) / 0.862727 (A2), i.e. **+16.8 points recovered of the −16.5 gap** vs Base-B (0.854558).
- Static audit formally proves the mechanism: in the broken model all 18 arm2 parameter tensors receive `grad None` (dead branch) and `seg_heads` reads a 7×7 input; in the fixed model arm2 receives real gradients and `seg_heads` reads the 56×56 fused feature.
- Class-level signature matches the mechanism: the collapse was concentrated in fine-structure classes (crazing −0.24, patches −0.32 vs Base-B) — exactly what a 7×7→200×200 main output destroys; both classes fully recover after the fix (crazing 0.747–0.757, patches 0.818–0.825 ≈ Base-B 0.746/0.815).
- Corollary: the broken entry had eSE/ACW **in** the graph, so the collapse could not have been caused by AAM itself — it was masked by the forward regression.

## 2. Does AAM (eSE + AdaptiveChannelWeight) help?

**Answer: 暂不支持（无显著增益；但也无伤害证据）.**

- A1 (0.857703) vs Base-B (0.854558): +0.31 points. The same-model run-to-run spread observed in this project is comparable (Base-B 0.854558 vs historical rs50 0.857853, Δ0.33 points, identical architecture & protocol, different seeds/environments). The AAM effect is therefore **within the noise band and cannot be distinguished from zero with one seed**.
- AAM is functionally wired (GradAudit: eSE/ACW gradients grow from ~1e-4 to ~0.37/0.18 by iter 1000 — the modules are being optimized, not dead).
- Per the task's decision rule: "A1 ≈ Base-B → AAM 至少没有明显伤害性能，但是否有增益尚无充分证据". Multi-seed runs are required before claiming any positive AAM contribution in the paper.

## 3. Does detail supervision help (on the correct forward)?

**Answer: 支持（数值上一致正向，幅度小，单seed，需多seed确认稳定性）.**

- A2 (0.862727) > A1 (0.857703): +0.50 points, with **all four classes improving** (bg +0.10, crazing +1.04, inclusion +0.25, patches +0.62) — no class regression.
- The gain concentrates in the boundary-sensitive classes (crazing +1.04 points), consistent with the intended role of the edge/detail branch.
- This contradicts the earlier suspicion (from the broken-forward era, where detail loss weight sweeps could not exceed ~0.68) that the detail loss itself was harmful: on the correct forward, detail supervision is neutral-to-positive.
- Magnitude caveat: +0.50 points from a single seed is "slight/numerical improvement", not established significance; GradAudit shows the detail head's gradient is well-behaved (L2 ≈ 2–4.5, no explosion, no conflict signature).

## 4. Class-level movement summary (unified 840 IoU)

| class | Base-B | Broken | A1 | A2 | main effect |
|---|---:|---:|---:|---:|---|
| background | 0.9757 | 0.9447 | 0.9756 | 0.9766 | fully recovered |
| crazing | 0.7458 | 0.5065 | 0.7471 | 0.7575 | fix recovers; detail adds +1.0pt |
| inclusion | 0.8817 | 0.8069 | 0.8896 | 0.8921 | recovered, small extra |
| patches | 0.8150 | 0.4996 | 0.8185 | 0.8247 | fix recovers; detail adds +0.6pt |

## 5. Route decision (per mandated options)

**主建议：方案A（继续当前 A2MS-DefectNet 路线，进入更长训练/多seed验证），但附带两个前提条件。**

理由：
1. forward regression 修复后，A2 = 0.862727 是当前 10k 协议下最好结果（> Base-B +0.82pt，> historical rs50 +0.49pt），AAM+detail 组合没有表现出伤害；
2. detail supervision 的增益方向一致且机理自洽（边界类受益最大）。

前提条件（在写进论文前必须满足）：
1. **多seed验证**（≥3 seeds）确认 A2>A1 与 A1≥Base-B 的方向稳定；当前单seed下 AAM 单独增益未被证实（问题2）。
2. 论文中关于 AAM 与联合监督的所有表述，必须以 fixed-forward 结果为准；历史 0.913–0.916 不得作为 AAM 或四路监督的证据（其模型既无 eSE/ACW 也未执行 detail loss）。

若多seed后 A1≈Base-B 且 A2−A1 缩至噪声内，则自动降级为方案C（保留AAM、重设计detail supervision）或方案D；当前证据不支持直接跳到 B/C/D。

不建议本轮之后立即做 20k/30k/240k：先完成多seed 10k 筛查（成本约 30 min/seed/组），再决定长训预算。
