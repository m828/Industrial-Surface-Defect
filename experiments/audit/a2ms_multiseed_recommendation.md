# A2MS Multiseed Recommendation

> Date: 2026-09-08. Decision-only document; no paper files modified.

## Verdicts (per mandated options)

- **AAM: 暂不支持** — ΔAAM mean +0.48 pt, positive 2/3 seeds, SD (0.79 pt) > mean; within run variability. Consistent small positive direction on crazing only (3/3).
- **Detail supervision: 暂不支持** — ΔDetail mean +0.05 pt, positive 1/3 seeds; indistinguishable from zero at 10k.
- **Full A2MS-B: 与 Base-B 基本相当** — ΔFull mean +0.53 pt, positive 2/3 seeds; mIoU-level gain unstable; class-level gains on crazing (+0.88 pt) and patches (+1.10 pt) are direction-consistent (3/3) but small.

## 主建议：方案B — 先补 Detail-only，形成完整 2×2 消融，再决定长训练

依据：

1. **ΔDetail 的测量被 AAM 混淆**：当前 detail supervision 只在"AAM之上"评估过，且结果不稳定（+0.50/−0.31/−0.04 pt）。Detail-only（Base-B 结构 + 第四路 detail supervision、无 eSE/ACW）是该效应唯一未被测过的单元；没有它，无法区分"detail 本身无效"与"detail 与 AAM 相互作用为负"。
2. **成本极低、信息量大**：只需 3 个 matched-seed 10k runs（沿用 seeds 1337/2026/3407 与完全锁定协议，约 35–40 分钟总时长），即可把 {B0, A1, D, A2} 拼成完整 2×2——这正是论文消融表所需结构，无论最终结论如何都必须补。
3. **其余选项当前都不成立**：
   - 方案A（直接设计长训练）：10k 下各组件效应尚未稳定（ΔFull SD 0.69 pt ≥ mean），在长训上花钱为时过早；长训决策应建立在 2×2 完成后的证据上。
   - 方案C（重新评估 AAM 设计）：ΔAAM mean 为正（+0.48 pt）且 crazing 3/3 同向，证据不支持现在宣判 AAM 需要重设计。
   - 方案D（重新评估 detail loss 设计）：ΔDetail mean ≈ 0 但非系统性为负；先测 D-only 再谈重设计。
   - 方案E（重新定义论文路线）：ΔFull mean +0.53 pt、2/3 seeds 为正、缺陷类一致小增益——"整体无稳定增益需重定义"强于当前证据；若 2×2 后 D-only 也 ≈ B0 且 A1 仍不稳定，则 E 自动成为下一轮主建议。

## 若方案B完成后的判读约定（预先登记，防止事后移动球门）

- D-only > B0 多数 seed 且 A2 ≥ D-only 多数 seed → detail 与 AAM 均有独立价值信号 → 进入方案A（长训练协议设计，matched-seed、统一终点评价）。
- D-only ≈ B0 且 A1 不稳定 → AAM/detail 均无独立稳定增益 → 转入方案E（重新定义论文方法路线；历史0.91x仍仅属Base-B族模型）。
- D-only > B0 但 A1 ≈ B0 → detail 有效、AAM 无效 → 转入方案C（围绕 detail 重构方法，移除或重设计 AAM）。

## 论文红线（不变）

- 本文件不改变任何论文内容；`fixed_existing_results.md` 不动。
- 论文不得声称 AAM 或 detail supervision "显著提升"（当前证据最多支持"在 10k 协议下对 crazing/patches 有方向一致的小幅数值改善"）。
- 历史 0.913–0.916 仅属于 Base-B 族（无 AAM、无 detail head、240k 长训），不得作为 A2MS-B 方法证据。
