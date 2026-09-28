# Leather 结果状态（历史链审计后）— 论文处理建议

> 日期：2026-09-28
> 状态：本文件为审计结论通知；**尚未修改 `paper_draft_cjig_v3.md`**。

## 一、v3 当前 Leather 数字的性质

v3 正文 Leather 结果（Base-B 87.21% / A2 84.99%，60k 固定终点、无 val 选择、cosine 调度）：
- 结果本身真实、可追溯（`experiments/audit/leather_results.json`）；
- 但协议与历史高分链不同，且复现链短于历史链（60k cosine vs 78k–130k 常数 LR + val-best），系统性偏低；
- **标记：PROVISIONAL — 建议在 v4 中替换。**

## 二、审计后的可用论文数字（统一 test468，含背景，batch1，TestRescale+ToTensor，无 Normalize）

| Dataset | Model | mIoU | 来源 |
|---|---|---:|---|
| Leather | Base-B | **0.899120** | 历史 0127 链 val235-best checkpoint（78k），本轮复评复核 |
| Leather | A2MS-DSMONet-B | **0.909713** | 历史 130k 链 val235-best checkpoint，正确 forward 复评 |
| Leather | Δ(A2−Base) | **+1.06 pt** | 同口径 |

per-class（A2 vs Base）：open_wound +2.73、scratch +4.67、brand_mark +1.29、hole +0.26、skin_disease +0.75、rotten_surface +2.60、wart −3.75、background −0.08（pt）。

## 三、写入 v4 时的强制要求

1. 实验设置必须写明：Leather 实验为"train 1638 训练 / val 235 选择 checkpoint / test 468 最终评价"。
2. claim 强度：A2 比 Base 提高 1.06 个百分点——用"提高/改善"，禁止"显著提升"。
3. 不得引用：91.0%（val 数字）、0.917625（822 变体，非论文模型）、0.860749（bug forward 伪低值）、84.99%/87.21%（60k 固定终点协议，除非单独标注协议差异）。
4. FPS/参数量沿用已审计的实测值（架构未变：29.37M/29.46M；768² FPS 35.84/37.11）。
5. wart 类 −3.7 pt 的回退不隐藏，可在类型差异分析中说明。

## 四、给 v4 的最小改动清单（待执行，需用户确认）

1. 2.4 节皮革表格：87.21/84.99 → 89.91/90.97，并补协议句。
2. 摘要中 Leather 相关句同步更新（如有数字）。
3. `leather_experiment_report.md` 增补一节指向本轮审计结论。
4. 讨论中"真实场景适用性"可保留，并补一句 A2 在皮革 7 类中 6 类改善、wart 回退的类型差异描述。
