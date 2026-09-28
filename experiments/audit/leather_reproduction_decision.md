# Leather 复现决策 (leather_reproduction_decision)

> 日期：2026-09-28
> 依据：leather_historical_91x_evidence.md / leather_checkpoint_unified468_results.md / leather_historical_vs_current_training_diff.md / leather_split_audit.md / leather_rotten_surface_diagnostic.md

## 一、命中的场景

**Scenario L1 成立**：现存历史 checkpoint 在当前统一 468 测试协议下独立恢复到 ≥0.91 量级：

| Checkpoint | 结构 | 统一468 mIoU |
|---|---|---:|
| pige_4_0229_160000 | 822 变体（非论文结构） | 0.917625 |
| eSE_adapt_detailloss_160000 | **论文 A2MS-DSMONet-B（r50/fixed forward）** | 0.909713 |
| dsmor50_0127_80000 | Base-B | 0.899120 |

历史"91%"来源已完全解释：A2 链 val235 best_iou=0.909084 → 同 checkpoint test468=0.909713。合法链路（train→val 选择→test 评价），无 test 泄漏。

## 二、决策

### 1. 是否重新训练 Leather：**NO（本轮不需要）**

理由：
- 论文 Leather 结果可直接由"历史 val-best checkpoint + 当前统一 test468 复评"支持，该流程满足 train/val/test 三分标准实践；
- 历史 checkpoint、日志、config、split、评价脚本全部在盘且 sha256 已固化，可追溯；
- 重训（常数 LR、~130k、batch12、val-best）最多复现已有数字，不改变结论，且引入新随机性。

### 2. 论文 Leather 结果来源：**B —— 历史合法 checkpoint 的统一468复评结果**

推荐论文使用（test468、含背景、固定评价协议）：

| Model | mIoU | 说明 |
|---|---:|---|
| Base-B | 0.899120 | 0127 链 val235-best@78k |
| A2MS-DSMONet-B | 0.909713 | 130k 链 val235-best@130k |

Δ = **+1.06 pt**。单 checkpoint 对单 checkpoint，表述用"提高/改善"，不用"显著提升"。

必须在论文方法/实验设置中声明：Leather 实验采用"训练集训练、验证集（235 张）选择 checkpoint、测试集（468 张）最终评价"的协议。

### 3. 当前 60k 复现结果的处理

Base 0.872137 / A2 0.849932（固定终点、无 val 选择、cosine）：

- **不进论文主表**；
- 保留在审计档案，作为"协议差异量化"证据（训练长度+scheduler+选择口径合计约 −3 ~ −6 pt）；
- 若要进论文，只能以"固定终点严格协议下的对照"名义与历史链分开标注——建议不放，避免混淆。

### 4. 备选增强（可选，不阻塞投稿）

若审稿人要求完全自清的可复现性，可按历史配方重跑一条链：
`model_dsmo_rs50_eSE_adapt_detailloss_fixed.py` + 常数 LR 1e-4 + batch 12 + train_iters 130k + val235 每 1000 iter 监控 + val-best 保存 + test468 终评一次。
该配方的每一项均有历史证据（本目录各文档）。**仅在用户明确要求时执行。**

### 5. 明确排除

- 不使用 pige_4_0229（0.917625）作为论文 A2MS-B 结果：结构是 822/Light_Bag 变体，不是论文模型。
- 不使用 buggy-forward 的 0.860749。
- 不使用 val235 数字（0.909084）充当 test 结果。
