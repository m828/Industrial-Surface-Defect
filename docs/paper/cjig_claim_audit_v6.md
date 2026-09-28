# CJIG v6 Claim 审查 (cjig_claim_audit_v6)

> 日期：2026-09-28
> 对象：`paper_draft_cjig_v6_dsmonet_a2ms_integrated.md`
> 原则：claim 强度 ≤ 证据强度；区分事实结果 / 相对比较 / 机制解释。

## Claim 分级

### Claim A — 事实结果（单一正式训练链，可直接陈述）

| # | Claim | 证据 | 状态 |
|---|---|---|---|
| A1 | DSMONet-B NEU 91.24% mIoU（840，240k 固定终点，seed1337） | `experiments/audit/neu_current_base_9124_manifest.json`，复评逐位一致 | 通过 |
| A2 | A2MS-DSMONet-B NEU 91.45% / 37.37 FPS / 29.46M | 240k 冻结终点审计 | 通过 |
| A3 | Leather：DSMONet-B 89.91% / A2MS 90.97%（test468） | `leather_paper_final_results.json`；Fig4 重渲一致性门复核通过 | 通过 |
| A4 | Leather 主流对比 84.21/86.20/88.24（Level A 同协议复评） | `mainstream_comparison_audit_v5.md` | 通过 |

### Claim B — 相对比较（各一套训练链，不写显著性）

| # | Claim | 限定 | 状态 |
|---|---|---|---|
| B1 | NEU +0.21 pt | 文中写"数值提高 0.21 个百分点"，并在 3.2 注明单种子、处于训练随机性量级 | 通过 |
| B2 | Leather +1.06 pt | 单权重对单权重；3.2 已注明训练随机性未量化 | 通过 |
| B3 | 参数量 +0.31%、MACs 持平、延迟 +2.1% | 实测表 8 | 通过 |

### Claim C — 机制/形态解释（分析性结论）

| # | Claim | 证据 | 状态 |
|---|---|---|---|
| C1 | 斑块类区域+边界同步改善（BIoU@1/2/3px +0.55/0.54/0.55，bootstrap CI 不含零） | `boundary_quality_results.json` | 通过 |
| C2 | 裂纹类区域完整性—边界定位权衡；高复杂度组 ΔIoU −1.74 | `morphology_exploratory_results.json` | 通过（文中注明探索性） |
| C3 | 皮革 6/7 类正向、刺猴 −3.75 pt | 表 5 | 通过 |
| C4 | 增强效果具有缺陷形态依赖性 | C1–C3 合并 | 通过 |

## 禁用/降档词扫描（全文 grep）

| 词 | 命中 | 处理 |
|---|---|---|
| 显著/大幅 | 0（仅"形态差异显著/明显"作描述性限定，非统计声称） | 通过 |
| 全面提升/普遍/所有类别 | 0；相反，文中明确"并非对所有缺陷类型产生一致增益" | 通过 |
| 泛化能力（强 claim） | 0；用"跨材质适用性考察/应用验证" | 通过 |
| 首次提出/全新 | 0 | 通过 |
| 优于/高于（比较级） | 仅与同协议复现方法比较处使用，均有表 4 支撑 | 通过 |

## 命名相关风险检查

- DSMONet-B 定义为"细节—语义互优化基础网络（ResNet-50 主干）"，未包装为新设计；贡献 1 措辞为"构建"，引言指出其为"本文首先构建"的基础架构——与旧撤稿稿关系经 Cover Letter 披露（见 disclosure v6），正文不展开。
- A2MS-DSMONet 定位严格为"在 DSMONet 基础上的面向工业缺陷增强"，两处（1.1、贡献2）一致。
- mask-aware loss 与 detail supervision 均明确标注已有来源（1.4 节 + 引言文献段），未声称原创。

## 结论

v6 全部 claim 与证据等级匹配，无越级表述；0 个 unsupported claim。
