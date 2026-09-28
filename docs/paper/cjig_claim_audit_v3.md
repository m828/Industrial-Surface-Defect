# CJIG v3 最终 Claim 审查（paper_draft_cjig_v3.md）

审查日期：2026-09-25。证据来源：`experiments/audit/`（long240k 终点、2×2 消融、机制评价、效率 v2、leather_results.json / leather_efficiency_benchmark.json）。

## A. 禁用表述全文扫描（grep 实测）

扫描词：`显著提升 / 大幅提高 / 全面优于 / 普遍提升 / 有效解决 / 边界增强 / 小目标优势 / 泛化能力 / 全面提升 / 所有类别 / 普遍增强`

**结果：0 处违规命中。** 唯一含"泛化能力"的句子为引言背景句"传统机器视觉方法……泛化能力不足"（描述传统方法的局限，非本文 claim，合规）；"跨材质泛化"仅出现在后续工作展望（"围绕……展开"，非能力断言）。

## B. 逐条数字核验（v3 全文）

| # | 位置 | 表述 | 审计依据 | 状态 |
|---|---|---|---|---|
| V1 | 摘要/表1/结论 | NEU: 91.45% / 91.24%，+0.21 pt | long240k_endpoint_results.json（0.914544/0.912408，统一840固定终点） | ✅ |
| V2 | 表1 | 四类 IoU（98.64/84.47/93.55/89.16 vs 98.62/84.15/93.49/88.71） | 同上 per-class | ✅ |
| V3 | 摘要/表7 | FPS 37.37/38.16；参数 29.46M/29.37M（+0.31%）；MACs 6.648G 持平；延迟 +2.1% | efficiency_benchmark_results.json（v2 交错测量，共享 GPU 已加注） | ✅ |
| V4 | 2.3/表2 | patches ΔBIoU@1/2/3px +0.55/+0.54/+0.55（macro）、+0.74/+0.85/+0.86（aggregate），CI 排除 0 | boundary_quality_results.json | ✅ |
| V5 | 2.3/表2 | crazing 聚集 +0.32 pt；BIoU@2px −0.45；BF1@2px −0.60 | paired_case_analysis.json | ✅（双向如实） |
| V6 | 2.3/表3 | 尺度分层 ΔIoU 九格 | scale_stratified_results.json | ✅ |
| V7 | 2.3 | 高复杂度裂纹组 ΔIoU −1.74 pt（标注"探索性"） | morphology_exploratory_results.json | ✅ |
| V8 | 2.5/表5 | 10k 四格 85.22±0.41/85.70±0.43/85.44±0.19/85.75±0.54；ΔAAM 2/3 正向 +0.48pt；ΔDetail 3/3 正向 +0.23pt | a2ms_2x2_factorial_analysis.md | ✅（"当前实验设置下"限定语已加） |
| V9 | 2.5/表6 | 240k 终点 91.24/91.45 | long240k_endpoint_results.json | ✅（单 seed 局限已声明） |
| V10 | 2.4/表4 | 皮革：Base-B 87.21% / A2MS 84.99%（A2 低 2.22 pt）；FPS 35.84/37.11（768×768）；参数 29.37/29.46M | leather_results.json + leather_efficiency_benchmark.json | ✅（负差异如实呈现并注明主因类别） |
| V11 | 摘要 | "进一步在真实皮革缺陷数据集上考察模型在不同工业材质下的适用性" | 实验已完成；"考察适用性"为允许级表述（未用"验证/证明泛化"） | ✅ |
| V12 | 图4说明 | 固定选例规则 + "不作为定量比较依据" | viz/leather_viz_manifest.json | ✅ |
| V13 | 图1 | 结构与 fixed 模型一致（AAM=eSE+ACW、细节监督仅训练、无 ELMM） | figures/Fig1_A2MS_DSMONet.* 目检 | ✅ |
| V14 | 参考文献 | 12 条完整著录 | reference_audit.md（18/18 检索到；精简保留 11+ResNet） | ✅（Zuo H/Li Q 完整作者列表排版前补全——已在文献注中声明） |
| V15 | 2.1 | 皮革划分"1638/235/468" | 磁盘 + leather_protocol_final_decision.md | ✅（更正了 v1/v2 的对调错误） |

## C. 已知限制声明（文中均已保留）

1. NEU 240k 终点为单 seed 比较（局限性 1）；
2. 皮革跨材质实验 A2 < Base（局限性 2 + 正文 2.4 如实呈现）；
3. FPS 为共享 GPU 交错测量（表 4/表 7 脚注）；
4. 裂纹边界定位折中（2.3、3.1、局限性 4）；
5. 主流方法对比待协议核验后补充（2.2 注、局限性 5）。

## D. 结论

v3 全文**无无证据强 claim**；所有实验数字可溯源至 `experiments/audit/` 冻结文件；皮革实验为真实新运行（非旧链数字）；过强表述 0 残留。达到投稿前收口标准。
