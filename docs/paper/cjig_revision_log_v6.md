# CJIG v6 修订日志 (cjig_revision_log_v6)

> 日期：2026-09-28
> 输入：`paper_draft_cjig_v5_submission_candidate.md`
> 输出：`paper_draft_cjig_v6_dsmonet_a2ms_integrated.md`
> 性质：技术主线重构 + 模型命名统一。**未改动任何实验数字、数据划分、评价协议。**

## 一、主线重构（v5 → v6）

| 维度 | v5 | v6 |
|---|---|---|
| 叙事结构 | Base-B + 两个模块（AAM/细节监督） | 两层方法体系：DSMONet（通用细节—语义互优化基础架构）→ A2MS-DSMONet（面向工业缺陷的增强） |
| 基础模型名 | Base-B（2026-05 实验整理期人工简称） | DSMONet-B（恢复历史本名；IDENTITY-A 已证实二者同一计算图，max_abs_diff=0.0） |
| 91.24% 的角色 | "基础网络 Base-B" 的基线数字 | DSMONet 通用架构本身的性能，正文明确其"较强基础分割能力"定位 |
| 贡献结构 | 模型 / 分析方法 / 验证 | ① 通用基础架构（DSMONet，含 91.24 结果）② 工业缺陷针对性增强（AAM+细节监督）③ 双数据集系统验证与形态分析 |
| 方法章节 | 1.1 总体结构 / 1.2 互优化机制（一段） | 1.1 总体框架（两级体系）/ 1.2 DSMONet 互优化网络（完整展开 + 概念式(1)，与真实 forward 逐项对应）/ 1.3 AAM（式2，强调嵌入互优化路径而非外挂）/ 1.4 辅助细节监督（明确非原创 loss）/ 1.5 联合优化（式3） |

## 二、命名统一执行清单

- 正文、摘要（中/英）、引言、方法、表 1–8、图 1/3/4 图注、消融、效率、可视化、讨论、结论：`Base-B` → `DSMONet-B`，共 30 处 `DSMONet-B`。
- 消融表 6 行名：`DSMONet-B` / `DSMONet-B + AAM` / `DSMONet-B + 细节监督` / `A2MS-DSMONet-B`。
- 首次定义处（1.1）：`采用 ResNet-50 主干的模型记为 DSMONet-B 与 A2MS-DSMONet-B`。
- 全文 grep：`Base-B` = 0 命中（内部 audit 文档不受影响）。

## 三、数字冻结核验（未改动）

NEU：DSMONet-B 91.24 / A2MS 91.45 / Δ+0.21 / 37.37 / 38.16 / 29.37 / 29.46 / +0.31%；
10k 消融：85.22±0.41 / 85.70±0.43 / 85.44±0.19 / 85.75±0.54；
Leather：89.91 / 90.97 / +1.06 / 468 / 235；主流对比 84.21 / 86.20 / 88.24；
效率：64.68 / 62.83 / 85.18 / 109.83 / 150.62 / 6.648 / 26.21 / 26.76 / 42.42 / 45.43 / 270.3；
边界/尺度/形态分析数值全部保留。
禁用残留 grep（87.21 / 84.99 / −2.22 / 12.46 / 127.9 / 115.6 / 71.6 / ELMM / 90.2）：0 命中。

## 四、图件更新

- Fig1 重绘（`repro_runs/mechanism_eval_240k/fig1_architecture_v6.py`）：新增 DSMONet 基础架构范围框（蓝）与 A2MS 增强标记（紫虚线 halo：①AAM；②细节监督分支，仅训练）；图注明确两层结构；计算图不变（仍严格对应 fixed 模型 forward）。
- Fig3/ Fig4 重渲（`visualize_cases_v6.py` / `leather_viz_final_v6.py` / `assemble_fig34_v6.py`）：面板列标题与总标题中 Base-B→DSMONet-B；选例规则、案例 ID、预测数据均不变；Fig4 重渲时一致性门复核通过（0.8991203572 / 0.9097127638，与锁定值逐位一致）。
- Fig2 不涉及命名，未改动。
- 全部输出 SVG + PDF + PNG（Fig1 600dpi，Fig3/4 600dpi）。

## 五、出版伦理配套更新

- `prior_publication_overlap_audit_v6.md`：DSMONet 完整写入正文后方法维度重叠由 MODERATE 升至 HIGH，如实记录；overall MODERATE。
- `prior_publication_disclosure_draft_v6.md`：更正"未正式发表"表述为"曾发表后撤稿"，撤稿原因按用户说明记录（同行评审流程问题，非学术不端）；披露内容按两层主线重写。
- 正文不讨论撤稿历史；披露仅在 Cover Letter/投稿系统层面。

## 六、未做事项（按指令冻结）

- 未重新训练、未调参、未改 split/协议；
- 未将大论文 90.2% 写入正文（仅存于 audit）；
- 未写"DSMONet 从 90.2 提升到 91.24"；
- 未声称 mask-aware loss / detail supervision 为本文原创损失；
- 标题未替换（两个备选见 `dsmonet_a2ms_storyline_v6.md`，待用户选择）。
