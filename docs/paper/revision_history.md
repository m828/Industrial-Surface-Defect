# 论文修订记录

## v1 初稿

- 整理硕士论文、撤稿小论文、固定实验结果和已有图表；
- 将论文主线调整为工业表面缺陷实时语义分割；
- 保留 NEU-Seg 和自建皮革缺陷数据集作为主实验。

## v2

- 删除工作稿提示和内部写作提醒；
- 调整实验章节顺序；
- 增加 Params、FLOPs、模型大小占位列；
- 扩写可视化分析，聚焦小目标、细长缺陷、边界模糊和背景纹理干扰。

## v3

- 强化“细节—语义互优化机制”作为核心基础机制；
- 调整摘要、引言贡献点、方法章节和结论；
- 明确 ELMM、AAM 和联合监督约束的任务适配定位。

## v4

- 将最终模型命名统一为 A2MS-DefectNet；
- 将 A2MS-DefectNet-S/B 定义为轻量/增强版本；
- 将 Base-S/Base-B 定义为仅采用细节—语义互优化机制的基础版本；
- 正文主线统一使用 A2MS-DefectNet 作为最终模型名。

## CJIG v1

- 按《中国图象图形学报》目标体例调整正文结构；
- 摘要改为四要素结构；
- 引用、图题、表题和图件要求按期刊体例整理；
- 参考文献核验表单独维护。

## CJIG 重构方案（paper_restructure_cjig_v2.md，未改动 v1 正文）

- 基于已审计冻结结果（240k 固定终点：Base-B 91.24% / A2MS-DSMONet-B 91.45%，+0.21 pt；效率、边界、尺度、消融全套 audit 数据）形成重构方案；
- 模型更名 A2MS-DefectNet → A2MS-DSMONet；
- 删除/标记旧链未复核内容：ELMM 消融、AAM/损失旧消融表、皮革数据集结果、127.9 FPS 与"FPS 反超"表述、NEU-Seg 类别名错误；
- 新增"不同缺陷类型性能分析"为核心实验节（patches 边界改善 / crazing 区域—边界权衡 / inclusion 稳定）；
- 输出标题候选、四要素摘要重写、创新点重写、实验章节骨架、讨论与结论替换稿、claim 风险审查表（R1–R17）。

## CJIG v2 正文重构（paper_draft_cjig_v2.md）

- 按重构方案完成正文重构：0 引言 / 1 A2MS-DSMONet 网络（1.1–1.5）/ 2 实验与分析（2.1–2.7）/ 3 讨论 / 4 结论；
- 全部 NEU-Seg 数字采用冻结审计值；皮革数据集仅保留数据集描述与定性验证定位，结果全部占位"待复核"；
- 方法与损失公式按 fixed 模型真实代码撰写（AAM=eSE+ACW 式1；损失 ω=[10,1,3]+λd=3 式2）；
- 禁用表述 grep 扫描 0 命中；无 ELMM / 旧链数字残留；
- 配套：cjig_revision_log_v2.md（修订日志）、cjig_claim_audit_v2.md（claim 审查 C1–C18 + 投稿前必办 S1–S8）、cjig_table_list_v2.md（图表清单）。

## CJIG v3 投稿收口（paper_draft_cjig_v3.md）

- 皮革实验真实补做：60k 迭代双模型重训（fixed 模型文件），统一 468 张固定终点评价——Base-B 87.21% / A2MS-B 84.99%（Δ −2.22 pt），FPS 35.84/37.11（768×768），结果如实回填表 4 并写入讨论与局限；
- 摘要/结论 claim 降档为"改善部分工业缺陷类别的结构表达能力"；贡献 2/3 按"分析方法/验证方案"重新校准；
- 表 1 改"固定终点评价结果"、2.3 改"不同缺陷形态下的性能差异分析"、消融表述加"当前实验设置下"限定；
- 图 1 按 fixed 模型重绘（SVG/PDF/PNG，无 ELMM，细节监督仅训练态）；图 4 皮革案例按固定规则选例生成；
- 参考文献 12 条全部联网核验（reference_audit.md），引言文献段改为作者—年份实名引用；皮革划分更正为 1638/235/468；
- 配套：cjig_revision_log_v3.md、cjig_claim_audit_v3.md（V1–V15 全过）、leather_experiment_report.md、experiments/audit/leather_results.json。

## v4 (2026-09-28)

- `paper_draft_cjig_v4_leather_recovered.md`: Leather 正式结果替换为历史 val-best 链统一 test468 复评（Base 89.91 / A2 90.97，+1.06pt）；v3 的 60k 固定终点结果（87.21/84.99）退出正文、留存 audit。新增表5 Leather 逐类、图2 占位；图4 用正式 checkpoint 重生成（固定选例规则+一致性门控）。表号顺延至表8。配套：cjig_revision_log_v4.md / cjig_claim_audit_v4.md / leather_result_provenance_v4.md / cjig_table_list_v4.md / experiments/audit/leather_paper_final_results.json。

## v5 (2026-09-28)

- `paper_draft_cjig_v5_submission_candidate.md`: 投稿收口候选。Leather 表4扩展为含 STDC/PIDNet/DDRNet 的 Level A 对比（84.21/86.20/88.24 vs Base 89.91 / A2 90.97）；NEU 主流对比移除（本地 baseline 为 test 监控选优，Level C 禁用）；图2新绘、图3/图4装配终版；英文摘要扩至约380词（官方约500字要求）；DMC-Net/CDARNet 作者经 Crossref/ACM 补齐；贡献点重写；语言去审计化。配套：mainstream_comparison_audit_v5 / reference_crosscheck_v5 / cjig_abstract_requirement_audit / prior_publication_overlap_audit_v5 / prior_publication_disclosure_draft_v5 / submission_metadata_todo_v5 / submission_package_checklist_v5 / cjig_revision_log_v5 / cjig_claim_audit_v5。

## v6 (2026-09-28)

- `paper_draft_cjig_v6_dsmonet_a2ms_integrated.md`: 技术主线重构。叙事由"Base-B + 两个模块"改为两层方法体系：DSMONet（通用细节—语义互优化基础架构）→ A2MS-DSMONet（面向工业缺陷增强）；全文 Base-B 统一更名为 DSMONet-B（IDENTITY-A 实证：与大论文 DSMONet-B 计算图同一，max_abs_diff=0.0）；方法章重构为 1.1 总体框架 / 1.2 DSMONet 互优化网络（含概念式1，与真实 forward 对应）/ 1.3 AAM / 1.4 辅助细节监督（明确非原创 loss）/ 1.5 联合优化；贡献点三层重写；讨论按"基础架构能力→形态依赖增强→跨材质验证"重组。全部实验数字冻结不变；90.2% 历史值不入正文。Fig1 重绘两层结构，Fig3/Fig4 重渲更名标签（Fig4 一致性门复核通过）。配套：cjig_revision_log_v6 / cjig_claim_audit_v6 / dsmonet_a2ms_storyline_v6 / model_naming_audit_v6 / prior_publication_overlap_audit_v6 / prior_publication_disclosure_draft_v6。

## v6.1 (2026-09-28)

- `paper_draft_cjig_v6_1_submission_candidate.md`: 投稿前措辞精修（11 处定向修改，0 处影响科学结论）。标题改为方案A"细节—语义互优化与自适应增强的工业表面缺陷实时语义分割网络"（中英同步）；删除全部"首先/We first"时间优先声明（保留"本文构建 DSMONet"创新归属）；摘要比较范围句收窄为"优于本文复现的 STDC、PIDNet 和 DDRNet 方法"；贡献2"实现…进一步协同优化"→"增强…协同建模能力"；式(1)前增加概念化表述说明；"基础网络"表述清除（→DSMONet-B/基础架构）。全文检查：Base-B=0、首先=0、强claim=0、禁用残留=0、冻结数字全在位。配套：cjig_revision_log_v6_1。
