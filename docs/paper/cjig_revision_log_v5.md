# CJIG v5 修订日志 (cjig_revision_log_v5)

> 文件：`docs/paper/paper_draft_cjig_v5_submission_candidate.md`（由 v4 定点收口，v4 保留）
> 日期：2026-09-28

## 相对 v4 的变更

### 1. 主流方法对比（第一优先级）
- **Leather：ADDED**。表 4 扩展为 5 行对比（DDRNet23-slim 84.21 / PIDNet-S 86.20 / STDC-Seg 88.24 / Base-B 89.91 / A2MS-B 90.97 + Params），全部 Level A（val235 选择 + test468 统一评价，两轮复评逐位一致）。
- **NEU：REMOVED**。删除"主流方法对比待补充"声明；2.2 改为统一协议内部对比 + 口径说明。理由：本地 NEU baseline 权重均为 test 监控选优（Level C，禁用），详见 `mainstream_comparison_audit_v5.md`。
- 2.6 补 768² 独占 A100 环境下 5 模型 FPS（含基线 85.18/109.83/150.62，如实呈现本文模型速度居后但仍实时）。

### 2. 图件
- 图 2 数据集示例：新绘（NEU 3 类 + 皮革 3 类，固定规则选图，无预测），PNG600/PDF/SVG + manifest。
- 图 3 NEU 可视化：由冻结案例面板装配为 6 行终版（crazing/patches × 改善/退化/相当）。
- 图 4 皮革可视化：由正式 checkpoint 案例面板装配为 6 行终版。
- 图 1 不变（质检通过：无 ELMM、detail 仅训练态、head_seg1=out）。

### 3. 摘要
- 英文摘要扩写至约 380 词（官方要求 500 字左右、中英无需对应，见 `cjig_abstract_requirement_audit.md`），加入 Leather 数字与基线对比，无强表述词。
- 中文摘要结果段补 Leather 基线对比一句。

### 4. 贡献点重写
按最终定位三条：网络本体（较低额外开销）/ 形态差异分析维度 / 双数据集验证与工程表现。删除"提出分析方法""构建验证方案"式表述。

### 5. 参考文献
- [8] DMC-Net：补全为 Zuo H, Zheng Y, Huang Q, et al.（Crossref 核验，全作者 5 人）。
- [10] CDARNet：补全为 Li Q, Ding C, Wang B, et al.（ACM DL 核验，全作者 6 人）。
- 双向核对：missing=0，orphan=0（`reference_crosscheck_v5.md`）。

### 6. 语言与术语
- 去审计化：正文不出现"锁定/复评/取证/manifest/冻结/审计"等词（"锁定的测试集"→"独立测试图像"，"统一复评"→"统一评价"）。
- 术语统一：A2MS-DSMONet(-B)、Base-B、细节—语义互优化、自适应语义增强模块（AAM，首次定义后锁定）、eSE、ACW、细节监督分支、Boundary IoU、BF-score。
- "AAM 计算开销可忽略"→"仅引入少量额外计算"。
- 逐病例→逐样本（样本级表述统一）。

### 7. 其他
- 删除"checkpoint"技术词在正文的直接使用（改"模型权重"，协议处保留必要说明）。
- 删除文件名 80000/160000 与实际迭代差异的写法（本轮纪律：不主动讨论训练轮次差异）。
- 局限性（6）更新为主流对比口径说明。

## 保持不变
NEU 全部数字、2.3 机制分析、2.5 消融、结论强度、12 条文献体系。
