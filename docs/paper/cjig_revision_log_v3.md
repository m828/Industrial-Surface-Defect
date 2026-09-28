# CJIG v3 修订日志（paper_draft_cjig_v2.md → paper_draft_cjig_v3.md）

依据：CJIG v3 收口任务书；v2 核心定位不变（轻量级语义—细节协同增强 + 精度/实时平衡 + 缺陷类型差异化分析 + 公开数据与真实工业场景验证）。

## 正文修改

| # | 位置 | v2 | v3 | 依据 |
|---|---|---|---|---|
| 1 | 中文摘要-结论 | "提升工业缺陷分割性能，并针对不同缺陷类型表现出差异化增强效果" | "改善部分工业缺陷类别的结构表达能力，并展现工程应用潜力" | 任务书 §三：claim 与证据对齐 |
| 2 | 英文摘要-Conclusion | 同上对应英文 | 同步改为 "improves the structural representation of partial industrial defect categories...showing potential for engineering applications" | 同上 |
| 3 | 引言贡献 2 | "面向缺陷形态差异的联合优化策略……多个维度" | "面向缺陷形态差异的联合优化分析方法，从类别、边界和尺度角度分析模块作用差异" | 任务书 §四：避免过度包装 |
| 4 | 引言贡献 3 | "构建公开基准与真实工业场景结合的实验验证体系" | "建立公开数据集与真实工业数据结合的实验验证方案" | 同上（"体系"→"方案"） |
| 5 | 2.1 皮革数据集描述 | "训练 1638、验证 468、测试 235" | **更正为"训练 1638、验证 235、测试 468"** | 磁盘与 2026-06 审计（`experiments/protocol/leather_protocol_final_decision.md`）确认 v1 沿袭的 val/test 数字对调错误；"泛化验证"改"应用验证" |
| 6 | 表 1 标题 | "长训练终点结果" | "固定终点评价结果" | 任务书 §五：强调评价协议而非训练长度 |
| 7 | 2.3 节标题 | "不同缺陷类型性能分析" | "不同缺陷形态下的性能差异分析" | 任务书 §五（实际含类别/边界/尺度/复杂度） |
| 8 | 2.5 消融表述 | "两个模块的组合能够取得最佳终点性能" | "在当前实验设置下，联合配置取得最高平均性能，并在长训练终点评价中优于基础网络" | 任务书 §五：避免暗示完整消融证明 |
| 9 | 引言文献两段 | 全部"作者待核验"，含 SPCS-Net/GCRANet/DASeg-Net/SAID/LETNet/BiSeNetV2/SeaFormer | 替换为已核验文献的作者—年份引用（Zhu/Zuo/Jeong/Li/Zhang；Yu/Fan/Hong/Peng/Xu/Wan），删除未通过精简筛选的条目 | `reference_audit.md` 逐条核验（18/18 检索到，多条细节更正） |
| 10 | 参考文献节 | 占位说明 | 12 条已核验完整著录（含 DOI/arXiv） | 同上 |
| 11 | 2.4 皮革节 | 占位表 + 复核说明 | 新增图 4（皮革验证案例）说明段：固定规则选例、不作定量依据 | 任务书 §六 |
| 12 | 图 1 | "建议重绘"说明 | 已重绘为 `figures/Fig1_A2MS_DSMONet.{svg,pdf,png}`，严格对应 fixed 模型 forward（含 AAM=eSE+ACW、细节监督分支仅训练、无 ELMM） | 任务书 §七 |

## 皮革实验（进行中 → 完成后回填）

- 训练：`repro_runs/leather_60k_seed1337/`，Base-B（`new/model_dsmo_rs50.py`）与 A2MS-DSMONet-B（`new/model_dsmo_rs50_eSE_adapt_detailloss_fixed.py`），60k 迭代、batch 8、Adam lr 1e-4、cosine T_max 48000、768×768、8 类、seed 1337、从头训练、固定终点、训练中不建测试/验证加载器；
- 脚本改造自已验证的 long240k 脚本（含非覆盖守卫、恢复 checkpoint、GradAudit），烟测通过；
- 完成后：统一 468 张测试评价 + 效率测量 → `experiments/audit/leather_results.json` + `leather_experiment_report.md` → 回填表 4 与图 4。

## 皮革实验完成记录（2026-09-25 补记）

- 两模型 60k 训练完成（wall 12.7h / 11.4h，无中断），统一 468 张固定终点评价：Base-B 87.21% / A2MS-B 84.99%（Δ −2.22 pt，主要来自烂面类 −12.46 pt）；FPS 35.84 / 37.11（768×768，交错测量）；参数 29.37M / 29.46M；
- 表 4 已回填真实数字并注明两数据集输入尺寸不同导致 FPS 不直接可比；皮革节新增训练协议说明；
- 图 4 素材已生成（`repro_runs/leather_60k_seed1337/viz/`，固定规则：268 张含缺陷测试图按 ΔmIoU 排序取改善/退化/近零各 2 例）；
- 因结果为 A2 < Base，按证据强度调整了相关表述：摘要"验证模型跨材质应用能力"→"考察模型在不同工业材质下的适用性"；3.1 节与结论如实报告皮革负差异；局限性（2）由"尚在复核"改为实际结论；
- 详细数据见 `docs/paper/leather_experiment_report.md` 与 `experiments/audit/leather_results.json`。

## 未改动

- 全部 NEU-Seg 数字（冻结值）；v2 建立的各节结构；术语台账。
- 未恢复：ELMM、127.9 FPS、+1.1 pt、"FPS 反超"、未核验皮革数字、一切过强表述。
