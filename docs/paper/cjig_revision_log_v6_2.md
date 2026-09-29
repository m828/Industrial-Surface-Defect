# CJIG v6.2 证据一致性修订记录

修订对象：[v6.1 投稿候选稿](paper_draft_cjig_v6_1_submission_candidate.md)；输出：[v6.2 投稿候选稿](paper_draft_cjig_v6_2_submission_candidate.md)。本轮只修订叙述、公式及证据适用范围；未训练模型，未改代码、数据划分、模型结构、图 1—4 或已报告的本文实验数值。

| 优先级 | 修改位置 | 修改原因与依据 | 是否影响科学结论 |
|---|---|---|---|
| P0-1 | 中英文摘要、引言末段及贡献点、1.1—1.2、结论 | 将“互优化”界定为语义进入细节融合、增强细节再进入最终联合表征的两阶段信息流；式（1）中的语义回流项本身不接收细节输入。区分架构组织方式与所用已有单元，不宣称细节直接更新深层语义张量。依据：[模型前向审计](../../experiments/audit/a2ms_b_fixed_forward_audit.md)及[基础架构身份审计](../../experiments/audit/neu_dsmonet_base_identity.md)。 | **不改变结果；收窄机制解释。** 保留 DSMONet → A2MS-DSMONet 两层方法体系。 |
| P0-2 | 1.4—1.5，式（3） | 按执行记录明确三路分割输出各含“多类别项＋前景/背景二类项”，路由依次为 OHEM CE、BCE-with-logits、OHEM CE；第四路仅为辅助细节损失。NEU 权重为 (10,1,3,3)，皮革细节权重为 1；基础模型无第四路。掩码感知与细节监督均不写成本文首次提出。依据：[实际损失展开](../../experiments/audit/historical_training_chain_diff.md)、[输出映射](../../experiments/audit/a2ms_b_fixed_forward_audit.md)和[冻结配置](../../experiments/configs/neu_a2ms_dsmonet_b_240k.yaml)。 | **修正方法定义，不改变冻结结果。** 预审报告 M2 对原式“双项结构”的判断过于简化：双项结构确实执行，需修正的是各路损失类型、目标变换与输出映射。 |
| P0-3 | 2.1 训练设置、2.5 消融解释 | 将 10k 三种子模块组合消融（余弦退火）与 240k 单种子固定终点主比较（恒定学习率）分开；不把短训模块效果外推到长训。依据：[10k 协议](../../experiments/audit/a2ms_multiseed_10k_protocol.md)、[240k 协议](../../experiments/audit/long240k_protocol_frozen.md)。 | **不改变数值结论；限定消融解释。** |
| P0-4 | 3.1 方法适用性 | STDC-Seg、PIDNet-S、DDRNet23-slim 的复现比较仅限定于皮革；NEU-Seg 只叙述 DSMONet-B 与完整模型的固定终点内部比较。依据：表 1、表 4 及[正式结果记录](../../experiments/audit/leather_paper_final_results.json)。 | **纠正无数据外推。** |
| P0-5 | 中英文摘要、2.1、2.4 表 4 注、3.1—3.2、结论 | 皮革正式权重依据验证监控选取，模型保存迭代不同；在统一 468 张测试图像上评价，不声称整个项目“只测试一次”。指出训练链与随机性未完全匹配，+1.06 pt 为该材质上的应用观察。依据：[皮革划分与监控](../../experiments/audit/leather_split_audit.md)、[正式权重记录](../../experiments/audit/leather_paper_final_results.json)。 | **不改变数值；收窄因果与跨材质解释。** |
| P0-6 | 2.1 评价协议、2.3 表 2 及正文、3.1—3.2 | 表 2 类别区域 IoU 为像素聚合；边界指标为有效图像—类别对的样本宏平均。裂纹聚合 IoU +0.32 pt 与逐样本均值 −0.71 pt 并存；配对自助法只表征固定权重下测试样本差异。尺度分层也按组内有效样本平均，不套用表 2 的聚合口径。依据：[配对样本分析](../../experiments/audit/paired_case_analysis.md)、[边界指标](../../experiments/audit/boundary_quality_results.md)、[尺度分层](../../experiments/audit/scale_stratified_results.md)。 | **不改变表中数值；纠正统计解释。** |
| P0-7 | 2.3、2.5、3.1、结论 | 斑块、裂纹变化归于完整 A2MS-DSMONet-B 配置；空洞填补仅为可视化案例支持的可能解释，不把 240k 差异单独归因于 AAM 或细节监督。10k 仅称联合配置在当前协议下取得最高平均 mIoU。依据：[2×2 消融](../../experiments/audit/a2ms_2x2_factorial_analysis.md)、[形态分析](../../experiments/audit/paired_case_analysis.md)。 | **不改变结果；删除单模块因果断言。** |
| P1 | 1.3 AAM 描述 | 分开 eSE 的输入相关通道响应与 ACW 的训练中学习得到的逐通道权重，避免将 ACW 写成逐图动态生成。依据：式（2）与[模型审计](../../experiments/audit/detail_only_model_audit.md)。 | **不影响科学结论。** |
| P1 | 2.6 效率描述 | 交错测量仅能降低共享 GPU 波动影响；实时性限定于指定 A100、输入尺寸和测量条件，不推断生产线节拍。保留本文两模型已核实的 FPS、参数量与延迟。原稿另列三种轻量方法的皮革 FPS，但本轮指定的审计资料未找到对应原始测量元数据，因此 v6.2 暂不以这组三数作横向速度结论。依据：[NEU 效率](../../experiments/audit/efficiency_benchmark_results.md)、[皮革模型效率](../../experiments/audit/leather_efficiency_benchmark_historical_ckpt.json)。 | **不改变本文模型结果；撤回无法在当前材料中独立核验的跨方法速度判断。** |
| P1 | 中英文摘要、引言、讨论、结论、文末 | 将“提升/改善”锚定当前固定终点或所报告皮革权重；英文边界改善明确为样本宏平均；去掉投稿正文中的内部参考文献核验备注，并适当收紧传统方法的概括。 | **不改变科学结果；提高口径和语言一致性。** |

## 投稿前仍需人工处理

- 作者、单位、中图分类号、收稿日期、基金信息仍为占位，需由作者据实填写；本轮未代填。
- 参考文献需按目标期刊著录格式统一；SqueezeBodyEdge 等沿用单元的精确原始文献来源仍需作者核对，本文不将其作为单元级原创。
- 若要恢复轻量方法的横向 FPS 比较，需先补齐可追溯的原始测量元数据，并确认同硬件、输入、精度和测试流程；本轮未重测或改写那些历史数值。

## 全文禁用表达检查

按任务列出的 8 项禁用词和数字，对 v6.2 稿件及本日志执行全文检索：**均为 0 命中**。
