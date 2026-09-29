# CJIG 投稿候选稿预审与风险审计

审查对象：`paper_draft_cjig_v6_1_submission_candidate.md`。依据仅为 `docs/PROJECT_CONTEXT.md`、该候选稿、`cjig_revision_log_v6_1.md`、`experiments/audit/` 和 `experiments/configs/`；`cjig_claim_audit_v6_1.md` 在当前仓库中不存在。本报告是投稿前文本与证据一致性审查，不重新判断冻结实验结果，也未核验期刊现行格式细则或外部文献优先权。

## 1. 总体评价

- **创新性：有条件成立。** 稿件已建立“DSMONet 细节—语义互优化架构 → A2MS 工业缺陷适配增强 → 类别、边界、尺度与形态分析”的完整论证链；标题、摘要、贡献点和方法章节基本一致。当前最脆弱的是“互优化”的精确计算含义和相对已有组件的结构差异。项目[模型身份审计](../../experiments/audit/neu_dsmonet_base_identity.md#L4-L20)确认现行 DSMONet-B 与早期同名实现是同一计算图，不能用命名变化或不同训练链的分数作为架构创新证据。
- **完整性：主体完整，存在必改的事实与口径错误。** NEU-Seg 的 91.24%/91.45% 与[240k 固定终点记录](../../experiments/audit/long240k_endpoint_results.md#L3-L18)一致；10k 四组依次为 85.22±0.41、85.70±0.43、85.44±0.19、85.75±0.54，与[2×2 消融记录](../../experiments/audit/a2ms_2x2_factorial_analysis.md#L9-L16)一致；皮革的 89.91%/90.97% 与[正式结果记录](../../experiments/audit/leather_paper_final_results.json#L5-L63)一致。损失公式、10k/240k 学习率说明、讨论中的比较范围和部分机制归因需要修正。
- **中文期刊适配度：较好，但尚不宜直接提交。** 题名为单一技术主题，中文摘要按目的、方法、结果、结论展开，第三条贡献也体现分析价值。投稿前应完成下列 P0 修改，并处理尚未填写的作者等元数据及参考文献格式。这里的“适配度”是文本判断，不等于已通过 CJIG 官方格式核验。

## 2. Major Concerns

### M1｜核心“互优化”叙事与式（1）未完全对齐（P0）

**问题：** 稿件称优化后的浅层细节“反向补偿深层语义特征”，但概念式中的 \(\mathbf F_s'\) 只依赖 \(\mathbf F_s\)，细节特征只在最终融合 \(\mathcal H(\mathbf F_d',\mathbf F_s')\) 中出现；对“更新深层语义”与“形成最终联合表征”的边界不清。

**证据：** [方法第 1.2 节](paper_draft_cjig_v6_1_submission_candidate.md#L73-L86)；[总体结构](paper_draft_cjig_v6_1_submission_candidate.md#L65-L69)。该节使用 DAPPM、UAFM、SqueezeBodyEdge 等已有单元，现有[相关工作段](paper_draft_cjig_v6_1_submission_candidate.md#L51-L53)主要靠“多依赖单向融合”概括差别。

**风险：** 审稿人可能把所谓双向互优化理解成已有组件的串接和两次融合，认为 DSMONet 只是换名；公式还会引发“反向补偿究竟发生在何处”的技术质疑。

**建议：** 仅用文字准确说明 arm1、细节融合、语义回流、arm2 各自的输入与输出，明确 arm2 使优化后的细节参与**最终语义—细节联合表征**，还是实际更新了某个深层语义张量；让式（1）和叙述同义。指出创新对象是这些信息流的组织方式，并逐一承认已有单元来源。避免将尚未量化的背景纹理抑制写成已经实证的效果，也避免笼统断定所引方法均缺双向交互。

### M2｜式（3）与冻结训练损失不符（P0）

**问题：** 式（3）写成三路输出各自同时计算分割损失与掩码感知损失，随后又称三路 \(\mathcal L_{seg}^{(i)}\) 均为 OHEM 交叉熵；实际配置是三路依次使用 OHEM、BCE-with-logits、OHEM，完整模型再加一路细节损失。

**证据：** [稿件式（3）及解释](paper_draft_cjig_v6_1_submission_candidate.md#L107-L118)；[NEU 完整模型配置](../../experiments/configs/neu_a2ms_dsmonet_b_240k.yaml#L23-L34)、[DSMONet 配置](../../experiments/configs/neu_dsmonet_b_240k.yaml#L23-L33)、[皮革配置](../../experiments/configs/leather_a2ms_dsmonet_b.yaml#L28-L34)。

**风险：** 方法定义与产生论文结果的训练目标不一致，直接损害可复现性；也可能被误认为另创了掩码感知损失。

**建议：** 按真实输出顺序逐项写损失函数及权重，标明第二路为已有的前景—背景 BCE 约束、第四路为辅助细节损失；保留已冻结的 NEU 权重与皮革细节权重，不改任何结果数字。

### M3｜10k 消融与 240k 主实验的学习率协议被合并描述（P0）

**问题：** “NEU-Seg 实验……学习率保持恒定”覆盖了全部 NEU 实验；实际上 240k 主结果采用恒定学习率，10k 三种子消融采用余弦退火。

**证据：** [稿件训练设置](paper_draft_cjig_v6_1_submission_candidate.md#L130-L132)及[表 6/7](paper_draft_cjig_v6_1_submission_candidate.md#L208-L230)；[10k 预设协议](../../experiments/audit/a2ms_multiseed_10k_protocol.md#L22-L33)；[240k 协议](../../experiments/audit/long240k_protocol_frozen.md#L28-L42)。

**风险：** 读者会误认为表 6 与表 7 仅有训练长度差异，且无法按文中说明复现实验。

**建议：** 在 2.1 节分开陈述两套已执行协议，明确 10k 表用于同预算模块组合比较，240k 表用于两模型固定终点评价；不要由短训消融外推出长训单模块效果。

### M4｜讨论把皮革方法对比扩展到了 NEU-Seg（P0）

**问题：** 讨论称 DSMONet-B 在两个数据集上“均高于同协议评价的 STDC、PIDNet、DDRNet”，但这些复现模型只列于皮革表 4；NEU-Seg 表 1 仅有两种本文模型，正文也明确以内部比较为主。

**证据：** [讨论 3.1 节](paper_draft_cjig_v6_1_submission_candidate.md#L257-L260)、[NEU 结果说明与表 1](paper_draft_cjig_v6_1_submission_candidate.md#L134-L143)、[皮革表 4](paper_draft_cjig_v6_1_submission_candidate.md#L177-L187)。

**风险：** 构成可直接核查的无数据比较，使审稿人怀疑其他结果叙述的准确性。

**建议：** 将“高于本文复现的三种方法”严格限定在皮革数据集；NEU-Seg 只描述同协议下 DSMONet-B 与完整模型的固定终点结果。

### M5｜皮革差异的可比性与测试过程需说准（P0）

**问题：** 两个正式皮革权重虽经同一验证划分选取，保存迭代分别为 78k 和 130k；历史训练链未构成严格同预算、同种子配对。稿件把 +1.06 个百分点直接解释成增强结构在不同材质下的正向作用，且多处使用“测试集仅评价一次”的绝对说法。验证划分有 235 张，但历史监控每次实际评价随机 228 张；项目审计也记录了多个历史权重在 468 张测试集上的评价。

**证据：** [稿件摘要](paper_draft_cjig_v6_1_submission_candidate.md#L23)、[2.1 节](paper_draft_cjig_v6_1_submission_candidate.md#L124-L132)、[表 4 注](paper_draft_cjig_v6_1_submission_candidate.md#L173-L187)、[讨论](paper_draft_cjig_v6_1_submission_candidate.md#L261-L267)；[正式权重迭代与选取记录](../../experiments/audit/leather_paper_final_results.json#L5-L50)、[验证监控与划分](../../experiments/audit/leather_split_audit.md#L19-L35)、[测试集历史权重评价台账](../../experiments/audit/leather_checkpoint_unified468_results.md#L7-L17)。

**风险：** +1.06 pt 不能被单独归因于 AAM 与细节监督；“仅评价一次”若指整个项目的测试行为，与审计台账不符。这里没有证据表明正式权重按测试集选取，但现行表述容易被质疑为过度净化实验过程。

**建议：** 保持“跨材质应用适用性考察”定位；说明正式权重依历史验证监控选取、保存迭代不同，并在局限性注明训练链及随机性未匹配。将测试描述限定为“报告的权重由验证集选择，在统一的 468 张测试图像上计算结果”，避免概括整个项目只运行过一次测试；视篇幅交代验证监控实际覆盖 228 张及其随机性。

### M6｜表 2 混用聚合和样本平均口径，裂纹方向可能被误读（P0）

**问题：** 表 2 的区域 \(\Delta\mathrm{IoU}\) 为全像素聚合差，边界指标为有效图像—类别对的样本宏平均差，但引言统一写成“840 张测试图像逐样本配对比较”。裂纹的聚合 IoU 为 +0.32 pt，而逐样本 IoU 平均为 −0.71 pt。

**证据：** [稿件表 2 与解释](paper_draft_cjig_v6_1_submission_candidate.md#L147-L159)；[逐样本分析](../../experiments/audit/paired_case_analysis.md#L3-L27)、[边界指标定义](../../experiments/audit/boundary_quality_results.md#L1-L5)。

**风险：** 两种汇总口径的符号相反却未在表附近解释，容易把少数大区域样本的聚合获益读成典型裂纹样本普遍改善；样本 bootstrap 区间也可能被误读为跨训练种子的稳定性证据。

**建议：** 在表注或紧邻正文分别写明区域 IoU 的聚合口径、边界指标的有效样本宏平均口径及各类有效样本数；点明裂纹“聚合正向、典型样本平均负向”的并存现象。说明现有配对置信区间只衡量固定权重下测试样本差异，不代表训练种子间稳定性。

### M7｜把联合模型观察结果归因于单个模块（P0）

**问题：** 高复杂度裂纹回退被解释为细节监督的定位局限，斑块边界改善被归因于 AAM 带来的语义一致性；240k 只比较 DSMONet-B 与完整 A2MS-DSMONet-B，不能分离两个模块作用。10k 的联合平均最高，但单模块/交互增益的种子方向并不一致。

**证据：** [稿件第 2.3、2.5 与 3.1 节](paper_draft_cjig_v6_1_submission_candidate.md#L159-L171)、[消融说明](paper_draft_cjig_v6_1_submission_candidate.md#L208-L230)、[讨论归因](paper_draft_cjig_v6_1_submission_candidate.md#L259-L263)；[2×2 效应分析](../../experiments/audit/a2ms_2x2_factorial_analysis.md#L20-L29)、[机制结论](../../experiments/audit/mechanism_eval_final_decision.md#L19-L27)。

**风险：** 审稿人会把贡献 2 理解为已证明 AAM 与细节监督协同改善边界，而数据只支持完整配置的形态相关观察。短训“最高平均值”也不等于稳定或可加的模块效果。

**建议：** 保留 AAM 与细节监督的设计动机及 10k 联合配置最高平均值；将斑块、裂纹结论归于**完整增强配置**，把空洞填补、语义一致性等解释写成示例支持的可能机制，不作单模块因果断言。明确高复杂度裂纹结果是单代理、单类、单种子的探索性分析。

## 3. Minor Concerns

1. **效率措辞（P1）。** [稿件 2.6 节](paper_draft_cjig_v6_1_submission_candidate.md#L232-L245)称交错测量“消除”共享 GPU 干扰、帧率“满足在线检测的实时性需求”；[效率记录](../../experiments/audit/efficiency_benchmark_results.md#L3-L5)只支持缓解相对比较偏倚，未给出具体产线节拍。宜写成指定 A100 环境下的实测速度与实时应用潜力，并保留两种输入尺寸不可横比的说明。
2. **AAM 用语（P1）。** [第 1.3 节](paper_draft_cjig_v6_1_submission_candidate.md#L88-L101)将 ACW 写成“动态调整”易被理解为每张输入图像都产生新权重；式（2）显示输入相关响应来自 eSE，ACW 是训练得到的逐通道参数。分别说明两者作用。
3. **题名长度判断（P1）。** [候选题名](paper_draft_cjig_v6_1_submission_candidate.md#L1)有 28 个汉字（含连接号共 29 个字符），与[修订日志“≤25 字”](cjig_revision_log_v6_1.md#L13)不符。题名本身是清楚的单一主题；应核对期刊实际字数口径，再决定是否压缩，不宜沿用未核实的合规判断。
4. **投稿信息与引文格式（P1）。** [作者、单位、中图分类号、日期和基金信息](paper_draft_cjig_v6_1_submission_candidate.md#L5-L15)仍是占位；[文末编辑备注](paper_draft_cjig_v6_1_submission_candidate.md#L297-L299)也明确参考文献格式待统一。投稿版本需补齐元数据并移除正文中的内部编辑备注。
5. **摘要与结论的适用范围（P1）。** [中文摘要结论](paper_draft_cjig_v6_1_submission_candidate.md#L23-L25)及[全文结论](paper_draft_cjig_v6_1_submission_candidate.md#L269-L271)概括为增强策略“能够……提升”，而 NEU 长训差值为单种子 +0.21 pt，[局限性](paper_draft_cjig_v6_1_submission_candidate.md#L265-L267)已承认随机性量级问题。建议在摘要和结论把性能陈述明确锚定于当前固定终点与皮革应用评价。

## 4. Innovation Evaluation

**判断：DSMONet + A2MS 可以形成一条完整的 CJIG 方法—应用—证据链，但当前文本尚不足以把每个模块都论证为独立原创贡献。** 第一层的独立价值是细节—语义信息流的架构组织及其可直接运行、独立评价的 DSMONet-B；第二层是把 AAM 接入互优化路径，并用仅训练阶段的辅助细节监督适配工业缺陷；第三层以双数据集及类别、边界、尺度分析限定适用范围。正文已在[贡献点](paper_draft_cjig_v6_1_submission_candidate.md#L57-L61)、[总体框架](paper_draft_cjig_v6_1_submission_candidate.md#L65-L69)与[细节监督说明](paper_draft_cjig_v6_1_submission_candidate.md#L103-L105)体现这三层，也没有把已有监督思想写成首次提出。

成立条件是完成 M1 和 M7：把“双向”对应到可检查的信息流，清楚区分已有单元与本文组合方式，并把 A2MS 的性能证据限定为完整配置的观察。91.24% 是当前协议下 DSMONet-B 的绝对结果；91.45% 与其差 +0.21 pt 是同一固定终点的数值比较，不能用不同训练链数字包装成方法提升。皮革 +1.06 pt 支持该材质上的应用结果，不证明跨材质泛化。仅凭本次允许的项目文件，无法独立认定外部文献中的新颖性优先权。

## 5. Claim Audit

禁用模型名和废弃实验数字在候选稿中未检出。绩效强语“显著提升”“大幅提高”“全面优于”“普遍提升”“稳定提升”“证明泛化”“首次提出”未见违规命中；“所有类别”出现在否定句，[皮革回退与局限性](paper_draft_cjig_v6_1_submission_candidate.md#L189-L202)亦有如实报告。需处理的是以下**语义强度**问题：

| 原表述或位置 | 证据边界与建议 |
|---|---|
| [“抑制背景纹理干扰”“解决……难以兼顾”](paper_draft_cjig_v6_1_submission_candidate.md#L19-L21)、[第 1.1 节](paper_draft_cjig_v6_1_submission_candidate.md#L65-L67) | 是结构动机，未见独立背景抑制指标；改为“旨在调节背景纹理响应”“缓解”。 |
| [“这些方法多依赖单向……仍缺乏充分双向互优化”](paper_draft_cjig_v6_1_submission_candidate.md#L53-L53) | 涵盖面超过逐项分析；针对具体融合路径说明本文差异。 |
| [“不依赖特定缺陷类别，可作为面向不同工业材质……”](paper_draft_cjig_v6_1_submission_candidate.md#L55-L55) | 可称结构未设置类别专用分支；经验范围只到带钢与皮革两种材质。 |
| [“ACW……动态调整”](paper_draft_cjig_v6_1_submission_candidate.md#L90-L101) | ACW 是学习到的通道权重，输入自适应来自 eSE；分开表述。 |
| [“对区域型缺陷具有更好的结构保持能力”](paper_draft_cjig_v6_1_submission_candidate.md#L159-L159) | 现有强证据限于本次 NEU 斑块类；避免推广到所有区域型缺陷。 |
| [“说明当前的细节监督约束……作用有限”](paper_draft_cjig_v6_1_submission_candidate.md#L171-L171) | 仅完整配置对比、探索性形态代理；改述完整模型的观察局限。 |
| [“单模块……更多体现为联合优化下的类型差异化改善”](paper_draft_cjig_v6_1_submission_candidate.md#L228-L230) | 10k 未证明单模块作用转化为长训类别收益；只保留联合配置的已观测结果。 |
| [“均高于同协议评价的……”](paper_draft_cjig_v6_1_submission_candidate.md#L259-L259) | 仅皮革数据有这组三种方法的对比；按 M4 修正。 |
| [“自适应通道重标定带来的语义一致性改善能够直接转化……”](paper_draft_cjig_v6_1_submission_candidate.md#L261-L261) | 缺少长训单模块归因与直接表征指标；写成待验证的解释，或只陈述完整配置的斑块结果。 |
| [“仍满足在线检测的实时性需求”](paper_draft_cjig_v6_1_submission_candidate.md#L243-L245) | 未给生产线时延阈值；报告指定硬件/输入下 FPS，不作普适满足判断。 |

## 6. Recommended Revision Priority

- **P0｜必须改：** M1 互优化信息流与创新边界；M2 损失公式；M3 两套 NEU 学习率协议；M4 讨论的比较范围；M5 皮革训练链及测试措辞；M6 表 2 统计口径；M7 完整模型与单模块的归因边界。全部是文本与公式说明修订，不需要改动冻结结果、图件或实验。
- **P1｜建议改：** 将摘要、讨论和结论的效果表述锚定具体评价；修正效率、ACW 和跨材质措辞；补齐投稿元数据、统一引文并删除内部备注；核实题名字数口径。
- **P2｜可优化：** 缩短摘要中重复的方法链描述，使“问题—方法—实验—意义”更紧凑；减少第 3、4 节重复陈述相同数值，在结论保留最能支持形态依赖性的结果。
