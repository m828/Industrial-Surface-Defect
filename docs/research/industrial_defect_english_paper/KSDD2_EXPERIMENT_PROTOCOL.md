> 阶段说明：以下为数据冻结前的历史准备记录。当前数据已交付KSDD2_DATA_FROZEN_V1；请以KSDD2_SOURCE_INTEGRATION_REPORT.md和KSDD2_SERVER_RUN.md为准。不要再按本文旧候选640×256输入或旧待冻结流程执行。

# KSDD2跨数据集验证候选协议

日期：2026-11-09。源码基底 `8abd026bb6b0fce051196379604f9c0ccabd6aac`；本轮使用同仓库独立工作树 `.ksdd2-prep-worktree`，不改历史实验。本文件为**训练前候选，不是已冻结运行配置或GPU批准**。配套机器可读草案：`research/industrial_defect_english/configs/ksdd2_protocol_candidate_v1.json`，明确标记DRAFT_NOT_RUNNABLE。

## 研究问题与矩阵

保持既有同一有界T在双侧融合、最终语义回流及细节监督中的共享关系，不为KSDD2增加或优化网络模块。研究在新工业数据集上能否保持：coupled相对原A2MS的增益、相对等容量对照的增益、相对最终重算T的增益，以及小缺陷/FP/FN的变化。允许正向、零和负向结果。

| 条件 | 训练seed |
|---|---|
| formal_a2ms | 1337、2027、3407 |
| capacity_control | 1337、2027、3407 |
| coupled_transport | 1337、2027、3407 |
| recomputed_final_transport | 1337、2027、3407 |

共12次新训练。NEU权重和成绩不充当KSDD2匹配基线，不重训或覆盖NEU。每seed配对报告coupled减三个对照的差值；汇总均值和样本标准差(ddof=1)，保留全部seed。三seed不能直接作为显著性证明。

## 数据身份和任务

[ViCoS官方数据页](https://www.vicos.si/resources/kolektorsdd2/)登记train2331（246阳性、2085正常），test1004（110阳性、894正常）；约230×630像素，原图实际H/W及编码仍待服务器检查。上述是官方发布元数据，**不是本轮服务器文件验收结果**。采用全监督二值像素分割，0背景、1缺陷；不使用NEU三类名称或四类权重。

引用 Božič J, Tabernik D, Skočaj D, *Mixed supervision for surface-defect detection: from weakly to fully supervised learning*, Computers in Industry, 2021。[作者论文记录](https://prints.vicos.si/publications/385)。本研究不把原文混合/弱监督检测指标同二值IoU直接排名。官方页标示CC BY-NC-SA 4.0；本轮不复制数据或开展再分发审计。

官方训练/测试边界保留。只从train建立一次约20%的开发验证集，优先按可得来源parent及精确内容同源分组；详见METRICS_AND_SPLIT_PLAN。全部模型和seed共享同一划分。测试集在配置、损失和预算冻结后才作另行授权的最终评价，本轮完全不评价它。

## 输入与初始化候选

优先原分辨率输入、不裁剪、不缩放；按RGB/255处理，右/下复制padding到共同的32倍数，用独立valid布尔mask标记原图区域。若服务器实际**官方训练图像**均能容纳，候选 **H×W=640×256**；这只是竖幅情况下的候选，不根据网页的“230×630”文字猜测数组方向。实际训练H/W不符则先报告、在训练前改定一次，不看验证分数或测试数据选训练尺寸。SqueezeBodyEdge两次无padding的3×3/stride2卷积要求足够深层空间尺寸；32倍数输入各轴至少224，本轮CPU正向使用224×256验证。不能把网络直接运行在64×96等过小张量上。

所有原始像素参与主分割损失和指标；padding像素不计入监督或评价。验证输出直接去除padding回原图，无缩放或形态学后处理。预先规定最终测试单图若超出训练pad框，则各轴取max(冻结训练框、原图轴长向上取32倍数)，单图原尺寸评价；不因此调整训练尺寸、裁剪GT或调参。该规则训练前统一冻结，四模型相同。训练增强候选：在padding前对RGB与GT同步水平翻转p=0.5；不引入新旋转或形变。损失细节分支另排除4px不确定边带，见下节。

同一训练seed的四模型从零初始化，三路分割输出均为2通道，细节头1通道。公共初始张量SHA必须相同；三个带transport的模型初始transport SHA相同。沿用isolated_v1采样/增强RNG，不宣称跨GPU逐位确定。候选不加载NEU checkpoint或ImageNet预训练；如作者选择另一初始化政策，四模型统一另行冻结。

## 损失候选与必要裁决

现有历史mask-aware代码对二类可计算，但binary_logits=[z0,z1]、binaryGT=y，与原分割三路相同，导致

`L_legacy_binary = 2*(w0*OHEM(z0,y)+w1*BCE(z1,onehot(y))+w2*OHEM(z2,y)) + w3*L_detail`。

因此“二类兼容”不意味着应不加说明地沿用。前三个w是输出路由权重，不是NEU四类别权重。历史空hard-pixel返回NaN的行为保持不改。

推荐显式新**任务适配候选** `ksdd2_binary_masked_ohem_detail_candidate_v1`，不是新算法贡献：

`L = w0*OHEM_valid(z0,y) + w1*BCE_valid(z1,onehot2(y)) + w2*OHEM_valid(z2,y) + w3*(BCE_detail_valid + Dice_detail_valid)`。

- 不再重复二类mask-aware三路，因其没有独立监督信息。所有四模型用同一协议。
- OHEM仅对valid像素挖掘，threshold=.7、min_kept=100000。若hard集合为空，**显式**返回valid像素平均CE，事件及次数必须入日志；它与NEU严格NaN协议不同，不声称同协议复现。
- 细节目标复用既有full/x2/x4及0.6/0.3/0.1构造；有效细节区为valid向内腐蚀半径4，避免padding及多尺度邻域不完整造成的假边界监督。原图边缘主分割监督保留。细节有效区全空则拒绝，不静默记零。
- 目标为正常图时，边界GT为空，BCE及带平滑的Dice仍计算；不是丢弃正常图。所有非有限loss/梯度仍停止留证。
- 候选路由权重[10,1,3,3]只作为待批起点，**不是二类class_weights**；不复制NEU类别权重。最终权重、是否采用该候选或严格重复历史方案须由作者训练前确认。

本轮实现位于`ksdd2_support.binary_joint_loss`，旧`models/losses.py`和vendor未修改。未将它接入NEU训练入口或改变旧默认值。空hard CE fallback及valid padding处理必须写入新的KSDD2协议ID，不能冒用neu_a2ms_mask_aware_candidate_v1。

## 公平训练候选及与NEU差异

| 项目 | 待冻结候选 |
|---|---|
| seed / split seed | 训练1337/2027/3407；划分1337，独立随机流 |
| batch / optimizer | batch16；Adam，lr1e-4，WD2e-6，constant |
| 采样 | 普通shuffle、epoch内无放回、drop_last；不做模型专属阳性过采样 |
| 预算 | 暂列10000真实更新、每1000更新验证；KSDD2成本验收及作者批准后才冻结 |
| 终点 | 只评价固定最终更新；不按验证性能、beta/T早停或选最佳checkpoint |
| 不完整运行 | 数值、资源、协议异常为INCOMPLETE，不能混入完整矩阵平均 |
| NEU不同处 | 数据/任务/二类头、原尺寸padding、valid mask、候选损失及原生坐标评价；绝不声称NEU同训练协议 |

若batch16不适合实际显存，只能在正式比较前统一调整四模型batch及预算并重新冻结，不能给单个模型专属设置。采样策略/预训练/权重不能因某模型成绩不利而修改。

## 成本评估与批准

12次训练是科学上合适的首个正式矩阵；**是否适合当前计算预算目前UNKNOWN**，本轮没有A100环境或实际步速数据。640×256候选每图163840像素，为NEU200×200的4.096倍；这是像素/近似激活量比，不能当作准确耗时或显存倍数。

后续先提交一次仅为成本估计的有限更新GPU smoke申请（四模型同条件，剔除warmup后计时，含同步、数据读入和实际loss），本轮不执行。每模型测得`s_m`秒/更新、`v_m`秒/完整开发验证、保存开销`c_m`后：

`estimated_allocated_gpu_hours = 3*sum_m(U*s_m + ceil(U/eval_every)*v_m + c_m)/3600`。

额外列计smoke、预检、共享GPU等待/占用的口径及可见争用；不把历史共享GPU墙钟直接作为独占成本。磁盘按实测checkpoint（含Adam状态）+保留日志/逐例统计×12计算，另留明确余量。仅限制正式输出checkpoint数量，不删原实验。所有数值来自未来实测，不编造小时数；不得默认复用旧NEU整批3小时上限。

## 本轮结论

模型和二值适配基础组件可用，数据/协议尚未真实冻结，且原NEU runner和导出门禁仍需最小任务分发适配。当前为**KSDD2_COMPONENTS_READY；尚未达到端到端KSDD2_SOURCE_READY**。具体缺口及下一步见CODE_READINESS与SERVER_HANDOFF；不以CPU合成PASS冒充服务器可训练状态。
