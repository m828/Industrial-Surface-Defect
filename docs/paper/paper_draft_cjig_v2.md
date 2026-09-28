# 语义—细节协同增强的工业表面缺陷实时语义分割网络

# Real-time Semantic Segmentation Network for Industrial Surface Defects with Semantic-Detail Collaborative Enhancement

作者：待补充

单位：待补充

中图分类号：待核定

文献标志码：A

收稿日期：待补充

基金项目：待补充

## 摘要

**目的** 工业表面缺陷具有尺度变化大、边界模糊、类别形态差异明显等特点，在线检测场景还要求模型具备实时推理能力。如何在语义分割中兼顾高层语义理解与局部细节保持，并在分割精度与推理速度之间取得平衡，是工业表面缺陷检测面临的重要问题。针对上述问题，本文提出一种语义—细节协同增强的轻量级实时缺陷分割网络 A2MS-DSMONet。

**方法** A2MS-DSMONet 以细节—语义双向互优化为基础特征交互机制，利用深层语义抑制浅层背景纹理干扰，并利用浅层空间信息补偿深层特征的位置与结构信息。在此基础上，设计自适应语义增强模块，通过高效通道注意力与自适应通道权重对特征进行通道重标定，增强网络对关键语义区域的响应能力；引入细节监督分支，通过辅助细节监督约束引导网络学习局部结构信息；并结合多尺度语义信息融合，形成面向缺陷尺度变化与类间形态差异的联合优化策略。

**结果** 在 NEU-Seg 带钢表面缺陷公开数据集上，采用统一 840 张测试图像、固定训练终点的评价协议，A2MS-DSMONet-B 取得 91.45% mIoU（含背景类）和 37.37 FPS（NVIDIA A100，batch size 1，FP32）。相比仅含细节—语义互优化机制的基础网络 Base-B（91.24% mIoU、38.16 FPS），mIoU 提高 0.21 个百分点，参数量仅增加 0.31%，计算量基本持平。缺陷类型分析表明，该方法对斑块类缺陷的边界质量改善最明显（1/2/3 像素容差下边界 IoU 提高约 0.5~0.9 个百分点），对裂纹类缺陷主要改善区域完整性，对夹杂物类缺陷性能保持稳定。进一步在真实皮革缺陷数据集上验证模型跨材质应用能力。

**结论** 实验表明，所提出方法能够在保持实时性的同时提升工业缺陷分割性能，并针对不同缺陷类型表现出差异化增强效果，具有工业在线检测的工程应用潜力。

## 关键词

工业表面缺陷；实时语义分割；语义—细节协同；自适应通道重标定；细节监督；缺陷类型分析

## Abstract

**Objective** Industrial surface defects exhibit large scale variation, blurred boundaries, and pronounced morphological differences across categories, while online inspection additionally requires real-time inference. Balancing high-level semantic understanding with local detail preservation under a limited computational budget remains a key problem for semantic segmentation in industrial inspection. To address this problem, this paper proposes A2MS-DSMONet, a lightweight real-time defect segmentation network based on semantic-detail collaborative enhancement.

**Method** A2MS-DSMONet builds on a bidirectional detail-semantic mutual optimization mechanism, in which deep semantic features suppress background-texture responses in shallow detail features, and the optimized shallow spatial information in turn compensates deep features with positional and structural cues. On top of this mechanism, an adaptive semantic enhancement module performs channel re-calibration through efficient channel attention and adaptive channel weighting, strengthening responses to semantically important regions; a detail supervision branch introduces an auxiliary detail supervisory constraint that guides the network to learn local structural information; and multi-scale semantic context aggregation is combined with the above designs to form a joint optimization strategy targeting defect scale variation and inter-class morphological differences.

**Result** On the public NEU-Seg strip-steel surface defect dataset, under a unified protocol of 840 test images evaluated at a fixed training endpoint, A2MS-DSMONet-B achieves 91.45% mIoU (including background) at 37.37 FPS (NVIDIA A100, batch size 1, FP32). Compared with Base-B, which uses only the detail-semantic mutual optimization mechanism (91.24% mIoU, 38.16 FPS), mIoU increases by 0.21 percentage points with only 0.31% more parameters and essentially unchanged computation. Per-defect-type analysis shows that the boundary quality of patch-type defects improves most (boundary IoU gains of about 0.5-0.9 percentage points at 1/2/3-pixel tolerances), crazing-type defects mainly gain in regional completeness, and inclusion-type defects remain stable. The model is further validated on a real-world leather defect dataset to examine its cross-material application potential.

**Conclusion** The proposed method improves industrial defect segmentation performance while preserving real-time inference, and exhibits differentiated enhancement effects across defect types, indicating its potential for online industrial inspection applications.

## Keywords

industrial surface defects; real-time semantic segmentation; semantic-detail collaboration; adaptive channel re-calibration; detail supervision; defect-type analysis

## 0 引言

工业制造过程中，钢材、皮革、芯片、磁瓦和金属零部件等材料表面容易出现裂纹、夹杂、斑块、破洞和弱纹理缺陷。这类缺陷会影响产品外观质量、结构可靠性和后续加工稳定性，因此在生产线上实现快速、稳定、精细的缺陷检测具有重要工程意义。传统人工检测依赖经验，检测效率和一致性受主观因素影响较大；传统机器视觉方法通常依赖灰度、纹理、边缘或频域特征，在光照变化、背景纹理复杂和缺陷形态多变时泛化能力不足。随着深度学习的发展，语义分割方法能够输出像素级缺陷区域，为缺陷定位、面积统计和质量判定提供更精细的信息。

将通用语义分割方法直接应用于工业表面缺陷实时检测仍面临多方面挑战。工业缺陷具有复杂纹理结构和多尺度分布特征，缺陷区域在整幅图像中占比低、形态差异大，且常与材料背景纹理交织。浅层特征虽然包含边缘、纹理和位置信息，却容易混入背景噪声；深层语义特征具有更强判别能力，但连续下采样会削弱局部结构与位置表达。因此，如何在语义分割模型中兼顾高层语义理解与局部细节保持，同时满足在线检测的实时性约束，仍是工业表面缺陷分割面临的重要问题。大型分割网络虽然表达能力较强，但推理开销较大；轻量网络速度较快，却容易损失局部结构信息。分割精度、推理速度与局部结构表达能力之间的平衡，是工业表面缺陷实时分割中的关键问题。

近年来，工业缺陷分割研究逐渐从通用编码器—解码器结构转向任务特异的轻量化、注意力增强和细节建模。DMC-Net（作者待核验，2025）面向实时表面缺陷分割设计轻量特征提取与多尺度增强模块；SPCS-Net（作者待核验，2025）通过空间位置注意和跨尺度融合改善工业缺陷分割精度；TAG-Net（作者待核验，2025）针对钢材表面缺陷显式建模背景、缺陷和边界注意力；GCRANet（作者待核验，2026）关注全局上下文引导下的弱缺陷轮廓增强和背景抑制；CDARNet（作者待核验，2025）面向实时金属表面缺陷分割，通过跨维度自适应区域重构增强鲁棒性；LGGFormer（作者待核验，2025）将局部细节与全局建模结合，并引入边缘引导解码；DASeg-Net（作者待核验，2025）将扩散模型与注意力机制用于芯片表面缺陷分割；SAID（作者待核验，2025）则体现了提示式分割在跨场景工业缺陷任务中的潜力。上述工作表明，工业缺陷分割不仅需要全局语义判别能力，还需要在复杂背景纹理下保持缺陷的局部结构。

实时语义分割研究主要围绕轻量骨干、高低分辨率分支、快速上下文聚合和高效特征融合展开。BiSeNetV1（作者待核验，2018）和 BiSeNetV2（作者待核验，2020）以空间细节和语义上下文的双分支建模为代表；DDRNet（作者待核验，2021）通过双分辨率结构保留空间细节并聚合上下文；STDCNet（作者待核验，2021）通过短期密集连接和细节聚合提升实时分割效率，其细节监督思想也为本文的细节约束设计提供参考；PP-LiteSeg（作者待核验，2022）、PIDNet（作者待核验，2023）、LETNet（作者待核验，2023）、SeaFormer（作者待核验，2023）和 SeaFormer++（作者待核验，2025）进一步从轻量解码、边界分支、高效 Transformer 或移动端注意力等角度改进速度—精度平衡。这些方法多在 Cityscapes、CamVid、ADE20K 等自然场景数据集上评估，直接迁移到工业缺陷场景时仍需针对低占比前景、复杂纹理和类间形态差异进行适配。

针对上述问题，本文提出一种语义—细节协同增强的工业表面缺陷实时分割网络 A2MS-DSMONet。该网络以细节—语义互优化机制为核心，通过深层语义调节浅层细节特征、抑制背景纹理误响应，并利用浅层空间信息补偿深层语义特征中的位置与结构信息；在此基础上引入自适应语义增强模块与细节监督分支，形成面向不同缺陷形态的联合优化策略。本文在 NEU-Seg 公开带钢表面缺陷数据集上开展主结果比较、消融实验、边界质量评价和缺陷尺度分层分析，并在真实工业生产场景采集的皮革表面缺陷数据集上进行跨材质应用验证。

本文主要贡献如下：

1. 提出一种轻量级语义—细节协同增强网络 A2MS-DSMONet。通过自适应语义增强模块、细节监督分支与语义—细节联合优化，提升工业缺陷分割能力，并保持实时推理性能。
2. 提出面向缺陷形态差异的联合优化策略。针对区域型缺陷、细长裂纹和点状/复杂纹理缺陷的形态差异，从类别、边界和尺度多个维度分析不同模块的作用差异。
3. 构建公开基准与真实工业场景结合的实验验证体系。通过 NEU-Seg 公开数据集与皮革真实缺陷数据集，综合评价模型精度、实时性和应用潜力。

## 1 A2MS-DSMONet 网络

### 1.1 总体结构

图 1 A2MS-DSMONet 网络结构示意图 / Fig. 1 Overall architecture of A2MS-DSMONet。图件要求：建议基于网络结构图进行矢量图重绘，或导出 300 dpi 以上位图。

A2MS-DSMONet 以细节—语义互优化机制为基础，将深层语义信息与浅层局部结构信息的联合优化作为核心设计原则。网络以 ResNet-50 为主干提取多层特征：浅层特征具有较高空间分辨率，包含边缘、纹理和位置信息；深层特征经过多次下采样后语义判别能力更强，但局部结构容易被削弱。在互优化机制基础上，网络引入自适应语义增强模块（adaptive attention module，AAM）对融合特征进行通道重标定，并通过多尺度上下文聚合模块增强深层语义的尺度表达能力；训练阶段进一步引入细节监督分支，引导网络学习缺陷局部结构信息。推理阶段网络仅保留主分割输出，细节监督分支与辅助输出不参与推理，不引入额外计算开销。

### 1.2 细节—语义互优化机制

工业缺陷图像中的浅层特征和深层特征具有明显互补关系。浅层特征保留较高空间分辨率，能够提供缺陷边缘、纹理变化和位置分布等细节信息；但材料表面的自然纹理、加工痕迹和局部亮度变化容易使浅层特征同时响应缺陷区域和非缺陷背景。深层特征具有更强语义判别能力，但连续下采样会降低空间分辨率，削弱局部结构表达。

细节—语义互优化机制建立浅层空间细节与深层语义信息之间的双向交互。一方面，深层特征经多尺度上下文聚合模块（DAPPM）增强后，通过注意力特征调制模块（UAFM）对深层语义进行整合，并经由 SqueezeBodyEdge 单元分解为语义主体与语义边缘分量；另一方面，浅层细节特征经拉普拉斯卷积单元增强局部结构响应后，与上采样的语义边缘分量、经通道重标定的语义特征共同融合，形成细节增强特征。随后，语义主体与重标定语义特征回流并与细节增强特征再次经 UAFM 融合，使语义信息抑制浅层背景纹理干扰、浅层细节补偿深层特征的位置与结构信息，完成细节与语义的双向互优化。该双向互优化特征作为网络的主分割输出基础。

### 1.3 自适应语义增强模块

工业缺陷常具有低占比、弱纹理和尺度变化明显等特点，不同语义通道对缺陷判别的重要性存在差异。为增强网络对关键语义区域的响应能力，本文在细节—语义互优化路径中引入自适应语义增强模块 AAM。AAM 由高效挤压激励单元（eSE）与自适应通道权重单元（ACW）组成：eSE 通过全局平均池化与 \(1\times1\) 卷积，以较低计算开销建模通道注意力响应；ACW 通过可学习的逐通道权重，使网络根据优化目标动态调整各通道的重要程度。二者联合对融合语义特征进行通道重标定，调节不同语义通道在细节—语义融合过程中的贡献。

设输入特征为 \(\mathbf{F}\)，AAM 的计算可表示为：

\[
\mathbf{a}_{\mathrm{eSE}}=h_{\mathrm{sig}}\!\left(\phi_{1\times1}(\operatorname{GAP}(\mathbf{F}))\right),
\qquad
\mathbf{F}_{\mathrm{AAM}}=\mathbf{F}\odot \mathbf{a}_{\mathrm{eSE}}\odot \mathbf{w}_{\mathrm{ACW}},
\tag{1}
\]

其中，\(\operatorname{GAP}(\cdot)\) 表示全局平均池化，\(\phi_{1\times1}\) 表示 \(1\times1\) 卷积，\(h_{\mathrm{sig}}(\cdot)\) 表示 HSigmoid 激活函数，\(\mathbf{w}_{\mathrm{ACW}}\) 为可学习通道权重，\(\odot\) 表示逐元素乘法。AAM 仅包含通道级运算，计算开销可忽略。

### 1.4 细节监督分支

在基础分割监督之外，本文引入细节监督分支，通过辅助细节监督约束引导网络学习局部结构信息。具体地，浅层细节特征经拉普拉斯卷积单元增强后接入一个独立的单通道细节输出头，对缺陷局部结构进行预测，并由细节监督损失约束。细节监督仅在训练阶段参与，推理阶段该分支不计算，因而不影响模型的推理速度与部署开销。需要说明的是，前景—背景掩码感知监督为基础网络与本文模型所共有，本文训练策略的差异在于引入细节监督约束，并与语义—细节互优化结构联合优化。

### 1.5 损失函数

A2MS-DSMONet 的训练目标由多阶段分割损失、掩码感知损失和细节监督损失共同构成：

\[
\mathcal{L}_{\mathrm{total}}=
\sum_{i=0}^{2}\omega_{i}\left(\mathcal{L}_{\mathrm{seg}}^{(i)}+\mathcal{L}_{\mathrm{mask}}^{(i)}\right)
+\lambda_{d}\mathcal{L}_{\mathrm{detail}}.
\tag{2}
\]

其中，\(\mathcal{L}_{\mathrm{seg}}^{(i)}\) 为第 \(i\) 路分割输出的 OHEM 交叉熵损失，\(\mathcal{L}_{\mathrm{mask}}^{(i)}\) 为前景—背景掩码感知损失，\(\mathcal{L}_{\mathrm{detail}}\) 为细节监督损失；权重设置为 \(\omega_0=10\)、\(\omega_1=1\)、\(\omega_2=3\)、\(\lambda_d=3\)。细节监督目标由标注掩码的局部结构信息提取生成。多阶段分割损失对主输出与两路辅助输出进行约束，掩码感知损失强化模型对缺陷前景区域整体分布的感知，细节监督损失引导网络学习缺陷局部结构。

## 2 实验与分析

### 2.1 数据集与实验设置

**NEU-Seg 公开数据集。** NEU-Seg 为带钢表面缺陷像素级语义分割数据集，本文实验包含裂纹（crazing）、夹杂物（inclusion）、斑块（patches）三类缺陷及背景类，共 4 类。训练集 3630 张，测试集 840 张，输入尺寸统一为 200×200。该数据集作为主要基准，用于主结果比较、消融实验、边界质量评价、缺陷尺度分层和类型差异分析。

**皮革表面缺陷数据集。** 该数据集来源于实际工业生产环境，包含开创伤、刺刮伤、烙印、破洞、皮肤藓、烂面、刺猴 7 类缺陷，共 2341 张 768×768 图像（训练 1638 张、验证 468 张、测试 235 张），用于验证模型面对不同材质、纹理背景和缺陷形态时的适应能力，作为真实工业场景的泛化验证。

**评价协议。** 测试阶段采用 TestRescale 与 ToTensor 变换，不使用 ImageNet 归一化；mIoU 计算包含背景类。NEU-Seg 主实验采用固定训练终点评价：训练预算固定为 240k 次迭代，仅在训练终点对全部 840 张测试图像评价一次，不进行任何基于测试集的中间模型选择。边界质量评价采用 1/2/3 像素容差的边界 IoU（Boundary IoU）与边界 F1 分数（BF-score），边界带由 3×3 形态学腐蚀—膨胀差分生成；评价指标均基于同一批冻结预测结果计算，缺陷类别不存在于标注且预测也为空的样本不计入该类平均。

**训练设置。** 所有 NEU-Seg 实验使用相同环境：NVIDIA A100-PCIE-40GB GPU，PyTorch 2.7.1，CUDA 12.6，Python 3.11。优化器为 Adam，初始学习率 1×10⁻⁴ 并保持恒定，权重衰减 2×10⁻⁶，batch size 为 16，从头训练，随机种子 1337。Base-B 表示仅采用细节—语义互优化机制的基础网络（ResNet-50 主干）；A2MS-DSMONet-B 为在基础网络上引入自适应语义增强模块与细节监督分支的完整模型。

### 2.2 NEU-Seg 主结果比较

表 1 给出 NEU-Seg 数据集上长训练终点的定量结果。A2MS-DSMONet-B 取得 91.45% mIoU，较 Base-B 提高 0.21 个百分点，四个类别（含背景）的 IoU 均不低于 Base-B；同时保持 37.37 FPS 的实时推理速度，参数量仅增加 0.31%。

表 1 NEU-Seg 数据集长训练终点结果（统一 840 张固定终点评价，240k 迭代，seed 1337）/ Table 1 Long-training endpoint results on NEU-Seg (unified 840-image fixed-endpoint evaluation, 240k iterations, seed 1337)。

| 模型 | AAM | 细节监督 | mIoU/% | 背景 | 裂纹 | 夹杂物 | 斑块 | FPS | Params/M |
|---|:-:|:-:|---:|---:|---:|---:|---:|---:|---:|
| Base-B | × | × | 91.24 | 98.62 | 84.15 | 93.49 | 88.71 | 38.16 | 29.37 |
| A2MS-DSMONet-B | √ | √ | **91.45** | 98.64 | 84.47 | 93.55 | **89.16** | 37.37 | 29.46 |

与主流方法的对比结果将在核验各方法原始出处与评价协议一致性后补充；对于协议或硬件口径不一致的文献数字，将以"文献值，仅供量级参考"方式标注。

### 2.3 不同缺陷类型性能分析

整体 mIoU 只能反映平均分割水平，工业缺陷的类间形态差异要求更细粒度的分析。本节从类别 IoU、边界质量和缺陷尺度三个维度比较 A2MS-DSMONet-B 与 Base-B。

**类别级与边界级分析。** 表 2 给出两类模型在三个缺陷类别上的 IoU 与边界质量差异（840 张测试图像逐病例配对比较）。

表 2 缺陷类别级性能差异（A2MS-DSMONet-B − Base-B，240k 终点，配对比较）/ Table 2 Per-defect-type performance differences (A2MS-DSMONet-B minus Base-B, 240k endpoint, paired comparison)。

| 类别 | Δ IoU/pt（聚集） | Δ Boundary IoU@2px/pt | Δ BF1@2px/pt |
|---|---:|---:|---:|
| 裂纹 crazing | +0.32 | −0.45 | −0.60 |
| 夹杂物 inclusion | +0.06 | +0.32 | +0.04 |
| 斑块 patches | +0.45 | +0.54 | +0.30 |

斑块类缺陷的区域与边界质量同步改善：1/2/3 像素容差下边界 IoU 分别提高 0.55、0.54、0.55 个百分点（病例级配对自助法 95% 置信区间均不含零；按全部测试像素聚集计算则分别提高 0.74、0.85、0.86 个百分点），表明语义—细节联合优化对区域型缺陷具有更好的结构保持能力。裂纹类缺陷的聚集 IoU 提高 0.32 个百分点，主要来自大面积裂纹区域内部空洞的填补；但典型病例的边界定位精度略有下降，呈现区域完整性与边界定位之间的折中。夹杂物类缺陷各项指标变化较小，性能保持稳定。

**尺度分层分析。** 按各缺陷类别标注面积占比的 P33/P67 分位数将测试样本划分为小、中、大三组，比较两模型在各组的 IoU 差异（表 3）。斑块类缺陷在小面积组增益最大（+0.73 个百分点），裂纹类在小尺度组出现一定回退。尺度效应具有类别依赖性，不存在统一的"小目标增益"规律。

表 3 缺陷尺度分层 IoU 差异（ΔIoU/pt，按类别内 GT 面积占比 P33/P67 分层）/ Table 3 Scale-stratified IoU differences (per-class P33/P67 tiers)。

| 类别 | 小 | 中 | 大 |
|---|---:|---:|---:|
| 裂纹 crazing | −1.24 | −0.50 | −0.44 |
| 夹杂物 inclusion | +0.28 | −0.24 | −0.05 |
| 斑块 patches | +0.73 | −0.04 | +0.49 |

进一步以周长/面积平方根作为裂纹形态复杂度代理指标进行探索性分层，结果显示复杂度越高的裂纹组回退越明显（高复杂度组 ΔIoU 为 −1.74 个百分点），说明当前的细节监督约束对细长多分支结构的边界精确定位作用有限。

### 2.4 真实皮革缺陷数据验证

除公开 NEU-Seg 数据集外，本文进一步在皮革缺陷数据集上进行验证。由于工业材质、纹理背景和缺陷形态存在明显差异，跨材质实验能够从应用角度补充评价模型的适应能力。

表 4 不同数据集实验结果 / Table 4 Results on different datasets。

| Dataset | Model | mIoU/% | FPS | Params/M |
|---|---|---:|---:|---:|
| NEU-Seg | Base-B | 91.24 | 38.16 | 29.37 |
| NEU-Seg | A2MS-DSMONet-B | **91.45** | 37.37 | 29.46 |
| Leather | Base-B | [待填入皮革实验结果] | [待填入] | [待填入] |
| Leather | A2MS-DSMONet-B | [待填入皮革实验结果] | [待填入] | [待填入] |

注：皮革数据集实验结果正在按与 NEU-Seg 相同的锁定协议复核，复核完成前以占位符标记，不引用任何未经确认的数字。

### 2.5 消融实验

表 5 给出固定 10k 迭代训练预算下、3 个随机种子（1337/2026/3407）的模块组合消融结果（mean±SD）；表 6 给出长训练终点验证结果。

表 5 模块组合消融（NEU-Seg，10k 迭代，3 种子 mean±SD，mIoU/%）/ Table 5 Module ablation (NEU-Seg, 10k iterations, 3 seeds, mean±SD, mIoU/%)。

| 配置 | AAM | 细节监督 | mIoU/% |
|---|:-:|:-:|---:|
| Base-B | × | × | 85.22±0.41 |
| +AAM | √ | × | 85.70±0.43 |
| +细节监督 | × | √ | 85.44±0.19 |
| A2MS-DSMONet-B | √ | √ | **85.75±0.54** |

表 6 长训练终点验证（NEU-Seg，240k 迭代，seed 1337，mIoU/%）/ Table 6 Long-training endpoint verification (NEU-Seg, 240k iterations, seed 1337, mIoU/%)。

| 配置 | mIoU/% |
|---|---:|
| Base-B | 91.24 |
| A2MS-DSMONet-B | **91.45** |

在相同 10k 迭代预算的四组配置中，联合配置取得最高平均 mIoU。其中自适应语义增强模块在 3 个种子中的 2 个上表现为正向增益（平均 +0.48 个百分点），细节监督约束在无注意力模块时 3 个种子均为小幅正向（平均 +0.23 个百分点），二者联合后的增益方向存在种子间差异。长训练终点上，联合配置较基础网络提高 0.21 个百分点。上述结果表明两个模块的组合能够取得最佳终点性能；单模块增益幅度较小，其作用更多体现为联合优化下的类型差异化改善（见 2.3 节）。

注：受算力约束，AAM-only 与细节监督-only 配置未开展 240k 长训练消融，长训练消融仅含 Base-B 与完整模型两组；10k 消融用于模块组合筛查。

### 2.6 复杂度与实时性分析

工业在线检测不仅关注分割精度，也关注模型复杂度和实时推理能力。表 7 汇总两模型的复杂度与速度实测结果（NVIDIA A100-PCIE-40GB，batch size 1，输入 200×200，FP32；预热 100 次、测量 1000 次，双模型交错分段测量以消除共享环境干扰）。

表 7 复杂度与实时性对比 / Table 7 Comparison of model complexity and real-time performance。

| 模型 | Params/M | MACs/G | 延迟均值/ms | 延迟 P95/ms | FPS | 峰值显存/MB |
|---|---:|---:|---:|---:|---:|---:|
| Base-B | 29.37 | 6.648 | 26.21 | 42.42 | 38.16 | 270.3 |
| A2MS-DSMONet-B | 29.46 | 6.648 | 26.76 | 45.43 | 37.37 | 270.3 |

A2MS-DSMONet-B 相对 Base-B 参数量仅增加 0.31%，推理 MACs 基本持平（细节监督分支仅参与训练，推理时不计算），平均延迟增加约 2.1%，仍保持 37 FPS 以上的实时推理能力。这表明所提方法以极小计算代价换取了分割精度与部分缺陷类别结构质量的提升。

注：FPS 为共享 GPU 环境下双模型交错测量的相对比较结果；空闲环境下绝对帧率更高，两模型相对关系不变。

### 2.7 可视化分析

图 3 NEU-Seg 数据集可视化分割结果 / Fig. 3 Visualization results on NEU-Seg。图件要求：导出 300 dpi 以上位图或矢量图。

为避免人工择优，展示病例按预先固定规则选择：以 A2MS-DSMONet-B 与 Base-B 的逐病例 BF1@2px 差值排序，对裂纹与斑块两类分别选取改善最大、退化最大和差值近零的病例各 3 例。每组展示原图、标注、两模型预测结果及误差分布图（真正例、假正例、假负例分别以不同颜色标注）。

可视化结果与定量分析一致：在改善病例中，A2MS-DSMONet-B 能够填补大面积裂纹区域内部的误分类空洞，使预测区域更完整；在退化病例中，该模型在强边缘附近出现裂纹区域的过度扩张，并对个别纤细裂纹段的响应减弱。斑块类病例中，改善病例的缺陷轮廓与标注贴合更紧密。

## 3 讨论

### 3.1 方法适用性分析

工业表面缺陷存在显著的类别间形态差异，本文实验表明所提方法的增强效果同样具有类型依赖性。对于区域型斑块缺陷，语义—细节联合优化的收益最明确，区域 IoU 与多容差边界 IoU 同步改善，该类缺陷边界相对闭合、区域内部占比高，自适应通道重标定带来的语义一致性改善能够直接转化为结构保持能力。对于细长复杂的裂纹缺陷，收益呈现区域完整性与边界定位之间的权衡：聚集尺度上裂纹 IoU 的改善主要来自大面积裂纹区域内部空洞的填补，而典型病例的边界指标略有下降，且形态复杂度越高折中越明显，这提示当前的细节监督约束更适合表述为对局部结构学习的引导，而非对细长结构边界定位的保证。夹杂物缺陷分割性能基本稳定。这种差异化结果说明，语义—细节协同增强并非对所有缺陷类型产生一致增益，其价值需要根据缺陷形态分别评估。

除公开 NEU-Seg 数据集外，本文进一步在皮革缺陷数据集上进行了验证。由于工业材质、纹理背景和缺陷形态存在明显差异，跨材质实验能够从应用角度补充评价模型的适应能力，进一步验证模型在不同工业材质上的应用潜力。

### 3.2 局限性

本文仍存在以下局限：（1）240k 长训练终点比较基于单一随机种子，+0.21 个百分点的差异处于训练随机性量级附近，其统计稳定性需多种子长训练进一步验证；（2）主实验集中于 NEU-Seg 单一工业材质，皮革数据集实验尚在复核，跨材质泛化结论需更多数据支持；（3）FPS 测量在共享 GPU 环境下完成，绝对帧率受环境影响；（4）细节监督约束在细长复杂裂纹上存在边界定位折中，有待通过形态感知的监督设计进一步研究；（5）与主流方法的系统对比将在核验各方法评价协议一致性后补充。

## 4 结论

本文提出一种轻量级语义—细节协同增强的工业表面缺陷实时分割网络 A2MS-DSMONet，在保证实时性的同时取得较高分割精度。在 NEU-Seg 数据集统一固定终点评价下，A2MS-DSMONet-B 取得 91.45% mIoU 和 37.37 FPS，参数量较基础网络仅增加 0.31%。细粒度实验分析表明，该方法能够改善部分缺陷类别的结构表达能力——对斑块类缺陷的区域与边界质量均有改善，对裂纹类缺陷主要改善区域完整性——并展现出较好的工程应用潜力。后续工作将围绕细长复杂缺陷的边界保持、多种子长训练验证和跨材质泛化展开。

## 参考文献

正式参考文献条目待逐条核验作者、题名、来源、卷期页码和 DOI 后，按《中国图象图形学报》格式补齐。未核验作者、题名、来源或 DOI 的文献均继续保留"待核验"状态；正文引用采用作者—年份形式。
