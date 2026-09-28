# 语义—细节协同增强的工业表面缺陷实时语义分割网络

# Real-time Semantic Segmentation Network for Industrial Surface Defects with Semantic-Detail Collaborative Enhancement

作者：待补充

单位：待补充

中图分类号：待核定（候选见投稿元数据待办文档）

文献标志码：A

收稿日期：待补充

基金项目：待补充

## 摘要

**目的** 工业表面缺陷具有尺度变化大、边界模糊、类别形态差异明显等特点，在线检测场景还要求模型具备实时推理能力。如何在语义分割中兼顾高层语义理解与局部细节保持，并在分割精度与推理速度之间取得平衡，是工业表面缺陷检测面临的重要问题。

**方法** 首先构建细节—语义相互优化网络 DSMONet，通过深层语义信息调节浅层细节特征、抑制背景纹理干扰，并利用优化后的浅层空间信息反向补偿深层语义特征，实现浅层细节与深层语义的双向互优化。在此基础上，针对工业表面缺陷低占比、弱纹理、多尺度和形态差异显著等特点，进一步设计自适应语义增强模块，通过高效通道注意力与自适应通道权重对参与互优化的语义特征进行通道重标定，并引入仅训练阶段的辅助细节监督约束局部结构学习，形成面向工业表面缺陷的 A2MS-DSMONet。

**结果** 在 NEU-Seg 带钢表面缺陷公开数据集上，采用 ResNet-50 主干的 DSMONet-B 在固定训练终点评价下取得 91.24% mIoU；进一步增强后的 A2MS-DSMONet-B 取得 91.45% mIoU 和 37.37 FPS，参数量仅增加 0.31%。缺陷形态分析表明，模型对斑块类缺陷的区域与边界质量改善较为明确，而裂纹类缺陷呈现区域完整性与边界定位之间的性能权衡。在真实皮革缺陷数据集上，经验证集选择的 A2MS-DSMONet-B 在 468 张独立测试图像上取得 90.97% mIoU，高于 DSMONet-B 的 89.91%，并高于相同协议下复现的 STDC、PIDNet、DDRNet 等实时分割方法。

**结论** 实验结果表明，细节—语义互优化架构具有较强基础分割能力，面向工业缺陷的自适应增强能够以较小附加计算开销进一步提升分割性能并保持实时性；细粒度分析表明，增强效果与缺陷形态存在一定关联。

## 关键词

工业表面缺陷；实时语义分割；细节—语义互优化；自适应通道重标定；细节监督；缺陷类型分析

## Abstract

**Objective** Industrial surface defects are characterized by large scale variation, blurred boundaries, and pronounced morphological differences across defect categories, while online inspection additionally requires real-time inference. How to balance high-level semantic understanding with local detail preservation, and how to reconcile segmentation accuracy with inference speed, remain key problems for semantic segmentation in industrial surface inspection.

**Method** We first construct DSMONet, a detail-semantic mutual optimization network in which deep semantic features modulate shallow detail features to suppress background-texture responses, while the refined shallow spatial information in turn compensates deep features with positional and structural cues, forming bidirectional mutual optimization between detail and semantics. Building on DSMONet, we further develop A2MS-DSMONet for industrial surface defects, which are typically low-contrast, weakly textured, multi-scale and morphologically diverse. An adaptive attention module (AAM) performs channel re-calibration of the semantic features involved in the mutual optimization path through an efficient squeeze-and-excitation unit and a learnable adaptive channel weighting unit, and an auxiliary detail supervision branch constrains local structural learning during training only, introducing no extra inference cost.

**Result** On the public NEU-Seg strip-steel surface defect dataset, with 840 test images evaluated under a unified fixed-endpoint protocol, DSMONet-B with a ResNet-50 backbone achieves 91.24% mIoU (including background), and the enhanced A2MS-DSMONet-B further reaches 91.45% mIoU at 37.37 FPS on an NVIDIA A100 (batch size 1, FP32), with only 0.31% more parameters. Per-defect-type analysis shows that patch-type defects gain the most, with both regional IoU and boundary quality improved (boundary IoU gains of about 0.5-0.9 percentage points at 1/2/3-pixel tolerances), while crazing-type defects exhibit a trade-off between regional completeness and boundary localization, and inclusion-type defects remain stable. On a real-world leather defect dataset with seven defect categories, the validation-selected A2MS-DSMONet-B achieves 90.97% mIoU on an independent 468-image test set, compared with 89.91% for DSMONet-B, and surpasses STDC, PIDNet and DDRNet models re-implemented and evaluated under the same protocol (88.24%, 86.20% and 84.21% mIoU respectively).

**Conclusion** The detail-semantic mutual optimization architecture provides a strong foundation for real-time segmentation, and the defect-oriented adaptive enhancement further improves segmentation performance with small additional computational cost. Fine-grained analysis shows that the enhancement is related to defect morphology: region-type defects benefit more clearly than slender, complex crack-type defects, which suggests that semantic-detail collaborative enhancement should be evaluated per defect morphology rather than assumed to help all categories uniformly.

## Keywords

industrial surface defects; real-time semantic segmentation; detail-semantic mutual optimization; adaptive channel re-calibration; detail supervision; defect-type analysis

## 0 引言

工业制造过程中，钢材、皮革、芯片、磁瓦和金属零部件等材料表面容易出现裂纹、夹杂、斑块、破洞和弱纹理缺陷。这类缺陷会影响产品外观质量、结构可靠性和后续加工稳定性，因此在生产线上实现快速、稳定、精细的缺陷检测具有重要工程意义。传统人工检测依赖经验，检测效率和一致性受主观因素影响较大；传统机器视觉方法通常依赖灰度、纹理、边缘或频域特征，在光照变化、背景纹理复杂和缺陷形态多变时泛化能力不足。随着深度学习的发展，语义分割方法能够输出像素级缺陷区域，为缺陷定位、面积统计和质量判定提供更精细的信息。

将通用语义分割方法直接应用于工业表面缺陷实时检测仍面临多方面挑战。工业缺陷具有复杂纹理结构和多尺度分布特征，缺陷区域在整幅图像中占比低、形态差异大，且常与材料背景纹理交织。浅层特征虽然包含边缘、纹理和位置信息，却容易混入背景噪声；深层语义特征具有更强判别能力，但连续下采样会削弱局部结构与位置表达。因此，如何在语义分割模型中兼顾高层语义理解与局部细节保持，同时满足在线检测的实时性约束，仍是工业表面缺陷分割面临的重要问题。大型分割网络虽然表达能力较强，但推理开销较大；轻量网络速度较快，却容易损失局部结构信息。分割精度、推理速度与局部结构表达能力之间的平衡，是工业表面缺陷实时分割中的关键问题。

近年来，工业缺陷分割研究逐渐从通用编码器—解码器结构转向任务特异的轻量化、注意力增强和细节建模。Zhu 等（2023）提出 Sub-region UNet，通过空间子区域特征提取与掩码感知损失分割金属表面弱缺陷；Zuo 等（2025）提出 DMC-Net，面向实时表面缺陷分割设计轻量特征提取与多尺度增强模块；Jeong 等（2025）提出 TAG-Net，针对钢材表面缺陷显式建模背景、缺陷和边界三重注意力；Li 等（2025）提出 CDARNet，面向实时金属表面缺陷分割，通过跨维度自适应区域重构增强鲁棒性；Zhang 等（2025）提出 LGGFormer，以双分支局部引导全局自注意力网络兼顾局部细节与全局建模。上述工作表明，工业缺陷分割不仅需要全局语义判别能力，还需要在复杂背景纹理下保持缺陷的局部结构。

实时语义分割研究主要围绕轻量骨干、高低分辨率分支、快速上下文聚合和高效特征融合展开。Yu 等（2018）提出 BiSeNet，以空间细节路径和语义上下文路径的双分支建模为代表；Fan 等（2021）提出 STDCNet，通过短期密集连接和细节聚合提升实时分割效率，其细节监督思想也为本文的细节约束设计提供参考；Hong 等（2023）提出 DDRNet，通过双分辨率结构保留空间细节并聚合上下文；Peng 等（2022）提出 PP-LiteSeg，其统一注意力融合模块（UAFM）为本文细节—语义交互模块提供参考；Xu 等（2023）提出 PIDNet，借鉴 PID 控制器设计三分支结构并引入边界注意力；Wan 等（2025）提出 SeaFormer++，以挤压增强轴向注意力改进移动端速度—精度平衡。这些方法多在 Cityscapes、CamVid、ADE20K 等自然场景数据集上评估，且多依赖单向的高低层特征融合或并行分支结构，深层语义与浅层空间细节之间仍缺乏充分的双向互优化；直接迁移到工业缺陷场景时，还需进一步应对低占比前景、复杂纹理背景和类间形态差异等任务特性。

针对上述问题，本文首先构建细节—语义相互优化网络 DSMONet，通过建立深层语义信息与浅层空间细节之间的双向信息交互，实现语义判别能力与局部结构信息的协同优化。该结构设计本身不依赖特定缺陷类别，可作为面向不同工业材质与缺陷类型的基础架构。在此基础上，本文进一步针对工业表面缺陷低占比、弱纹理、尺度变化和类别形态差异显著等特点，引入自适应语义增强模块与辅助细节监督机制，形成面向工业表面缺陷的 A2MS-DSMONet。本文在 NEU-Seg 公开带钢表面缺陷数据集上开展主结果比较、消融实验、边界质量评价和缺陷尺度分层分析，并在真实工业生产场景采集的皮革表面缺陷数据集上进行应用验证。

本文主要贡献如下：

1. 构建细节—语义相互优化网络 DSMONet，通过深层语义信息对浅层空间细节进行语义约束，并利用优化后的浅层结构信息反向补偿深层特征的位置与结构表达，实现浅层细节与深层语义的双向协同优化。在当前统一训练与评价协议下，采用 ResNet-50 主干的 DSMONet-B 在 NEU-Seg 数据集上取得 91.24% mIoU。
2. 面向工业表面缺陷低占比、弱纹理、多尺度和形态差异明显等特征，在 DSMONet 基础上进一步设计 A2MS-DSMONet，引入由高效通道注意力与自适应通道权重组成的自适应语义增强模块，并结合辅助细节监督对局部结构学习进行约束，实现高层语义表征与缺陷局部结构信息的进一步协同优化。
3. 在 NEU-Seg 公开带钢缺陷数据集和真实皮革缺陷数据集上进行系统验证，并从类别、边界、尺度和形态复杂度等维度分析模型表现。A2MS-DSMONet-B 分别取得 91.45% 和 90.97% 的 mIoU，同时保持实时推理能力；细粒度结果进一步表明模型增强效果具有缺陷形态依赖性。

## 1 DSMONet 与面向工业缺陷的协同增强网络

### 1.1 总体框架

本文网络采用两级方法体系，总体结构如图 1 所示。第一级为细节—语义相互优化网络 DSMONet：以 ResNet-50 为主干提取多层特征，通过深层语义信息与浅层空间细节之间的双向交互实现协同建模，解决实时语义分割中高层语义判别能力与局部结构保持难以兼顾的问题。第二级为面向工业表面缺陷的增强：针对缺陷低占比、弱纹理、多尺度和形态差异显著等任务特性，在 DSMONet 的互优化路径中引入自适应语义增强模块（adaptive attention module，AAM）对参与交互的语义特征进行通道重标定，并在训练阶段引入辅助细节监督分支约束局部结构学习，构成 A2MS-DSMONet。采用 ResNet-50 主干的模型记为 DSMONet-B 与 A2MS-DSMONet-B。推理阶段网络仅保留主分割输出，辅助输出与细节监督分支不参与推理，不引入额外计算开销。

图 1 A2MS-DSMONet 网络结构示意图。DSMONet 基础架构实现细节—语义相互优化，AAM 与辅助细节监督为面向工业缺陷的增强 / Fig. 1 Overall architecture of A2MS-DSMONet. DSMONet forms the detail-semantic mutual optimization backbone, while AAM and auxiliary detail supervision constitute the defect-oriented enhancements。

### 1.2 DSMONet 细节—语义相互优化网络

工业缺陷图像中的浅层特征和深层特征具有明显互补关系。浅层特征保留较高空间分辨率，能够提供缺陷边缘、纹理变化和位置分布等细节信息；但材料表面的自然纹理、加工痕迹和局部亮度变化容易使浅层特征同时响应缺陷区域和非缺陷背景。深层特征具有更强语义判别能力，但连续下采样会降低空间分辨率，削弱局部结构表达。常规的单向融合方式（如直接将浅层特征与上采样的深层特征拼接或相加）仅对两类特征作一次性组合，浅层特征中的背景纹理响应难以被抑制，深层特征也难以获得局部结构的反向补偿。

DSMONet 建立浅层空间细节与深层语义信息之间的双向互优化。具体地，主干网络提取的多层特征中，最深层特征经多尺度上下文聚合模块（DAPPM）增强后，与自身一并送入注意力特征调制模块（UAFM，记为 arm1）整合为语义特征；该语义特征经 SqueezeBodyEdge 单元分解为语义主体与语义边缘分量，并经通道重标定单元调节各通道响应。浅层特征经步进卷积降采样与拉普拉斯卷积单元增强局部结构响应后，与上采样的语义边缘分量和通道重标定语义特征共同进入细节融合单元，形成细节增强特征——深层语义由此对浅层细节实现语义约束。随后，语义主体与重标定语义特征经逐元素相加回流，并与细节增强特征再次经 UAFM（记为 arm2）融合，使优化后的浅层细节反向补偿深层语义的位置与结构信息，完成细节与语义的双向互优化。互优化特征经分割头产生主分割输出。

上述过程可概念化表示为：

\[
\mathbf{F}_{d}'=\mathcal{G}_{d}(\mathbf{F}_{d},\mathbf{F}_{s}),\qquad
\mathbf{F}_{s}'=\mathbf{F}_{s}\oplus\mathcal{B}(\mathbf{F}_{s})\oplus\mathcal{R}(\mathbf{F}_{s}),\qquad
\mathbf{F}_{\mathrm{out}}=\mathcal{H}(\mathbf{F}_{d}',\mathbf{F}_{s}'),
\tag{1}
\]

其中，\(\mathbf{F}_{d}\) 与 \(\mathbf{F}_{s}\) 分别表示浅层细节特征与深层语义特征；\(\mathcal{G}_{d}(\cdot)\) 表示语义对细节的调制，由语义边缘分量、通道重标定语义特征与浅层细节的细节融合实现；\(\mathcal{B}(\cdot)\) 与 \(\mathcal{R}(\cdot)\) 分别表示语义主体分解与通道重标定，二者与原始语义特征逐元素相加构成语义回流；\(\mathcal{H}(\cdot)\) 为基于 UAFM 的细节—语义融合，使优化后的细节信息反向补偿语义表达。与一次性拼接或相加不同，该结构先以语义约束细节、再以优化细节补偿语义，形成两阶段的双向互优化。

### 1.3 自适应语义增强模块

工业缺陷常具有低占比、弱纹理和尺度变化明显等特点，不同语义通道对缺陷判别的重要性存在差异。为增强网络对关键语义区域的响应能力，本文在细节—语义互优化路径的通道重标定位置引入自适应语义增强模块 AAM，替代常规通道注意力单元。AAM 由高效挤压激励单元（eSE）与自适应通道权重单元（ACW）组成：eSE 通过全局平均池化与 \(1\times1\) 卷积，以较低计算开销建模通道注意力响应；ACW 通过可学习的逐通道权重，使网络根据优化目标动态调整各通道的重要程度。二者联合对参与互优化的语义特征进行通道重标定，调节不同语义通道在细节—语义融合过程中的贡献。AAM 并非独立于主干的外挂模块，而是嵌入互优化路径，其输出同时参与细节融合与语义回流两个方向的交互。

设输入特征为 \(\mathbf{F}\)，AAM 的计算可表示为：

\[
\mathbf{a}_{\mathrm{eSE}}=h_{\mathrm{sig}}\!\left(\phi_{1\times1}(\operatorname{GAP}(\mathbf{F}))\right),
\qquad
\mathbf{F}_{\mathrm{AAM}}=\mathbf{F}\odot \mathbf{a}_{\mathrm{eSE}}\odot \mathbf{w}_{\mathrm{ACW}},
\tag{2}
\]

其中，\(\operatorname{GAP}(\cdot)\) 表示全局平均池化，\(\phi_{1\times1}\) 表示 \(1\times1\) 卷积，\(h_{\mathrm{sig}}(\cdot)\) 表示 HSigmoid 激活函数，\(\mathbf{w}_{\mathrm{ACW}}\) 为可学习通道权重，\(\odot\) 表示逐元素乘法。AAM 主要由轻量级通道操作构成，仅引入少量额外计算。

### 1.4 面向局部结构学习的辅助细节监督

在主分割监督之外，A2MS-DSMONet 引入辅助细节监督分支，引导网络学习缺陷的局部结构信息。具体地，浅层细节特征经拉普拉斯卷积单元增强后接入一个独立的单通道细节输出头，对缺陷局部结构进行预测，并由细节监督损失约束；细节监督目标由标注掩码的局部结构信息提取生成。该分支仅在训练阶段参与，推理阶段不启用，因而不影响模型的推理速度与部署开销。需要说明的是，细节监督思想已有研究基础（如 STDC 系列工作），本文的差异在于将该约束接入细节—语义互优化结构的浅层细节路径，并与自适应语义增强联合优化；前景—背景掩码感知监督同样沿用已有方法，并非本文新提出的损失函数。

### 1.5 联合优化与损失函数

A2MS-DSMONet 的训练目标由多阶段分割损失、掩码感知损失和细节监督损失共同构成：

\[
\mathcal{L}_{\mathrm{total}}=
\sum_{i=0}^{2}\omega_{i}\left(\mathcal{L}_{\mathrm{seg}}^{(i)}+\mathcal{L}_{\mathrm{mask}}^{(i)}\right)
+\lambda_{d}\mathcal{L}_{\mathrm{detail}}.
\tag{3}
\]

其中，\(\mathcal{L}_{\mathrm{seg}}^{(i)}\) 为第 \(i\) 路分割输出的 OHEM 交叉熵损失，\(\mathcal{L}_{\mathrm{mask}}^{(i)}\) 为前景—背景掩码感知损失，\(\mathcal{L}_{\mathrm{detail}}\) 为细节监督损失；NEU-Seg 实验的权重设置为 \(\omega_0=10\)、\(\omega_1=1\)、\(\omega_2=3\)、\(\lambda_d=3\)，皮革实验沿用该数据集既有训练配置（\(\lambda_d=1\)，见 2.1 节）。多阶段分割损失对主输出与两路辅助输出进行约束，掩码感知损失强化模型对缺陷前景区域整体分布的感知，细节监督损失引导网络学习缺陷局部结构。

## 2 实验与分析

### 2.1 数据集与实验设置

**NEU-Seg 公开数据集。** NEU-Seg 为带钢表面缺陷像素级语义分割数据集，本文实验包含裂纹（crazing）、夹杂物（inclusion）、斑块（patches）三类缺陷及背景类，共 4 类。训练集 3630 张，测试集 840 张，输入尺寸统一为 200×200。该数据集作为主要基准，用于主结果比较、消融实验、边界质量评价、缺陷尺度分层和类型差异分析。

**皮革表面缺陷数据集。** 该数据集来源于实际工业生产环境，包含开创伤、刺刮伤、烙印、破洞、皮肤藓、烂面、刺猴 7 类缺陷，共 2341 张 768×768 图像，划分为训练集 1638 张、验证集 235 张、测试集 468 张。该数据集用于验证模型面对不同材质、纹理背景和缺陷形态时的适用性，作为真实工业场景的应用验证。其中验证集仅用于训练过程中的模型选择，测试集仅用于最终独立评价。两个数据集的图像与标注示例如图 2 所示。

图 2 数据集示例 / Fig. 2 Examples of the NEU-Seg and leather defect datasets。

**评价协议。** 测试阶段采用 TestRescale 与 ToTensor 变换，不使用 ImageNet 归一化；mIoU 计算包含背景类。两个数据集的评价流程有所不同：对 NEU-Seg，本文采用固定 240k 次迭代的训练预算，并在训练终点对全部 840 张测试图像统一评价一次，训练中不使用验证集或测试集进行模型选择；对皮革数据集，沿用该数据集的既有训练协议，以 235 张验证集选择最优模型权重，最终仅在 468 张独立测试图像上评价一次。两种协议均不使用测试集进行模型选择。边界质量评价采用 1/2/3 像素容差的边界 IoU（Boundary IoU）与边界 F1 分数（BF-score），边界带由 3×3 形态学腐蚀—膨胀差分生成；评价指标均基于同一批保存的预测结果计算，缺陷类别不存在于标注且预测也为空的样本不计入该类平均。

**训练设置。** NEU-Seg 实验使用相同环境：NVIDIA A100-PCIE-40GB GPU，PyTorch 2.7.1，CUDA 12.6，Python 3.11。优化器为 Adam，初始学习率 1×10⁻⁴ 并保持恒定，权重衰减 2×10⁻⁶，batch size 为 16，从头训练，随机种子 1337。DSMONet-B 为采用细节—语义互优化机制的基础网络（ResNet-50 主干）；A2MS-DSMONet-B 为在 DSMONet-B 上引入自适应语义增强模块与辅助细节监督分支的完整模型。皮革实验沿用该数据集的既有训练链：其训练配置记录为 Adam 优化器、学习率 1×10⁻⁴ 恒定、batch size 约 12，训练中每 1000 次迭代在 235 张验证集上监控 mIoU 并仅按验证集最优保存模型权重。皮革对比方法（STDC、PIDNet、DDRNet）在同一训练集、验证集和测试集划分下训练与评价。

### 2.2 NEU-Seg 总体性能比较

表 1 给出 NEU-Seg 数据集上长训练终点的定量结果。DSMONet-B 在统一协议下取得 91.24% mIoU，表明细节—语义互优化基础架构具有较强的缺陷分割能力；A2MS-DSMONet-B 进一步提高至 91.45% mIoU，数值提高 0.21 个百分点，其中斑块类 IoU 提升相对更明显；同时保持 37.37 FPS 的实时推理速度，参数量仅增加 0.31%。需要说明的是，不同文献在 NEU 系数据集上的数据划分与评价口径存在差异，直接引用文献数字难以保证公平比较，因此本文主要报告统一协议下 DSMONet-B 与完整模型的内部对比，并以消融实验与细粒度分析支撑结论。

表 1 NEU-Seg 数据集固定终点评价结果（统一 840 张测试图像，240k 迭代训练终点，seed 1337）/ Table 1 Fixed-endpoint evaluation results on NEU-Seg (unified 840 test images, 240k-iteration training endpoint, seed 1337)。

| 模型 | AAM | 细节监督 | mIoU/% | 背景 | 裂纹 | 夹杂物 | 斑块 | FPS | Params/M |
|---|:-:|:-:|---:|---:|---:|---:|---:|---:|---:|
| DSMONet-B | × | × | 91.24 | 98.62 | 84.15 | 93.49 | 88.71 | 38.16 | 29.37 |
| A2MS-DSMONet-B | √ | √ | **91.45** | 98.64 | 84.47 | 93.55 | **89.16** | 37.37 | 29.46 |

### 2.3 不同缺陷形态下的性能差异分析

整体 mIoU 只能反映平均分割水平，工业缺陷的类间形态差异要求更细粒度的分析。本节从类别 IoU、边界质量和缺陷尺度三个维度比较 A2MS-DSMONet-B 与 DSMONet-B。

**类别级与边界级分析。** 表 2 给出两类模型在三个缺陷类别上的 IoU 与边界质量差异（840 张测试图像逐样本配对比较）。

表 2 缺陷类别级性能差异（A2MS-DSMONet-B − DSMONet-B，240k 终点，配对比较）/ Table 2 Per-defect-type performance differences (A2MS-DSMONet-B minus DSMONet-B, 240k endpoint, paired comparison)。

| 类别 | Δ IoU/pt（聚集） | Δ Boundary IoU@2px/pt | Δ BF1@2px/pt |
|---|---:|---:|---:|
| 裂纹 crazing | +0.32 | −0.45 | −0.60 |
| 夹杂物 inclusion | +0.06 | +0.32 | +0.04 |
| 斑块 patches | +0.45 | +0.54 | +0.30 |

斑块类缺陷的区域与边界质量同步改善：1/2/3 像素容差下边界 IoU 分别提高 0.55、0.54、0.55 个百分点（样本级配对自助法 95% 置信区间均不含零；按全部测试像素聚合计算则分别提高 0.74、0.85、0.86 个百分点），表明语义—细节联合优化对区域型缺陷具有更好的结构保持能力。裂纹类缺陷的聚合 IoU 提高 0.32 个百分点，主要来自大面积裂纹区域内部空洞的填补；但典型样本的边界定位精度略有下降，呈现区域完整性与边界定位之间的折中。夹杂物类缺陷各项指标变化较小，性能保持稳定。

**尺度分层分析。** 按各缺陷类别标注面积占比的 P33/P67 分位数将测试样本划分为小、中、大三组，比较两模型在各组的 IoU 差异（表 3）。斑块类缺陷在小面积组增益最大（+0.73 个百分点），裂纹类在小尺度组出现一定回退。尺度效应具有类别依赖性，不存在统一的"小目标增益"规律。

表 3 缺陷尺度分层 IoU 差异（ΔIoU/pt，按类别内 GT 面积占比 P33/P67 分层）/ Table 3 Scale-stratified IoU differences (per-class P33/P67 tiers)。

| 类别 | 小 | 中 | 大 |
|---|---:|---:|---:|
| 裂纹 crazing | −1.24 | −0.50 | −0.44 |
| 夹杂物 inclusion | +0.28 | −0.24 | −0.05 |
| 斑块 patches | +0.73 | −0.04 | +0.49 |

进一步以周长/面积平方根作为裂纹形态复杂度代理指标进行探索性分层，结果显示复杂度越高的裂纹组回退越明显（高复杂度组 ΔIoU 为 −1.74 个百分点），说明当前的细节监督约束对细长多分支结构的边界精确定位作用有限。

### 2.4 真实皮革缺陷数据验证

除公开 NEU-Seg 数据集外，本文进一步在皮革缺陷数据集上验证模型在不同工业材质下的适用性。如 2.1 节所述，皮革实验中各模型均在 235 张验证集上进行模型选择，最终在 468 张独立测试图像上统一评价；不使用测试集进行任何模型选择。定量结果如表 4 所示。

表 4 皮革数据集上不同方法的分割性能比较（468 张独立测试图像）/ Table 4 Segmentation performance of different methods on the leather defect dataset (468 independent test images)。

| 方法 | 来源 | mIoU/% | Params/M |
|---|---|---:|---:|
| DDRNet23-slim（内部复现） | 本文评价 | 84.21 | 20.30 |
| PIDNet-S（内部复现） | 本文评价 | 86.20 | 7.72 |
| STDC-Seg（STDCNet1446，内部复现） | 本文评价 | 88.24 | 16.08 |
| DSMONet-B | 本文 | 89.91 | 29.37 |
| A2MS-DSMONet-B | 本文 | **90.97** | 29.46 |

注：所有方法均在相同的训练/验证/测试划分下训练，以验证集最优结果选择模型权重，并在同一 468 张测试图像上评价一次；mIoU 包含背景类。内部复现指基于原作者公开代码框架的本地实现，与官方结果可能存在实现差异。

表 5 给出逐类别结果。在 7 类皮革缺陷中，A2MS-DSMONet-B 在 6 类上取得正向变化，其中刺刮伤、开创伤、烂面三类改善相对明显；刺猴类 IoU 较 DSMONet-B 下降约 3.7 个百分点，说明自适应语义增强与细节监督的收益并非对所有缺陷类别一致。该现象与 NEU-Seg 中的类别依赖性结果相互呼应。

表 5 皮革数据集逐类别 IoU 对比（统一 468 张测试图像，%）/ Table 5 Per-class IoU comparison on the leather dataset (unified 468 test images, %)。

| 类别 | DSMONet-B | A2MS-DSMONet-B | Δ/pt |
|---|---:|---:|---:|
| 开创伤 open wound | 76.48 | 79.21 | +2.73 |
| 刺刮伤 scratch | 70.05 | 74.72 | +4.67 |
| 烙印 brand mark | 88.25 | 89.54 | +1.29 |
| 破洞 hole | 99.07 | 99.33 | +0.26 |
| 皮肤藓 skin disease | 97.12 | 97.87 | +0.75 |
| 烂面 rotten surface | 91.59 | 94.19 | +2.60 |
| 刺猴 wart | 97.41 | 93.66 | −3.75 |
| mIoU（含背景） | 89.91 | **90.97** | **+1.06** |

皮革数据集上的分割可视化案例如图 4 所示。

图 4 皮革真实缺陷验证案例 / Fig. 4 Validation cases on real-world leather defects。为避免人工择优，展示案例按预先确定的规则选择：在 468 张测试图像中取标注含缺陷的 268 张，按两模型逐图 mIoU 差值（A2MS-DSMONet-B − DSMONet-B）排序，分别选取改善最大、退化最大与差值近零的案例各 2 例，每例包含原图、标注、DSMONet-B 预测与 A2MS-DSMONet-B 预测。在 268 张含缺陷图像中，A2MS-DSMONet-B 有 211 张优于 DSMONet-B、45 张劣于 DSMONet-B、12 张基本持平；改善案例中 A2MS-DSMONet-B 对缺陷区域的覆盖更完整，退化案例则与其在个别类别（如刺猴）上的回退一致。该图用于展示真实工业场景下模型预测效果，不作为定量比较依据。

### 2.5 消融实验

表 6 给出固定 10k 迭代训练预算下、3 个随机种子（1337/2026/3407）的模块组合消融结果（mean±SD）；表 7 给出长训练终点验证结果。

表 6 模块组合消融（NEU-Seg，10k 迭代，3 种子 mean±SD，mIoU/%）/ Table 6 Module ablation (NEU-Seg, 10k iterations, 3 seeds, mean±SD, mIoU/%)。

| 配置 | AAM | 细节监督 | mIoU/% |
|---|:-:|:-:|---:|
| DSMONet-B | × | × | 85.22±0.41 |
| DSMONet-B + AAM | √ | × | 85.70±0.43 |
| DSMONet-B + 细节监督 | × | √ | 85.44±0.19 |
| A2MS-DSMONet-B | √ | √ | **85.75±0.54** |

表 7 长训练终点验证（NEU-Seg，240k 迭代，seed 1337，mIoU/%）/ Table 7 Long-training endpoint verification (NEU-Seg, 240k iterations, seed 1337, mIoU/%)。

| 配置 | mIoU/% |
|---|---:|
| DSMONet-B | 91.24 |
| A2MS-DSMONet-B | **91.45** |

在相同 10k 迭代预算的四组配置中，联合配置取得最高平均 mIoU。其中自适应语义增强模块在 3 个种子中的 2 个上表现为正向增益（平均 +0.48 个百分点），细节监督约束在无注意力模块时 3 个种子均为小幅正向（平均 +0.23 个百分点），二者联合后的增益方向存在种子间差异。长训练终点上，联合配置较基础网络提高 0.21 个百分点。上述结果表明，在当前实验设置下，联合配置取得最高平均性能，并在长训练终点评价中优于基础网络；单模块增益幅度较小，其作用更多体现为联合优化下的类型差异化改善（见 2.3 节）。

注：受算力约束，AAM-only 与细节监督-only 配置未开展 240k 长训练消融，长训练消融仅含 DSMONet-B 与完整模型两组；10k 消融用于模块组合筛查。

### 2.6 复杂度与实时性分析

工业在线检测不仅关注分割精度，也关注模型复杂度和实时推理能力。表 8 汇总两模型的复杂度与速度实测结果（NVIDIA A100-PCIE-40GB，batch size 1，输入 200×200，FP32；预热 100 次、测量 1000 次，双模型交错分段测量以消除共享环境干扰）。

表 8 复杂度与实时性对比 / Table 8 Comparison of model complexity and real-time performance。

| 模型 | Params/M | MACs/G | 延迟均值/ms | 延迟 P95/ms | FPS | 峰值显存/MB |
|---|---:|---:|---:|---:|---:|---:|
| DSMONet-B | 29.37 | 6.648 | 26.21 | 42.42 | 38.16 | 270.3 |
| A2MS-DSMONet-B | 29.46 | 6.648 | 26.76 | 45.43 | 37.37 | 270.3 |

A2MS-DSMONet-B 相对 DSMONet-B 参数量仅增加 0.31%，推理 MACs 基本持平（细节监督分支仅参与训练，推理时不启用），平均延迟增加约 2.1%，仍保持 37 FPS 以上的实时推理能力。在皮革实验的 768×768 输入下（独占 A100、FP32、batch size 1），DSMONet-B 与 A2MS-DSMONet-B 实测分别为 64.68 FPS 与 62.83 FPS；同环境下 STDC-Seg、PIDNet-S、DDRNet23-slim 分别为 85.18、109.83、150.62 FPS，本文模型帧率低于上述轻量方法，但绝对帧率仍满足在线检测的实时性需求。这表明所提方法以较小计算代价换取了分割精度与部分缺陷类别结构质量的提升。

注：NEU-Seg 输入下的 FPS 为共享 GPU 环境下双模型交错测量的相对比较结果，绝对帧率受环境影响；皮革输入下的 FPS 为独占 GPU 环境实测值。两种输入尺寸不同，帧率不横向比较。

### 2.7 可视化分析

图 3 NEU-Seg 数据集可视化分割结果 / Fig. 3 Visualization results on NEU-Seg。

NEU-Seg 上的分割可视化结果如图 3 所示。为避免人工择优，展示病例按预先确定规则选择：以 A2MS-DSMONet-B 与 DSMONet-B 的逐样本 BF1@2px 差值排序，对裂纹与斑块两类分别选取改善最大、退化最大和差值近零的代表性病例。每行依次展示原图、标注、两模型预测结果及误差分布图（真正例、假正例、假负例分别以不同颜色标注）。

可视化结果与定量分析一致：在改善病例中，A2MS-DSMONet-B 能够填补大面积裂纹区域内部的误分类空洞，使预测区域更完整；在退化病例中，该模型在强边缘附近出现裂纹区域的过度扩张，并对个别纤细裂纹段的响应减弱。斑块类病例中，改善病例的缺陷轮廓与标注贴合更紧密。

## 3 讨论

### 3.1 方法适用性分析

DSMONet 的细节—语义互优化架构在两个工业缺陷数据集上均表现出较强的基础分割能力：在 NEU-Seg 统一协议下取得 91.24% mIoU，在皮革数据集上取得 89.91% mIoU，且均高于同协议评价的 STDC、PIDNet、DDRNet 等轻量实时分割方法。这说明深层语义与浅层细节的双向互优化能够为工业表面缺陷分割提供有效的特征基础，其结构设计不依赖特定缺陷类别。

在此基础上，面向工业缺陷的自适应增强同样呈现类型依赖性。对于区域型斑块缺陷，语义—细节联合优化的收益最明确，区域 IoU 与多容差边界 IoU 同步改善，该类缺陷边界相对闭合、区域内部占比高，自适应通道重标定带来的语义一致性改善能够直接转化为结构保持能力。对于细长复杂的裂纹缺陷，收益呈现区域完整性与边界定位之间的权衡：聚合尺度上裂纹 IoU 的改善主要来自大面积裂纹区域内部空洞的填补，而典型样本的边界指标略有下降，且形态复杂度越高折中越明显，这提示当前的细节监督约束更适合表述为对局部结构学习的引导，而非对细长结构边界定位的保证。夹杂物缺陷分割性能基本稳定。这种差异化结果说明，语义—细节协同增强并非对所有缺陷类型产生一致增益，其价值需要根据缺陷形态分别评估。

除公开 NEU-Seg 数据集外，本文进一步在皮革缺陷数据集上考察模型的跨材质适用性。由于工业材质、纹理背景和缺陷形态存在明显差异，跨材质实验能够从应用角度补充评价模型的适应能力。实验结果显示，A2MS-DSMONet-B 在 468 张独立测试图像上取得 90.97% mIoU，较 DSMONet-B 提高 1.06 个百分点，说明所提增强结构在不同工业材质下仍能够获得正向总体结果。但逐类别分析显示，不同缺陷形态的响应并不一致，例如刺猴类别存在一定回退。这与 NEU-Seg 中的类别差异分析一致，提示语义—细节协同增强更适合从缺陷形态角度评价，而不宜简单概括为所有类别的统一提升。

### 3.2 局限性

本文仍存在以下局限：（1）NEU-Seg 240k 长训练终点比较基于单一随机种子，+0.21 个百分点的差异处于训练随机性量级附近，其统计稳定性需多种子长训练进一步验证；（2）皮革实验采用验证集最优模型选择协议，与 NEU-Seg 的固定训练终点协议不同，且皮革结果为单权重对单权重比较，训练随机性对差异幅度的影响未量化；（3）不同缺陷类别的收益存在形态依赖性，细长复杂裂纹存在区域完整性与边界定位的折中，刺猴类在皮革上出现回退；（4）FPS 测量受 GPU 环境（共享/独占）影响，文中已分别标注；（5）当前验证仅覆盖钢材与皮革两种工业材质；（6）受评价协议一致性约束，本文未直接引用其他文献在 NEU 系数据集上的报告值进行对比，NEU-Seg 上以统一协议下的内部对比为主。

## 4 结论

本文围绕实时语义分割中高层语义与局部空间细节协同建模问题，构建细节—语义相互优化网络 DSMONet，并在此基础上针对工业表面缺陷低占比、弱纹理、多尺度与形态差异显著等特点，引入自适应语义增强模块与辅助细节监督机制，形成面向工业表面缺陷的 A2MS-DSMONet。NEU-Seg 固定终点评价结果显示，DSMONet-B 取得 91.24% mIoU，A2MS-DSMONet-B 进一步提高至 91.45% mIoU 并保持 37.37 FPS 的实时推理能力；细粒度分析进一步表明，斑块类缺陷的区域与边界质量改善较为明确，而细长裂纹存在区域完整性与边界定位之间的权衡。在真实皮革缺陷数据集上，经验证集选择的 A2MS-DSMONet-B 在 468 张独立测试图像上取得 90.97% mIoU，较 DSMONet-B 的 89.91% 提高 1.06 个百分点。结果表明，细节—语义互优化架构具有较强基础分割能力，所提增强策略能够以较小附加计算开销进一步提升分割性能并保持实时性，但增强效果仍具有缺陷形态依赖性。后续工作将围绕细长复杂缺陷的边界保持、多种子长训练验证和更多工业材质上的适用性评估展开。

## 参考文献

[1] Yu C, Wang J, Peng C, et al. BiSeNet: bilateral segmentation network for real-time semantic segmentation[C]//Proceedings of the European Conference on Computer Vision (ECCV). 2018: 334-349. DOI: 10.1007/978-3-030-01261-8_20.

[2] Fan M, Lai S, Huang J, et al. Rethinking BiSeNet for real-time semantic segmentation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2021: 9716-9725. DOI: 10.1109/CVPR46437.2021.00959.

[3] Hong Y, Pan H, Sun W, et al. Deep dual-resolution networks for real-time and accurate semantic segmentation of traffic scenes[J]. IEEE Transactions on Intelligent Transportation Systems, 2023, 24(3): 3448-3460. DOI: 10.1109/TITS.2022.3228042.

[4] Peng J, Liu Y, Tang S, et al. PP-LiteSeg: a superior real-time semantic segmentation model[EB/OL]. arXiv preprint arXiv:2204.02681, 2022.

[5] Xu J, Xiong Z, Bhattacharyya S P. PIDNet: a real-time semantic segmentation network inspired by PID controllers[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2023: 19529-19539. DOI: 10.1109/CVPR52729.2023.01871.

[6] Wan Q, Huang Z, Lu J, et al. SeaFormer++: squeeze-enhanced axial transformer for mobile visual recognition[J]. International Journal of Computer Vision, 2025, 133(6): 3645-3666. DOI: 10.1007/s11263-025-02345-2.

[7] Zhu W, Liang R, Yang J, et al. A sub-region Unet for weak defects segmentation with global information and mask-aware loss[J]. Engineering Applications of Artificial Intelligence, 2023, 122: 106011. DOI: 10.1016/j.engappai.2023.106011.

[8] Zuo H, Zheng Y, Huang Q, et al. DMC-Net: a lightweight network for real-time surface defect segmentation[J]. Journal of Real-Time Image Processing, 2025, 22(2). DOI: 10.1007/s11554-025-01639-5.

[9] Jeong S, Song J, Lee S J. TAG-Net: triple attention guided network for inspecting surface defects on steel products[J]. International Journal of Control, Automation and Systems, 2025, 23. DOI: 10.1007/s12555-024-0550-8.

[10] Li Q, Ding C, Wang B, et al. CDARNet: a robust cross-dimensional adaptive region reconstruction network for real-time metal surface defect segmentation[J]. Advanced Engineering Informatics, 2025, 67: 103514. DOI: 10.1016/j.aei.2025.103514.

[11] Zhang G, Lu Y, Jiang X, et al. LGGFormer: a dual-branch local-guided global self-attention network for surface defect segmentation[J]. Advanced Engineering Informatics, 2025, 64: 103099. DOI: 10.1016/j.aei.2024.103099.

[12] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016: 770-778. DOI: 10.1109/CVPR.2016.90.

注：以上条目已通过出版方页面/DOI/Crossref 逐条核验（见 docs/paper/reference_audit.md 与 reference_crosscheck_v5.md）。投稿前需按期刊著录格式统一调整。
