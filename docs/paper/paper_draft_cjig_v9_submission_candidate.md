# 细节—语义互优化与自适应增强的工业表面缺陷实时语义分割网络

# Real-time Semantic Segmentation Network for Industrial Surface Defects via Detail-Semantic Mutual Optimization and Adaptive Enhancement

作者：待补充

单位：待补充

中图分类号：待核定

文献标志码：A

收稿日期：待补充

基金项目：待补充

## 摘要

**目的** 工业表面缺陷具有尺度变化大、边界模糊、类别形态差异明显等特点，在线检测场景还要求模型具备实时推理能力。如何在语义分割中兼顾高层语义理解与局部细节保持，并在分割精度与推理速度之间取得平衡，是工业表面缺陷检测面临的重要问题。

**方法** 本文构建细节—语义相互优化网络 DSMONet：深层语义特征参与浅层细节融合，调节细节路径中的背景纹理响应；经语义约束的细节特征再与语义回流特征融合，形成兼顾类别判别与空间结构的联合表征。在此基础上，针对工业表面缺陷低占比、弱纹理、多尺度和形态差异明显等特点，引入由高效通道注意力与可学习逐通道权重组成的自适应语义增强模块，并以仅用于训练的辅助细节监督约束浅层结构学习，形成 A2MS-DSMONet。

**结果** 在 NEU-Seg 带钢表面缺陷公开数据集上，本文按统一 240k 固定终点协议比较经典语义分割、通用实时分割、具有公开可复现实现的工业缺陷参照方法与本文两级模型，共 10 个模型。DSMONet-B 与 A2MS-DSMONet-B 分别取得 91.24% 和 91.45% mIoU，完整增强配置的参数量为 29.46M、推理 MACs 为 6.65G；在指定 A100 GPU 前向测试中，A2MS-DSMONet-B 的单图等效帧率为 95.41 FPS（batch=1，FP32），该结果仅反映模型前向计算效率。两模型的 mIoU 数值高于本次统一复现的 8 种非本文提出的对比实现，但复杂度高于所列轻量方法。形态分析中，斑块类区域与边界指标呈正向变化，裂纹类存在聚合区域与样本平均边界结果之间的权衡。在另一工业材质皮革缺陷数据集上，两模型分别取得 89.91% 和 90.97% mIoU，结果作为跨材质应用场景下的补充评价。

**结论** 当前结果支持 DSMONet 的独立分割能力和完整工业增强配置在所测协议下的数值改善，并提供指定硬件下的前向计时依据。增强效果具有缺陷形态依赖性，文献报告值与本文统一复现值需分开解释；单图前向等效帧率不能替代生产线端到端吞吐评价。

## 关键词

工业表面缺陷；实时语义分割；细节—语义互优化；自适应通道重标定；细节监督；缺陷类型分析

## Abstract

**Objective** Industrial surface defects vary in size, texture and morphology, and their boundaries can be difficult to distinguish from material backgrounds. Pixel-level segmentation is useful for locating defects and describing their regions, but the features required for class discrimination and precise spatial localization are not interchangeable. Shallow features preserve position and fine structure while also responding to background texture; deep features provide contextual information but lose spatial detail through downsampling. This study investigates how these complementary features can be organized within a segmentation architecture and adapted to industrial defects with low foreground occupancy, weak texture and diverse shapes. Real-time segmentation is the intended application direction. The evaluation covers segmentation performance, model complexity and a unified forward inference benchmark on a shared GPU. End-to-end deployment throughput and production-line timing require further validation.

**Method** The method consists of two levels. DSMONet first organizes detail-semantic interaction using a ResNet-50 backbone. Deep semantic features undergo context aggregation and body-edge decomposition, and the resulting semantic information enters shallow-detail fusion. The refined detail features are subsequently fused with a semantic return path to form the final joint representation used by the main segmentation head. Mutual optimization refers to the joint training of these interacting paths under the segmentation objective; the refined details contribute to the final representation rather than directly updating a deep semantic tensor. A2MS-DSMONet then adapts this architecture to industrial defects. Its adaptive attention module combines input-dependent efficient squeeze-and-excitation responses with channel-wise weights learned during training. The recalibrated semantic features participate in both detail fusion and the semantic return path. Auxiliary detail supervision constrains local structural learning during training and is omitted from the inference path. Existing detail-supervision and mask-aware loss ideas are used as supervision components rather than claimed as new loss functions. For NEU-Seg, ten models are compared: two classical segmentation methods, five general real-time segmentation methods, one industrial-defect representative with publicly reproducible implementation under the study protocol and the two proposed configurations. The comparison uses 3,630 training images and 840 test images, four classes including background, training from scratch, seed 1337, Adam with a constant learning rate of 0.0001 and weight decay of 0.000002, batch size 16, and a fixed endpoint of 240,000 iterations. The test set is not used for checkpoint selection. Model-specific output heads and supervision losses are retained. This scheme improves comparability without representing each method's best original recipe. Literature-reported scores are discussed separately and are not pooled with the unified reimplementation results, because split, category, checkpoint-selection and metric definitions can differ. U-Net and LETNet internally pad the task input from 200 by 200 pixels to 208 by 208, while BiSeNetV2 pads it to 224 by 224; their outputs are cropped to the common evaluation size. A separate three-seed, 10,000-iteration ablation uses cosine learning-rate scheduling. The two proposed leather models use validation-monitored checkpoint selection and are evaluated on 468 common test images.

**Result** Under the NEU-Seg fixed-endpoint protocol, DSMONet-B and the complete A2MS-DSMONet-B configuration achieve 91.24% and 91.45% mIoU, respectively. Both proposed models score numerically higher than eight non-proposed comparison implementations in mIoU under this unified fixed-endpoint protocol. The 0.21-percentage-point difference between the two proposed models is an observed single-seed endpoint difference, not evidence of a statistically stable training gain. Among the classical implementations, DeepLabv3+ and U-Net achieve 90.95% and 90.62% mIoU. Their parameter counts are 59.34M and 31.04M, and their inference computation is 14.29G and 36.10G MACs, respectively. DSMONet-B has 29.37M parameters, while A2MS-DSMONet-B has 29.46M; both have 6.65G inference MACs at the reported precision. The general real-time comparison methods have lower parameter counts and computation but lower mIoU in this training scheme. FDSNet, the industrial-defect representative with publicly reproducible implementation, achieves 87.08% mIoU under our protocol. Parameter counts include training-specific heads, whereas MACs and timing use the specified inference paths with internal padding. BiSeNetV2 and FDSNet are timed after removal of auxiliary heads. Their main outputs agree exactly with the original configurations for a batch of two random inputs loaded with the same weights; deployment mIoU has not been recomputed on the full test set. In the specified shared A100 GPU forward inference benchmark (200 by 200 input, batch=1, FP32), DSMONet-B and A2MS-DSMONet-B achieve single-image equivalent FPS values of 95.65 and 95.41, respectively. These median-latency-based values reflect model forward computation efficiency only. In the short-training ablation, the complete configuration has the highest mean mIoU of 85.75% with a standard deviation of 0.54 percentage points, although the sign of its difference from DSMONet-B varies across seeds. Morphology analysis therefore focuses on observations of the complete configuration. For NEU-Seg patches, case-averaged boundary IoU differences at one-, two- and three-pixel tolerances are positive, at 0.55, 0.54 and 0.55 percentage points. Crazing has a positive pixel-aggregated region-IoU difference of 0.32 percentage points but a negative average per-case region-IoU difference of 0.71 percentage points, together with lower average boundary scores. The opposing regional summaries indicate that aggregate improvement does not imply improvement for a typical image. Paired bootstrap intervals describe test-case variation for fixed trained weights, not training-seed uncertainty. On the leather defect dataset, DSMONet-B and A2MS-DSMONet-B achieve 89.91% and 90.97% mIoU, respectively, providing a supplementary application evaluation on another industrial material. Six of seven defect classes have positive numerical differences, while the wart category decreases.

**Conclusion** The results support DSMONet as an independently evaluated detail-semantic architecture and show a numerical improvement for the complete industrial enhancement under the current fixed-endpoint protocol. The classical and general real-time implementations provide structural comparators; FDSNet, the industrial-defect representative with publicly reproducible implementation, provides a task-specific comparator. Other recent industrial methods remain literature context rather than quantitative comparisons. Lower MACs than the two classical implementations do not imply uniformly higher measured speed. The specified A100 forward inference benchmark provides an efficiency observation, while actual deployment depends on hardware, input load and the complete inspection pipeline. The defect-type findings also limit the enhancement claim: positive patch-region and boundary changes do not extend uniformly to complex slender cracks or every leather category. Further work should expand uniform reimplementation of industrial methods, assess long-training seed variability and validate target-device end-to-end performance.

## Keywords

industrial surface defects; real-time semantic segmentation; detail-semantic mutual optimization; adaptive channel re-calibration; detail supervision; defect-type analysis

## 0 引言

工业制造过程中，钢材、皮革、芯片、磁瓦和金属零部件等材料表面容易出现裂纹、夹杂、斑块、破洞和弱纹理缺陷。这类缺陷会影响产品外观质量、结构可靠性和后续加工稳定性，因此在生产线上实现快速、稳定、精细的缺陷检测具有重要工程意义。传统人工检测依赖经验，检测效率和一致性受主观因素影响较大；依赖灰度、纹理、边缘或频域特征的传统机器视觉方法，在光照变化、背景纹理复杂和缺陷形态多变时也面临适应性挑战。随着深度学习的发展，语义分割方法能够输出像素级缺陷区域，为缺陷定位、面积统计和质量判定提供更精细的信息。

将通用语义分割方法直接应用于工业表面缺陷实时检测仍面临多方面挑战。工业缺陷具有复杂纹理结构和多尺度分布特征，缺陷区域在整幅图像中占比低、形态差异大，且常与材料背景纹理交织。浅层特征虽然包含边缘、纹理和位置信息，却容易混入背景噪声；深层语义特征具有更强判别能力，但连续下采样会削弱局部结构与位置表达。因此，如何在语义分割模型中兼顾高层语义理解与局部细节保持，同时满足在线检测的实时性约束，仍是工业表面缺陷分割面临的重要问题。较大规模的分割网络通常需要更多参数与计算；轻量设计旨在控制这些开销，其实际速度和局部结构保持能力仍依赖具体实现与部署环境。分割精度、推理速度与局部结构表达能力之间的平衡，是工业表面缺陷实时分割中的关键问题。

近年来，工业缺陷分割研究逐渐转向任务特异的轻量化、注意力增强和细节建模。Zhang J 等（2022）提出 FDSNet，以轻量特征提取和辅助监督处理表面缺陷；Zhu 等（2023）提出 Sub-region UNet，通过空间子区域特征提取与掩码感知损失分割弱缺陷。Zuo 等（2025）的 DMC-Net 面向实时表面缺陷分割，采用轻量特征提取与多尺度增强；Jeong 等（2025）的 TAG-Net 显式建模背景、缺陷和边界三重注意力；Li Q C 等（2025）的 CDARNet 采用跨维度自适应区域重构；Zhang G W 等（2025）的 LGGFormer 结合局部引导与全局自注意力。Chen B 等（2024）的 A-Net 从轻量结构组织角度传递细节与语义信息；Lu 等（2026）的 DSGNet 结合深度可分离与 Ghost 卷积控制复杂度。上述研究表明，工业纹理判别与局部结构保持可以通过不同路径组织实现，需结合具体评价协议检验各自的性能与开销。

经典语义分割为区域判别与结构恢复提供了两类参照。Ronneberger 等（2015）的 U-Net 通过编码器—解码器与跨层连接恢复空间信息；Chen 等（2018）的 DeepLabv3+ 结合空洞卷积、多尺度上下文与解码细化。本文将这两类设计作为精度与复杂度的参照，而不预设其在工业任务中的优劣。

实时语义分割研究围绕分支分工、上下文聚合与高效融合展开。Yu 等（2018）的 BiSeNet 区分空间路径和上下文路径；Yu 等（2021）的 BiSeNetV2 通过引导聚合连接细节与语义分支。Fan 等（2021）的 STDCNet 采用短期密集连接和细节聚合，其细节监督思想也为本文提供参考；Pan 等（2023）的 DDRNet 通过双分辨率结构交换空间与上下文信息；Xu J C 等（2023）的 PIDNet 引入三分支设计与边界注意力。Xu G A 等（2023）的 LETNet 结合轻量 CNN 与 Transformer，属于通用实时分割设计；Wan 等（2025）的 SeaFormer++ 以挤压增强轴向注意力处理移动端视觉任务。Peng 等（2022）的 PP-LiteSeg 所用 UAFM 为本文特征融合提供参考。已有方法已包含多种跨分支交互，本文关注的是“深层语义进入浅层细节融合—所得细节再进入最终联合表征”的具体路径组织，并以工业缺陷的类别与边界表现检验这一组织方式。

针对上述问题，本文构建细节—语义相互优化网络 DSMONet，按“语义约束细节—优化细节参与最终融合”的顺序组织深浅层信息流，形成细节—语义联合表征。该架构未设置针对某一缺陷类别的专用分支，并作为本文第一层方法独立评价。在此基础上，本文针对工业表面缺陷低占比、弱纹理、尺度变化和类别形态差异明显等特点，引入自适应语义增强模块与辅助细节监督机制，形成面向工业表面缺陷的 A2MS-DSMONet。本文在 NEU-Seg 公开带钢表面缺陷数据集上开展主结果比较、消融实验、边界质量评价和缺陷尺度分层分析，并在真实工业生产场景采集的皮革表面缺陷数据集上考察跨材质应用适用性。

本文主要贡献如下：

1. 构建可独立进行像素级分割的细节—语义相互优化网络 DSMONet，按“语义约束细节—细节参与最终融合”的顺序组织深浅层信息流，在统一分割目标下形成联合表征。在当前固定终点评价协议下，采用 ResNet-50 主干的 DSMONet-B 在 NEU-Seg 数据集上取得 91.24% mIoU。
2. 面向工业表面缺陷低占比、弱纹理、多尺度和形态差异明显等特征，在 DSMONet 的交互路径中接入由输入相关的高效通道注意力与可学习逐通道权重组成的 AAM，并在浅层细节路径加入仅训练阶段的辅助监督，形成工业缺陷适配模型 A2MS-DSMONet。
3. 在 NEU-Seg 公开带钢缺陷数据集上，按共同训练预算与评价口径比较 10 个模型；在真实皮革数据集上考察另一工业材质下的应用适用性。结合类别、边界、尺度和形态复杂度分析，刻画完整增强配置的表现与局限，并报告精度、模型复杂度与指定硬件前向速度的取舍。

## 1 DSMONet 与面向工业缺陷的协同增强网络

### 1.1 总体框架

本文网络采用两级方法体系，总体结构如图 1 所示。第一级为细节—语义相互优化网络 DSMONet：以 ResNet-50（He 等，2016）为主干提取多层特征，先以深层语义信息调节浅层细节，再将细节增强特征与语义特征融合，旨在兼顾类别判别与局部结构保持。第二级为面向工业表面缺陷的增强：针对缺陷低占比、弱纹理、多尺度和形态差异明显等任务特性，在 DSMONet 的互优化路径中引入自适应语义增强模块（adaptive attention module，AAM）对参与交互的语义特征进行通道重标定，并在训练阶段引入辅助细节监督分支约束局部结构学习，构成 A2MS-DSMONet。采用 ResNet-50 主干的模型记为 DSMONet-B 与 A2MS-DSMONet-B。推理阶段网络仅保留主分割输出，辅助输出与细节监督分支不参与推理；AAM 位于推理主路径，其计算计入表 2 的推理 MACs。

图 1 A2MS-DSMONet 网络结构示意图。DSMONet 基础架构实现细节—语义相互优化，AAM 与辅助细节监督为面向工业缺陷的增强 / Fig. 1 Overall architecture of A2MS-DSMONet. DSMONet forms the detail-semantic mutual optimization backbone, while AAM and auxiliary detail supervision constitute the defect-oriented enhancements。

### 1.2 DSMONet 细节—语义相互优化网络

工业缺陷图像中的浅层特征和深层特征具有明显互补关系。浅层特征保留较高空间分辨率，能够提供缺陷边缘、纹理变化和位置分布等细节信息；但材料表面的自然纹理、加工痕迹和局部亮度变化容易使浅层特征同时响应缺陷区域和非缺陷背景。深层特征具有更强语义判别能力，但连续下采样会降低空间分辨率，削弱局部结构表达。直接拼接或相加能够融合两类特征，但未显式区分“语义调节浅层细节”与“细节参与最终语义融合”两个处理阶段。本文据此组织连续的跨层信息流。

DSMONet 的细节—语义互优化体现为两阶段的信息流组织。具体地，主干网络提取的多层特征中，最深层特征经多尺度上下文聚合模块（DAPPM，参考 Pan 等（2023））增强后，与自身一并送入注意力特征调制模块（UAFM，参考 Peng 等（2022），记为 arm1）整合为语义特征；该语义特征经主体—边缘分解单元 SqueezeBodyEdge（参考 Li X T 等（2020）的分解思想）分解为语义主体与语义边缘分量，并经通道重标定单元调节各通道响应。浅层特征经步进卷积降采样与拉普拉斯卷积单元增强局部结构响应后，与上采样的语义边缘分量和通道重标定语义特征共同进入细节融合单元，形成受语义调节的细节增强特征。随后，原始语义特征、语义主体分量与重标定语义特征经逐元素相加形成语义回流特征；UAFM（记为 arm2）将该特征与细节增强特征融合，得到用于主分割输出的细节—语义联合表征。本文所称互优化，是指语义信息进入细节融合路径、所得细节增强特征再进入最终联合表征，两类信息流在统一分割目标下共同优化；该架构的设计重点是上述交互路径的组织方式，而非某个单独算子。SqueezeBodyEdge 与拉普拉斯卷积作为已有单元使用，不将其本身表述为本文原创。

为便于描述细节—语义交互过程，本文采用概念化形式表示该两阶段交互过程：

\[
\mathbf{F}_{d}'=\mathcal{G}_{d}(\mathbf{F}_{d},\mathbf{F}_{s}),\qquad
\mathbf{F}_{s}'=\mathbf{F}_{s}\oplus\mathcal{B}(\mathbf{F}_{s})\oplus\mathcal{R}(\mathbf{F}_{s}),\qquad
\mathbf{F}_{\mathrm{out}}=\mathcal{H}(\mathbf{F}_{d}',\mathbf{F}_{s}'),
\tag{1}
\]

其中，\(\mathbf{F}_{d}\) 与 \(\mathbf{F}_{s}\) 分别表示浅层细节特征与深层语义特征；\(\mathcal{G}_{d}(\cdot)\) 表示语义对细节的调制，由语义边缘分量、通道重标定语义特征与浅层细节的细节融合实现；\(\mathcal{B}(\cdot)\) 与 \(\mathcal{R}(\cdot)\) 分别表示语义主体分解与通道重标定，二者与原始语义特征逐元素相加构成语义回流；\(\mathcal{H}(\cdot)\) 为基于 UAFM 的细节—语义融合，使优化后的细节信息参与最终联合表征。式（1）中的 \(\mathbf{F}_{s}'\) 表示语义分支形成的回流特征，细节信息在 \(\mathcal{H}(\mathbf{F}_{d}',\mathbf{F}_{s}')\) 中进入输出表征。两阶段交互共同服务于分割目标，不表示细节特征对深层语义张量的直接迭代更新。

### 1.3 自适应语义增强模块

工业缺陷常具有低占比、弱纹理和尺度变化明显等特点，不同语义通道对缺陷判别的重要性存在差异。为调节参与交互的语义特征，本文在细节—语义互优化路径的通道重标定位置引入自适应语义增强模块 AAM，替代常规通道注意力单元。AAM 由高效挤压激励单元（eSE）与自适应通道权重单元（ACW）组成：eSE 通过全局平均池化与 \(1\times1\) 卷积，根据输入特征生成逐通道响应；ACW 在训练中学习逐通道权重。二者联合对语义特征进行通道重标定，调节其在细节融合与语义回流路径中的贡献。AAM 嵌入互优化路径，其重标定结果同时参与这两个交互阶段。

设输入特征为 \(\mathbf{F}\)，AAM 的计算可表示为：

\[
\mathbf{a}_{\mathrm{eSE}}=h_{\mathrm{sig}}\!\left(\phi_{1\times1}(\operatorname{GAP}(\mathbf{F}))\right),
\qquad
\mathbf{F}_{\mathrm{AAM}}=\mathbf{F}\odot \mathbf{a}_{\mathrm{eSE}}\odot \mathbf{w}_{\mathrm{ACW}},
\tag{2}
\]

其中，\(\operatorname{GAP}(\cdot)\) 表示全局平均池化，\(\phi_{1\times1}\) 表示 \(1\times1\) 卷积，\(h_{\mathrm{sig}}(\cdot)\) 表示 HSigmoid 激活函数，\(\mathbf{w}_{\mathrm{ACW}}\) 为训练中学习得到的逐通道权重，\(\odot\) 表示逐元素乘法。输入相关的通道变化由 \(\mathbf{a}_{\mathrm{eSE}}\) 提供；AAM 主要由轻量级通道操作构成，仅引入少量额外计算。

### 1.4 面向局部结构学习的辅助细节监督

在主分割监督之外，A2MS-DSMONet 引入辅助细节监督分支，引导网络学习缺陷的局部结构信息。具体地，浅层细节特征经拉普拉斯卷积单元增强后接入一个独立的单通道细节输出头，对缺陷局部结构进行预测，并由细节监督损失约束；细节监督目标由标注掩码的局部结构信息提取生成。该分支仅在训练阶段参与，推理阶段不启用，因此该分支本身不增加推理开销。细节监督思想已有研究基础（如 Fan 等（2021）的 STDC 工作）；本文将该约束接入细节—语义互优化结构的浅层细节路径。前景—背景掩码感知监督沿用 Zhu 等（2023）的已有方法思想，并非本文新提出的损失函数。

### 1.5 联合优化与损失函数

A2MS-DSMONet 训练时输出主分割结果 \(\mathbf{o}_0\)、细节融合辅助分割结果 \(\mathbf{o}_1\)、语义辅助分割结果 \(\mathbf{o}_2\) 和细节监督结果 \(\mathbf{o}_3\)。前三路各计算一次多类别预测损失，并对其前景—背景二类化结果计算一次掩码感知损失；第四路仅计算辅助细节损失。实际执行的训练目标为：

\[
\mathcal{L}_{\mathrm{total}}=
\sum_{i=0}^{2}\omega_i\left[
\ell_i\bigl(\mathbf{o}_i,T_i(\mathbf{y})\bigr)
+\ell_i\bigl(P(\mathbf{o}_i),\mathbf{y}_{\mathrm{bin}}\bigr)
\right]
+\lambda_d\mathcal{L}_{\mathrm{detail}}(\mathbf{o}_3,\mathbf{y}).
\tag{3}
\]

其中，\(\mathbf{y}\) 为多类别标注，\(\mathbf{y}_{\mathrm{bin}}\) 为缺陷前景—背景二类标注；\(T_0(\mathbf{y})=T_2(\mathbf{y})=\mathbf{y}\)，\(T_1(\mathbf{y})\) 为标注的 one-hot 形式。\(P(\mathbf{o})\) 将背景通道输出与所有缺陷类别通道输出之和组成二通道预测。第 0 路（seg1）和第 2 路（seg3）的 \(\ell_i\) 为在线难例挖掘交叉熵（OHEM CE）；第 1 路（seg2）的 \(\ell_i\) 为 BCE-with-logits，其二类标注也在损失计算中转换为 one-hot 形式。第四路细节损失根据标注掩码生成边界目标，包含 BCE 与 Dice 约束。式（3）中的二类项沿用已有掩码感知监督思想，并非本文新提出的损失函数。NEU-Seg 的 \((\omega_0,\omega_1,\omega_2,\lambda_d)=(10,1,3,3)\)；皮革实验的细节分支权重为 \(\lambda_d=1\)，前三路权重保持为 \((10,1,3)\)。DSMONet-B 不含细节监督分支，仅保留式（3）的前三路多类别项与相应二类项。

## 2 实验与分析

### 2.1 数据集与实验设置

**NEU-Seg 公开数据集。** NEU-Seg 为带钢表面缺陷像素级语义分割数据集，本文实验包含裂纹（crazing）、夹杂物（inclusion）、斑块（patches）三类缺陷及背景类，共 4 类。训练集 3630 张，测试集 840 张，输入尺寸统一为 200×200。该数据集作为主要基准，用于主结果比较、消融实验、边界质量评价、缺陷尺度分层和类型差异分析。

**皮革表面缺陷数据集。** 该数据集来源于实际工业生产环境，包含开创伤、刺刮伤、烙印、破洞、皮肤藓、烂面、刺猴 7 类缺陷，共 2341 张 768×768 图像，划分为训练集 1638 张、验证集 235 张、测试集 468 张。该数据集用于考察模型面对不同工业材质、纹理背景和缺陷形态时的适用性，作为真实工业场景的应用评价。本文两模型的正式权重依据训练过程中的验证监控结果选取，并在统一的 468 张测试图像上进行评价；测试集不参与权重选择。皮革数据集另补充复现 U-Net、DeepLabv3+ 与 FDSNet，采用独立的固定终点方案，相关设置与结果见 2.6 节。两个数据集的图像与标注示例如图 2 所示。

图 2 数据集示例 / Fig. 2 Examples of the NEU-Seg and leather defect datasets。


**评价协议。** 测试阶段采用 TestRescale 与 ToTensor 变换，不使用 ImageNet 归一化；mIoU 计算包含背景类。两个数据集的评价流程有所不同：对 NEU-Seg，主结果采用固定 240k 次迭代的训练预算，并在训练终点对全部 840 张测试图像统一评价，训练中不使用验证集或测试集进行模型选择；对皮革数据集的主实验，沿用该数据集的既有训练协议，本文两模型的正式权重依据验证监控指标选取，在统一的 468 张独立测试图像上进行评价。NEU-Seg 十模型与皮革本文两模型所报告权重均不按测试结果选择；皮革三种历史对比方法在相同 468 张测试图像上评价，其完整训练与选权链未在此比较中统一。边界质量评价采用 1/2/3 像素容差的边界 IoU（Boundary IoU）与边界 F1 分数（BF-score），边界带由 3×3 形态学腐蚀—膨胀差分生成。表 3 的类别区域 IoU 按测试集像素聚合，边界指标对有效图像—类别对取宏平均；尺度分层与逐样本分析则按各组有效图像取宏平均。这些分析均基于同一批保存的预测结果计算，缺陷类别不存在于标注且预测也为空的样本不计入该类边界平均。

**训练设置。** NEU-Seg 十模型主比较统一采用从头训练、随机种子 1337、Adam 优化器、恒定学习率 1×10⁻⁴、权重衰减 2×10⁻⁶、batch size 16 和 240k 迭代固定终点；共享同一训练集、增强入口及最终评价口径，测试集不参与权重选择。各方法保留相应输出头与监督损失，因此共同训练方案不表示训练目标完全相同，也不代表各方法在原始最佳专用训练方案下的最高性能。本文两模型的训练环境为 NVIDIA A100-PCIE-40GB、PyTorch 2.7.1、CUDA 12.6、Python 3.11。表 8 的模块组合消融采用另一个 10k 协议：随机种子为 1337/2026/3407，优化器与批量设置相同，学习率采用余弦退火（\(T_{\max}=48k\)，最低学习率 1×10⁻⁶），四组配置仅在该短训协议内比较。皮革中本文两模型沿用既有训练链，采用 Adam、恒定学习率 1×10⁻⁴、batch size 12，依据每 1000 次迭代的验证监控 mIoU 选择保存权重；不同模型的保存迭代和训练链未完全匹配。

### 2.2 对比方法与公平比较原则

NEU-Seg 对比方法按结构代表性、工业任务相关性和可追溯实现选取，组成表 1 所示的四类体系。U-Net（Ronneberger 等，2015）与 DeepLabv3+（Chen 等，2018）分别提供跨层细节恢复和多尺度语义聚合参照；BiSeNetV2（Yu 等，2021）、STDC-Seg（Fan 等，2021）、PIDNet-S（Xu J C 等，2023）、DDRNet-23（Pan 等，2023）与 LETNet（Xu G A 等，2023）覆盖不同通用实时分割设计；选取具有公开可复现实现的 FDSNet（Zhang J 等，2022）作为本轮工业缺陷参照方法。DSMONet-B 与 A2MS-DSMONet-B 则对应本文两级方法。该选择用于观察不同信息组织方式的精度与效率代价，发表年份不是唯一筛选条件；“通用实时”分类依据原方法定位，不预先判定其在本文硬件上的实测速度。

表 1 对比方法分类及评价作用 / Table 1 Categories and evaluation roles of the comparison methods。

| 类别 | 模型 | 比较作用 | 表中结果来源 |
|---|---|---|---|
| 经典语义分割（Classical） | U-Net、DeepLabv3+ | 跨层细节恢复与多尺度语义聚合的结构参照 | 本文统一复现 |
| 通用实时语义分割（Real-time） | BiSeNetV2、STDC-Seg、PIDNet-S、DDRNet-23、LETNet | 不同分支组织与轻量特征融合设计的参照 | 本文统一复现 |
| 工业表面缺陷分割（Industrial） | FDSNet | 具有公开可复现实现、已完成同协议重训的工业缺陷参照 | 本文统一复现 |
| 本文方法（Ours） | DSMONet-B、A2MS-DSMONet-B | 分别评价细节—语义互优化架构与完整工业增强配置 | 本文固定终点评价 |

本文明确区分两类精度证据。文献报告值来自原论文的数据划分、类别集合、训练预算与权重选择方案，用于描述相关工作；统一复现值来自本文共同数据划分、训练预算和评价口径，用于表 2 的定量比较。即使文献采用相同数据集名称或相同样本数量，也需核对具体样本、评价类别、背景是否参与均值以及是否使用测试集选择权重。由于这些协议的同一性未获充分确认，文献报告的精度与 FPS 不直接并入本文统一结果表，也不据此计算本文相对文献的提升幅度。

共同训练方案提高了本轮实现间的可比性，各方法仍保留相应监督损失和必要输入适配，因此表 2 不是各原始方法在专用最优训练方案下的排名。U-Net 使用本地 VGG 式五级编解码实现，DeepLabv3+ 使用 ResNet-50、输出步长 16；其他实现和推理路径差异见表注。工业专用量化对比采用具有公开可复现实现、已完成本协议重训的 FDSNet 作为本轮任务参照，尚未统一复现的方法在相关工作中讨论，其文献报告值不用于补齐实验表。

### 2.3 NEU-Seg 总体性能比较

表 2 汇总十模型在相同数据划分、240k 训练预算和含背景四类 mIoU 口径下的结果。经典方法中，DeepLabv3+ 与 U-Net 分别取得 90.95% 和 90.62% mIoU，接近本文两模型。参数量方面，二者分别为 59.34M 与 31.04M，均高于 DSMONet-B 的 29.37M；计算量方面，二者分别为 14.29G 与 36.10G MACs，高于本文两模型的 6.65G。这表明细节—语义互优化架构在当前任务和实现条件下具有竞争力的区域分割精度，并具有低于这两种经典实现的 MACs；参数规模与实际速度的取舍还需结合表 9 分析。

通用实时方法中，BiSeNetV2、STDC-Seg、PIDNet-S、DDRNet-23 与 LETNet 的参数量和推理 MACs 均低于本文两模型，本次 mIoU 数值也较低。具有公开可复现实现的本轮工业缺陷参照 FDSNet 取得 87.08% mIoU，提供了较小参数规模和推理计算量下的任务参照。这些结果反映的是本次统一复现范围内的精度—复杂度关系，不外推至未复现的近期工业方法，也不由 MACs 推断跨硬件速度排序。

DSMONet-B 取得 91.24% mIoU，完整 A2MS-DSMONet-B 取得 91.45%，二者 mIoU 数值均高于表中 8 种非本文提出的统一复现对比实现。DSMONet 建立了可独立评价的细节—语义互优化分割能力，A2MS 在当前单种子固定终点评价下取得 +0.21 个百分点的数值改善。后续形态分析用于说明完整配置的类别与边界表现；这一差值的训练随机性稳定程度仍需多种子长训练检验。

表 2 不同语义分割方法在 NEU-Seg 数据集上的性能比较 / Table 2 Performance comparison of semantic segmentation methods on NEU-Seg。

| 模型 | 类型 | mIoU/% | 背景 IoU/% | 裂纹 IoU/% | 夹杂物 IoU/% | 斑块 IoU/% | Params/M | MACs/G |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| U-Net | 经典分割 | 90.62 | 98.53 | 83.06 | 93.15 | 87.73 | 31.04 | 36.10 |
| DeepLabv3+ | 经典分割 | 90.95 | 98.55 | 84.01 | 93.08 | 88.15 | 59.34 | 14.29 |
| BiSeNetV2 | 通用实时 | 88.75 | 98.27 | 80.72 | 93.00 | 83.00 | 5.20 | 2.36 |
| STDC-Seg | 通用实时 | 88.16 | 98.17 | 80.59 | 92.54 | 81.33 | 16.07 | 6.02 |
| PIDNet-S | 通用实时 | 88.82 | 98.28 | 81.44 | 92.84 | 82.70 | 7.72 | 0.97 |
| DDRNet-23 | 通用实时 | 89.24 | 98.33 | 82.03 | 93.44 | 83.17 | 20.30 | 2.90 |
| LETNet | 通用实时 | 86.36 | 97.59 | 74.77 | 89.62 | 83.47 | 0.95 | 1.21 |
| FDSNet | 工业缺陷 | 87.08 | 97.95 | 78.69 | 91.46 | 80.24 | 2.59 | 0.17 |
| DSMONet-B | 本文方法 | 91.24 | 98.62 | 84.15 | 93.49 | 88.71 | 29.37 | 6.65 |
| A2MS-DSMONet-B | 本文方法 | 91.45 | 98.64 | 84.47 | 93.55 | 89.16 | 29.46 | 6.65 |

注：① 所列精度均为本文统一复现或本文方法评价结果，不含原论文报告值；mIoU 包含背景，IoU 按同一 840 张测试图像的像素聚合计算。② 任务输入为 200×200；U-Net 与 LETNet 内部补齐至 208×208，BiSeNetV2 补齐至 224×224，输出统一裁回 200×200，补齐标注区域不参与损失；MACs 包含内部尺寸适配开销。③ Params 统计完整实现，包含训练专用头；MACs 对应推理主路径及实际执行的分支，STDC-Seg 当前实现仍执行的辅助分支计入。④ BiSeNetV2/FDSNet 的精度来自保留辅助头构型的主输出；BiSeNetV2 的 MACs 和表 9 测速采用 aux_mode='eval'，FDSNet 的 MACs 和测速采用 aux=False。同权重转换后，在一组两张随机输入上主输出逐元素一致，部署构型的完整测试集 mIoU 未另行重算。⑤ DDRNet 采用 DDRNet-23。LETNet 为经导入与通道兼容修正的公开实现，FDSNet 的边缘及图像级语义辅助目标由本地标注生成。

### 2.4 不同缺陷形态下的性能差异分析

整体 mIoU 只能反映平均分割水平，工业缺陷的类间形态差异要求更细粒度的分析。本节从类别 IoU、边界质量和缺陷尺度三个维度比较 A2MS-DSMONet-B 与 DSMONet-B。

**类别级与边界级分析。** 表 3 给出同一批 840 张测试图像上两模型的类别区域 IoU 与边界质量差异。其中区域 IoU 按各类别的测试集像素聚合计算，边界指标先对有效图像—类别对配对求差，再取样本宏平均；两类汇总口径不能解释为统一的“840 张逐样本平均”。

表 3 缺陷类别级性能差异（A2MS-DSMONet-B − DSMONet-B，240k 终点，配对比较）/ Table 3 Per-defect-type performance differences (A2MS-DSMONet-B minus DSMONet-B, 240k endpoint, paired comparison)。

| 类别 | Δ IoU/pt（像素聚合） | Δ Boundary IoU@2px/pt（样本宏平均） | Δ BF1@2px/pt（样本宏平均） |
|---|---:|---:|---:|
| 裂纹 crazing | +0.32 | −0.45 | −0.60 |
| 夹杂物 inclusion | +0.06 | +0.32 | +0.04 |
| 斑块 patches | +0.45 | +0.54 | +0.30 |

注：有效图像—类别对数分别为裂纹 368、夹杂物 361、斑块 353；标注与预测均为空的图像—类别对不计入边界宏平均。配对自助法置信区间仅反映固定 seed 1337 模型下测试样本的差异，不代表训练随机性。

在当前固定权重比较中，完整 A2MS-DSMONet-B 配置对 NEU-Seg 斑块类的聚合区域 IoU 和样本宏平均边界 IoU 均呈正向差异：1/2/3 像素容差下边界 IoU 分别高 0.55、0.54、0.55 个百分点（样本级配对自助法 95% 置信区间均不含零；按全部测试像素聚合计算则分别高 0.74、0.85、0.86 个百分点）。裂纹类的像素聚合 IoU 高 0.32 个百分点，但逐样本 IoU 差值的平均值为 −0.71 个百分点，样本宏平均边界指标也下降；两种 IoU 汇总口径的方向不同，不能将聚合结果解释为典型样本普遍改善。部分可视化改善病例可见大面积裂纹区域内部空洞减少，这可作为聚合结果的一种可能解释，但不能据此单独归因于 AAM 或细节监督。夹杂物类多数指标的数值差异较小。

**尺度分层分析。** 按各缺陷类别标注面积占比的 P33/P67 分位数将测试样本划分为小、中、大三组，比较两模型在各组的 IoU 差异（表 4）。斑块类缺陷在小面积组增益最大（+0.73 个百分点），裂纹类在小尺度组出现一定回退。尺度效应具有类别依赖性，不存在统一的"小目标增益"规律。

表 4 缺陷尺度分层 IoU 差异（ΔIoU/pt，按类别内 GT 面积占比 P33/P67 分层）/ Table 4 Scale-stratified IoU differences (per-class P33/P67 tiers)。

| 类别 | 小 | 中 | 大 |
|---|---:|---:|---:|
| 裂纹 crazing | −1.24 | −0.50 | −0.44 |
| 夹杂物 inclusion | +0.28 | −0.24 | −0.05 |
| 斑块 patches | +0.73 | −0.04 | +0.49 |

进一步以周长/面积平方根作为裂纹形态复杂度代理指标进行探索性分层，在当前固定权重下观察到高复杂度裂纹组 ΔIoU 为 −1.74 个百分点。该结果提示完整增强配置对复杂裂纹的分割仍有限制，不能分离出辅助细节监督的独立作用，也不能推广为所有复杂缺陷的规律。

**可视化分析。**

图 3 NEU-Seg 数据集可视化分割结果 / Fig. 3 Visualization results on NEU-Seg。

NEU-Seg 上的分割可视化结果如图 3 所示。为避免人工择优，展示病例按预先确定规则选择：以 A2MS-DSMONet-B 与 DSMONet-B 的逐样本 BF1@2px 差值排序，对裂纹与斑块两类分别选取改善最大、退化最大和差值近零的代表性病例。每行依次展示原图、标注、两模型预测结果及误差分布图（真正例、假正例、假负例分别以不同颜色标注）。

可视化结果与定量分析一致：在改善病例中，A2MS-DSMONet-B 能够填补大面积裂纹区域内部的误分类空洞，使预测区域更完整；在退化病例中，该模型在强边缘附近出现裂纹区域的过度扩张，并对个别纤细裂纹段的响应减弱。斑块类病例中，改善病例的缺陷轮廓与标注贴合更紧密。

### 2.5 真实皮革缺陷数据验证

除公开 NEU-Seg 数据集外，本文进一步在皮革缺陷数据集上考察模型在不同工业材质下的适用性。如 2.1 节所述，本文两模型的正式权重来自验证监控选择，保存迭代不同；在统一的 468 张独立测试图像上进行评价，测试集不参与所报告权重的选择。该比较不构成严格匹配训练链与随机种子的因果消融。定量结果如表 5 所示。

表 5 皮革数据集上不同方法的分割性能比较（468 张独立测试图像）/ Table 5 Segmentation performance of different methods on the leather defect dataset (468 independent test images)。

| 方法 | 来源 | mIoU/% | Params/M |
|---|---|---:|---:|
| DDRNet-23（内部复现） | 本文评价 | 84.21 | 20.30 |
| PIDNet-S（内部复现） | 本文评价 | 86.20 | 7.72 |
| STDC-Seg（STDCNet1446，内部复现） | 本文评价 | 88.24 | 16.08 |
| DSMONet-B | 本文 | 89.91 | 29.37 |
| A2MS-DSMONet-B | 本文 | **90.97** | 29.46 |

注：所有方法均在统一 468 张测试图像上评价，mIoU 包含背景。本文两模型的权重依据验证监控结果选取，保存迭代与训练链未完全匹配；三种对比方法使用既有本地实现的历史权重，其完整训练与选权链未构成本文统一重训协议。内部复现指项目既有本地实现，与原作者公开实现可能存在差异。皮革 DDRNet 权重由与 NEU-Seg 比较相同的模型入口加载，其宽度配置对应 DDRNet-23。

表 6 给出逐类别结果。在 7 类皮革缺陷中，完整 A2MS-DSMONet-B 配置相对 DSMONet-B 在 6 类上取得正向数值差异，其中刺刮伤、开创伤、烂面三类差异相对明显；刺猴类 IoU 下降约 3.7 个百分点，说明所报告的完整配置表现并非所有缺陷类别一致。由于两模型的训练链与随机性未完全匹配，这些差异只作为该材质上的应用观察，不单独归因于某个模块。

表 6 皮革数据集逐类别 IoU 对比（统一 468 张测试图像，%）/ Table 6 Per-class IoU comparison on the leather dataset (unified 468 test images, %)。

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

图 4 皮革真实缺陷验证案例 / Fig. 4 Test-set examples of real-world leather defects。为避免人工择优，展示案例按预先确定的规则选择：在 468 张测试图像中取标注含缺陷的 268 张，按两模型逐图 mIoU 差值（A2MS-DSMONet-B − DSMONet-B）排序，分别选取改善最大、退化最大与差值近零的案例各 2 例，每例包含原图、标注、DSMONet-B 预测与 A2MS-DSMONet-B 预测。在 268 张含缺陷图像中，A2MS-DSMONet-B 相对 DSMONet-B 有 211 张逐图 mIoU 数值较高、45 张较低、12 张近似持平；改善案例中 A2MS-DSMONet-B 对缺陷区域的覆盖更完整，退化案例显示漏检与类别误分等局部预测错误。该图用于展示真实工业场景下模型预测效果，不作为定量比较依据。

### 2.6 皮革缺陷数据集的补充评价

为进一步覆盖经典语义分割结构和具有公开可复现实现的工业缺陷参照方法，本文在皮革数据集上补充复现 U-Net、DeepLabv3+ 与 FDSNet。前两者作为经典分割参照，FDSNet 作为工业缺陷参照，不代表工业专用方法的整体水平。表 7 报告的是本文补充复现结果，而非原论文报告值。

三种补充模型均使用既定的 1638 张训练图像和 468 张测试图像，输入为原生 768×768，评价包含背景在内的 8 类。模型从头训练，采用随机种子 1337、Adam 优化器、初始学习率 1×10⁻⁴、权重衰减 2×10⁻⁶ 和 batch size 8；学习率按迭代采用余弦调度（\(T_{\max}=48000\)，最低学习率 1×10⁻⁶），训练至 60k 固定终点。训练过程中不使用验证集或测试集进行监控及模型选择，各模型保留相应监督目标。

表 7 皮革缺陷数据集上的补充基线评价（60k 固定终点）/ Table 7 Additional evaluation of reference implementations on the leather defect dataset (60k fixed endpoint)。

| 模型 | 类别 | mIoU/% |
|---|---|---:|
| U-Net | Classical | 82.42 |
| DeepLabv3+ | Classical | 87.60 |
| FDSNet | Industrial defect | 78.19 |

注：结果均为本文补充复现，在统一 468 张测试图像上评价，mIoU 包含背景。Classical 为经典语义分割，Industrial defect 为具有公开可复现实现的工业缺陷参照。与表 5 中本文两模型的既有训练链相比，本轮补充实验在训练预算、批量设置、学习率调度与权重选择方式上不同；表 5 中其他历史对比方法的完整训练与选权链也未在本轮统一。因此，两表不合并为统一训练协议下的性能排名，也不据此计算本文方法相对新增基线的提升幅度。本表不报告补充模型的参数量或 FPS。

补充实验进一步考察了不同类型分割方法在皮革缺陷场景中的表现差异，为原有应用评价补充经典结构与工业任务参照。该结果仅用于当前固定终点协议下的补充分析，不构成对本文增强策略的因果消融或跨材质泛化能力的检验。

### 2.7 消融实验

表 8 给出固定 10k 迭代预算下、3 个随机种子（1337/2026/3407）的模块组合消融结果（mean±SD）；240k 固定终点的 DSMONet-B 与完整模型结果已列于表 2。

表 8 模块组合消融（NEU-Seg，10k 迭代，3 种子 mean±SD，mIoU/%）/ Table 8 Module ablation (NEU-Seg, 10k iterations, 3 seeds, mean±SD, mIoU/%)。

| 配置 | AAM | 细节监督 | mIoU/% |
|---|:-:|:-:|---:|
| DSMONet-B | × | × | 85.22±0.41 |
| DSMONet-B + AAM | √ | × | 85.70±0.43 |
| DSMONet-B + 细节监督 | × | √ | 85.44±0.19 |
| A2MS-DSMONet-B | √ | √ | **85.75±0.54** |

在相同 10k 迭代预算的四组配置中，联合配置取得最高平均 mIoU。单独加入 AAM 的配置在 3 个种子中的 2 个上高于 DSMONet-B，平均差为 +0.48 个百分点；单独加入细节监督的配置在 3 个种子中均高于 DSMONet-B，平均差为 +0.23 个百分点；联合配置的差值方向存在种子间差异。240k 固定训练终点上，仅比较 DSMONet-B 与完整 A2MS-DSMONet-B，后者数值高 0.21 个百分点。10k 与 240k 不仅训练长度不同，学习率调度也不同；短训结果只支持当前协议内的模块组合比较，不用于推断长训单模块的独立贡献或跨种子一致性。

注：受算力约束，AAM-only 与细节监督-only 配置未开展 240k 长训练消融，长训练消融仅含 DSMONet-B 与完整模型两组；10k 消融用于模块组合筛查。

### 2.8 模型复杂度与推理效率

参数量与推理 MACs 见表 2，分别表征完整实现的参数规模与所指定推理构型的计算量。本文两模型的参数量为 29.37M 与 29.46M，增加约 0.31%；推理 MACs 均按两位小数报告为 6.65G，舍入后的相同数值不表示计算图完全相同。AAM 位于推理主路径，相关计算计入；辅助细节监督仅训练阶段启用，不计入推理 MACs。完整实现的参数量也不直接等于移除训练辅助头后的部署存储占用。

表 9 给出已完成的十模型统一前向计时结果。测试在共享 NVIDIA A100-PCIE-40GB 上进行，采用 batch=1、FP32、模型 eval 状态和无梯度计算，关闭 cuDNN benchmark。任务入口为一张固定 200×200 随机张量，包含各模型自身的 padding/crop 封装。每模型预热 100 次；正式计时分为 10 个区块，每区块按固定顺序对各模型分别连续前向 100 次，每次前后进行 CUDA 同步，共获得每模型 1000 次计时记录。FPS 由未舍入的单图延迟中位数换算，表示该前向路径的单图等效帧率，未计入图像读取、CPU 预处理、数据传输与业务后处理。

10 个测量区块均监控到外部 GPU 进程，最终统计采用每模型全部 1000 次记录。区块间及重复运行计时接近，数据经人工复核后采纳；该记录未达到零外部进程条件，重复性也不能证明不存在干扰。因此表 9 的速度结论限定于本次共享环境与输入设置，不代表生产线端到端吞吐。

表 9 指定 A100 环境下的单图前向计时结果 / Table 9 Single-image forward timing under the specified A100 conditions。

| 模型 | 中位延迟/ms | 单图等效 FPS |
|---|---:|---:|
| U-Net | 2.76 | 361.82 |
| DeepLabv3+ | 11.69 | 85.54 |
| BiSeNetV2 | 5.40 | 185.21 |
| STDC-Seg | 8.81 | 113.50 |
| PIDNet-S | 8.29 | 120.63 |
| DDRNet-23 | 6.16 | 162.29 |
| LETNet | 44.74 | 22.35 |
| FDSNet | 4.42 | 226.00 |
| DSMONet-B | 10.45 | 95.65 |
| A2MS-DSMONet-B | 10.48 | 95.41 |

注：中位延迟保留两位小数，FPS 使用未舍入中位数换算，不应由表内舍入延迟再次反算。BiSeNetV2/FDSNet 按去除训练辅助头的构型测量，STDC-Seg 仍执行的辅助分支按实际路径计时。固定随机输入的重复计时次数不表示测试图像数量。

DSMONet-B 与 A2MS-DSMONet-B 单图等效帧率分别为 95.65 和 95.41 FPS，本次条件下数值接近，不能据此提出增强配置的速度改善或独立速度差异主张。U-Net 虽具有较高 MACs，却取得较高实测帧率；LETNet 的 MACs 较低，帧率却较低，说明计算量与该环境的实测速度不呈统一的单调关系。本文结果支持将精度、完整参数规模、推理 MACs 与指定硬件前向速度分别评价；真实工业部署仍需结合目标设备、输入分辨率与检测节拍验证。


## 3 讨论

### 3.1 对比范围与基线选择依据

对比方法的选择服务于本文两级架构的评价问题：既需要考察细节恢复和语义聚合结构的区域分割表现，也需要观察通用实时与工业任务设计的计算及速度代价。仅按发表时间筛选方法，不能保证其数据、类别和训练评价协议与本文一致，也不能替代统一复现。表 1 所列经典、通用实时与本轮工业缺陷参照方法形成互补参照；近期研究用于分析相关设计进展和当前比较尚未覆盖的范围，本文不将不同协议的文献报告值与统一复现值排序。

保留经典模型有明确的结构和实验依据。U-Net 的跨层连接提供细节恢复参照，DeepLabv3+ 的多尺度上下文提供语义聚合参照，与 DSMONet 所关注的细节—语义组织问题直接相关。在当前协议下，两者的区域分割结果接近本文模型，不能将它们视为仅用于抬高相对增益的弱基线。其参数量、MACs 和实际速度表现又有所不同，使特征组织与开销之间的取舍能够被分别观察。

NEU-Seg 工业专用量化比较目前仅纳入具有公开可复现实现的 FDSNet，该方法已经形成本文共同协议下可追溯的重训与评价结果，作为本轮任务参照。近期方法的实现条件并不相同：本次核查的部分方法暂未获得可直接运行的完整公开实现，而 Chen B 等（2024）的 A-Net 已公开网络代码，Lu 等（2026）的 DSGNet 已公开训练与评价代码。这些可获得代码的方法尚未在本文协议下完成统一复现，不能仅以原论文结果补入表 2。工业专用代表数量有限是本研究比较范围的局限，不能由 FDSNet 一项推断本文超过近期工业分割方法的整体水平。

### 3.2 精度、复杂度与速度的取舍

DSMONet-B 与 A2MS-DSMONet-B 的固定终点结果分别评价了架构本身及完整工业增强配置。+0.21 个百分点的差异只说明当前权重下的数值变化；10k 多种子模块组合结果和 240k 两模型比较所用学习率调度不同，不提供长训单模块因果分解。

本文两模型的推理 MACs 低于所列经典实现，但参数量和计算量高于所列轻量方法。表 9 则显示，U-Net 在本次 A100 条件下更快，DeepLabv3+ 的帧率较低，不能把“经典方法计算量较高”概括为“经典方法推理较慢”。同样，LETNet 较低 MACs 与较低帧率并存，模型类型或计算量均不能替代指定硬件的实际计时。本文在该测量条件下给出约 95 FPS 的单图等效帧率和所报告的区域分割精度，这是一组有明确条件的取舍结果，尚不支持速度最优、整体效率支配或生产线节拍已满足的结论。

### 3.3 形态差异与皮革应用适用性

在此基础上，完整 A2MS-DSMONet-B 配置的观察结果呈现类型依赖性。在 NEU-Seg 斑块类中，聚合区域 IoU 与多容差样本宏平均边界 IoU 均为正向差异；这可以与较闭合的缺陷区域及其内部占比共同讨论，但现有证据不能把变化单独归因于 AAM 的语义重标定。裂纹类呈现聚合区域 IoU 为正、逐样本区域 IoU 平均值和边界指标为负的并存现象；部分可视化案例中的内部空洞填补提供一种可能解释，高复杂度分组结果仍属探索性观察，不能视为辅助细节监督对细长结构作用的单独检验。夹杂物类别多数指标变化较小。这些结果提示完整增强配置的表现随缺陷形态而异，不宜概括为所有类别的一致增益。

除公开 NEU-Seg 数据集外，本文进一步在皮革缺陷数据集上考察两级方法在另一工业材质下的适用性。基于验证监控选取的正式权重，在统一 468 张测试图像上，A2MS-DSMONet-B 取得 90.97% mIoU，数值高于 DSMONet-B 的 89.91%，差为 +1.06 个百分点。由于两模型的保存迭代不同，训练链和随机性未完全匹配，该差值是本次皮革应用比较的结果，不能独立证明增强结构的因果收益或跨材质泛化。逐类别结果中刺猴类别仍有回退，与 NEU-Seg 中观察到的形态相关差异共同提示需要按缺陷类别评估表现。

本文以 NEU-Seg 的共同训练与固定终点评价作为主要基准，Leather 则作为真实工业场景中另一材质的应用补充。表 7 新增的经典与工业缺陷参照扩展了皮革场景中的方法类型覆盖，但采用独立的 60k 固定终点协议，不与表 5 的主实验结果作跨表排名。各模型均在相应材质的训练集上训练后评价；这里的跨材质应用考察不是将钢材训练权重直接用于未见过的皮革材质，也不构成域泛化验证。

### 3.4 局限性

本文仍存在以下局限：（1）十模型主比较及 240k 增量结果基于单一随机种子，+0.21 个百分点在训练随机性下的稳定性未被长训多种子检验；（2）共同训练方案提高了比较的可比性，但监督损失和必要实现适配不同，结果不代表各原始方法最佳配方的最高性能；（3）工业专用量化参照仅 FDSNet，已有公开代码的其他近期方法尚未统一复现，领域覆盖仍有限；（4）皮革两模型采用验证监控选择权重、保存迭代不同，训练链和随机性未完全匹配，+1.06 个百分点不构成严格因果消融；新增三种皮革基线采用独立固定终点协议，不能与主实验结果合并排名；（5）配对自助法区间只刻画固定权重下的测试样本差异，复杂裂纹和皮革刺猴类仍有回退；（6）前向计时采用固定随机输入与共享 GPU，不能替代独占设备、真实输入负载与端到端生产线测试；（7）BiSeNetV2/FDSNet 的去辅助头检查仅覆盖一组随机输入，完整测试集的部署构型精度未另行评价；当前应用评价也仅覆盖钢材与皮革两种材质。

## 4 结论

本文构建细节—语义互优化架构 DSMONet，使深层语义约束浅层细节，并使所得细节信息参与最终联合表征；进一步结合 AAM 与训练阶段辅助细节监督，形成工业任务适配模型 A2MS-DSMONet。在十模型共同训练与固定终点评价中，DSMONet-B 和 A2MS-DSMONet-B 分别取得 91.24% 和 91.45% mIoU。前向推理基准中，两模型在指定共享 A100、200×200、batch=1、FP32 条件下的单图等效帧率分别为 95.65 和 95.41 FPS，呈现当前协议下的精度与开销取舍。完整增强配置的效果具有形态依赖性：斑块类区域与边界指标呈正向变化，复杂裂纹则存在聚合区域与样本平均边界结果之间的权衡。皮革统一 468 张测试图像上的 89.91% 和 90.97% mIoU 补充了另一材质的应用观察，训练链未完全匹配的限制仍保留。本文结论限于所测实现与协议，不将跨论文报告值纳入排序，也不由单图前向等效帧率推断生产线吞吐；后续应扩大工业专用方法的统一复现范围，检验长训随机性，并完成目标设备上的端到端部署评价。

## 参考文献

Chen B, Niu T Z, Yu W Y, Zhang R Q, Wang Z R and Li B. 2024. A-Net: an A-shape lightweight neural network for real-time surface defect segmentation. IEEE Transactions on Instrumentation and Measurement, 73: 5001314 [DOI: 10.1109/TIM.2023.3341115].

Chen L C, Zhu Y K, Papandreou G, Schroff F and Adam H. 2018. Encoder-decoder with atrous separable convolution for semantic image segmentation//Computer Vision—ECCV 2018. Cham: Springer: 833-851 [DOI: 10.1007/978-3-030-01234-2_49].

Fan M Y, Lai S Q, Huang J S, Wei X M, Chai Z H, Luo J F, et al. 2021. Rethinking BiSeNet for real-time semantic segmentation//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE: 9716-9725 [DOI: 10.1109/CVPR46437.2021.00959].

He K M, Zhang X Y, Ren S Q and Sun J. 2016. Deep residual learning for image recognition//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. IEEE: 770-778 [DOI: 10.1109/CVPR.2016.90].

Jeong S, Song J and Lee S J. 2025. TAG-Net: triple attention guided network for inspecting surface defects on steel products. International Journal of Control, Automation and Systems, 23: 441-448 [DOI: 10.1007/s12555-024-0550-8].

Li Q C, Ding C C, Wang B X, Jiao J Y, Huang W G and Zhu Z K. 2025. CDARNet: a robust cross-dimensional adaptive region reconstruction network for real-time metal surface defect segmentation. Advanced Engineering Informatics, 67: 103514 [DOI: 10.1016/j.aei.2025.103514].

Li X T, Li X, Zhang L, Cheng G L, Shi J P, Lin Z C, et al. 2020. Improving semantic segmentation via decoupled body and edge supervision//Computer Vision—ECCV 2020. Cham: Springer: 435-452 [DOI: 10.1007/978-3-030-58520-4_26].

Lu H, Zhao Y Y, Yang G and Wu S L. 2026. DSGNet: a lightweight network integrating depthwise separable and Ghost convolutions for real-time surface defect segmentation. Expert Systems, 43(2): e70187 [DOI: 10.1111/exsy.70187].

Pan H H, Hong Y D, Sun W C and Jia Y S. 2023. Deep dual-resolution networks for real-time and accurate semantic segmentation of traffic scenes. IEEE Transactions on Intelligent Transportation Systems, 24(3): 3448-3460 [DOI: 10.1109/TITS.2022.3228042].

Peng J C, Liu Y, Tang S Y, Hao Y Y, Chu L T, Chen G W, et al. 2022. PP-LiteSeg: a superior real-time semantic segmentation model[EB/OL]. [2026-10-10]. https://arxiv.org/abs/2204.02681 [DOI: 10.48550/arXiv.2204.02681].

Ronneberger O, Fischer P and Brox T. 2015. U-Net: convolutional networks for biomedical image segmentation//Medical Image Computing and Computer-Assisted Intervention—MICCAI 2015. Cham: Springer: 234-241 [DOI: 10.1007/978-3-319-24574-4_28].

Wan Q, Huang Z L, Lu J C, Yu G and Zhang L. 2025. SeaFormer++: squeeze-enhanced axial transformer for mobile visual recognition. International Journal of Computer Vision, 133(6): 3645-3666 [DOI: 10.1007/s11263-025-02345-2].

Xu G A, Li J C, Gao G W, Lu H M, Yang J and Yue D. 2023. Lightweight real-time semantic segmentation network with efficient transformer and CNN. IEEE Transactions on Intelligent Transportation Systems, 24(12): 15897-15906 [DOI: 10.1109/TITS.2023.3248089].

Xu J C, Xiong Z X and Bhattacharyya S P. 2023. PIDNet: a real-time semantic segmentation network inspired by PID controllers//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE: 19529-19539 [DOI: 10.1109/CVPR52729.2023.01871].

Yu C Q, Wang J B, Peng C, Gao C X, Yu G and Sang N. 2018. BiSeNet: bilateral segmentation network for real-time semantic segmentation//Computer Vision—ECCV 2018. Cham: Springer: 334-349 [DOI: 10.1007/978-3-030-01261-8_20].

Yu C Q, Gao C X, Wang J B, Yu G, Shen C H and Sang N. 2021. BiSeNet V2: bilateral network with guided aggregation for real-time semantic segmentation. International Journal of Computer Vision, 129(11): 3051-3068 [DOI: 10.1007/s11263-021-01515-2].

Zhang G W, Lu Y, Jiang X H, Jin S H, Li S P and Xu M L. 2025. LGGFormer: a dual-branch local-guided global self-attention network for surface defect segmentation. Advanced Engineering Informatics, 64: 103099 [DOI: 10.1016/j.aei.2024.103099].

Zhang J, Ding R W, Ban M J and Guo T Y. 2022. FDSNeT: an accurate real-time surface defect segmentation network//Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing. IEEE: 3803-3807 [DOI: 10.1109/ICASSP43922.2022.9747311].

Zhu W B, Liang R, Yang J X, Cao Y L, Fu G Z and Cao Y P. 2023. A sub-region Unet for weak defects segmentation with global information and mask-aware loss. Engineering Applications of Artificial Intelligence, 122: 106011 [DOI: 10.1016/j.engappai.2023.106011].

Zuo H Q, Zheng Y B, Huang Q Z, Du Z H and Wang H. 2025. DMC-Net: a lightweight network for real-time surface defect segmentation. Journal of Real-Time Image Processing, 22: 61 [DOI: 10.1007/s11554-025-01639-5].
