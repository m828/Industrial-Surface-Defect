# 细节—语义互优化与自适应增强的工业表面缺陷实时语义分割网络

# Real-time Semantic Segmentation Network for Industrial Surface Defects via Detail-Semantic Mutual Optimization and Adaptive Enhancement

作者：待补充

单位：待补充

中图分类号：待核定（候选见投稿元数据待办文档）

文献标志码：A

收稿日期：待补充

基金项目：待补充

## 摘要

**目的** 工业表面缺陷具有尺度变化大、边界模糊、类别形态差异明显等特点，在线检测场景还要求模型具备实时推理能力。如何在语义分割中兼顾高层语义理解与局部细节保持，并在分割精度与推理速度之间取得平衡，是工业表面缺陷检测面临的重要问题。

**方法** 本文构建细节—语义相互优化网络 DSMONet：深层语义特征参与浅层细节融合，调节细节路径中的背景纹理响应；经语义约束的细节特征再与语义回流特征融合，形成兼顾类别判别与空间结构的联合表征。在此基础上，针对工业表面缺陷低占比、弱纹理、多尺度和形态差异明显等特点，引入由高效通道注意力与可学习逐通道权重组成的自适应语义增强模块，并以仅用于训练的辅助细节监督约束浅层结构学习，形成 A2MS-DSMONet。

**结果** 在 NEU-Seg 带钢表面缺陷公开数据集上，采用 ResNet-50 主干的 DSMONet-B 在固定训练终点评价下取得 91.24% mIoU；同协议下完整增强配置 A2MS-DSMONet-B 取得 91.45% mIoU，在指定 A100 环境下测得 37.37 FPS，参数量增加 0.31%。缺陷形态分析表明，完整配置在斑块类缺陷上观察到区域与边界质量改善，而裂纹类缺陷呈现聚合区域结果与样本级边界结果之间的性能权衡。在真实皮革缺陷数据集上，依据验证监控选取、保存迭代不同的两模型权重在统一 468 张测试图像上分别取得 89.91% 和 90.97% mIoU；A2MS-DSMONet-B 的结果也高于本文在该数据集复现的 STDC、PIDNet 和 DDRNet 方法。

**结论** 在当前固定终点评价与皮革应用评价条件下，DSMONet 提供了有效的细节—语义交互分割架构；完整增强配置以较小附加计算开销取得数值较高的分割结果，并在指定 A100 环境下保持所测推理速度，其效果随缺陷形态而变化。

## 关键词

工业表面缺陷；实时语义分割；细节—语义互优化；自适应通道重标定；细节监督；缺陷类型分析

## Abstract

**Objective** Industrial surface defects are characterized by large scale variation, blurred boundaries, and pronounced morphological differences across defect categories, while online inspection additionally requires real-time inference. How to balance high-level semantic understanding with local detail preservation, and how to reconcile segmentation accuracy with inference speed, remain key problems for semantic segmentation in industrial surface inspection.

**Method** We construct DSMONet to organize detail-semantic interaction in two stages. Deep semantic features guide shallow-detail fusion, and the resulting detail features are subsequently fused with a semantic return path to form the final joint segmentation representation. Building on DSMONet, we develop A2MS-DSMONet for low-occupancy, weakly textured, multi-scale and morphologically diverse industrial defects. An adaptive attention module (AAM) combines input-dependent efficient squeeze-and-excitation responses with learned channel-wise weights in the semantic interaction path. Auxiliary detail supervision constrains shallow structural learning during training only; this branch adds no inference-time computation.

**Result** On the public NEU-Seg strip-steel surface defect dataset, with 840 test images evaluated under a unified fixed-endpoint protocol, DSMONet-B with a ResNet-50 backbone achieves 91.24% mIoU (including background), and the complete A2MS-DSMONet-B configuration reaches 91.45% mIoU at 37.37 FPS on an NVIDIA A100 (batch size 1, FP32), with 0.31% more parameters. Patch-type defects show positive regional and boundary changes (case-averaged boundary IoU gains of about 0.5 percentage points at 1/2/3-pixel tolerances), whereas crazing shows a trade-off between pixel-aggregated regional IoU and case-averaged boundary quality; inclusion remains approximately unchanged. On a real-world leather defect dataset with seven defect categories, the validation-monitored DSMONet-B and A2MS-DSMONet-B checkpoints, saved at different iterations, achieve 89.91% and 90.97% mIoU, respectively, on the same 468-image test set. The latter also exceeds the STDC, PIDNet and DDRNet models re-implemented and evaluated on this leather test set (88.24%, 86.20% and 84.21% mIoU, respectively).

**Conclusion** Under the evaluated protocols, DSMONet provides a detail-semantic interaction architecture with independently measured segmentation performance. The complete defect-oriented configuration has numerically higher mIoU in the reported comparisons, with small additional computational cost at the measured inference speed. The observed patch-type improvement does not extend uniformly to slender, complex cracks.

## Keywords

industrial surface defects; real-time semantic segmentation; detail-semantic mutual optimization; adaptive channel re-calibration; detail supervision; defect-type analysis

## 0 引言

工业制造过程中，钢材、皮革、芯片、磁瓦和金属零部件等材料表面容易出现裂纹、夹杂、斑块、破洞和弱纹理缺陷。这类缺陷会影响产品外观质量、结构可靠性和后续加工稳定性，因此在生产线上实现快速、稳定、精细的缺陷检测具有重要工程意义。传统人工检测依赖经验，检测效率和一致性受主观因素影响较大；依赖灰度、纹理、边缘或频域特征的传统机器视觉方法，在光照变化、背景纹理复杂和缺陷形态多变时也面临适应性挑战。随着深度学习的发展，语义分割方法能够输出像素级缺陷区域，为缺陷定位、面积统计和质量判定提供更精细的信息。

将通用语义分割方法直接应用于工业表面缺陷实时检测仍面临多方面挑战。工业缺陷具有复杂纹理结构和多尺度分布特征，缺陷区域在整幅图像中占比低、形态差异大，且常与材料背景纹理交织。浅层特征虽然包含边缘、纹理和位置信息，却容易混入背景噪声；深层语义特征具有更强判别能力，但连续下采样会削弱局部结构与位置表达。因此，如何在语义分割模型中兼顾高层语义理解与局部细节保持，同时满足在线检测的实时性约束，仍是工业表面缺陷分割面临的重要问题。大型分割网络虽然表达能力较强，但推理开销较大；轻量网络速度较快，却容易损失局部结构信息。分割精度、推理速度与局部结构表达能力之间的平衡，是工业表面缺陷实时分割中的关键问题。

近年来，工业缺陷分割研究逐渐从通用编码器—解码器结构转向任务特异的轻量化、注意力增强和细节建模。Zhu 等（2023）提出 Sub-region UNet，通过空间子区域特征提取与掩码感知损失分割金属表面弱缺陷；Zuo 等（2025）提出 DMC-Net，面向实时表面缺陷分割设计轻量特征提取与多尺度增强模块；Jeong 等（2025）提出 TAG-Net，针对钢材表面缺陷显式建模背景、缺陷和边界三重注意力；Li 等（2025）提出 CDARNet，面向实时金属表面缺陷分割，通过跨维度自适应区域重构增强鲁棒性；Zhang 等（2025）提出 LGGFormer，以双分支局部引导全局自注意力网络兼顾局部细节与全局建模。上述工作表明，工业缺陷分割不仅需要全局语义判别能力，还需要在复杂背景纹理下保持缺陷的局部结构。

实时语义分割研究主要围绕轻量骨干、高低分辨率分支、快速上下文聚合和高效特征融合展开。Yu 等（2018）提出 BiSeNet，以空间细节路径和语义上下文路径的双分支建模为代表；Fan 等（2021）提出 STDCNet，通过短期密集连接和细节聚合提升实时分割效率，其细节监督思想也为本文的细节约束设计提供参考；Hong 等（2023）提出 DDRNet，通过双分辨率结构保留空间细节并聚合上下文；Peng 等（2022）提出 PP-LiteSeg，其统一注意力融合模块（UAFM）为本文细节—语义交互模块提供参考；Xu 等（2023）提出 PIDNet，借鉴 PID 控制器设计三分支结构并引入边界注意力；Wan 等（2025）提出 SeaFormer++，以挤压增强轴向注意力改进移动端速度—精度平衡。上述方法采用不同的细节保持与语义融合路径，为实时分割提供了有效设计。本文关注的结构组织方式是先将深层语义用于浅层细节融合，再使所得细节特征参与最终语义—细节融合；工业缺陷的低占比前景、复杂纹理背景和类间形态差异还要求对这一信息流的类别与边界表现进行评价。

针对上述问题，本文构建细节—语义相互优化网络 DSMONet，按“语义约束细节—优化细节参与最终融合”的顺序组织深浅层信息流，形成细节—语义联合表征。该架构未设置针对某一缺陷类别的专用分支，并作为本文第一层方法独立评价。在此基础上，本文针对工业表面缺陷低占比、弱纹理、尺度变化和类别形态差异明显等特点，引入自适应语义增强模块与辅助细节监督机制，形成面向工业表面缺陷的 A2MS-DSMONet。本文在 NEU-Seg 公开带钢表面缺陷数据集上开展主结果比较、消融实验、边界质量评价和缺陷尺度分层分析，并在真实工业生产场景采集的皮革表面缺陷数据集上考察跨材质应用适用性。

本文主要贡献如下：

1. 构建可独立进行像素级分割的细节—语义相互优化网络 DSMONet，按“语义约束细节—细节参与最终融合”的顺序组织深浅层信息流，在统一分割目标下形成联合表征。在当前固定终点评价协议下，采用 ResNet-50 主干的 DSMONet-B 在 NEU-Seg 数据集上取得 91.24% mIoU。
2. 面向工业表面缺陷低占比、弱纹理、多尺度和形态差异明显等特征，在 DSMONet 的交互路径中接入由输入相关的高效通道注意力与可学习逐通道权重组成的 AAM，并在浅层细节路径加入仅训练阶段的辅助监督，形成工业缺陷适配模型 A2MS-DSMONet。
3. 在 NEU-Seg 公开带钢缺陷数据集和真实皮革缺陷数据集上评价两级方法，并从类别、边界、尺度和形态复杂度等维度分析完整增强配置的表现与局限。A2MS-DSMONet-B 分别取得 91.45% 和 90.97% 的 mIoU；在指定 A100 环境下测得推理速度，细粒度结果显示模型差异具有缺陷形态依赖性。

## 1 DSMONet 与面向工业缺陷的协同增强网络

### 1.1 总体框架

本文网络采用两级方法体系，总体结构如图 1 所示。第一级为细节—语义相互优化网络 DSMONet：以 ResNet-50 为主干提取多层特征，先以深层语义信息调节浅层细节，再将细节增强特征与语义特征融合，旨在兼顾类别判别与局部结构保持。第二级为面向工业表面缺陷的增强：针对缺陷低占比、弱纹理、多尺度和形态差异明显等任务特性，在 DSMONet 的互优化路径中引入自适应语义增强模块（adaptive attention module，AAM）对参与交互的语义特征进行通道重标定，并在训练阶段引入辅助细节监督分支约束局部结构学习，构成 A2MS-DSMONet。采用 ResNet-50 主干的模型记为 DSMONet-B 与 A2MS-DSMONet-B。推理阶段网络仅保留主分割输出，辅助输出与细节监督分支不参与推理；AAM 的实际推理开销见 2.6 节。

图 1 A2MS-DSMONet 网络结构示意图。DSMONet 基础架构实现细节—语义相互优化，AAM 与辅助细节监督为面向工业缺陷的增强 / Fig. 1 Overall architecture of A2MS-DSMONet. DSMONet forms the detail-semantic mutual optimization backbone, while AAM and auxiliary detail supervision constitute the defect-oriented enhancements。

### 1.2 DSMONet 细节—语义相互优化网络

工业缺陷图像中的浅层特征和深层特征具有明显互补关系。浅层特征保留较高空间分辨率，能够提供缺陷边缘、纹理变化和位置分布等细节信息；但材料表面的自然纹理、加工痕迹和局部亮度变化容易使浅层特征同时响应缺陷区域和非缺陷背景。深层特征具有更强语义判别能力，但连续下采样会降低空间分辨率，削弱局部结构表达。直接拼接或相加能够融合两类特征，但未显式区分“语义调节浅层细节”与“细节参与最终语义融合”两个处理阶段。本文据此组织连续的跨层信息流。

DSMONet 的细节—语义互优化体现为两阶段的信息流组织。具体地，主干网络提取的多层特征中，最深层特征经多尺度上下文聚合模块（DAPPM，参考 DDRNet）增强后，与自身一并送入注意力特征调制模块（UAFM，参考 PP-LiteSeg，记为 arm1）整合为语义特征；该语义特征经 SqueezeBodyEdge 单元分解为语义主体与语义边缘分量，并经通道重标定单元调节各通道响应。浅层特征经步进卷积降采样与拉普拉斯卷积单元增强局部结构响应后，与上采样的语义边缘分量和通道重标定语义特征共同进入细节融合单元，形成受语义调节的细节增强特征。随后，原始语义特征、语义主体分量与重标定语义特征经逐元素相加形成语义回流特征；UAFM（记为 arm2）将该特征与细节增强特征融合，得到用于主分割输出的细节—语义联合表征。本文所称互优化，是指语义信息进入细节融合路径、所得细节增强特征再进入最终联合表征，两类信息流在统一分割目标下共同优化；该架构的设计重点是上述交互路径的组织方式，而非某个单独算子。SqueezeBodyEdge 与拉普拉斯卷积作为已有单元使用，不将其本身表述为本文原创。

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

在主分割监督之外，A2MS-DSMONet 引入辅助细节监督分支，引导网络学习缺陷的局部结构信息。具体地，浅层细节特征经拉普拉斯卷积单元增强后接入一个独立的单通道细节输出头，对缺陷局部结构进行预测，并由细节监督损失约束；细节监督目标由标注掩码的局部结构信息提取生成。该分支仅在训练阶段参与，推理阶段不启用，因此该分支本身不增加推理开销。细节监督思想已有研究基础（如 STDC 系列工作）；本文将该约束接入细节—语义互优化结构的浅层细节路径。前景—背景掩码感知监督沿用 Zhu 等采用的已有方法思想，并非本文新提出的损失函数。

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

**皮革表面缺陷数据集。** 该数据集来源于实际工业生产环境，包含开创伤、刺刮伤、烙印、破洞、皮肤藓、烂面、刺猴 7 类缺陷，共 2341 张 768×768 图像，划分为训练集 1638 张、验证集 235 张、测试集 468 张。该数据集用于考察模型面对不同工业材质、纹理背景和缺陷形态时的适用性，作为真实工业场景的应用评价。正式报告的模型权重依据训练过程中的验证监控结果选取，并在统一的 468 张测试图像上进行评价；测试集不参与权重选择。两个数据集的图像与标注示例如图 2 所示。

图 2 数据集示例 / Fig. 2 Examples of the NEU-Seg and leather defect datasets。

**评价协议。** 测试阶段采用 TestRescale 与 ToTensor 变换，不使用 ImageNet 归一化；mIoU 计算包含背景类。两个数据集的评价流程有所不同：对 NEU-Seg，主结果采用固定 240k 次迭代的训练预算，并在训练终点对全部 840 张测试图像统一评价，训练中不使用验证集或测试集进行模型选择；对皮革数据集，沿用该数据集的既有训练协议，正式模型权重依据验证监控指标选取，在统一的 468 张独立测试图像上进行评价。两种协议下报告的权重均不按测试结果选择。边界质量评价采用 1/2/3 像素容差的边界 IoU（Boundary IoU）与边界 F1 分数（BF-score），边界带由 3×3 形态学腐蚀—膨胀差分生成。表 2 的类别区域 IoU 按测试集像素聚合，边界指标对有效图像—类别对取宏平均；尺度分层与逐样本分析则按各组有效图像取宏平均。这些分析均基于同一批保存的预测结果计算，缺陷类别不存在于标注且预测也为空的样本不计入该类边界平均。

**训练设置。** NEU-Seg 实验使用 NVIDIA A100-PCIE-40GB GPU，PyTorch 2.7.1，CUDA 12.6，Python 3.11；两类协议均采用 Adam 优化器、初始学习率 1×10⁻⁴、权重衰减 2×10⁻⁶、batch size 16，并从头训练。主结果比较为 240k 固定终点协议，随机种子为 1337，学习率保持恒定，用于评价 DSMONet-B 与 A2MS-DSMONet-B 的最终性能。表 6 的模块组合消融为 10k 多种子协议，采用 1337/2026/3407 三个种子与余弦退火学习率（\(T_{\max}=48k\)，最低学习率 1×10⁻⁶）；四组配置仅在这一 10k 协议内比较。DSMONet-B 为细节—语义互优化基础架构的 ResNet-50 主干模型；A2MS-DSMONet-B 为在 DSMONet-B 上引入 AAM 与辅助细节监督分支的完整模型。皮革实验沿用该数据集的既有训练链：其训练配置记录为 Adam 优化器、恒定学习率 1×10⁻⁴、batch size 12，并依据每 1000 次迭代的验证监控 mIoU 选择保存权重；两种正式模型的保存迭代不同。皮革对比方法（STDC、PIDNet、DDRNet）在同一训练集、验证集和测试集划分下训练与评价，但不据此推断训练链完全匹配。

### 2.2 NEU-Seg 总体性能比较

表 1 给出 NEU-Seg 数据集上 240k 固定训练终点的定量结果。DSMONet-B 在统一协议下取得 91.24% mIoU，表明细节—语义互优化基础架构在该数据集上具备分割能力；同协议下 A2MS-DSMONet-B 取得 91.45% mIoU，数值高 0.21 个百分点，其中斑块类 IoU 的数值差异相对更明显；在指定 A100 环境下测得 37.37 FPS，参数量增加 0.31%。需要说明的是，不同文献在 NEU 系数据集上的数据划分与评价口径存在差异，直接引用文献数字难以保证公平比较，因此本文主要报告统一协议下 DSMONet-B 与完整模型的内部对比，并以消融实验与细粒度分析限定结论范围。

表 1 NEU-Seg 数据集固定终点评价结果（统一 840 张测试图像，240k 迭代训练终点，seed 1337）/ Table 1 Fixed-endpoint evaluation results on NEU-Seg (unified 840 test images, 240k-iteration training endpoint, seed 1337)。

| 模型 | AAM | 细节监督 | mIoU/% | 背景 | 裂纹 | 夹杂物 | 斑块 | FPS | Params/M |
|---|:-:|:-:|---:|---:|---:|---:|---:|---:|---:|
| DSMONet-B | × | × | 91.24 | 98.62 | 84.15 | 93.49 | 88.71 | 38.16 | 29.37 |
| A2MS-DSMONet-B | √ | √ | **91.45** | 98.64 | 84.47 | 93.55 | **89.16** | 37.37 | 29.46 |

### 2.3 不同缺陷形态下的性能差异分析

整体 mIoU 只能反映平均分割水平，工业缺陷的类间形态差异要求更细粒度的分析。本节从类别 IoU、边界质量和缺陷尺度三个维度比较 A2MS-DSMONet-B 与 DSMONet-B。

**类别级与边界级分析。** 表 2 给出同一批 840 张测试图像上两模型的类别区域 IoU 与边界质量差异。其中区域 IoU 按各类别的测试集像素聚合计算，边界指标先对有效图像—类别对配对求差，再取样本宏平均；两类汇总口径不能解释为统一的“840 张逐样本平均”。

表 2 缺陷类别级性能差异（A2MS-DSMONet-B − DSMONet-B，240k 终点，配对比较）/ Table 2 Per-defect-type performance differences (A2MS-DSMONet-B minus DSMONet-B, 240k endpoint, paired comparison)。

| 类别 | Δ IoU/pt（像素聚合） | Δ Boundary IoU@2px/pt（样本宏平均） | Δ BF1@2px/pt（样本宏平均） |
|---|---:|---:|---:|
| 裂纹 crazing | +0.32 | −0.45 | −0.60 |
| 夹杂物 inclusion | +0.06 | +0.32 | +0.04 |
| 斑块 patches | +0.45 | +0.54 | +0.30 |

注：有效图像—类别对数分别为裂纹 368、夹杂物 361、斑块 353；标注与预测均为空的图像—类别对不计入边界宏平均。配对自助法置信区间仅反映固定 seed 1337 模型下测试样本的差异，不代表训练随机性。

在当前固定权重比较中，完整 A2MS-DSMONet-B 配置对 NEU-Seg 斑块类的聚合区域 IoU 和样本宏平均边界 IoU 均呈正向差异：1/2/3 像素容差下边界 IoU 分别高 0.55、0.54、0.55 个百分点（样本级配对自助法 95% 置信区间均不含零；按全部测试像素聚合计算则分别高 0.74、0.85、0.86 个百分点）。裂纹类的像素聚合 IoU 高 0.32 个百分点，但逐样本 IoU 差值的平均值为 −0.71 个百分点，样本宏平均边界指标也下降；两种 IoU 汇总口径的方向不同，不能将聚合结果解释为典型样本普遍改善。部分可视化改善病例可见大面积裂纹区域内部空洞减少，这可作为聚合结果的一种可能解释，但不能据此单独归因于 AAM 或细节监督。夹杂物类多数指标的数值差异较小。

**尺度分层分析。** 按各缺陷类别标注面积占比的 P33/P67 分位数将测试样本划分为小、中、大三组，比较两模型在各组的 IoU 差异（表 3）。斑块类缺陷在小面积组增益最大（+0.73 个百分点），裂纹类在小尺度组出现一定回退。尺度效应具有类别依赖性，不存在统一的"小目标增益"规律。

表 3 缺陷尺度分层 IoU 差异（ΔIoU/pt，按类别内 GT 面积占比 P33/P67 分层）/ Table 3 Scale-stratified IoU differences (per-class P33/P67 tiers)。

| 类别 | 小 | 中 | 大 |
|---|---:|---:|---:|
| 裂纹 crazing | −1.24 | −0.50 | −0.44 |
| 夹杂物 inclusion | +0.28 | −0.24 | −0.05 |
| 斑块 patches | +0.73 | −0.04 | +0.49 |

进一步以周长/面积平方根作为裂纹形态复杂度代理指标进行探索性分层，在当前固定权重下观察到高复杂度裂纹组 ΔIoU 为 −1.74 个百分点。该结果提示完整增强配置对复杂裂纹的分割仍有限制，不能分离出辅助细节监督的独立作用，也不能推广为所有复杂缺陷的规律。

### 2.4 真实皮革缺陷数据验证

除公开 NEU-Seg 数据集外，本文进一步在皮革缺陷数据集上考察模型在不同工业材质下的适用性。如 2.1 节所述，正式报告的模型权重来自验证监控选择，不同模型的保存迭代不同；在统一的 468 张独立测试图像上进行评价，测试集不参与所报告权重的选择。该比较不构成严格匹配训练链与随机种子的因果消融。定量结果如表 4 所示。

表 4 皮革数据集上不同方法的分割性能比较（468 张独立测试图像）/ Table 4 Segmentation performance of different methods on the leather defect dataset (468 independent test images)。

| 方法 | 来源 | mIoU/% | Params/M |
|---|---|---:|---:|
| DDRNet23-slim（内部复现） | 本文评价 | 84.21 | 20.30 |
| PIDNet-S（内部复现） | 本文评价 | 86.20 | 7.72 |
| STDC-Seg（STDCNet1446，内部复现） | 本文评价 | 88.24 | 16.08 |
| DSMONet-B | 本文 | 89.91 | 29.37 |
| A2MS-DSMONet-B | 本文 | **90.97** | 29.46 |

注：所有方法均使用相同的训练/验证/测试划分，所报告权重依据验证监控结果选取，并在统一 468 张测试图像上评价；不同模型的保存迭代及训练链不完全匹配。mIoU 包含背景类。内部复现指基于原作者公开代码框架的本地实现，与官方结果可能存在实现差异。

表 5 给出逐类别结果。在 7 类皮革缺陷中，完整 A2MS-DSMONet-B 配置相对 DSMONet-B 在 6 类上取得正向数值差异，其中刺刮伤、开创伤、烂面三类差异相对明显；刺猴类 IoU 下降约 3.7 个百分点，说明所报告的完整配置表现并非所有缺陷类别一致。由于两模型的训练链与随机性未完全匹配，这些差异只作为该材质上的应用观察，不单独归因于某个模块。

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

在相同 10k 迭代预算的四组配置中，联合配置取得最高平均 mIoU。单独加入 AAM 的配置在 3 个种子中的 2 个上高于 DSMONet-B，平均差为 +0.48 个百分点；单独加入细节监督的配置在 3 个种子中均高于 DSMONet-B，平均差为 +0.23 个百分点；联合配置的差值方向存在种子间差异。240k 固定训练终点上，仅比较 DSMONet-B 与完整 A2MS-DSMONet-B，后者数值高 0.21 个百分点。10k 与 240k 不仅训练长度不同，学习率调度也不同；短训结果只支持当前协议内的模块组合比较，不用于推断长训单模块的独立贡献或跨种子一致性。

注：受算力约束，AAM-only 与细节监督-only 配置未开展 240k 长训练消融，长训练消融仅含 DSMONet-B 与完整模型两组；10k 消融用于模块组合筛查。

### 2.6 复杂度与实时性分析

工业在线检测不仅关注分割精度，也关注模型复杂度和实时推理能力。表 8 汇总两模型的复杂度与速度实测结果（NVIDIA A100-PCIE-40GB，batch size 1，输入 200×200，FP32；预热 100 次、测量 1000 次，双模型交错分段测量以降低共享 GPU 负载波动对相对比较的影响）。

表 8 复杂度与实时性对比 / Table 8 Comparison of model complexity and real-time performance。

| 模型 | Params/M | MACs/G | 延迟均值/ms | 延迟 P95/ms | FPS | 峰值显存/MB |
|---|---:|---:|---:|---:|---:|---:|
| DSMONet-B | 29.37 | 6.648 | 26.21 | 42.42 | 38.16 | 270.3 |
| A2MS-DSMONet-B | 29.46 | 6.648 | 26.76 | 45.43 | 37.37 | 270.3 |

A2MS-DSMONet-B 相对 DSMONet-B 参数量增加 0.31%，推理 MACs 在所报精度下基本持平（细节监督分支仅参与训练，推理时不启用），平均延迟增加约 2.1%；在上述 A100 环境和 200×200 输入下测得 37.37 FPS。在皮革实验的 768×768 输入下（独占 A100、FP32、batch size 1），DSMONet-B 与 A2MS-DSMONet-B 实测分别为 64.68 FPS 与 62.83 FPS。在所测硬件和输入条件下，本文模型具备实时应用潜力；具体生产线适用性仍取决于检测节拍与部署环境。效率与精度的关系仅指本文所列协议下的数值比较。

注：NEU-Seg 输入下的 FPS 为共享 GPU 环境下双模型交错测量的相对比较结果，绝对帧率受环境影响；皮革输入下的 FPS 为独占 GPU 环境实测值。两种输入尺寸不同，帧率不横向比较。

### 2.7 可视化分析

图 3 NEU-Seg 数据集可视化分割结果 / Fig. 3 Visualization results on NEU-Seg。

NEU-Seg 上的分割可视化结果如图 3 所示。为避免人工择优，展示病例按预先确定规则选择：以 A2MS-DSMONet-B 与 DSMONet-B 的逐样本 BF1@2px 差值排序，对裂纹与斑块两类分别选取改善最大、退化最大和差值近零的代表性病例。每行依次展示原图、标注、两模型预测结果及误差分布图（真正例、假正例、假负例分别以不同颜色标注）。

可视化结果与定量分析一致：在改善病例中，A2MS-DSMONet-B 能够填补大面积裂纹区域内部的误分类空洞，使预测区域更完整；在退化病例中，该模型在强边缘附近出现裂纹区域的过度扩张，并对个别纤细裂纹段的响应减弱。斑块类病例中，改善病例的缺陷轮廓与标注贴合更紧密。

## 3 讨论

### 3.1 方法适用性分析

DSMONet 的细节—语义互优化架构在两个工业缺陷数据集上均给出可独立评价的分割结果：在 NEU-Seg 240k 固定终点协议下取得 91.24% mIoU，在皮革数据集报告权重的统一 468 张测试图像上取得 89.91% mIoU。在皮革数据集上，后者高于本文复现并在相同测试划分评价的 STDC-Seg、PIDNet-S、DDRNet23-slim；NEU-Seg 未开展这三种方法的同协议比较，只报告 DSMONet-B 与完整 A2MS-DSMONet-B 的内部对比。DSMONet 的结构未设置特定缺陷类别专用分支，但其经验适用范围仍限于本文测试的材质与协议。

在此基础上，完整 A2MS-DSMONet-B 配置的观察结果呈现类型依赖性。在 NEU-Seg 斑块类中，聚合区域 IoU 与多容差样本宏平均边界 IoU 均为正向差异；这可以与较闭合的缺陷区域及其内部占比共同讨论，但现有证据不能把变化单独归因于 AAM 的语义重标定。裂纹类呈现聚合区域 IoU 为正、逐样本区域 IoU 平均值和边界指标为负的并存现象；部分可视化案例中的内部空洞填补提供一种可能解释，高复杂度分组结果仍属探索性观察，不能视为辅助细节监督对细长结构作用的单独检验。夹杂物类别多数指标变化较小。这些结果提示完整增强配置的表现随缺陷形态而异，不宜概括为所有类别的一致增益。

除公开 NEU-Seg 数据集外，本文进一步在皮革缺陷数据集上考察两级方法在另一工业材质下的适用性。基于验证监控选取的正式权重，在统一 468 张测试图像上，A2MS-DSMONet-B 取得 90.97% mIoU，数值高于 DSMONet-B 的 89.91%，差为 +1.06 个百分点。由于两模型的保存迭代不同，训练链和随机性未完全匹配，该差值是本次皮革应用比较的结果，不能独立证明增强结构的因果收益或跨材质泛化。逐类别结果中刺猴类别仍有回退，与 NEU-Seg 中观察到的形态相关差异共同提示需要按缺陷类别评估表现。

### 3.2 局限性

本文仍存在以下局限：（1）NEU-Seg 240k 固定终点比较基于单一随机种子，+0.21 个百分点的差异较小，训练随机性下的稳定性尚需多种子长训练验证；（2）皮革实验采用验证监控选择权重，与 NEU-Seg 的固定终点协议不同；皮革两模型保存迭代不同，历史训练链和随机性未完全匹配，+1.06 个百分点仅是所报告权重的应用比较，不能单独归因于增强模块；（3）边界指标的配对自助法区间只刻画固定权重下的测试样本差异，不能替代训练随机性检验；（4）完整配置在不同缺陷类别上的数值变化具有形态依赖性，复杂裂纹和皮革刺猴类存在回退；（5）FPS 测量受 GPU 环境（共享/独占）与输入尺寸影响，不能据此直接推断生产线节拍；（6）当前评价仅覆盖钢材与皮革两种工业材质，NEU-Seg 上也未对文献方法开展同协议复现比较。

## 4 结论

本文围绕实时语义分割中高层语义与局部空间细节协同建模问题，构建细节—语义相互优化网络 DSMONet，并在此基础上针对工业表面缺陷低占比、弱纹理、多尺度与形态差异明显等特点，引入 AAM 与辅助细节监督，形成 A2MS-DSMONet。互优化体现为语义信息调节浅层细节、增强后的细节参与最终联合表征。在 NEU-Seg 当前 240k 固定终点评价协议下，DSMONet-B 与完整 A2MS-DSMONet-B 分别取得 91.24% 和 91.45% mIoU，差为 +0.21 个百分点；后者在指定 A100 条件下测得 37.37 FPS。斑块类区域与边界指标呈正向差异，裂纹类则表现为像素聚合区域与逐样本边界结果之间的权衡。在真实皮革缺陷数据集上，基于验证监控选择的两模型权重在统一 468 张测试图像上分别取得 89.91% 和 90.97% mIoU，差为 +1.06 个百分点；由于训练链与随机性未完全匹配，该比较用于考察另一工业材质下的应用表现。本文结果支持 DSMONet 的独立分割能力与完整增强配置在所测条件下的数值表现，尚不能将各类别变化单独归因于某个模块，也不能推断未测材质或生产线节拍下的效果。后续工作将关注复杂裂纹边界、多种子长训练验证和更多工业材质的适用性评估。

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
