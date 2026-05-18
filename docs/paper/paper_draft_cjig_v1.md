# 基于自适应注意力与细节—语义互优化的工业表面缺陷实时分割方法

# Real-time Industrial Surface Defect Segmentation Based on Adaptive Attention and Detail-Semantic Mutual Optimization

作者：待补充

单位：待补充

中图分类号：待核定

文献标志码：A

收稿日期：待补充

基金项目：待补充

## 摘要

**目的** 工业表面缺陷语义分割需要在复杂材料纹理背景下准确定位不同尺度、形态和边界模糊程度的缺陷区域，同时满足在线检测场景对实时性的要求。针对缺陷区域占比低、细小或细长缺陷易漏检、背景纹理干扰强以及边界不清晰等问题，本文提出一种基于自适应注意力与细节—语义互优化的工业表面缺陷实时分割网络 A2MS-DefectNet。

**方法** A2MS-DefectNet 以细节—语义互优化机制为核心，通过浅层空间细节与深层语义信息的双向交互，利用深层语义抑制浅层背景纹理噪声，并利用浅层细节补偿深层特征中缺失的边界和位置信息。在此基础上，设计高效轻量映射模块 ELMM，以简化深层语义到浅层细节的映射路径，降低特征解耦与重构开销；引入自适应注意力模块 AAM，以增强模型对多尺度、低占比和弱纹理缺陷区域的通道响应；同时借鉴 STDCNet 的边缘监督思想和 Sub-region UNet 的掩码感知监督思想，将边缘细节损失与掩码感知损失联合引入训练过程，形成面向缺陷边界和前景区域的联合监督约束。

**结果** 在 NEU-Seg 带钢表面缺陷数据集和自建皮革缺陷数据集上的实验结果表明，A2MS-DefectNet-B 分别取得 91.3% mIoU、127.9 FPS 和 91.0% mIoU、71.6 FPS。与仅采用细节—语义互优化机制的 Base-B 相比，A2MS-DefectNet-B 在 NEU-Seg 数据集上的 mIoU 提升 1.1 个百分点、FPS 提升 12.3；在自建皮革缺陷数据集上的 mIoU 提升 1.8 个百分点、FPS 提升 4.3。消融实验显示，ELMM、AAM 以及边缘细节损失和掩码感知损失构成的联合监督约束均能提升模型对工业缺陷区域的分割性能。

**结论** A2MS-DefectNet 能够在保持实时推理能力的同时提高多尺度、低占比、细长和边界模糊缺陷的像素级分割精度。该方法适用于钢材、皮革等工业表面缺陷在线检测场景，但仍需进一步补充模型参数量、FLOPs、模型大小、跨设备速度测试以及更多工业场景泛化实验。

## 关键词

工业表面缺陷；实时语义分割；细节—语义互优化；自适应注意力；轻量化网络；边界监督；掩码感知损失

## Abstract

**Objective** Industrial surface defect segmentation is a pixel-level visual inspection task in which a model must identify defect regions with different scales, shapes, textures, and boundary clarity under complex material backgrounds. Compared with image-level classification or bounding-box detection, semantic segmentation provides more direct information for defect localization, area measurement, boundary analysis, and downstream quality assessment. However, industrial surface images are difficult for real-time segmentation networks because defect regions often occupy only a small fraction of the image, many defects are elongated or discontinuous, and the visual difference between a true defect and a normal material texture can be subtle. Online inspection systems also require high inference speed, which makes it impractical to simply use heavy segmentation networks. To address these issues, this paper proposes A2MS-DefectNet, a real-time industrial surface defect segmentation network based on adaptive attention and detail-semantic mutual optimization.

**Method** The core of A2MS-DefectNet is a detail-semantic mutual optimization mechanism. Shallow features preserve spatial details such as edges, local texture changes, and positional cues, but they are easily disturbed by background texture, illumination variation, and normal manufacturing marks. Deep features have stronger semantic discrimination, but repeated down-sampling weakens fine boundaries and small defect structures. The proposed network therefore builds a bidirectional interaction between shallow detail features and deep semantic features. Deep semantic information is used to regulate shallow detail responses and suppress false activations caused by background textures, while optimized shallow spatial information is fed back to compensate deep semantic features with boundary and location information. On top of this mechanism, an efficient lightweight mapping module, ELMM, is designed to simplify the mapping path from deep semantic features to shallow detail features and reduce the cost of feature decoupling and reconstruction. An adaptive attention module, AAM, is further introduced to combine adaptive channel weighting and efficient channel attention, strengthening the response to multi-scale, low-ratio, and weak-texture defects. In training, the network uses a joint supervision strategy. The edge detail loss is introduced with reference to the boundary supervision idea of STDCNet, and the mask-aware loss is introduced with reference to Sub-region UNet. These two losses are combined with the basic segmentation loss so that the model is supervised simultaneously at the pixel classification, boundary structure, and foreground region levels.

**Result** Experiments are conducted on NEU-Seg and a self-built leather defect dataset. NEU-Seg is used to evaluate the ability of the model to segment steel surface defects such as inclusions, patches, and scratches under metal texture backgrounds. The leather dataset is collected from a practical industrial production scenario and contains seven categories of leather defects, including open wound, scratch, brand mark, hole, skin disease, rotten surface, and wart-like defect. On NEU-Seg, A2MS-DefectNet-B achieves 91.3% mIoU and 127.9 FPS. On the self-built leather defect dataset, it achieves 91.0% mIoU and 71.6 FPS. Compared with Base-B, which only uses the detail-semantic mutual optimization mechanism, A2MS-DefectNet-B improves mIoU by 1.1 percentage points and FPS by 12.3 on NEU-Seg, and improves mIoU by 1.8 percentage points and FPS by 4.3 on the leather dataset. Ablation studies show that ELMM improves the efficiency of semantic-to-detail mapping, AAM enhances defect-related channel responses, and the joint supervision strategy improves the learning of defect boundaries and foreground regions. Visual comparisons further indicate that the proposed method produces more continuous predictions for elongated scratches, reduces missed detection of small defects, and better preserves ambiguous defect boundaries under strong texture interference.

**Conclusion** A2MS-DefectNet provides an effective real-time segmentation framework for industrial surface defects by combining detail-semantic mutual optimization, lightweight feature mapping, adaptive attention, and joint boundary/foreground supervision. The method improves the balance between segmentation accuracy and inference speed on steel and leather defect datasets, and it is especially useful for low-ratio, small-scale, elongated, and boundary-blurred defects. Future work should complete Params, FLOPs, and model-size statistics, evaluate cross-device real-time performance, and validate the method on more industrial materials and production scenarios. These extensions will make the efficiency analysis, deployment assessment, and generalization evaluation more complete for practical industrial inspection applications.

## Keywords

industrial surface defects; real-time semantic segmentation; detail-semantic mutual optimization; adaptive attention; lightweight network; boundary supervision; mask-aware loss

## 0 引言

工业制造过程中，钢材、皮革、芯片、磁瓦和金属零部件等材料表面容易出现划痕、夹杂、破洞、斑块、磨损和弱纹理缺陷。这类缺陷会影响产品外观质量、结构可靠性和后续加工稳定性，因此在生产线上实现快速、稳定、精细的缺陷检测具有重要工程意义。传统人工检测依赖经验，检测效率和一致性受主观因素影响较大；传统机器视觉方法通常依赖灰度、纹理、边缘或频域特征，在光照变化、背景纹理复杂和缺陷形态多变时泛化能力不足。随着深度学习的发展，语义分割方法能够输出像素级缺陷区域，为缺陷定位、面积统计和质量判定提供更精细的信息。

将通用语义分割方法直接应用于工业表面缺陷实时检测仍面临多方面挑战。首先，工业缺陷尺度差异明显，点状缺陷、细长划痕和不规则斑块可能同时出现，模型需要兼顾局部细节与全局语义。其次，缺陷区域在整幅图像中占比较低，大面积背景纹理容易干扰模型对前景缺陷的判别。再次，工业材料表面纹理往往与缺陷边缘交织，浅层特征虽然包含边缘、纹理和位置信息，却也容易混入背景噪声；深层语义特征具有更强判别能力，但连续下采样会削弱小目标和边界位置表达。最后，在线检测场景通常要求模型在保证像素级精度的同时具有较高推理速度。大型分割网络虽然具有较强表达能力，但推理开销较大；轻量网络速度较快，却容易损失边缘细节和小目标信息。因此，如何在分割精度、推理速度和缺陷边界保持能力之间取得平衡，是工业表面缺陷实时分割中的关键问题。

近年来，工业缺陷分割研究逐渐从通用编码器—解码器结构转向任务特异的轻量化、注意力增强和边界细节建模。DMC-Net（作者待核验，2025）面向实时表面缺陷分割设计轻量特征提取与多尺度增强模块，任务与本文工业实时分割主线较为接近；SPCS-Net（作者待核验，2025）通过空间位置注意和跨尺度融合改善工业缺陷分割精度；TAG-Net（作者待核验，2025）针对钢材表面缺陷显式建模背景、缺陷和边界注意力；GCRANet（作者待核验，2026）关注全局上下文引导下的弱缺陷轮廓增强和背景抑制；CDARNet（作者待核验，2025）面向实时金属表面缺陷分割，通过跨维度自适应区域重构增强鲁棒性；LGGFormer（作者待核验，2025）将局部细节与全局建模结合，并引入边缘引导解码；DASeg-Net（作者待核验，2025）将扩散模型与注意力机制用于芯片表面缺陷分割；SAID（作者待核验，2025）则体现了提示式分割在跨场景工业缺陷任务中的潜力。上述工作表明，工业缺陷分割不仅需要全局语义判别能力，还需要在复杂背景纹理下保持细小缺陷和模糊边界的局部结构。

实时语义分割研究主要围绕轻量骨干、高低分辨率分支、快速上下文聚合和高效特征融合展开。BiSeNetV1（作者待核验，2018）和 BiSeNetV2（作者待核验，2020）以空间细节和语义上下文的双分支建模为代表；DDRNet（作者待核验，2021）通过双分辨率结构保留空间细节并聚合上下文；STDCNet（作者待核验，2021）通过短期密集连接和细节聚合提升实时分割效率，其边缘监督思想也为缺陷边界学习提供参考；PP-LiteSeg（作者待核验，2022）、PIDNet（作者待核验，2023）、LETNet（作者待核验，2023）、SeaFormer（作者待核验，2023）和 SeaFormer++（作者待核验，2025）进一步从轻量解码、边界分支、高效 Transformer 或移动端注意力等角度改进速度—精度平衡。这些方法多在 Cityscapes、CamVid、ADE20K 等自然场景数据集上评估，直接迁移到工业缺陷场景时仍需针对低占比前景、复杂纹理和边界模糊问题进行适配。

针对上述问题，本文提出一种基于自适应注意力与细节—语义互优化的工业表面缺陷实时分割网络 A2MS-DefectNet。该网络以细节—语义互优化机制为核心，通过深层语义调节浅层细节特征，抑制背景纹理误响应，并利用浅层空间信息补偿深层语义特征中的边界与位置信息。在此基础上，本文进一步设计高效轻量映射模块 ELMM、自适应注意力模块 AAM，并引入边缘细节损失与掩码感知损失构成联合监督约束。本文主实验聚焦 NEU-Seg 带钢表面缺陷数据集和自建皮革缺陷数据集，城市街景等通用场景仅作为实时语义分割方法背景，不作为论文主线。

本文主要贡献如下：

1. 提出以细节—语义互优化机制为核心的工业表面缺陷实时分割网络 A2MS-DefectNet。针对工业缺陷图像中浅层纹理噪声强、深层语义空间细节不足的问题，引入浅层空间细节与深层语义信息的双向交互，使深层语义能够抑制浅层背景纹理干扰，并利用浅层细节补偿深层特征的边界与位置信息。
2. 设计高效轻量映射模块 ELMM。通过简化深层语义特征到浅层细节特征的映射路径，降低复杂特征解耦与重构过程带来的计算开销，在保持细节—语义交互能力的同时提升推理效率。
3. 引入自适应注意力模块 AAM。结合自适应通道权重与高效通道注意力机制，增强模型对多尺度、低占比和弱纹理缺陷区域的响应能力，抑制复杂背景纹理干扰。
4. 构建面向缺陷边界与前景区域的联合监督约束。借鉴 STDCNet（作者待核验，2021）的边缘监督思想和 Sub-region UNet（作者待核验，2023）的掩码感知监督思想，将边缘细节损失与掩码感知损失联合引入训练过程，提高模型对小目标缺陷、细长缺陷和边界模糊缺陷的分割能力。

## 1 方法

### 1.1 问题定义

给定工业表面图像 \(I \in \mathbb{R}^{H\times W\times C}\)，语义分割模型需要预测每个像素所属类别，输出与输入图像空间尺寸对应的预测结果 \(\hat{Y} \in \mathbb{R}^{H\times W}\)。对于 NEU-Seg 与自建皮革缺陷数据集，模型需要在背景和多类缺陷之间完成像素级分类。本文采用平均交并比 mIoU 衡量分割精度，采用 FPS 衡量模型推理速度，并结合 Params、FLOPs 和模型大小分析复杂度。

像素级预测和 mIoU 可表示为：

\[
\hat{Y}=\arg\max_{c} f_{\theta}(I)_{h,w,c}, \qquad
\mathrm{mIoU}=\frac{1}{K}\sum_{k=1}^{K}\frac{TP_k}{TP_k+FP_k+FN_k}.
\]

其中，\(f_{\theta}\) 表示待训练分割网络，\(K\) 为类别数，\(TP_k\)、\(FP_k\) 和 \(FN_k\) 分别表示第 \(k\) 类的真正例、假正例和假负例。公式待核验：基础交叉熵或 OHEM 损失的完整形式需依据原始 Word 文档或代码实现确认。

### 1.2 A2MS-DefectNet 总体结构

图 1 A2MS-DefectNet 网络结构示意图 / Fig. 1 Overall architecture of A2MS-DefectNet。图件要求：建议基于原始网络结构图进行矢量图重绘，或导出 300 dpi 以上位图。

A2MS-DefectNet 以细节—语义互优化机制为核心，通过 ELMM、AAM 和联合监督约束面向工业表面缺陷实时分割任务进行结构与训练策略设计。网络首先通过主干网络提取多层特征，其中浅层特征具有较高空间分辨率，包含边缘、纹理和位置信息；深层特征经过多次下采样后语义判别能力更强，但小目标位置和边界细节容易被削弱。工业材料表面纹理复杂，浅层特征中的背景纹理常与缺陷边缘相互混杂，若缺少语义约束，模型容易把正常纹理误判为缺陷；而仅依赖深层语义又难以精确恢复细长缺陷、低占比缺陷和模糊边界。因此，A2MS-DefectNet 将浅层空间细节与深层语义信息的双向交互作为基础特征表达单元，使模型在早期保留空间定位信息的同时，利用深层语义抑制无效纹理响应。

在该核心机制基础上，A2MS-DefectNet 面向工业缺陷场景进一步进行三方面适配。首先，针对原复杂映射路径中可能存在的特征解耦和重构开销，设计 ELMM，以轻量通道映射和融合方式完成深层语义对浅层细节的调制。其次，针对多尺度、低占比和弱纹理缺陷区域响应不足的问题，引入 AAM，通过自适应通道权重与高效通道注意力增强缺陷相关特征。最后，针对缺陷边界模糊和前景区域占比低的问题，将边缘细节损失与掩码感知损失联合引入训练过程，使模型同时受到像素分类、边界结构和前景区域分布约束。由此，A2MS-DefectNet 以细节—语义互优化机制为基础，形成面向工业缺陷实时分割的轻量化、注意力增强和联合监督网络。

### 1.3 细节—语义互优化机制

工业缺陷图像中的浅层特征和深层特征具有明显互补关系，也存在各自局限。浅层特征保留较高空间分辨率，能够提供缺陷边缘、纹理变化、位置分布等细节信息，对细长划痕、小面积夹杂物和边界不规则缺陷尤为重要。然而，钢材、皮革等材料表面往往存在自然纹理、加工痕迹和局部亮度变化，浅层特征容易同时响应缺陷区域和非缺陷背景纹理，导致误分割。深层特征经过更大感受野和多层非线性变换，具有更强语义判别能力，能够帮助模型区分缺陷与复杂背景；但连续下采样会降低空间分辨率，使小目标位置、细长结构连续性和边界细节发生损失。

细节—语义互优化机制的核心作用在于建立浅层空间细节与深层语义信息之间的双向交互。首先，利用深层语义信息调节浅层细节特征，使浅层特征在保留边缘和纹理表达的同时减少对背景纹理噪声的响应。对于材料纹理复杂或缺陷与背景对比度较低的区域，语义调节能够为浅层细节提供类别判别方向，降低正常纹理被误识别为缺陷的概率。其次，将经过语义约束优化后的浅层空间信息反向补偿深层特征，为深层语义恢复缺陷边界、细长结构和局部位置信息。通过这种双向交互，模型能够同时获得语义判别能力和空间定位能力，从而提升对小目标、细长缺陷和边界模糊缺陷的分割质量。

### 1.4 高效轻量映射模块 ELMM

原始细节—语义映射路径包含较复杂的特征解耦和重构过程，在工业缺陷实时分割中可能带来额外计算开销。考虑到工业图像中缺陷区域通常较小，且在线检测对推理速度敏感，本文设计高效轻量映射模块 ELMM，以更简洁的方式实现深层语义对浅层细节的调制。ELMM 并不削弱细节—语义互优化机制，而是在保留深层语义调节浅层细节这一核心功能的同时，减少不必要的映射复杂度，使该机制更适合工业实时检测场景。

设浅层细节特征为 \(\mathbf{F}_{d}\)，深层语义特征为 \(\mathbf{F}_{s}\)。ELMM 首先通过 \(1\times1\) 卷积调整语义特征通道，并将其映射为权重响应；随后通过逐元素加权强化浅层细节特征中与缺陷语义相关的区域；最后通过轻量融合得到输出特征。其占位表达式如下：

\[
\mathbf{M}_{s}=\sigma(\phi_{1\times1}(\operatorname{Up}(\mathbf{F}_{s}))),
\]

\[
\mathbf{F}_{d}^{\prime}=\mathbf{F}_{d}\odot \mathbf{M}_{s},
\]

\[
\mathbf{F}_{\mathrm{ELMM}}=\psi_{1\times1}\left(\left[\mathbf{F}_{d}^{\prime};\operatorname{Up}(\mathbf{F}_{s})\right]\right).
\tag{1}
\]

其中，\(\operatorname{Up}(\cdot)\) 表示上采样，\(\phi_{1\times1}\) 和 \(\psi_{1\times1}\) 表示 \(1\times1\) 卷积映射，\(\sigma(\cdot)\) 表示 Sigmoid 函数，\(\odot\) 表示逐元素乘法，\([\cdot;\cdot]\) 表示特征拼接。公式待核验：式（1）的通道变换、融合方式和符号命名需依据原始 Word 文档或代码实现确认。

### 1.5 自适应注意力模块 AAM

工业缺陷常具有低占比、弱纹理和尺度变化明显等特点。若网络对背景纹理响应过强，细小缺陷和细长划痕容易被忽略。为增强模型对关键通道和缺陷相关特征的选择能力，本文引入自适应注意力模块 AAM。AAM 由自适应通道权重单元 ACW 与高效挤压激励单元 eSE 组成。ACW 通过可学习权重建模不同通道的重要性，使网络能够根据训练目标动态调整通道关注程度；eSE 以较低计算开销获得通道注意力响应。二者结合后，模型能够进一步增强缺陷区域的特征表达，尤其适用于小目标、弱纹理和边界不清晰的缺陷。

设输入特征为 \(\mathbf{F}\)，全局池化向量为 \(\mathbf{z}=\operatorname{GAP}(\mathbf{F})\)。AAM 的占位表达式如下：

\[
\mathbf{a}_{\mathrm{ACW}}=\operatorname{ACW}(\mathbf{z}), \qquad
\mathbf{a}_{\mathrm{eSE}}=\operatorname{eSE}(\mathbf{F}),
\]

\[
\mathbf{F}_{\mathrm{AAM}}=\mathbf{F}\odot \mathbf{a}_{\mathrm{ACW}}\odot \mathbf{a}_{\mathrm{eSE}}.
\tag{2}
\]

其中，\(\mathbf{a}_{\mathrm{ACW}}\) 与 \(\mathbf{a}_{\mathrm{eSE}}\) 分别表示自适应通道权重和高效通道注意力响应。公式待核验：ACW、eSE 以及 AAM 输出的完整公式需依据原始 Word 文档或代码实现确认。

### 1.6 联合监督约束

A2MS-DefectNet 的训练目标由基础分割损失、边缘细节损失和掩码感知损失共同构成。基础分割损失沿用多阶段监督思路，对不同输出层进行语义分割约束。边缘细节损失借鉴 STDCNet（作者待核验，2021）的边缘监督思想，通过从真实标签中提取边界信息增强模型对缺陷轮廓的学习能力。掩码感知损失借鉴 Sub-region UNet（作者待核验，2023）的 mask-aware loss 思想，将背景与非背景缺陷区域转化为前景—背景监督，以强化模型对缺陷区域整体分布的感知。

联合损失的占位表达式如下：

\[
\mathcal{L}_{\mathrm{seg}}=\sum_{i=0}^{2}\omega_i\mathcal{L}_{\mathrm{seg}}^{(i)},
\]

\[
\mathcal{L}_{\mathrm{total}}=
\mathcal{L}_{\mathrm{seg}}
+\lambda_{e}\mathcal{L}_{\mathrm{edge}}
+\lambda_{m}\mathcal{L}_{\mathrm{mask}}.
\tag{3}
\]

其中，\(\mathcal{L}_{\mathrm{seg}}\) 为多阶段基础分割损失，\(\mathcal{L}_{\mathrm{edge}}\) 为边缘细节损失，\(\mathcal{L}_{\mathrm{mask}}\) 为掩码感知损失，\(\lambda_e\) 与 \(\lambda_m\) 分别为对应权重。公式待核验：\(\omega_i\)、\(\lambda_e\)、\(\lambda_m\) 的具体取值、边缘标签生成方式以及掩码感知损失的完整形式需依据原始 Word 文档或代码实现确认。

需要说明的是，边缘细节损失和掩码感知损失分别来源于已有边缘监督和 mask-aware loss 思想。本文贡献在于将二者共同引入 A2MS-DefectNet，并与基础分割损失构成面向工业缺陷任务的联合监督约束，使模型同时关注类别级像素预测、缺陷边界结构和前景区域分布。

## 2 实验与分析

### 2.1 数据集与实验设置

本文主实验采用 NEU-Seg 与自建皮革缺陷数据集。NEU-Seg 是带钢表面缺陷语义分割数据集，包括六类带钢表面缺陷。本文选取其中夹杂物、补丁和划痕三类进行实验。由于当前固定结果文件中尚未登记 NEU-Seg 的样本划分和图像尺寸，相关信息在终稿中需结合原始数据集说明或实验记录进一步补充。该数据集用于评估模型在金属表面纹理背景下对小尺度、细长形态和边界不规则缺陷的分割能力。

自建皮革缺陷数据集来自实际工业生产场景，包含开创伤、刺刮伤、烙印、破洞、皮肤藓、烂面、刺猴 7 类缺陷，共 2341 张 \(768\times768\) 图像。其中训练集 1638 张、验证集 468 张、测试集 235 张。该数据集用于评估模型在真实皮革纹理背景下对多类别、多尺度和边界模糊缺陷的分割能力。

图 2 主实验数据集样例 / Fig. 2 Examples of the main experimental datasets。图件要求：建议使用原始数据集样例重新排版，终稿导出 300 dpi 以上位图或矢量图。

表 1 主实验数据集与实验设置 / Table 1 Datasets and experimental settings。
表格格式：终稿按三线表排版，仅保留顶线、表头线和底线；Markdown 预览中以普通表格显示。

| 项目 | NEU-Seg | 自建皮革缺陷数据集 |
|---|---|---|
| 任务类型 | 带钢表面缺陷语义分割 | 皮革表面缺陷语义分割 |
| 使用类别 | 夹杂物、补丁、划痕 | 开创伤、刺刮伤、烙印、破洞、皮肤藓、烂面、刺猴 |
| 图像数量 | 待补充 | 2341 张 |
| 数据划分 | 待补充 | 训练 1638 张、验证 468 张、测试 235 张 |
| 图像尺寸 | 待补充 | \(768\times768\) |
| 评价指标 | mIoU、FPS、Params、FLOPs、模型大小 | mIoU、FPS、Params、FLOPs、模型大小 |

所有实验使用相同训练和推理环境。硬件平台为 Intel Core i9-10900X CPU、NVIDIA GeForce RTX 3090 GPU 和 24 GB 内存；软件环境为 Ubuntu 18.04、PyTorch 1.10、CUDA 11.2、cuDNN 8.2 和 Python 3.9.7。训练时 batch size 设置为 16，最大迭代次数为 80000，优化器为 Adam，初始学习率为 0.0001，权重衰减为 \(2\times10^{-6}\)。数据增强包括随机旋转、随机水平翻转和随机垂直翻转等操作。

实验表中的 Base-S 和 Base-B 表示仅采用细节—语义互优化机制的基础版本，S/B 分别对应不同主干配置；A2MS-DefectNet-S 和 A2MS-DefectNet-B 表示在基础版本上进一步引入 ELMM、AAM 和联合监督约束后的最终模型配置。

### 2.2 与主流方法对比

表 2 给出 NEU-Seg 数据集上的定量结果。A2MS-DefectNet-B 取得 91.3% mIoU 和 127.9 FPS；A2MS-DefectNet-S 取得 89.7% mIoU 和 196.5 FPS。与 Base-B 相比，A2MS-DefectNet-B 的 mIoU 提升 1.1 个百分点，FPS 提升 12.3。该结果表明，在细节—语义互优化机制提供基础特征交互能力的前提下，ELMM、AAM 和联合监督约束能够进一步提升工业缺陷分割精度，并保持较好的实时性。

表 2 NEU-Seg 数据集定量实验结果 / Table 2 Quantitative results on NEU-Seg。
表格格式：终稿按三线表排版，仅保留顶线、表头线和底线；Markdown 预览中以普通表格显示。

| Dataset | Model | Backbone | Params/M | FLOPs/G | 模型大小/MB | mIoU/% | FPS |
|---|---|---|---:|---:|---:|---:|---:|
| NEU-Seg | FDSNet | - | 待补充 | 待补充 | 待补充 | 78.8 | 186.1 |
| NEU-Seg | PGA-Net | - | 待补充 | 待补充 | 待补充 | 82.2 | 48.5 |
| NEU-Seg | Sub-region UNet | - | 待补充 | 待补充 | 待补充 | 88.5 | 285.1 |
| NEU-Seg | BiSeNetV1-L | ResNet-18 | 待补充 | 待补充 | 待补充 | 87.5 | 218.7 |
| NEU-Seg | BiSeNetV2-L | - | 待补充 | 待补充 | 待补充 | 85.1 | 223.3 |
| NEU-Seg | SFNet | DF2 | 待补充 | 待补充 | 待补充 | 87.5 | 151.0 |
| NEU-Seg | SFNet | ResNet-50 | 待补充 | 待补充 | 待补充 | 90.5 | 126.3 |
| NEU-Seg | STDC1-Seg | STDC1 | 待补充 | 待补充 | 待补充 | 87.5 | 255.1 |
| NEU-Seg | STDC2-Seg | STDC2 | 待补充 | 待补充 | 待补充 | 87.7 | 169.7 |
| NEU-Seg | PP-LiteSeg-T | STDC1 | 待补充 | 待补充 | 待补充 | 80.5 | 142.2 |
| NEU-Seg | PP-LiteSeg-B | STDC2 | 待补充 | 待补充 | 待补充 | 81.1 | 105.6 |
| NEU-Seg | DDRNet23slim | - | 20.3 | 待补充 | 待补充 | 86.9 | 160.4 |
| NEU-Seg | Base-S | ResNet-18 | 14.0 | 待补充 | 待补充 | 88.4 | 178.5 |
| NEU-Seg | Base-B | ResNet-50 | 29.3 | 待补充 | 待补充 | 90.2 | 115.6 |
| NEU-Seg | A2MS-DefectNet-S | ResNet-18 | 待补充 | 待补充 | 待补充 | 89.7 | 196.5 |
| NEU-Seg | A2MS-DefectNet-B | ResNet-50 | 待补充 | 待补充 | 待补充 | 91.3 | 127.9 |

表 3 给出自建皮革缺陷数据集上的定量结果。A2MS-DefectNet-B 取得 91.0% mIoU 和 71.6 FPS；A2MS-DefectNet-S 取得 89.7% mIoU 和 188.7 FPS。与 Base-B 相比，A2MS-DefectNet-B 的 mIoU 提升 1.8 个百分点，FPS 提升 4.3。该结果说明，在真实皮革纹理背景下，以细节—语义互优化机制为核心并进一步进行工业缺陷任务适配的 A2MS-DefectNet 对多尺度缺陷具有更好的分割性能。

表 3 皮革缺陷数据集定量实验结果 / Table 3 Quantitative results on the leather defect dataset。
表格格式：终稿按三线表排版，仅保留顶线、表头线和底线；Markdown 预览中以普通表格显示。

| Dataset | Model | Backbone | Params/M | FLOPs/G | 模型大小/MB | mIoU/% | FPS |
|---|---|---|---:|---:|---:|---:|---:|
| Leather | Sub-region UNet | - | 待补充 | 待补充 | 待补充 | 84.6 | 55.3 |
| Leather | BiSeNetV1-L | ResNet-18 | 待补充 | 待补充 | 待补充 | 86.9 | 180.8 |
| Leather | BiSeNetV2-L | - | 待补充 | 待补充 | 待补充 | 87.1 | 213.1 |
| Leather | SFNet | DF2 | 待补充 | 待补充 | 待补充 | 89.9 | 142.6 |
| Leather | SFNet | ResNet-50 | 待补充 | 待补充 | 待补充 | 89.1 | 44.9 |
| Leather | STDC1-Seg | STDC1 | 待补充 | 待补充 | 待补充 | 85.0 | 163.4 |
| Leather | STDC2-Seg | STDC2 | 待补充 | 待补充 | 待补充 | 85.3 | 135.1 |
| Leather | PP-LiteSeg-T | STDC1 | 待补充 | 待补充 | 待补充 | 82.2 | 122.7 |
| Leather | PP-LiteSeg-B | STDC2 | 待补充 | 待补充 | 待补充 | 85.5 | 86.9 |
| Leather | DDRNet23slim | - | 20.3 | 待补充 | 待补充 | 88.1 | 149.5 |
| Leather | Base-S | ResNet-18 | 14.0 | 待补充 | 待补充 | 88.1 | 167.2 |
| Leather | Base-B | ResNet-50 | 29.3 | 待补充 | 待补充 | 89.2 | 67.3 |
| Leather | A2MS-DefectNet-S | ResNet-18 | 待补充 | 待补充 | 待补充 | 89.7 | 188.7 |
| Leather | A2MS-DefectNet-B | ResNet-50 | 待补充 | 待补充 | 待补充 | 91.0 | 71.6 |

### 2.3 消融实验

表 4 给出高效轻量映射模块 ELMM 的消融结果。Baseline 未使用细节—语义互优化映射，DM 为原复杂映射模块，ELMM 为本文轻量映射模块。ELMM 在获得 91.3% mIoU 的同时保持 126.7 FPS。与 DM 相比，ELMM 的 mIoU 提升 1.2 个百分点，FPS 提升 9.1。该结果说明，在细节—语义互优化机制中，深层语义对浅层细节的调制是提升缺陷分割性能的重要基础；同时，用轻量映射替代复杂映射路径能够减少交互开销，使该基础机制更适合工业实时分割场景。

表 4 高效轻量映射模块消融实验 / Table 4 Ablation study of the efficient lightweight mapping module。
表格格式：终稿按三线表排版，仅保留顶线、表头线和底线；Markdown 预览中以普通表格显示。

| 方法 | Params/M | FLOPs/G | 模型大小/MB | mIoU/% | FPS |
|---|---:|---:|---:|---:|---:|
| Baseline | 待补充 | 待补充 | 待补充 | 85.1 | 128.6 |
| DM | 待补充 | 待补充 | 待补充 | 90.1 | 117.6 |
| ELMM | 待补充 | 待补充 | 待补充 | 91.3 | 126.7 |

表 5 给出自适应注意力模块 AAM 的消融结果。与 SE 和 eSE 相比，AAM 在保持较高速度的同时进一步提升 mIoU。AAM 插入 2 次时取得 91.3% mIoU 和 126.7 FPS，是当前消融表中的最佳设置。该结果表明，在细节—语义互优化机制形成基础特征交互后，AAM 能够进一步强化多尺度、低占比和弱纹理缺陷相关通道响应，并抑制背景纹理带来的无效激活。

表 5 自适应注意力模块消融实验 / Table 5 Ablation study of the adaptive attention module。
表格格式：终稿按三线表排版，仅保留顶线、表头线和底线；Markdown 预览中以普通表格显示。

| 方法 | 数量 | Params/M | FLOPs/G | 模型大小/MB | mIoU/% | FPS |
|---|---:|---:|---:|---:|---:|---:|
| SE | 1 | 待补充 | 待补充 | 待补充 | 90.5 | 122.9 |
| eSE | 1 | 待补充 | 待补充 | 待补充 | 90.7 | 126.3 |
| AAM | 1 | 待补充 | 待补充 | 待补充 | 91.1 | 126.9 |
| AAM | 2 | 待补充 | 待补充 | 待补充 | 91.3 | 126.7 |

表 6 给出联合损失函数的消融结果。仅使用基础多阶段分割损失 \(l_0+l_1+l_2\) 时，模型 mIoU 为 90.4%；加入边缘细节损失后，mIoU 提升至 90.7%；加入掩码感知损失后，mIoU 为 90.8%；同时使用边缘细节损失和掩码感知损失时，mIoU 达到 91.3%。结果表明，边缘约束和前景掩码约束能够从不同角度增强模型对缺陷区域的学习。需要强调的是，边缘细节损失和掩码感知损失分别借鉴已有边缘监督与 mask-aware loss 思想，本文将二者联合引入 A2MS-DefectNet，是对工业缺陷边界细节和前景区域学习需求的任务适配。

表 6 联合损失函数消融实验 / Table 6 Ablation study of the joint loss function。
表格格式：终稿按三线表排版，仅保留顶线、表头线和底线；Markdown 预览中以普通表格显示。

| \(l_0+l_1+l_2\) | \(l_{\mathrm{edge}}\) | \(l_{\mathrm{mask}}\) | Params/M | FLOPs/G | 模型大小/MB | mIoU/% |
|---|---|---|---:|---:|---:|---:|
| √ |  |  | 待补充 | 待补充 | 待补充 | 90.4 |
| √ | √ |  | 待补充 | 待补充 | 待补充 | 90.7 |
| √ |  | √ | 待补充 | 待补充 | 待补充 | 90.8 |
| √ | √ | √ | 待补充 | 待补充 | 待补充 | 91.3 |

### 2.4 可视化分析

图 3 NEU-Seg 数据集可视化分割结果 / Fig. 3 Visualization results on NEU-Seg。图件要求：建议基于原图重新排版，终稿导出 300 dpi 以上位图或矢量图。

从可视化结果可以看出，A2MS-DefectNet 对带钢表面缺陷中的细长划痕、小面积夹杂物和不规则补丁具有较好的响应能力。对于细长缺陷，部分对比方法容易出现分割断裂或端部缺失，而本文方法能够更连续地保留长条状结构；对于小目标缺陷，本文方法能够减少漏检现象，使预测区域更接近真实标注；对于边界模糊区域，联合边缘监督有助于提升缺陷轮廓的完整性，降低预测边界过度扩张或收缩的问题。

图 4 皮革缺陷数据集可视化分割结果 / Fig. 4 Visualization results on the leather defect dataset。图件要求：建议基于原图重新排版，终稿导出 300 dpi 以上位图或矢量图。

皮革图像具有更强的天然纹理干扰，缺陷区域常与背景纹理相互交织。可视化结果表明，A2MS-DefectNet 能够在复杂纹理背景下保持较好的前景响应，对细小点状缺陷、划痕状缺陷、多类别混合缺陷和相邻边界模糊缺陷均具有较稳定的分割表现。与部分对比方法相比，本文方法在背景纹理较强时误分割区域更少，在缺陷边缘过渡不明显时能更好保持目标整体形状。

上述现象与本文结构设计具有一致性。细节—语义互优化机制为浅层空间信息和深层语义信息建立双向交互，使模型既能保留边界位置，又能抑制复杂背景纹理噪声；ELMM 在该机制中以轻量语义调制增强浅层细节中的有效缺陷响应；AAM 强化缺陷相关通道并抑制背景纹理干扰；边缘细节损失与掩码感知损失则分别从边界结构和前景区域两个角度约束模型训练。因此，A2MS-DefectNet 在小目标、细长缺陷、边界模糊和背景纹理干扰较强的场景下表现出更好的视觉分割质量。

### 2.5 复杂度与实时性分析

工业在线检测不仅关注分割精度，也关注模型复杂度和实时推理能力。表 7 汇总了部分模型的复杂度与速度结果。已有结果中包含 Base-S、Base-B 和 DDRNet23slim 的参数量，其余模型的 Params、FLOPs 和模型大小需进一步从原始实验记录或代码统计结果中补充。为避免引入未经核验的数据，表中未确认项统一标记为“待补充”。

表 7 复杂度与实时性对比 / Table 7 Comparison of model complexity and real-time performance。
表格格式：终稿按三线表排版，仅保留顶线、表头线和底线；Markdown 预览中以普通表格显示。

| Model | Backbone | Params/M | FLOPs/G | 模型大小/MB | NEU-Seg FPS | Leather FPS |
|---|---|---:|---:|---:|---:|---:|
| DDRNet23slim | - | 20.3 | 待补充 | 待补充 | 160.4 | 149.5 |
| Base-S | ResNet-18 | 14.0 | 待补充 | 待补充 | 178.5 | 167.2 |
| Base-B | ResNet-50 | 29.3 | 待补充 | 待补充 | 115.6 | 67.3 |
| A2MS-DefectNet-S | ResNet-18 | 待补充 | 待补充 | 待补充 | 196.5 | 188.7 |
| A2MS-DefectNet-B | ResNet-50 | 待补充 | 待补充 | 待补充 | 127.9 | 71.6 |

从推理速度看，A2MS-DefectNet-B 在 NEU-Seg 和皮革缺陷数据集上的 FPS 均高于 Base-B，说明围绕细节—语义互优化机制进行的轻量映射和注意力增强设计没有削弱实时性，反而改善了实际推理效率。A2MS-DefectNet-S 在两个数据集上均保持较高 FPS，适合对速度要求更高的在线检测场景。后续需补充 A2MS-DefectNet-S 和 A2MS-DefectNet-B 的 Params、FLOPs 和模型大小，以形成更完整的速度—精度—复杂度分析。

## 3 讨论

A2MS-DefectNet 的实验结果表明，面向工业缺陷分割任务的网络设计需要同时处理语义判别、空间细节和实时推理三类约束。与通用实时语义分割网络相比，工业缺陷图像中低占比前景、材料纹理干扰和模糊边界更为突出，单纯提高深层语义表达并不足以保证缺陷区域的完整预测。本文采用细节—语义互优化机制，使浅层细节和深层语义形成双向信息补偿，为工业缺陷实时分割提供基础特征交互能力。

ELMM 的作用主要体现在降低映射开销和保持语义调制能力。消融结果显示，轻量化映射不只是减少计算路径，也能够避免复杂映射过程对实时性造成额外负担。AAM 的作用主要体现在增强缺陷相关通道响应，尤其适用于多尺度、低占比和弱纹理缺陷区域。联合监督约束则从训练目标层面补充边界和前景信息，使模型在主分割损失之外获得更直接的缺陷轮廓和区域分布约束。

与近年工业缺陷分割方法相比，DMC-Net（作者待核验，2025）、CDARNet（作者待核验，2025）和 GCRANet（作者待核验，2026）等工作同样关注实时性、轻量化或背景抑制；TAG-Net（作者待核验，2025）、DASeg-Net（作者待核验，2025）和 LGGFormer（作者待核验，2025）则更多体现注意力和边界增强趋势。本文与这些工作的共同点在于均关注复杂工业纹理下的像素级缺陷定位，不同点在于本文围绕细节—语义互优化机制构建基础特征交互，并将轻量映射、自适应注意力和联合监督约束统一到一个实时分割框架中。SAID（作者待核验，2025）等提示式分割方法具有跨场景潜力，但其评估范式与全监督实时语义分割网络不同，当前更适合作为拓展方向而非主实验对比。

当前稿件仍存在若干需要补充和核验的内容。首先，A2MS-DefectNet-S/B 的 Params、FLOPs 和模型大小尚未完成统计，复杂度分析仍不完整。其次，NEU-Seg 的样本划分、图像尺寸、类别级 IoU、小目标或细长缺陷分组评价仍需从数据集说明或实验记录中补齐。再次，部分图件仍需按照期刊要求重绘或导出 300 dpi 以上版本。最后，正文引用中使用的近年文献还需逐条核验作者、题名、期刊或会议、DOI 与代码可用性，未核验条目不应进入正式参考文献列表。

## 4 结论

本文面向工业表面缺陷实时语义分割任务，提出以细节—语义互优化机制为核心的 A2MS-DefectNet。该方法利用浅层空间细节与深层语义信息的双向交互提升基础特征表达，并进一步通过 ELMM、AAM 和联合监督约束增强模型对多尺度、低占比和边界模糊缺陷的分割能力。实验结果表明，A2MS-DefectNet-B 在 NEU-Seg 数据集上达到 91.3% mIoU 和 127.9 FPS，在自建皮革缺陷数据集上达到 91.0% mIoU 和 71.6 FPS；与 Base-B 相比，A2MS-DefectNet-B 在 NEU-Seg 上 mIoU 提升 1.1 个百分点、FPS 提升 12.3，在皮革缺陷数据集上 mIoU 提升 1.8 个百分点、FPS 提升 4.3。消融实验进一步验证了 ELMM、AAM 和联合监督约束在细节—语义互优化基础结构上的增益作用。可视化分析显示，本文方法在小目标、细长缺陷、边界模糊和背景纹理干扰场景下具有较好的分割稳定性。

后续研究将进一步补充 A2MS-DefectNet 的 Params、FLOPs 和模型大小等复杂度统计，开展跨设备实时性测试，并在更多工业材料和生产场景中验证模型泛化能力，以提升方法评估的完整性和可复现性。

## 参考文献

正式参考文献条目待逐条核验作者、题名、来源、卷期页码和 DOI 后，按《中国图象图形学报》格式补齐。未核验作者、题名、来源或 DOI 的文献均继续保留“待核验”状态；正文引用采用作者—年份形式。
