# 近三年工业缺陷分割与轻量实时语义分割文献调研

面向《中国图象图形学报》投稿，本文献表仅围绕“工业表面缺陷实时语义分割”和 A2MS-DefectNet 主线整理。表中“是否有代码”只记录已核验到的论文页或官方仓库信息；没有核到官方仓库时写“未检索到官方代码”，不以第三方复现仓库替代。

模板附件提示：投稿稿件应采用中英文题名、摘要、关键词结构；英文摘要宜按 Objective、Method、Result、Conclusion 组织；图件需保证清晰度，模板示例出现 300 dpi 要求；参考文献需给出作者、年份、题名、来源、页码和 DOI/URL 等可追溯信息。

## 候选方法总表

| 类别 | 方法 | 年份 | 任务类型 | 是否工业缺陷 | 是否实时/轻量 | 是否像素级分割 | 使用数据集 | 是否有代码 | 与本文关系 | 建议用途 |
|---|---|---:|---|---|---|---|---|---|---|---|
| 工业缺陷分割 | [DMC-Net](https://link.springer.com/article/10.1007/s11554-025-01639-5) | 2025 | 表面缺陷语义分割 | 是 | 是 | 是 | NEU-SEG | 有，论文页给出 [GitHub](https://github.com/Michaelzyb/DMC-Net.git) | 轻量实时缺陷分割，任务与本文高度一致，可补强近年工业缺陷对比 | 加入实验对比 |
| 工业缺陷分割 | [SPCS-Net](https://ouci.dntb.gov.ua/en/works/lxLL18wV/) | 2025 | 工业表面缺陷分割 | 是 | 部分涉及效率 | 是 | NEU-Seg、MBP-Seg、USB-Seg | 未检索到官方代码 | 空间位置注意与跨尺度融合，适合写入近年高精度缺陷分割 | 加入相关工作 |
| 工业缺陷分割 | [GCRANet](https://www.sciencedirect.com/science/article/pii/S0031320325015560) | 2026 | 像素级表面缺陷检测/分割 | 是 | 是 | 是 | SD-saliency-900、Magnetic tile、DAGM 2007、CrackSeg9k、MVTec AD | 未检索到官方代码 | 全局上下文、轻量注意力、背景抑制，与本文细节—语义互优化和轻量化主线相关 | 加入相关工作 |
| 工业缺陷分割 | [TAG-Net](https://pure.jbnu.ac.kr/en/publications/tag-net-triple-attention-guided-network-for-inspecting-surface-de/) | 2025 | 钢材表面缺陷语义分割 | 是 | 未突出轻量 | 是 | NEU-Seg | 未检索到官方代码 | 三重注意力显式建模背景、缺陷和边界，适合支撑边界/注意力相关工作 | 加入相关工作 |
| 工业缺陷分割 | [DASeg-Net](https://www.sciencedirect.com/science/article/pii/S0952197625011327) | 2025 | 芯片表面缺陷分割 | 是 | 否，扩散模型成本较高 | 是 | 芯片缺陷分割及其他表面缺陷数据集 | 未检索到官方代码 | 扩散模型与 HiLo 注意力、边缘辅助训练，可作为注意力/边界增强背景 | 加入相关工作 |
| 工业缺陷分割 | [SAID](https://www.mdpi.com/1424-8220/25/16/4929) | 2025 | 工业缺陷提示式分割 | 是 | 否，偏基础模型/提示学习 | 是 | Industrial-5i | 未检索到官方代码 | 面向跨场景工业缺陷分割，但范式是提示式/少标注，不宜直接作为常规实时网络主对比 | 仅作为背景 |
| 工业缺陷分割 | [LGGFormer](https://www.sciencedirect.com/science/article/pii/S147403462400750X) | 2025 | 表面缺陷分割 | 是 | 部分涉及效率 | 是 | 3 个公开缺陷数据集 | 未检索到官方代码 | 局部引导全局自注意力、双分支和边缘引导解码，适合支撑全局/局部与边界建模 | 加入相关工作 |
| 工业缺陷分割 | [CDARNet](https://www.sciencedirect.com/science/article/pii/S1474034625004070) | 2025 | 金属表面缺陷实时分割 | 是 | 是 | 是 | NEU-Seg 等金属缺陷数据集 | 未检索到官方代码 | 明确实时金属缺陷分割，并给出 NEU-Seg 训练/测试划分说明；若代码可得，实验价值高 | 加入相关工作 |
| 工业缺陷分割 | [TSEDNet](https://www.sciencedirect.com/science/article/pii/S026322412401323X) | 2025 | 带钢表面缺陷像素级分割 | 是 | 未突出轻量 | 是 | X-SDD、SD-saliency-900 | 未检索到官方代码 | 面向类别不平衡和任务特异解码，适合补充工业场景复杂性 | 加入相关工作 |
| 工业缺陷分割 | [DDSNet](https://www.researchgate.net/publication/382293246_DDSNet_Deep_Dual-branch_Networks_for_Surface_Defect_Segmentation) | 2024 | 表面缺陷语义分割 | 是 | 未核验 | 是 | 公开表面缺陷数据集，细节待核验 | 未检索到官方代码 | 双分支缺陷分割，可作为近年工业分割补充；需进一步核对 IEEE TIM 正式页 | 加入相关工作 |
| 工业缺陷分割 | [Sub-region UNet](https://www.sciencedirect.com/science/article/pii/S0952197623001951) | 2023 | 弱缺陷分割 | 是 | 否 | 是 | 金属表面弱缺陷数据集 | 未检索到官方代码 | mask-aware loss 来源之一，已与本文联合监督约束直接相关 | 加入实验对比 |
| 工业缺陷分割 | [MLR-Net](https://www.sciencedirect.com/science/article/pii/S0952197623011910) | 2023 | 皮革缺陷分割 | 是 | 未突出实时 | 是 | 皮革缺陷数据集 | 未检索到官方代码 | 与本文自建皮革缺陷数据集任务接近，但数据集和类别需对齐 | 加入相关工作 |
| 小目标/边界增强 | [Residual Shape Adaptive Dense-nested UNet](https://www.sciencedirect.com/science/article/pii/S0031320323007707) | 2024 | 金属表面微小缺陷像素级检查 | 是 | 有剪枝加速设计 | 是 | 金属表面微小缺陷数据集 | 未检索到官方代码 | 小目标、形状自适应和剪枝加速，可支撑小缺陷/边界分析 | 加入相关工作 |
| 轻量实时语义分割 | [PIDNet](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html) | 2023 | 实时语义分割 | 否 | 是 | 是 | Cityscapes、CamVid | 有，[GitHub](https://github.com/XuJiacong/PIDNet) | 三分支细节/上下文/边界融合，与本文边界和实时性分析高度相关 | 加入实验对比 |
| 轻量实时语义分割 | [LETNet](https://arxiv.org/abs/2302.10484) | 2023 | 轻量实时语义分割 | 否 | 是 | 是 | Cityscapes、CamVid | 有，[GitHub](https://github.com/IVIPLab/LETNet) | Transformer+CNN 轻量实时网络，可作为近年通用实时分割补充对比 | 加入实验对比 |
| 轻量实时语义分割 | [SeaFormer](https://iclr.cc/virtual/2023/poster/11994) | 2023 | 移动端语义分割 | 否 | 是 | 是 | ADE20K、Pascal Context、COCO-Stuff、Cityscapes | 有，[GitHub](https://github.com/fudan-zvg/SeaFormer) | 移动端轴向注意力网络，可作为轻量注意力背景或可选复现 | 加入实验对比 |
| 轻量实时语义分割 | [SeaFormer++](https://github.com/fudan-zvg/SeaFormer) | 2025 | 移动视觉识别/语义分割骨干扩展 | 否 | 是 | 是 | ImageNet、ADE20K、Cityscapes 等 | 有，[GitHub](https://github.com/fudan-zvg/SeaFormer) | IJCV 版本扩展 SeaFormer，适合作为近年轻量骨干背景 | 仅作为背景 |
| 轻量实时语义分割 | [PP-LiteSeg](https://arxiv.org/abs/2204.02681) | 2022 | 实时语义分割 | 否 | 是 | 是 | Cityscapes | 有，[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) | 已在现有结果表中作为实时分割对比，建议保留 | 加入实验对比 |
| 轻量实时语义分割 | [STDCNet/STDC-Seg](https://openaccess.thecvf.com/content/CVPR2021/html/Fan_Rethinking_BiSeNet_for_Real-Time_Semantic_Segmentation_CVPR_2021_paper.html) | 2021 | 实时语义分割 | 否 | 是 | 是 | Cityscapes、CamVid | 有，[GitHub](https://github.com/MichaelFan01/STDC-Seg) | 已在现有结果表中对比，且边缘监督思想为本文联合损失提供来源 | 加入实验对比 |
| 轻量实时语义分割 | [DDRNet](https://arxiv.org/abs/2101.06085) | 2021 | 实时语义分割 | 否 | 是 | 是 | Cityscapes、CamVid | 有，[GitHub](https://github.com/ydhongHIT/DDRNet) | 已在现有结果表中对比，适合作为双分辨率实时网络代表 | 加入实验对比 |
| 轻量实时语义分割 | [BiSeNetV1](https://openaccess.thecvf.com/content_ECCV_2018/papers/Changqian_Yu_BiSeNet_Bilateral_Segmentation_ECCV_2018_paper.pdf) | 2018 | 实时语义分割 | 否 | 是 | 是 | Cityscapes、CamVid、COCO-Stuff | 有，官方/复现代码需在最终参考文献前核验 | 已在现有结果表中对比，是双分支实时分割经典基线 | 加入实验对比 |
| 轻量实时语义分割 | [BiSeNetV2](https://arxiv.org/abs/2004.02147) | 2020 | 实时语义分割 | 否 | 是 | 是 | Cityscapes | 有，论文页给出代码短链，最终需核验可访问性 | 已在现有结果表中对比，是细节/语义双分支思想的重要背景 | 加入实验对比 |

## 不建议进入主对比的边界

- 仅做目标检测、分类、异常检测且不输出像素级掩码的方法，不进入主实验对比；可在引言或相关工作中作为工业检测背景一笔带过。
- DFFNet 等基于 NEU-DET、mAP、边界框定位的轻量检测方法虽然与钢材缺陷相关，但不是语义分割任务，不建议放入 A2MS-DefectNet 主对比表。
- SAID 属于提示式/跨场景分割范式，任务是像素级分割，但评估协议和常规全监督实时语义分割不同，建议作为背景或拓展讨论，不进入主实验。

## 已核验来源索引

- DMC-Net: Journal of Real-Time Image Processing, 2025, DOI: 10.1007/s11554-025-01639-5.
- SPCS-Net: AIP Advances, 2025, DOI: 10.1063/5.0274903.
- GCRANet: Pattern Recognition, 2026, DOI: 10.1016/j.patcog.2025.112893.
- TAG-Net: International Journal of Control, Automation and Systems, 2025, DOI: 10.1007/s12555-024-0550-8.
- DASeg-Net: Engineering Applications of Artificial Intelligence, 2025, DOI: 10.1016/j.engappai.2025.111131.
- SAID: Sensors, 2025, DOI: 10.3390/s25164929.
- LGGFormer: Advanced Engineering Informatics, 2025, DOI: 10.1016/j.aei.2024.103099.
- CDARNet: Advanced Engineering Informatics, 2025, DOI: 10.1016/j.aei.2025.103514.
- TSEDNet: Measurement, 2025, DOI: 10.1016/j.measurement.2024.115438.
- Sub-region UNet: Engineering Applications of Artificial Intelligence, 2023, DOI: 10.1016/j.engappai.2023.106011.
- MLR-Net: Engineering Applications of Artificial Intelligence, 2023, DOI: 10.1016/j.engappai.2023.107007.
- PIDNet: CVPR 2023, DOI: 10.1109/CVPR52729.2023.01871.
- SeaFormer: ICLR 2023; SeaFormer++: IJCV 2025, official repository lists both entries.
- LETNet: IEEE Transactions on Intelligent Transportation Systems, arXiv:2302.10484.
- PP-LiteSeg: arXiv:2204.02681, PaddleSeg.
- STDCNet/STDC-Seg: CVPR 2021.
- DDRNet: arXiv:2101.06085.
- BiSeNetV1: ECCV 2018; BiSeNetV2: arXiv:2004.02147.
