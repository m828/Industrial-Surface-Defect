# 参考文献核验报告（CJIG v3）

核验日期：2026-09-24。方式：逐条联网检索（出版方页面/DOI/arXiv）。18 条全部检索到，无"不存在"条目；多条需要修正细节。

## A. 逐条核验结果

| # | 原引用（v2 正文） | 核验结果 | 修正意见 |
|---|---|---|---|
| 1 | BiSeNetV1（作者待核验，2018） | ✅ BiSeNet: Bilateral Segmentation Network for Real-Time Semantic Segmentation. Yu C, Wang J, Peng C, et al. ECCV 2018, LNCS 11217, pp. 334–349（Springer 官方页码；部分文献误作 325–341）. DOI: 10.1007/978-3-030-01261-8_20 | 保留，补全作者与页码 |
| 2 | BiSeNetV2（作者待核验，2020） | ⚠️ BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation. Yu C, **Gao C**, Wang J, et al. IJCV 129(11):3051–3068, **2021**（arXiv 2020）. DOI: 10.1007/s11263-021-01515-2 | 年份改 2021；若精简可删（被 DDRNet/PIDNet 覆盖） |
| 3 | DDRNet（作者待核验，2021） | ⚠️ 期刊版标题为 Deep Dual-Resolution Networks for Real-Time and Accurate Semantic Segmentation of **Traffic Scenes**. Hong Y, Pan H, Sun W, Jia Y. IEEE T-ITS 24(3):3448–3460, **2023**. DOI: 10.1109/TITS.2022.3228042 | 标题与年份按期刊版更正 |
| 4 | STDCNet（作者待核验，2021） | ✅ Rethinking BiSeNet For Real-time Semantic Segmentation. Fan M, Lai S, Huang J, et al. CVPR 2021:9716–9725. DOI: 10.1109/CVPR46437.2021.00959 | 保留（细节监督思想来源） |
| 5 | PP-LiteSeg（作者待核验，2022） | ✅ PP-LiteSeg: A Superior Real-Time Semantic Segmentation Model. Peng J, Liu Y, Tang S, et al. **仅 arXiv 预印本** arXiv:2204.02681（2022） | 保留，标注 arXiv preprint（UAFM 出处——本文 arm1/arm2 即 UAFM，建议引用） |
| 6 | PIDNet（作者待核验，2023） | ✅ PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers. Xu J, Xiong Z, Bhattacharyya SP. CVPR 2023:19529–19539. DOI: 10.1109/CVPR52729.2023.01871 | 保留 |
| 7 | LETNet（作者待核验，2023） | ⚠️ 正确标题：Lightweight Real-Time Semantic Segmentation Network With Efficient Transformer and CNN. Xu G, Li J, Gao G, et al. IEEE T-ITS 24(12), 2023. DOI: 10.1109/TITS.2023.3248089 | 若保留须改正标题；与 SeaFormer++ 定位重叠，建议删 |
| 8 | SeaFormer（作者待核验，2023） | ✅ SeaFormer: Squeeze-enhanced Axial Transformer for Mobile Semantic Segmentation. Wan Q, Huang Z, Lu J, et al. ICLR 2023. arXiv:2301.13156 | 可由 SeaFormer++ 一条覆盖 |
| 9 | SeaFormer++（作者待核验，2025） | ✅ SeaFormer++: Squeeze-enhanced Axial Transformer for Mobile **Visual Recognition**. Wan Q, et al. IJCV 133(6):3645–3666, 2025. DOI: 10.1007/s11263-025-02345-2 | 保留（覆盖 SeaFormer） |
| 10 | DMC-Net（作者待核验，2025） | ✅ DMC-Net: a lightweight network for real-time surface defect segmentation. Zuo H, et al. Journal of Real-Time Image Processing 22(2), 2025. DOI: 10.1007/s11554-025-01639-5（NEU-SEG 上 73.74% mIoU、211.7 FPS） | 保留——与本文定位最贴近的 2025 实时缺陷分割工作 |
| 11 | SPCS-Net（作者待核验，2025） | ⚠️ 论文标题为 A high-precision segmentation network for industrial surface defect detection（SPCS-Net 为文中网络名）. Chen H, Min B-W. AIP Advances 15(5):055118, 2025. DOI: 10.1063/5.0274903 | 期刊层级较弱，建议优先删；若保留须用真实论文标题 |
| 12 | TAG-Net（作者待核验，2025） | ✅ TAG-Net: Triple Attention Guided Network for Inspecting Surface Defects on Steel Products. Jeong S, Song J, Lee SJ. IJCAS 23, 2025. DOI: 10.1007/s12555-024-0550-8 | 保留（钢材缺陷三重注意力：背景/缺陷/边界） |
| 13 | GCRANet（作者待核验，2026） | ⚠️ 正确标题：Global Context Guided Refinement and Aggregation Network for Lightweight Surface Defect Detection. Yan F, Jiang X, et al. Pattern Recognition, 112893, **2025**（非 2026）. DOI: 10.1016/j.patcog.2025.112893 | 年份改 2025；描述按正式标题改写（非"弱缺陷轮廓增强"） |
| 14 | CDARNet（作者待核验，2025） | ✅ CDARNet: A robust cross-dimensional adaptive region reconstruction network for real-time metal surface defect segmentation. Li Q, et al. Advanced Engineering Informatics 67:103514, 2025. DOI: 10.1016/j.aei.2025.103514 | 保留 |
| 15 | LGGFormer（作者待核验，2025） | ⚠️ 正确标题：LGGFormer: A Dual-Branch Local-Guided Global Self-Attention Network for Surface Defect Segmentation. Zhang G, Lu Y, Jiang X, et al. AEI 64:103099, 2025. DOI: 10.1016/j.aei.2024.103099 | 保留；描述改为"双分支局部引导全局自注意力"（非"边缘引导解码"） |
| 16 | DASeg-Net（作者待核验，2025） | ✅ 论文标题：The end-to-end chip surface defect segmentation method based on the diffusion model and attention mechanism. Xia Z, et al. EAAI, 111131, 2025. DOI: 10.1016/j.engappai.2025.111131 | 与"实时"主线有张力，建议删或一句带过 |
| 17 | SAID（作者待核验，2025） | ✅ SAID: Segment All Industrial Defects with Scene Prompts. Huang Y, Zhu J, Zhong X, Deng Y. Sensors 25(16):4929, 2025. DOI: 10.3390/s25164929 | 提示式/SAM 路线，与全监督实时主线不同，建议删或作为展望一句 |
| 18 | Sub-region UNet（作者待核验，2023） | ✅ A sub-region Unet for weak defects segmentation with global information and mask-aware loss. Zhu W, Liang R, Yang J, et al. **Engineering Applications of Artificial Intelligence** 122:106011, 2023. DOI: 10.1016/j.engappai.2023.106011 | 保留（掩码感知损失思想来源；注意勿误写为 Expert Systems with Applications） |

## B. 建议参考文献清单（精简后 11 篇 + 基础引用）

实时语义分割（6 篇）：BiSeNet (ECCV 2018)、STDCNet (CVPR 2021)、DDRNet (T-ITS 2023)、PP-LiteSeg (arXiv 2022)、PIDNet (CVPR 2023)、SeaFormer++ (IJCV 2025)。
工业缺陷分割（5 篇）：Sub-region UNet (EAAI 2023)、DMC-Net (JRTIP 2025)、CDARNet (AEI 2025)、LGGFormer (AEI 2025)、TAG-Net (IJCAS 2025)。
建议删除：BiSeNetV2（可并入 BiSeNet 一句）、LETNet、SPCS-Net、DASeg-Net、SAID、GCRANet（如需 Pattern Recognition 层级文献可保留并改正描述）。
另需补充基础引用（v1 未列但正文实际依赖）：ResNet（He 等，CVPR 2016，主干）；OHEM（Shrivastava 等，CVPR 2016，若正文提到 OHEM）；HSigmoid/MobileNetV3（Howard 等，ICCV 2019，eSE 激活出处，可选）。

## C. 正文引用改动

- 删除正文中被删文献的引用句；保留文献的"作者，年份"按上表更正（DDRNet→2023、GCRANet→2025 或删、LGGFormer 描述更正等）。
- 1.2/1.5 节中 UAFM、DAPPM、细节监督等模块出处建议标注 PP-LiteSeg / DDRNet / STDCNet（均为已核验真实文献），避免"提出他人模块"的归属风险——正文表述应保持"借鉴/参考"而非"提出"。
