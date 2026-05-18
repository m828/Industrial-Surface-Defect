# reference_check_table

本表根据 `recent_work_survey.md` 逐条整理。字段“是否已核验”仅表示当前文件中是否已有可追溯来源、DOI、arXiv 或正式论文页线索；作者、题名、期刊/会议和 DOI 任一项未完整确认时，仍保留“待核验”。本表不新增未核验参考文献信息。

| 方法 | 作者 | 年份 | 题名 | 期刊/会议 | DOI | 是否已核验 | 是否进入正文 | 是否进入实验 |
|---|---|---:|---|---|---|---|---|---|
| DMC-Net | 待核验 | 2025 | DMC-Net: a lightweight network for real-time surface defect segmentation | Journal of Real-Time Image Processing | 10.1007/s11554-025-01639-5 | 来源和 DOI 已核验，作者待核验 | 是 | 否，建议后续复现后进入实验 |
| SPCS-Net | 待核验 | 2025 | SPCS-Net: high-precision segmentation network for industrial surface defect segmentation | AIP Advances | 10.1063/5.0274903 | 来源和 DOI 已核验，作者待核验 | 是 | 否，当前无官方代码和复现实验 |
| GCRANet | 待核验 | 2026 | Global Context Guided Refinement and Aggregation Network，完整题名待核验 | Pattern Recognition | 10.1016/j.patcog.2025.112893 | 来源和 DOI 已核验，作者待核验 | 是 | 否，当前未进入结果表 |
| TAG-Net | 待核验 | 2025 | TAG-Net: Triple Attention Guided Network for inspecting surface defects | International Journal of Control, Automation and Systems | 10.1007/s12555-024-0550-8 | 来源和 DOI 已核验，作者待核验 | 是 | 否，当前未进入结果表 |
| DASeg-Net | 待核验 | 2025 | DASeg-Net: diffusion model and attention mechanism for chip surface defect segmentation，完整题名待核验 | Engineering Applications of Artificial Intelligence | 10.1016/j.engappai.2025.111131 | 来源和 DOI 已核验，作者待核验 | 是 | 否，扩散式方法成本和协议与本文不同 |
| SAID | 待核验 | 2025 | Segment All Industrial Defects with Scene Prompts | Sensors | 10.3390/s25164929 | 来源和 DOI 已核验，作者待核验 | 是 | 否，仅作为提示式分割背景 |
| LGGFormer | 待核验 | 2025 | LGGFormer，完整题名待核验 | Advanced Engineering Informatics | 10.1016/j.aei.2024.103099 | 来源和 DOI 已核验，作者待核验 | 是 | 否，当前未进入结果表 |
| CDARNet | 待核验 | 2025 | CDARNet，完整题名待核验 | Advanced Engineering Informatics | 10.1016/j.aei.2025.103514 | 来源和 DOI 已核验，作者待核验 | 是 | 否，当前未进入结果表 |
| TSEDNet | 待核验 | 2025 | TSEDNet，完整题名待核验 | Measurement | 10.1016/j.measurement.2024.115438 | 来源和 DOI 已核验，作者待核验 | 是 | 否，当前未进入结果表 |
| DDSNet | 待核验 | 2024 | DDSNet: Deep Dual-branch Networks for Surface Defect Segmentation | 待核验 | 待核验 | 待核验，仅有非正式线索 | 否 | 否 |
| Sub-region UNet | 待核验 | 2023 | 题名待核验 | Engineering Applications of Artificial Intelligence | 10.1016/j.engappai.2023.106011 | 来源和 DOI 已核验，作者与题名待核验 | 是 | 是，已进入现有 NEU-Seg 和皮革实验表 |
| MLR-Net | 待核验 | 2023 | 题名待核验 | Engineering Applications of Artificial Intelligence | 10.1016/j.engappai.2023.107007 | 来源和 DOI 已核验，作者与题名待核验 | 否 | 否，皮革数据集和类别需对齐 |
| Residual Shape Adaptive Dense-nested UNet | 待核验 | 2024 | Residual Shape Adaptive Dense-nested UNet，完整题名待核验 | Pattern Recognition，待核验 | 待核验 | 待核验 | 否 | 否 |
| PIDNet | 待核验 | 2023 | PIDNet: A Real-Time Semantic Segmentation Network Inspired by PID Controllers | CVPR 2023 | 10.1109/CVPR52729.2023.01871 | 来源和 DOI 已核验，作者待核验 | 是 | 否，建议后续复现后进入实验 |
| LETNet | 待核验 | 2023 | LETNet，完整题名待核验 | IEEE Transactions on Intelligent Transportation Systems / arXiv，待核验 | arXiv:2302.10484；DOI 待核验 | 来源线索已核验，正式条目待核验 | 是 | 否，建议后续复现后进入实验 |
| SeaFormer | 待核验 | 2023 | SeaFormer: Squeeze-enhanced Axial Transformer for Mobile Semantic Segmentation，待核验 | ICLR 2023 | 待核验 | 来源线索已核验，正式条目待核验 | 是 | 否，建议后续复现后进入实验 |
| SeaFormer++ | 待核验 | 2025 | SeaFormer++，完整题名待核验 | IJCV 2025，待核验 | 待核验 | 来源线索已核验，正式条目待核验 | 是 | 否，仅作为轻量骨干背景 |
| PP-LiteSeg | 待核验 | 2022 | PP-LiteSeg: A Superior Real-Time Semantic Segmentation Model | arXiv / PaddleSeg，正式来源待核验 | arXiv:2204.02681；DOI 待核验 | 来源线索已核验，正式条目待核验 | 是 | 是，已进入现有 NEU-Seg 和皮革实验表 |
| STDCNet/STDC-Seg | 待核验 | 2021 | Rethinking BiSeNet for Real-Time Semantic Segmentation | CVPR 2021 | 待核验 | 来源已核验，DOI 和作者待核验 | 是 | 是，已进入现有 NEU-Seg 和皮革实验表；同时作为边缘监督来源 |
| DDRNet | 待核验 | 2021 | Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes | arXiv，正式来源待核验 | arXiv:2101.06085；DOI 待核验 | 来源线索已核验，正式条目待核验 | 是 | 是，已进入现有 NEU-Seg 和皮革实验表 |
| BiSeNetV1 | 待核验 | 2018 | BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation | ECCV 2018 | 待核验 | 来源已核验，DOI 和作者待核验 | 是 | 是，已进入现有 NEU-Seg 和皮革实验表 |
| BiSeNetV2 | 待核验 | 2020 | BiSeNetV2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation，待核验 | arXiv，正式来源待核验 | arXiv:2004.02147；DOI 待核验 | 来源线索已核验，正式条目待核验 | 是 | 是，已进入现有 NEU-Seg 和皮革实验表 |

## 后续核验建议

1. 对“作者待核验”的条目，应从 DOI 页面、会议官网、期刊官网或作者正式预印本中补齐作者列表。
2. 对“题名待核验”的条目，不应直接进入正式参考文献列表；需先核对英文题名、卷期页码和 DOI。
3. 对 DMC-Net、PIDNet、LETNet、SeaFormer 等建议实验对比方法，只有在完成同一数据划分、同一评价指标和同一硬件/软件环境复现后，才可进入主实验表。
4. 对 SAID、DASeg-Net 等范式差异较大的方法，当前建议只进入相关工作或讨论，不进入主实验对比。
