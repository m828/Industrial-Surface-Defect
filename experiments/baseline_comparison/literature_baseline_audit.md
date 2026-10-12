# NEU-Seg 文献 baseline 审计（CJIG 投稿前领域覆盖补充）

日期：2026-10-12
性质：**纯文献核验**，本轮未训练任何模型、未修改任何冻结结果、未修改论文正文。
目的：回应"缺少工业缺陷领域方法比较"的潜在审稿意见——盘点 NEU-Seg 上已有公开方法的报告结果与协议，论证哪些可比较、哪些只能进相关工作。

> 铁律：**文献报告值 ≠ 本文统一复现值**。本文 10 模型对比表（`results/final_baseline_table_v3.md`）只含统一协议复现结果；本文件中的所有文献报告值**禁止混入该表**。

本文统一协议（参照系）：train 3630 / test 840、4 类含 background、200×200、from scratch、seed 1337、Adam lr1e-4、batch 16、240k 固定终点、无 test 选择、mIoU 含 background。

---

## 1. 逐方法审计

### 1.1 Sub-region U-Net（重点：与本文存在谱系关系）

| 字段 | 内容 |
|---|---|
| Paper | A sub-region Unet for weak defects segmentation with global information and mask-aware loss |
| 作者/年份/来源 | W. Zhu, R. Liang, J. Yang, Y. Cao, G. Fu, Y. Cao；2023；Engineering Applications of Artificial Intelligence, vol. 122, 106011 |
| 模型类别 | 工业缺陷专用分割（金属表面弱缺陷；sub-region u 形结构 + 全局辅助信息 + mask-aware loss） |
| NEU-Seg split | 论文实验使用 NEU-Seg 与 USB-Seg（摘要页已核验）；具体划分与图像数量未能在线核验原文表格 |
| 指标定义 | mIoU（原文 Table 数值未能在线核验，摘要页无数据） |
| 报告结果 | **无法在线核验**（ScienceDirect 仅摘要开放） |
| 是否可直接比较 | **否**——split/指标口径未核验 |
| 是否建议进 Table X | **否** |

**特别标记（披露相关，事实）**：本项目代码谱系与该文存在直接历史渊源——服务器历史目录 `subregion unet/` 即本文 DSMONet 的历史开发入口（含 mask-aware loss、subregion 机制的历史实现）；本文稿件已按纪律将 mask-aware loss 表述为已有方法（非原创 loss）。因此 Sub-region U-Net **不是独立第三方 baseline**，其报告值不进对比表；与本文的方法重叠由 `docs/paper/prior_publication_overlap_audit_v6.md` 系列文件负责披露，不在本文件展开。

### 1.2 FDSNet（已进入统一复现表的唯一工业方法）

| 字段 | 内容 |
|---|---|
| Paper | FDSNet: An Accurate Real-Time Surface Defect Segmentation Network |
| 年份/来源 | 2022；ICASSP（官方代码 github.com/jianzhang96/fdsnet，完整可运行） |
| 模型类别 | 工业缺陷专用分割（Fast-SCNN 式轻量骨干 + 边界/语义辅助任务） |
| NEU-Seg split | 官方将 NEU-Seg 转 PASCAL VOC 格式，trainval-test 划分（与本文 3630/840 不同） |
| 指标定义 | mIoU（VOC 口径，含 background） |
| 报告结果 | 官方 README：NEU-Seg mIoU **78.8**、FPS 186.1（GTX 1080Ti，其自有训练协议：150 epochs、batch 8、OHEM） |
| 是否可直接比较 | **否**——split、训练预算、评价口径均不同 |
| 是否建议进 Table X | **已纳入**——但以本文统一协议复现值 **87.08**（240k 固定终点）入表；文献 78.8 仅作本文件的参考记录，不入表 |

**特别标记**：统一复现值 87.08 高于官方报告值 78.8。差异来源为协议不同（本文 3630/840 扩充划分 + 240k 训练预算 vs 官方 trainval/test + 150 epochs），不构成"超过原方法"的主张；论文中只引用统一协议下的复现值。

### 1.3 PGA-Net

| 字段 | 内容 |
|---|---|
| Paper | PGA-Net: Pyramid Feature Fusion and Global Context Attention Network for Automated Surface Defect Detection |
| 作者/年份/来源 | H. Dong, K. Song, Y. He, J. Xu, Y. Yan, Q. Meng；2020；IEEE Transactions on Industrial Informatics, 16(12): 7448–7458 |
| 模型类别 | 工业缺陷专用分割（金字塔特征融合 + 全局上下文注意力） |
| NEU-Seg split | 论文在 NEU-Seg 上有实验（原文 Table III），具体划分未能在线核验（原文 PDF 受限） |
| 指标定义 | mIoU（原文表格未在线核验） |
| 报告结果 | **无法在线核验**；未检索到官方代码（2026-10-12 复核） |
| 是否可直接比较 | **否** |
| 是否建议进 Table X | **否**（无官方代码，无法可核验复现；列入相关工作引用） |

### 1.4 A-Net（文献协议差异的典型例子）

| 字段 | 内容 |
|---|---|
| Paper | A-Net: An A-shape Lightweight Neural Network for Real-time Surface Defect Segmentation |
| 作者/年份/来源 | B. Chen, T. Niu, W. Yu 等；2023；IEEE Transactions on Instrumentation and Measurement（论文称代码公开） |
| 模型类别 | 轻量工业缺陷分割 |
| NEU-Seg split | **与本文不同**：自选 inclusion/patches/**scratches** 三类（非 crazing），resize 224×224，每类 250 训练 / 50 测试，另划 15% 验证，2000 epochs |
| 指标定义 | 逐缺陷类 IoU（**不含 background 的 4 类 mIoU 口径**） |
| 报告结果 | A-Net：inclusion 82.86 / patches 75.99 / scratches 88.03（IoU%），0.39M 参数；同协议下 U-Net 约 77.22（按摘要 -0.21% 反推） |
| 是否可直接比较 | **否**——类别集合（scratches 替代 crazing）、划分、输入尺寸、指标口径全部不同 |
| 是否建议进 Table X | **否** |

### 1.5 DSGNet（代码公开的近年方法）

| 字段 | 内容 |
|---|---|
| Paper | DSGNet: A Lightweight Network Integrating Depthwise Separable and Ghost Convolutions for Real-Time Surface Defect Segmentation |
| 作者/年份/来源 | H. Lu 等；2025；Expert Systems（Wiley），DOI 10.1111/exsy.70187；代码公开（GitHub: young-zyy/DSGNet） |
| 模型类别 | 轻量工业缺陷分割 |
| NEU-Seg split | README 未明确划分细节（报告 NEU-Seg/MSD/MT 三数据集结果） |
| 指标定义 | mIoU（口径未核验是否含 background） |
| 报告结果 | NEU-Seg mIoU **83.39**，0.49M 参数，0.29G FLOPs（官方 README） |
| 是否可直接比较 | **否**——协议未核验一致 |
| 是否建议进 Table X | 本轮**否**（任务规定不新增训练）；代码公开，列为"未来可复现候选" |

### 1.6 AASFormer（与本文同一 3630/840 划分的文献）

| 字段 | 内容 |
|---|---|
| Paper | AASFormer: Adaptive Axial Squeeze Transformer Network for Industrial Surface Defect Segmentation |
| 作者/年份/来源 | H. Yang, D. Zhang 等；Big Data Mining and Analytics（2026 在线），DOI 10.26599/BDMA.2025.9020095 |
| 模型类别 | CNN-Transformer 混合工业缺陷分割 |
| NEU-Seg split | **3630 训练 / 840 验证、200×200（与本文同一扩充版 NEU-Seg 划分，原文 §4.1 明确写出）** |
| 指标定义 | MIoU；但其网络输出为 Sigmoid 单通道（缺陷/背景二分类式），MIoU 是否为本文 4 类含背景口径**无法从公开文本确认** |
| 报告结果 | NEU-Seg MIoU **84.85%**（其 Table 1；同表 SegMAN 85.11%） |
| 是否可直接比较 | **否**——尽管划分相同，指标类别口径未核验一致（疑似二类口径），且表述为 validation 而非 held-out test；未检索到官方代码 |
| 是否建议进 Table X | **否**；可在相关工作/讨论中以"文献报告值"提及并注明口径差异 |

**附带发现（事实）**：3630/840 这一扩充版 NEU-Seg 划分并非本文自创，近年文献（如 AASFormer）已在使用；本文协议的特殊性在于 4 类含背景 mIoU + 240k 固定终点 + 无 test 选择的统一训练，而非划分本身。

### 1.7 其余近年工业方法（沿用既有审计结论）

DMC-Net（官方仓自我断链）、TAG-Net / CDARNet / LGGFormer / SPCS-Net / GCRANet（均无官方代码）——逐项证据见 `experiments/baseline_comparison/recent_industrial_methods_audit.md`（2026-10-10），结论不变：不具备可核验的公平复现条件，仅进相关工作。

---

## 2. 汇总表

| 方法 | 年份/来源 | 类别 | NEU-Seg 报告值 | 官方代码 | 与本文协议可比？ | 进 Table X？ |
|---|---|---|---|---|---|---|
| Sub-region U-Net | 2023 EAAI | 工业专用 | 未能在线核验 | 无官方发布（本文历史谱系相关，见 §1.1 标记） | 否 | 否（谱系披露对象，非独立 baseline） |
| FDSNet | 2022 ICASSP | 工业专用 | 78.8（其协议） | 完整公开 | 否（故本文统一复现 87.08 入表） | **是（统一复现值）** |
| PGA-Net | 2020 IEEE TII | 工业专用 | 未能在线核验 | 无 | 否 | 否 |
| A-Net | 2023 IEEE TIM | 轻量工业 | 缺陷类 IoU（协议不同） | 论文称公开 | 否 | 否 |
| DSGNet | 2025 Expert Systems | 轻量工业 | 83.39（其协议） | 公开 | 否 | 否（未来候选） |
| AASFormer | 2026 BDMA | CNN-Transformer | 84.85（同划分，指标口径未核验一致） | 未检索到 | 否 | 否 |

## 3. 结论

1. **当前 10 模型对比体系维持不变**：工业缺陷层仍以统一协议复现的 FDSNet 为代表；本轮文献审计未发现必须立刻补训的方法（唯一代码公开且较新的 DSGNet 留作未来候选，需先核验其协议）。
2. **文献值与统一复现值严格分离**：本文件中的 78.8 / 83.39 / 84.85 等均为各文献自有协议下的报告值，禁止与 `final_baseline_table_v3.md` 混合引用。
3. **审稿回应口径**：近年工业缺陷方法中，官方代码完整且任务对口的可复现方法稀缺（DMC-Net 断链、TAG-Net/CDARNet/LGGFormer/PGA-Net 无代码）；本文选择官方代码完整的 FDSNet 统一复现作为工业专用方法代表，其余列入相关工作——此说法有本文件与 `recent_industrial_methods_audit.md` 双重证据支撑。
4. **谱系披露**：Sub-region U-Net 与本文方法存在历史谱系关系，已在 §1.1 如实标记，披露细节以 `docs/paper/prior_publication_overlap_audit_v6.md` 为准。

## 4. 核验来源

- Sub-region U-Net 摘要页（ScienceDirect S0952197623001951，2026-10-12 访问）：确认 NEU-Seg + USB-Seg、sub-region/mask-aware loss 设计；表格数值未开放。
- FDSNet 官方 README（github.com/jianzhang96/fdsnet，2026-10-12 访问）：NEU-Seg 78.8 / FPS 186.1、VOC 转换与 trainval-test 约定。
- A-Net 论文全文 PDF（max-chenb.github.io/mypaper/ANet.pdf，2026-10-12 访问）：NEU-Seg 协议（三类、250/50、224×224）与 Table IV 数值。
- DSGNet 官方 README（github.com/young-zyy/DSGNet，2026-10-12 访问）：NEU-Seg 83.39 / 0.49M / 0.29G；出处页（Ulster University Pure 与作者单位主页）确认 Expert Systems 2025。
- AASFormer 全文 PDF（sciopen.com，DOI 10.26599/BDMA.2025.9020095，2026-10-12 访问）：NEU-Seg 3630/840 划分原文表述与 Table 1 数值（84.85 / SegMAN 85.11）。
- PGA-Net 题录（NASA/ADS 与多处引用页，2026-10-12 访问）：IEEE TII 16(12):7448–7458, 2020；原文表格未获取。
