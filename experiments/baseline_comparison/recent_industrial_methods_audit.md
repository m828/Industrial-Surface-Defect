# 近年工业表面缺陷分割方法可复现性审计（2024–2026）

日期：2026-10-10
性质：资料核验 + 代码可用性调查（**本轮未训练任何新模型**）。
方法：仓库既有文献记录（`docs/literature/comparison_candidate_list.md`、`recent_work_survey.md`）+ 2026-10-10 在线核验（GitHub API / 出版方页面）。
范围限定：仅像素级语义分割方法；目标检测、异常检测、图像分类方法不作为 baseline 候选。

## 1. 任务点名复核的四个方法

| 方法 | 年份 / 来源 | 像素级语义分割？ | 适用 NEU-Seg / Leather？ | 官方代码 | 网络核心文件完整？ | 训练代码完整？ | 迁移成本 | 适合公平对比？ |
|---|---|---|---|---|---|---|---|---|
| DMC-Net | 2025，轻量实时表面缺陷分割（NEU-SEG 报告 mIoU+FPS） | 是 | NEU 高度一致 | 公开（github.com/Michaelzyb/DMC-Net）但**残缺** | **否**——`net_factory.py` 引用 `models/total_supvised/dmcnet.py`，但该文件在本地克隆（2026-09-29）与在线核验（2026-10-10，GitHub API 全目录清单）中均不存在，仓库自我断链 | 主程序因断链无法运行 | —（不可能） | **否（不可复现）** |
| TAG-Net | 2025，三重注意力钢材表面缺陷分割（NEU-Seg 上评估） | 是 | NEU 一致 | **未检索到官方代码**（2026-10-10 复核仍未发现） | — | — | 高（需自行复现，结构细节不全风险大） | 否（自行实现无法保证与原方法一致） |
| CDARNet | 2025，实时金属表面缺陷分割（Cross-Dimensional Adaptive Region reconstruction Net，经 SFCF-Net 论文引用确认存在） | 是 | NEU 一致（论文含 NEU 划分说明） | **未检索到官方代码** | — | — | 高 | 否（同上） |
| LGGFormer | 2025，双分支 local-guided global self-attention 表面缺陷分割（ScienceDirect S147403462400750X） | 是 | 同属表面缺陷分割 | **未检索到官方代码** | — | — | 高 | 否（同上） |

## 2. 仓库已记录的其余 2024–2026 工业方法（沿用既有调查结论，本次复核无变化）

| 方法 | 年份/来源 | 官方代码 | 结论 |
|---|---|---|---|
| SPCS-Net | 2025（NEU/MBP/USB-Seg） | 未检索到 | 仅相关工作 |
| GCRANet | 2026 Pattern Recognition | 未检索到 | 仅相关工作 |
| TSEDNet | 长尾带钢缺陷 | 未检索到 | 仅相关工作 |
| DDSNet | IEEE TIM（待核验） | 未检索到 | 仅相关工作 |
| DASeg-Net | 芯片缺陷扩散分割 | 成本高、协议不一致 | 仅相关工作 |
| SAID | 提示式缺陷分割 | 评估协议不同（提示学习） | 不宜横向比较 |
| Residual Shape Adaptive Dense-nested UNet | 2024 Pattern Recognition | 未检索到 | 仅相关工作 |
| MLR-Net | 2023 皮革缺陷 | 未检索到 | 仅相关工作 |

## 3. 唯一结论

> **当前 10 模型对比体系足够，无须补训。**

证据与理由：

1. 2024–2026 年工业缺陷语义分割方法中，唯一官方代码公开且任务与 NEU-Seg 完全对口的是 DMC-Net，但其官方仓库**自我断链**（核心文件 `dmcnet.py` 至今缺失，2026-10-10 GitHub API 复核确认），无法复现；其余方法（TAG-Net / CDARNet / LGGFormer / SPCS-Net / GCRANet 等）均未公开官方代码，自行复现既无法保证与原方法一致，也不满足公平对比的可核验要求。
2. 现有对比已覆盖三个层次且每层有代表：Classical（U-Net、DeepLabv3+）；Real-time（BiSeNetV2、STDC-Seg、PIDNet-S、DDRNet-23、LETNet）；Industrial（FDSNet，ICASSP 2022，官方代码完整且已按统一协议复现）。
3. 论文可在相关工作/讨论中如实说明："近年部分工业缺陷方法未公开代码或代码不完整（如 DMC-Net 官方仓库缺失核心模型文件），本文选取官方代码完整可复现的 FDSNet 作为工业缺陷专用方法代表，其余方法列入相关工作。"——这是可辩护的审稿回应。
4. 若未来 TAG-Net / CDARNet / DMC-Net 官方代码补全，可按既有 240k 统一协议随时补训（脚本模板齐备，单模型约 12–15h）。
