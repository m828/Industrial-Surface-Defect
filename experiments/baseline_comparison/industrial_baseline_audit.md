# 工业缺陷领域 Baseline 审计（NEU-Seg 补充实验）

日期：2026-09-30
目的：回应"缺少工业表面缺陷专用方法比较"的潜在审稿意见。

## 1. DMC-Net —— 不可复现（记录，不猜测）

**结论：UNREPRODUCIBLE —— 官方仓库自身缺失模型文件。**

证据链：

1. 本地快照 `third_party/DMC-Net/`（vendored，随主仓库 git 管理）：`models/net_factory.py:2` 有 `from models.total_supvised.dmcnet import *`，但 `models/total_supvised/` 下**不存在** `dmcnet.py`。
2. 官方仓库 `github.com/Michaelzyb/DMC-Net`（作者 Yubo Zheng，与快照 README 邮箱一致；创建时间 2025-01-16，与本地 pyc 时间戳吻合）：master 分支递归列举全部 79 个文件，**无任何 dmc 字样文件**，`models/total_supvised/` 同样缺 `dmcnet.py`。缺失是上游问题，非本地快照损坏。
3. 本地 `__pycache__` 中无 `dmcnet.*.pyc`，无法从编译产物恢复。
4. 论文（DMC-Net, 2025, NEU-SEG mIoU 73.74%）未附其他代码获取渠道。

处置：不凭空重写该模型（无法保证与原版一致，引入不公平比较风险）。在论文表述中以"官方代码未公开完整模型定义"如实记录；工业缺陷 baseline 改由 FDSNet 承担。

## 2. FDSNet —— 纳入（工业缺陷专用，官方代码完整）

| 项 | 值 |
|---|---|
| 论文 | FDSNet: An Accurate Real-Time Surface Defect Segmentation Network (ICASSP 2022) |
| 仓库 | github.com/jianzhang96/fdsnet |
| commit | `ced4d0d8d0de2f3a58ad3d675a5827de26829e38`（2024-10-17） |
| 本地 | `third_party/FDSNet/` |
| 官方 NEU-Seg 报告 | 78.8 mIoU / 186.1 FPS（注意：官方用自有 trainval/test 划分，与本文 3630/840 划分不同，不可直接比较） |
| 结构 | Fast-SCNN 式：Encoder(x8/x16/x32) + GCU 全局上下文 + FeatureFusion + 分类头上采样至输入分辨率；aux 模式含 edge aux（BCE 对边缘图）+ semantic aux（BCE 对图像级类别存在向量） |
| 输入 | 200×200 原生可行（官方 NEU 训练即 scale-ratio=None） |
| 原生训练链 | Adam lr 1e-4, wd 1e-5, OHEM CE + edge BCE + semantic BCE，损失权重 [1.0, 0.5, 0.5]，batch 8，150 epochs（自有划分） |

适配方案：

- 数据：本文统一 train3630/test840 列表（VOC 式自有划分不用）；
- 辅助监督目标**在线生成**（官方 AuxiliaryGT 生成脚本未随仓库发布）：
  - edge GT：对整幅多类标注图做形态学边界（3×3 膨胀≠腐蚀），与官方 loss 中 `(target_edge>0).float()` 的二值约定一致；
  - semantic GT：类别 1–3 的图像级存在向量（multi-hot）；
- 协议统一：seed 1337、Adam lr 1e-4 恒定、wd 2e-6、batch 16、240k 固定终点、训练全程无 val/test、仅保存终点 checkpoint；
- 损失按 FDSNet 链习惯：[main OHEM CE, edge BCE, semantic BCE] × [1.0, 0.5, 0.5]。

## 3. LETNet —— 纳入（轻量实时，已有适配基础）

| 项 | 值 |
|---|---|
| 仓库 | `third_party/LETNet/`（嵌套 git 仓库，基线 commit `faaa065`） |
| 已有适配 | `results/letnet_adaptation_log.md`：4 个上游 blocker 修复 + NEU 数据集适配（200→208 pad，图像补 0 / 标注补 255，评价 crop 回 200），smoke 通过 |
| 遗留决策 | 原生 recipe（Adam 5e-4、warmpoly、epoch 制、训练中 val）→ 本轮按任务要求切换到统一协议 |

本轮补充：

- 协议化训练入口（恒定 lr 1e-4、wd 2e-6、seed 1337、240k、无 val/test、仅终点保存）；
- 统一评价注册（208 pad / 200 crop wrapper，mIoU 口径与其余模型完全一致）；
- 论文表注预留："LETNet 按原模型输入要求采用 208×208 padding，其余评价协议保持一致"。

## 4. 协议一致性检查单

- [x] 数据一致：NEU-Seg train3630 / test840（同一列表文件）
- [x] split 一致：不新建 val
- [x] evaluation 一致：统一 `tools/evaluate_class_iou.py`（含背景 4 类 mIoU，200×200，TestRescale+ToTensor，无 Normalize）
- [x] 无 test 选择：训练全程不加载 test，仅终点 checkpoint 评价
- [x] 优化器统一：Adam lr 1e-4 恒定，wd 2e-6，seed 1337，batch 16，240k iters

## 5. 备注

- 对 `tools/evaluate_class_iou.py` 的唯一改动是**新增** FDSNet/LETNet 的模型注册项（builder + 别名）；评价核心逻辑（预处理、混淆矩阵、IoU 计算）零改动。
- 训练目录：`repro_runs/baseline_neu_240k_industrial/{fdsnet,letnet}/`，结果归入 `experiments/baseline_comparison/{fdsnet,letnet}/`。
