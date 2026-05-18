# Industrial Surface Defect Segmentation with A2MS-DefectNet

本项目用于整理工业表面缺陷实时语义分割论文、实验代码、实验结果和投稿材料。当前目标期刊为《中国图象图形学报》；若后续投稿受阻，备选期刊为《计算机工程与应用》。

## 当前论文方法

A2MS-DefectNet 是一种以细节—语义互优化机制为核心的工业表面缺陷实时分割网络，进一步引入：

- ELMM 高效轻量映射模块；
- AAM 自适应注意力模块；
- 边缘细节损失与掩码感知损失组成的联合监督约束。

当前命名规则如下：

- A2MS-DefectNet：本文面向工业表面缺陷实时分割任务的最终模型名称；
- A2MS-DefectNet-S：轻量版本；
- A2MS-DefectNet-B：增强版本；
- Base-S / Base-B：仅采用细节—语义互优化机制的基础版本；
- 正文主线统一使用 A2MS-DefectNet 作为最终模型名。

## 数据集

- NEU-Seg；
- 自建皮革缺陷数据集。

## 当前状态

- 已有论文初稿与 A2MS-DefectNet 第四版论文稿；
- 已有 NEU-Seg 和皮革数据集主实验结果；
- 已有 ELMM、AAM、联合损失消融实验；
- 正在补充复杂度统计、类别级 IoU、小目标/细长缺陷分析和近年方法对比。

## 目录说明

- `docs/paper/`：论文稿、投稿体例要点、修订记录和论文结果更新协议；
- `docs/literature/`：近年相关工作调研、实验候选清单、相关工作改写和参考文献核验表；
- `docs/experiment_plan/`：后续实验补充计划、消融、复杂度、类别 IoU、小目标可视化计划；
- `experiments/results/`：固定结果、新结果待核验表、复杂度/类别 IoU/小目标分组结果模板；
- `experiments/scripts/`：服务器已有脚本用途梳理；
- `figures/`：结构图、模块图、数据样例和定性结果图占位；
- `src/`：服务器遗留代码说明，不直接提交服务器大文件或未整理训练代码。

## 重要说明

- 本仓库中大文件、训练权重、数据集不要直接提交到 GitHub；
- 权重文件建议使用 Git LFS、Release、网盘或服务器路径管理；
- 所有新实验结果必须先写入 `experiments/results/new_results_pending.md`，并核验后再进入论文；
- 不允许将口头结果、未确认日志或未核验表格直接写入摘要、结论或主实验表。
