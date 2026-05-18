# Experiments

## 实验目的

本目录用于记录 A2MS-DefectNet 论文后续服务器实验、固定结果和待核验结果。实验目标是为《中国图象图形学报》投稿补充复杂度统计、类别级 IoU、小目标/细长缺陷分析和近年方法对比，不重新设计模型。

## 数据集说明

- NEU-Seg：工业表面缺陷语义分割主实验数据集；
- 自建皮革缺陷数据集：工业表面缺陷语义分割主实验数据集。

原始图片、标签和数据集压缩包不得提交到 GitHub，应保留在服务器数据目录、数据管理平台或其他受控位置。

## 主要模型

- A2MS-DefectNet-S：轻量版本；
- A2MS-DefectNet-B：增强版本；
- Base-S / Base-B：仅采用细节—语义互优化机制的基础版本；
- 对比方法：PP-LiteSeg、STDC、DDRNet、BiSeNet、Sub-region UNet、PIDNet、DMC-Net，以及后续确认可复现的 LETNet 或 SeaFormer。

## 服务器代码边界

服务器代码路径记录为 `/workspace/Industrial Surface Defect/`。当前本地环境未读取到该路径，因此本仓库只记录脚本用途和实验流程，不移动、不删除、不改写服务器原始训练代码。

## 实验结果记录规范

1. 新实验完成后，先写入 `experiments/results/new_results_pending.md`；
2. 同步写入对应结构化结果文件，例如 `complexity_results.csv`、`class_iou_results.csv` 或 `small_object_group_results.csv`；
3. 核对训练脚本、测试脚本、权重路径、日志路径和输出表格；
4. 结果可复查后，再同步到 `experiments/results/fixed_existing_results.md`；
5. 最后更新论文实验表、摘要、结论或图件。

## 禁止事项

- 不允许口头结果直接写入论文；
- 不允许未核验结果进入摘要、结论或主实验表；
- 不提交 `.pth`、`.pkl`、`.pt`、`.ckpt`、`.onnx`、数据集、缓存、运行目录和大压缩包；
- 不把未确认复现的文献方法写入主实验表。
