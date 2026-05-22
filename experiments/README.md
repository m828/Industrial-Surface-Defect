# Experiments

## 实验目的

本目录用于记录 A2MS-DefectNet 论文后续服务器实验、固定结果和待核验结果。实验目标是为《中国图象图形学报》投稿补充复杂度统计、类别级 IoU、小目标/细长缺陷分析和近年方法对比，不重新设计模型。

## 目录结构

```
experiments/
├── README.md                          # 本文件
├── experiment_entry_mapping.md        # 论文名称 ↔ 代码文件映射表
├── server_code_audit.md               # 服务器代码审计报告
├── configs/                           # 实验配置（待补充）
├── scripts/
│   ├── README.md                      # 服务器脚本用途梳理
│   ├── complexity_stats.py            # 复杂度统计脚本
│   ├── measure_complexity_plan.md     # 复杂度统计实验计划
│   ├── evaluate_class_iou_plan.md     # 类别级 IoU 实验计划
│   ├── small_object_eval_plan.md      # 小目标分组评价计划
│   └── qualitative_visualization_plan.md  # 可视化分析计划
├── logs/                              # 实验日志目录
│   └── README.md
└── results/
    ├── fixed_existing_results.md      # 固定已核验结果（论文数值基准）
    ├── new_results_pending.md         # 待核验新结果
    ├── complexity_results.csv         # 复杂度统计结构化数据
    ├── class_iou_results.csv          # 类别级 IoU 结构化数据
    ├── small_object_group_results.csv # 小目标分组评价结构化数据
    └── experiment_log.md              # 实验运行日志
```

## 数据集说明

- **NEU-Seg**：工业表面缺陷语义分割主实验数据集，带钢表面缺陷（夹杂物、补丁、划痕三类），图像尺寸 200x200。**当前状态：数据集文件未部署到服务器，需人工确认位置。**
- **自建皮革缺陷数据集**：工业表面缺陷语义分割主实验数据集，皮革缺陷（开创伤、刺刮伤、烙印、破洞、皮肤藓、烂面、刺猴七类），图像尺寸 768x768，总量 2341 张。**当前状态：位于 `new (copy)/dataset/pige/`，可直接使用。**

原始图片、标签和数据集压缩包不得提交到 GitHub，应保留在服务器数据目录、数据管理平台或其他受控位置。

## 主要模型

| 论文名称 | Backbone | 对应代码文件 | 状态 |
|---|---|---|---|
| A2MS-DefectNet-S | ResNet-18 | 需人工确认（见 `experiment_entry_mapping.md`） | 需确认 |
| A2MS-DefectNet-B | ResNet-50 | 需人工确认（见 `experiment_entry_mapping.md`） | 需确认 |
| Base-S | ResNet-18 | 需人工确认（见 `experiment_entry_mapping.md`） | 需确认 |
| Base-B | ResNet-50 | 需人工确认（见 `experiment_entry_mapping.md`） | 需确认 |

对比方法：PP-LiteSeg、STDC、DDRNet、BiSeNet、Sub-region UNet、PIDNet、DMC-Net，以及后续确认可复现的 LETNet 或 SeaFormer。

## 实验结果记录规范

### 写入流程

1. 新实验完成后，先写入 `results/new_results_pending.md`；
2. 同步写入对应结构化结果文件（`complexity_results.csv`、`class_iou_results.csv` 或 `small_object_group_results.csv`）；
3. 在 `results/experiment_log.md` 追加本次运行记录；
4. 核对训练脚本、测试脚本、权重路径、日志路径和输出表格；
5. 结果可复查后，再同步到 `results/fixed_existing_results.md`；
6. 最后更新论文实验表、摘要、结论或图件。

### 记录要求

- **FPS 必须注明**：硬件型号、batch size、输入尺寸、是否包含后处理；
- **FLOPs 必须注明**：输入尺寸（如 200x200 或 768x768）；
- **模型大小必须注明**：权重文件路径和统计方式（参数+缓冲区 or 仅参数）；
- **Params 必须注明**：可训练参数 or 全部参数；
- **类别级 IoU 必须注明**：每个类别的 IoU、Dice、Recall、Precision；
- **历史结果**：来自硕士论文或旧日志的结果标注"历史结果，待核验"；
- **不确定项**：所有不确定的脚本、权重、模型对应关系标注"需人工确认"。

### 命名规范

- 模型统一使用：A2MS-DefectNet-S, A2MS-DefectNet-B, Base-S, Base-B
- 不再使用 A2MS-DSMONet 作为正文主模型名
- Baseline 使用论文中统一的名称（如 BiSeNetV1, DDRNet23slim, STDC2-Seg 等）

## 禁止事项

- 不允许口头结果直接写入论文；
- 不允许未核验结果进入摘要、结论或主实验表；
- 不提交 `.pth`、`.pkl`、`.pt`、`.ckpt`、`.onnx`、数据集、缓存、运行目录和大压缩包；
- 不把未确认复现的文献方法写入主实验表；
- 不编造实验结果；
- 不移动或删除服务器原始代码；
- 不直接修改论文摘要和结论中的实验数值（除非对应结果已完成核验并写入 `fixed_existing_results.md`）。
