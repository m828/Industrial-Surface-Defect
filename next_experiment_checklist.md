# 下一步服务器实验清单

## 可直接支撑论文更新的实验

| 优先级 | 实验 | 建议脚本/入口 | 输入 | 输出 | 记录文件 | 完成后论文位置 |
|---|---|---|---|---|---|---|
| P1 | A2MS-DefectNet-S/B 复杂度统计 | `new/model_dsmo_512_eSE.py`、`new/model_dsmo_rs50_eSE_adapt_*.py`，具体文件需人工确认 | 模型定义、对应权重、固定输入尺寸 | Params、FLOPs、模型大小、FPS | `experiments/results/complexity_results.csv`、`new_results_pending.md` | 复杂度与实时性分析表 |
| P1 | Base-S/Base-B 复杂度统计 | `new/model_dsmo_*.py`，具体文件需人工确认 | 模型定义、对应权重、固定输入尺寸 | Params、FLOPs、模型大小、FPS | `complexity_results.csv`、`new_results_pending.md` | 复杂度与实时性分析表 |
| P1 | NEU-Seg 类别级 IoU | `test_neu_*.py`，具体脚本需人工确认 | NEU-Seg 测试集、权重 | 每类 IoU、Dice、Recall、Precision | `class_iou_results.csv`、`new_results_pending.md` | 类别级性能表或附录表 |
| P1 | 皮革数据集类别级 IoU | `test_pige_*.py`，具体脚本需人工确认 | 皮革测试集、权重 | 每类 IoU、Dice、Recall、Precision | `class_iou_results.csv`、`new_results_pending.md` | 类别级性能表或附录表 |
| P1 | 小目标/细长缺陷可视化 | 现有测试或可视化脚本需人工确认 | 代表性样本、Base-B/A2MS-DefectNet-B 权重 | 原图、标注、预测、局部放大图 | `figures/qualitative_results/`、`new_results_pending.md` | 可视化分析图 |
| P1 | DMC-Net 复现 | 官方代码或自建适配脚本，需先确认 | NEU-Seg 训练/测试集 | mIoU、FPS、可选复杂度 | `new_results_pending.md` | 近年方法对比表 |
| P1 | PIDNet 复现 | `pid.py`、`train_neu_pid.py`、`train_pige_pid.py`，需人工确认 | NEU-Seg/皮革训练测试集 | mIoU、FPS、可选复杂度 | `new_results_pending.md` | 实时分割对比表 |

## 备用或视时间加入的实验

| 优先级 | 实验 | 建议处理 | 记录文件 | 用途 |
|---|---|---|---|---|
| P2 | LETNet 或 SeaFormer 二选一 | 优先选择代码可运行、依赖较少、输入适配成本低的方法 | `new_results_pending.md` | 可选进入实验对比 |
| P2 | 小目标/细长缺陷分组评价 | 先定义面积占比或长宽比规则，再统计样本数 | `small_object_group_results.csv` | 支撑细粒度分析 |
| P2 | Boundary F1 或边界 IoU | 确认实现和阈值后再跑 | `small_object_group_results.csv` 或新增边界指标表 | 支撑边界改进结论 |
| P2 | 复杂度与实时性统一表 | 汇总所有已核验 Params/FLOPs/FPS | `complexity_results.csv` | 论文复杂度表 |

## 只作为相关工作的内容

SPCS-Net、TAG-Net、GCRANet、CDARNet、DASeg-Net、SAID 等若没有可靠官方代码、复现成本过高或任务设置不完全一致，先只进入相关工作，不进入主实验表。后续若找到官方代码并跑通 NEU-Seg 或皮革数据集，再转入待核验实验。
