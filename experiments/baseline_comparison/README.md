# Baseline Comparison Experiments

用于后续 STDC / PIDNet / DDRNet 等基线对比实验的工作目录。

## 目录结构

- `configs/`：各基线方法的训练/评测配置文件
- `results/`：统一协议下的评测结果（仅小型结果文件，如 CSV/JSON/MD 汇总）
- `logs/`：训练与评测日志的摘要记录（完整大日志不入库）

## 约定

- 基线实验在服务器上执行训练；本目录只保存配置、结果汇总与日志摘要。
- 所有基线评测必须遵循 `experiments/eval_protocol_lock.md` 与 `experiments/protocol/` 中冻结的统一评价协议。
- 不上传数据集、模型权重（`*.pth` / `*.pt` / `*.pkl` / `*.ckpt`）与大体积日志文件。
- 当前论文冻结数字（CJIG v6.2）：NEU-Seg 上 DSMONet-B 91.24% / A2MS-DSMONet-B 91.45% mIoU；Leather 上 DSMONet-B 89.91% / A2MS-DSMONet-B 90.97% mIoU。新基线结果不得改动上述已冻结数字。
