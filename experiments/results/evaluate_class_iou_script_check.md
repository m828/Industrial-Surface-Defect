# evaluate_class_iou.py 脚本检查记录

> 生成日期：2026-06-15

## 本轮修复范围

- 改为单模型统一 CLI，支持 `--dataset`、`--test-list`、`--model-name`、`--model-file`、`--weight`、`--num-classes`、`--input-size`、`--output-csv`、`--save-log`。
- 保留旧命令计划别名：`--model_type`、`--weight_path`、`--test_txt`、`--input_size`、`--num_classes`、`--output_dir`。
- 使用独立 PIL 数据集读取逻辑，Leather 按 `image label` 解析，NEU 按 `label image` 解析。
- 修复 NEU loader 缺 `Image` 导入的问题。
- Base-B、A2MS-DefectNet-B、STDC1-Seg、DDRNet23slim、PIDNet-S 均有独立 loader。
- DDRNet23slim loader 改为优先使用 `new/model_ddr.py::DualResNet_imagenet`；若权重仍不匹配，脚本会报出需单独适配，不影响其他模型。
- 每次评估输出 confusion matrix、per-class IoU/Dice/Precision/Recall、mIoU、测试样本数、权重路径、模型入口、日期和设备。

## 检查结果

| 检查项 | 结果 |
|---|---|
| `python3 -m py_compile tools/evaluate_class_iou.py` | 通过 |
| `python3 tools/evaluate_class_iou.py --help` | 通过 |
| 是否训练 | 否 |
| 是否修改模型结构 | 否 |
| 是否写入/覆盖权重 | 否 |

## Smoke test 验证结果

- Base-B Leather 0127 权重加载成功；
- `experiments/protocol/leather_valid_test_list.txt` 的 468 个样本完整评估成功；
- mIoU=0.899120，与上一轮统一评估 Base-B(0127)=0.8991 一致；
- 输出：`experiments/results/smoke_base_b_leather_class_iou.csv`；
- 日志：`experiments/logs/smoke_base_b_leather.log`。
