# Smoke Test 后下一步判断

> 生成日期：2026-06-15

## 一、Leather test 样本数最终结论

结论：**468**。

依据：`experiments/protocol/leather_test_split_audit.md` 已确认：

- `new (copy)/dataset/pige/test.txt` 总行数 468；
- 非空行数 468；
- 无空行；
- 无重复行；
- 无格式错误；
- 468 个 image/label 均真实存在；
- `new/dataset/pige/test.txt` 与 `new (copy)/dataset/pige/test.txt` 内容完全一致。

`experiments/eval_protocol_lock.md` 已修正 Leather/Pige test 样本数为 468。

## 二、evaluate_class_iou.py 是否已稳定

当前判断：**对 Base-B Leather smoke test 已稳定通过；对其余模型已有 loader，但仍需逐个 smoke test。**

已完成：

- 语法检查通过；
- `--help` 通过；
- 支持新 CLI：`--dataset`、`--test-list`、`--model-name`、`--model-file`、`--weight`、`--num-classes`、`--input-size`、`--output-csv`、`--save-log`；
- 兼容旧参数别名：`--model_type`、`--weight_path`、`--test_txt`、`--input_size`、`--num_classes`、`--output_dir`；
- NEU loader 缺 `Image` 导入问题已通过独立 PIL dataset 规避/修复；
- DDRNet loader 已改为 `model_ddr.py::DualResNet_imagenet`，但尚未跑 DDRNet smoke test。

## 三、Base-B Leather smoke test 是否通过

结论：**通过**。

| 项目 | 值 |
|---|---|
| 模型 | Base-B |
| 权重 | `new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl` |
| 测试样本数 | 468 |
| mIoU | 0.899120 |
| 输出 CSV | `experiments/results/smoke_base_b_leather_class_iou.csv` |
| 日志 | `experiments/logs/smoke_base_b_leather.log` |

## 四、Base-B Leather 当前 mIoU 是否可追溯

结论：**可追溯，但仍需人工确认是否作为论文 Base-B Leather 主结果权重。**

可追溯链条：

- 测试列表：`experiments/protocol/leather_valid_test_list.txt`；
- 模型文件：`/workspace/Industrial Surface Defect/new/model_dsmo_rs50.py`；
- 模型入口：`DSMONet(num_classes=8, backbone=resnet50())`；
- 权重：`/workspace/Industrial Surface Defect/new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl`；
- 脚本：`tools/evaluate_class_iou.py`；
- CSV：`experiments/results/smoke_base_b_leather_class_iou.csv`；
- 日志：`experiments/logs/smoke_base_b_leather.log`。

## 五、是否可以继续评估 A2MS-DefectNet-B Leather

结论：**可以继续评估**。

建议下一条命令使用：

- 模型：`A2MS-DefectNet-B`；
- 模型文件：`new/model_dsmo_rs50_eSE_adapt_detailloss.py`；
- 权重：`new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl`；
- 测试列表：`experiments/protocol/leather_valid_test_list.txt`。

注意：仍然不能把 checkpoint `best_iou=0.9091` 当成 test 结果。

## 六、是否可以继续评估 STDC1、DDRNet、PIDNet

- STDC1-Seg：可以继续 smoke test，但需人工确认 `sdtdcnet_pige_stdc2_pige.pkl` 文件名含 `stdc2` 是否确认为 STDC1/STDCNet1446。
- PIDNet-S：可以继续 smoke test。
- DDRNet23slim：可以继续尝试 smoke test；loader 已修正为 `model_ddr.py::DualResNet_imagenet`。如果权重加载失败，应标记为“需单独适配”，不要阻塞其他模型。

## 七、是否可以开始 A2MS-DefectNet-B NEU-Seg 重训

当前判断：**技术上可以进入重训准备，但不要立即训练，需先完成人工确认。**

重训前仍需确认：

- 是否能找回历史 91.3 权重；
- A2MS-B NEU 的训练脚本和配置是否与历史一致；
- 新权重和日志输出路径，必须避免覆盖现有权重；
- 是否使用 `experiments/retrain_a2ms_b_neu/` 作为新训练记录目录；
- 是否先用修复后的 `evaluate_class_iou.py` 对 NEU Base-B 做 smoke test。

## 八、仍需人工确认的问题

1. 是否正式把 Leather/Pige test 样本数从 467 修正为 468 并同步所有文档？
2. 是否采用 Base-B 0127 权重作为 Leather Base-B 主结果权重？
3. 是否继续评估 A2MS-B Leather，并接受若结果仍低于 Base-B 则改写 Leather 结论？
4. STDC1 权重文件名含 `stdc2` 的身份是否确认？
5. 是否同意下一步按顺序 smoke test A2MS-B、STDC1、PIDNet、DDRNet？
6. 是否允许后续启动 A2MS-B NEU 重训准备，但在训练前再次确认命令和输出路径？
