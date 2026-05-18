# 类别级 IoU 统计计划

## 目标数据集

- NEU-Seg；
- 自建皮革缺陷数据集。

## 目标模型

优先统计 A2MS-DefectNet-S、A2MS-DefectNet-B、Base-S 和 Base-B；若时间允许，再统计已纳入主表的代表性基线。

## 输出指标

- class_name；
- IoU；
- Dice；
- Recall；
- Precision；
- 测试脚本；
- 权重路径；
- 日期；
- 是否已核验。

## 记录文件

结果先写入 `experiments/results/class_iou_results.csv`，并在 `experiments/results/new_results_pending.md` 中登记。只有完成脚本、权重、日志和输出表格核验后，才能进入论文类别级分析表或附录表。
