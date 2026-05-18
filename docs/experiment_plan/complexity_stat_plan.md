# 复杂度统计计划

## 目标模型

- A2MS-DefectNet-S；
- A2MS-DefectNet-B；
- Base-S；
- Base-B。

## 需统计字段

- Params；
- FLOPs；
- 模型大小；
- FPS；
- 输入尺寸；
- batch size；
- 测试设备；
- 测试脚本；
- 权重路径；
- 是否包含后处理。

## 记录文件

统计完成后先写入 `experiments/results/complexity_results.csv`，再在 `experiments/results/new_results_pending.md` 中登记核验状态。核验后同步到 `experiments/results/fixed_existing_results.md`，最后更新论文复杂度与实时性分析表。

## 注意事项

- FLOPs 必须注明输入尺寸；
- FPS 必须注明设备、batch size、预热轮数、计时轮数和是否包含数据读取；
- 模型大小应明确来自权重文件大小还是参数序列化文件大小；
- 不得补写未经脚本输出支持的数值。
