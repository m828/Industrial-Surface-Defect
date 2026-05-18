# 下一阶段任务

## 第一优先级

1. 统计 A2MS-DefectNet-S/B 的 Params、FLOPs、模型大小；
2. 统计 Base-S/Base-B 的 Params、FLOPs、模型大小；
3. 输出 NEU-Seg 和皮革数据集类别级 IoU；
4. 整理小目标/细长缺陷可视化结果；
5. 复现 DMC-Net，并优先在 NEU-Seg 上跑通；
6. 复现 PIDNet，作为 CVPR 2023 实时分割代表。

## 第二优先级

1. LETNet 或 SeaFormer 二选一；
2. 小目标/细长缺陷分组评价；
3. Boundary F1 或边界 IoU；
4. 复杂度与实时性统一表。

## 第三优先级

1. SPCS-Net、TAG-Net、GCRANet、CDARNet、DASeg-Net、SAID 等文献只进入相关工作；
2. 若后续找到官方代码，再考虑进入实验对比。

## 执行约束

- 不编造实验结果；
- 不移动或删除服务器原始代码；
- 不提交数据集、权重、缓存和大压缩包；
- 新结果先进入 `experiments/results/new_results_pending.md`，核验后再进入 `fixed_existing_results.md` 和论文。
