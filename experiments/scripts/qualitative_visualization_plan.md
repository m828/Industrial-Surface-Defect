# 可视化分析计划

## 目标

整理小目标缺陷、细长缺陷、边界模糊和复杂纹理干扰的可视化结果，用于论文定性分析。

## 前置条件

- [ ] 确认 A2MS-DefectNet-B 和 Base-B 的训练权重路径
- [ ] 确认测试脚本是否支持可视化输出
- [ ] 选取代表性样本（小目标、细长、边界模糊、复杂纹理）

## 可视化内容

### 1. 主模型 vs Baseline 定性对比

对同一样本展示：
- 原图 (Original)
- 真实标注 (Ground Truth)
- A2MS-DefectNet-B 预测
- Base-B 预测
- （可选）其他 baseline 预测

### 2. 特定缺陷类型分析

| 缺陷类型 | 选取标准 | 数据集 | 期望展示 |
|---|---|---|---|
| 小目标缺陷 | 面积占比 < 1% | NEU-Seg / Leather | 模型对小缺陷的检出能力 |
| 细长缺陷 | 长宽比 > 5:1 | NEU-Seg (划痕) / Leather (刺刮伤) | 模型对细长结构的连续性保持 |
| 边界模糊缺陷 | 缺陷与背景对比度低 | NEU-Seg / Leather | 模型对模糊边界的分割精度 |
| 复杂纹理干扰 | 背景纹理与缺陷相似 | NEU-Seg (带钢纹理) | 模型对背景噪声的抑制能力 |

### 3. 消融实验可视化（可选）

对比不同模块对定性结果的影响：
- Baseline vs +ELMM vs +ELMM+AAM vs +ELMM+AAM+Joint Loss

## 输出目录

```
figures/
├── qualitative_results/
│   ├── neu_small_object/          # NEU-Seg 小目标
│   ├── neu_elongated/             # NEU-Seg 细长缺陷
│   ├── neu_boundary_blur/         # NEU-Seg 边界模糊
│   ├── neu_texture_interference/  # NEU-Seg 纹理干扰
│   ├── leather_small_object/      # Leather 小目标
│   ├── leather_elongated/         # Leather 细长缺陷
│   ├── leather_boundary_blur/     # Leather 边界模糊
│   └── leather_texture_interference/  # Leather 纹理干扰
```

## 可视化格式

- 分辨率：300 dpi 以上
- 格式：PNG 或 PDF（矢量图优先）
- 标注：每个子图需有标签（Original / GT / A2MS-DefectNet-B / Base-B）
- 局部放大：对小目标和细长缺陷区域提供局部放大框

## 实现方案

### 方案 A：修改现有测试脚本

在 `test_neu_dsmonet.py` 或 `test_pige_resnet.py` 中添加可视化输出逻辑：
1. 保存原图、标注、预测的拼接图
2. 对特定样本添加局部放大
3. 使用不同颜色映射区分不同类别

### 方案 B：独立可视化脚本

编写独立脚本：
1. 加载模型权重
2. 对选定样本进行推理
3. 使用 matplotlib 或 PIL 生成对比图
4. 保存到 `figures/qualitative_results/`

## 样本选取方法

1. 先在测试集上运行推理，统计每张图的 mIoU
2. 按缺陷类型和难度筛选代表性样本
3. 优先选取：
   - A2MS-DefectNet-B 正确但 Base-B 错误的样本
   - 两种模型都失败但 A2MS-DefectNet-B 更接近的样本
   - 典型成功和失败案例

## 注意事项

- 可视化结果先进入 `new_results_pending.md` 记录
- 图片文件不提交到 GitHub（通过 `.gitignore` 排除）
- 论文中使用的图件需在 `figures/qualitative_results/` 中保留源文件
- 需人工确认选取的样本是否具有代表性
