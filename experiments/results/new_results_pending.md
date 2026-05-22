# 新实验结果待核验记录

> 所有新实验先写入本表。只有"是否已核验"和"是否可写入论文"均明确为"是"后，才允许同步到 `fixed_existing_results.md` 和论文实验表。
>
> 历史结果（来自硕士论文或旧日志）标注为"历史结果，待核验"，不自动写入论文。

## 字段说明

| 字段 | 含义 | 要求 |
|---|---|---|
| 日期 | 实验完成日期 | YYYY-MM-DD |
| 数据集 | NEU-Seg / Leather / Yachi | - |
| 模型 | 论文统一命名 | A2MS-DefectNet-S/B, Base-S/B, 或 baseline 名称 |
| 主干 | Backbone | ResNet-18, ResNet-50, STDC1, STDC2, - 等 |
| 训练脚本 | 服务器训练脚本相对路径 | 如 `new/train_neu_resnet.py` |
| 测试脚本 | 服务器测试脚本相对路径 | 如 `new/test_neu_dsmonet.py` |
| 配置文件 | yml 配置文件路径 | 如 `new/dsmonet_resnet.yml` |
| 权重路径 | .pkl/.pth 文件路径 | 不提交到 GitHub |
| Params/M | 参数量（百万） | 精确到小数点后两位 |
| FLOPs/G | 浮点运算量（GFLOPs） | 注明输入尺寸 |
| 模型大小/MB | 模型文件大小 | 精确到小数点后两位 |
| mIoU | 平均交并比 (%) | 精确到小数点后一位 |
| FPS | 推理速度 | 注明硬件、batch size、输入尺寸、是否含后处理 |
| 类别级 IoU | 各类别 IoU | 格式：类名1:iou1;类名2:iou2 |
| 是否已核验 | 人工核验状态 | 是/否 |
| 是否可写入论文 | 可否进入论文主表 | 是/否 |
| 备注 | 补充说明 | - |

## 待核验结果

### 复杂度统计（2026-05-18，A100，batch=1，随机初始化权重）

| 日期 | 数据集 | 模型 | 主干 | 训练脚本 | 测试脚本 | 配置文件 | 权重路径 | Params/M | FLOPs/G | 模型大小/MB | mIoU | FPS | 类别级 IoU | 是否已核验 | 是否可写入论文 | 备注 |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| 2026-05-18 | NEU-Seg (200x200) | A2MS-DefectNet-S | ResNet-18 | 需人工确认 | 需人工确认 | 需人工确认 | N/A (随机初始化) | 14.03 | 2.34 | 53.61 | - | 170.8 | - | 否 | 否 | 复杂度统计，A100，batch=1，脚本 tools/measure_complexity.py |
| 2026-05-18 | Leather (768x768) | A2MS-DefectNet-S | ResNet-18 | 需人工确认 | 需人工确认 | 需人工确认 | N/A (随机初始化) | 14.03 | - | 53.61 | - | 169.1 | - | 否 | 否 | 复杂度统计，A100，batch=1 |
| 2026-05-18 | NEU-Seg (200x200) | A2MS-DefectNet-B | ResNet-50 | 需人工确认 | 需人工确认 | 需人工确认 | N/A (随机初始化) | 29.46 | 6.42 | 112.70 | - | 100.3 | - | 否 | 否 | 复杂度统计，A100，batch=1 |
| 2026-05-18 | Leather (768x768) | A2MS-DefectNet-B | ResNet-50 | 需人工确认 | 需人工确认 | 需人工确认 | N/A (随机初始化) | 29.46 | - | 112.70 | - | 99.5 | - | 否 | 否 | 复杂度统计，A100，batch=1 |
| 2026-05-18 | NEU-Seg (200x200) | Base-S | ResNet-18 | 需人工确认 | 需人工确认 | 需人工确认 | N/A (随机初始化) | 14.00 | 2.51 | 53.49 | - | 149.7 | - | 否 | 否 | 复杂度统计，A100，batch=1 |
| 2026-05-18 | Leather (768x768) | Base-S | ResNet-18 | 需人工确认 | 需人工确认 | 需人工确认 | N/A (随机初始化) | 14.00 | - | 53.49 | - | 150.5 | - | 否 | 否 | 复杂度统计，A100，batch=1 |
| 2026-05-18 | NEU-Seg (200x200) | Base-B | ResNet-50 | 需人工确认 | 需人工确认 | 需人工确认 | N/A (随机初始化) | 29.37 | 6.65 | 112.36 | - | 99.9 | - | 否 | 否 | 复杂度统计，A100，batch=1 |
| 2026-05-18 | Leather (768x768) | Base-B | ResNet-50 | 需人工确认 | 需人工确认 | 需人工确认 | N/A (随机初始化) | 29.37 | - | 112.36 | - | 102.6 | - | 否 | 否 | 复杂度统计，A100，batch=1 |
| 2026-05-18 | NEU-Seg (200x200) | DDRNet23slim | DDRNet-Internal | 需人工确认 | 需人工确认 | - | N/A (随机初始化) | 6.71 | 2.07 | 25.67 | - | 167.7 | - | 否 | 否 | 复杂度统计，A100，batch=1 |
| 2026-05-18 | Leather (768x768) | DDRNet23slim | DDRNet-Internal | 需人工确认 | 需人工确认 | - | N/A (随机初始化) | 6.71 | - | 25.67 | - | 166.6 | - | 否 | 否 | 复杂度统计，A100，batch=1 |
| 2026-05-18 | NEU-Seg (200x200) | STDC1-Seg | STDCNet1446 | 需人工确认 | 需人工确认 | - | N/A (随机初始化) | 16.07 | 6.02 | 61.40 | - | 121.5 | - | 否 | 否 | 复杂度统计，A100，batch=1；BiSeNet+STDCNet1446 |
| 2026-05-18 | Leather (768x768) | STDC1-Seg | STDCNet1446 | 需人工确认 | 需人工确认 | - | N/A (随机初始化) | 16.07 | - | 61.40 | - | 123.7 | - | 否 | 否 | 复杂度统计，A100，batch=1 |
| 2026-05-18 | NEU-Seg (200x200) | STDC2-Seg | STDCNet813 | 需人工确认 | 需人工确认 | - | N/A (随机初始化) | 12.04 | 3.83 | 46.01 | - | 180.9 | - | 否 | 否 | 复杂度统计，A100，batch=1；BiSeNet+STDCNet813 |
| 2026-05-18 | Leather (768x768) | STDC2-Seg | STDCNet813 | 需人工确认 | 需人工确认 | - | N/A (随机初始化) | 12.04 | - | 46.01 | - | 178.6 | - | 否 | 否 | 复杂度统计，A100，batch=1 |
| 2026-05-18 | NEU-Seg (200x200) | PP-LiteSeg-B | ResNet-50 | 需人工确认 | 需人工确认 | - | N/A (随机初始化) | 30.84 | 6.63 | 117.86 | - | 110.2 | - | 否 | 否 | 复杂度统计，A100，batch=1 |
| 2026-05-18 | Leather (768x768) | PP-LiteSeg-B | ResNet-50 | 需人工确认 | 需人工确认 | - | N/A (随机初始化) | 30.84 | - | 117.86 | - | 110.4 | - | 否 | 否 | 复杂度统计，A100，batch=1 |
| 2026-05-18 | NEU-Seg (200x200) | BiSeNetV1-L | ResNet-50 | 需人工确认 | 需人工确认 | - | N/A (随机初始化) | 29.38 | 6.52 | 112.28 | - | 127.8 | - | 否 | 否 | 复杂度统计，A100，batch=1；CSFCN from csfcn_yuan.py |
| 2026-05-18 | Leather (768x768) | BiSeNetV1-L | ResNet-50 | 需人工确认 | 需人工确认 | - | N/A (随机初始化) | 29.38 | - | 112.28 | - | 99.2 | - | 否 | 否 | 复杂度统计，A100，batch=1 |
| 2026-05-18 | NEU-Seg (200x200) | PIDNet-S | PIDNet-Internal | `new/train_neu_pid.py` | - | `ddr.yml` | N/A (随机初始化) | 7.62 | 0.97 | 29.31 | - | 125.1 | - | 否 | 否 | 复杂度统计，A100，batch=1；CVPR 2023 |
| 2026-05-18 | Leather (768x768) | PIDNet-S | PIDNet-Internal | `new/train_pige_pid.py` | - | - | N/A (随机初始化) | 7.62 | - | 29.31 | - | 126.1 | - | 否 | 否 | 复杂度统计，A100，batch=1 |
| 2026-05-18 | NEU-Seg (200x200) | PIDNet-M | PIDNet-Internal | `new/train_neu_pid.py` | - | `ddr.yml` | N/A (随机初始化) | 28.54 | 3.63 | 109.18 | - | 122.6 | - | 否 | 否 | 复杂度统计，A100，batch=1；CVPR 2023 |
| 2026-05-18 | NEU-Seg (200x200) | PIDNet-L | PIDNet-Internal | `new/train_neu_pid.py` | - | `ddr.yml` | N/A (随机初始化) | 36.94 | 5.56 | 141.25 | - | 107.1 | - | 否 | 否 | 复杂度统计，A100，batch=1；CVPR 2023 |

### 类别级 IoU 评估（2026-05-18，A100，Leather 测试集 468 张）

| 日期 | 数据集 | 模型 | 主干 | 权重路径 | mIoU | background | open_wound | scratch | brand_mark | hole | skin_disease | rotten_surface | wart | 是否已核验 | 是否可写入论文 | 备注 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 2026-05-18 | Leather | Base-B | ResNet-50 | dsmonet_resnet_pascal_pige_dsmor50_0126.pkl | 0.8806 | 0.9915 | 0.7467 | 0.6583 | 0.8839 | 0.9872 | 0.9572 | 0.8728 | 0.9472 | 否 | 否 | 评估脚本 tools/evaluate_class_iou.py |
| 2026-05-18 | Leather | A2MS-DefectNet-B | ResNet-50 | dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl | 0.8607 | 0.9902 | 0.7326 | 0.5445 | 0.8841 | 0.9884 | 0.9660 | 0.8524 | 0.9276 | 否 | 否 | 评估脚本 tools/evaluate_class_iou.py |
| 2026-05-18 | Leather | STDC1-Seg | STDCNet1446 | sdtdcnet_pige_stdc2_pige.pkl | 0.8824 | 0.9925 | 0.7469 | 0.6300 | 0.8843 | 0.9873 | 0.9525 | 0.8955 | 0.9701 | 否 | 否 | 评估脚本 tools/evaluate_class_iou.py；权重文件名含"stdc2"但实际为 STDCNet1446 |

### 类别级 IoU 评估（2026-05-18，A100，NEU-Seg 测试集 840 张）

| 日期 | 数据集 | 模型 | 主干 | 训练脚本 | 配置文件 | 权重路径 | Params/M | FLOPs/G | 模型大小/MB | mIoU | FPS (200x200) | background | inclusion | patch | scratch | 是否已核验 | 是否可写入论文 | 备注 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 2026-05-18 | NEU-Seg | PIDNet-S | PIDNet-Internal | `new/train_neu_pid.py` | `pid_neu.yml` | `new (copy)/model_savePaths/pid_neu_0119_neu_pid.pkl` | 7.62 | 0.97 | 29.31 | 0.8667 | 125.1 | 0.9786 | 0.7768 | 0.9111 | 0.8004 | 否 | 否 | 训练 60000 iters, batch=16, lr=1e-4, Adam, cosine_annealing; best mIoU at iter 58500; 评估脚本 tools/evaluate_pidnet_neu.py |

### 类别级 IoU 评估跳过记录

| 模型 | 原因 |
|---|---|
| Base-S | 缺少 Leather 权重 |
| A2MS-DefectNet-S | 缺少 Leather 权重 |
| PP-LiteSeg-B | 缺少 Leather 权重 |
| DDRNet23slim | 缺少 Leather 权重 |
| BiSeNetV2-L | 缺少 Leather 权重和代码 |
| Sub-region UNet | 缺少 Leather 权重 |
| PIDNet-S | 缺少 Leather 权重 |

### 小目标与细长缺陷分组评价（2026-05-18，A100，Leather 测试集 468 张）

| 日期 | 数据集 | 模型 | 主干 | 小目标 mIoU | 中等目标 mIoU | 大目标 mIoU | 细长缺陷 mIoU | 多缺陷 mIoU | 无缺陷 mIoU | 是否已核验 | 是否可写入论文 | 备注 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 2026-05-18 | Leather | Base-B | ResNet-50 | 0.5141 | 0.8801 | 0.6876 | 0.8531 | 0.8739 | 0.1247 | 否 | 否 | 分组评价脚本 tools/evaluate_small_object_groups.py |
| 2026-05-18 | Leather | A2MS-DefectNet-B | ResNet-50 | 0.4084 | 0.8657 | 0.6721 | 0.8334 | 0.8586 | 0.1244 | 否 | 否 | 分组评价脚本 tools/evaluate_small_object_groups.py |
| 2026-05-18 | Leather | STDC1-Seg | STDCNet1446 | 0.4549 | 0.8770 | 0.6796 | 0.8469 | 0.8705 | 0.1250 | 否 | 否 | 分组评价脚本 tools/evaluate_small_object_groups.py |

**分组规则**：小目标=面积占比<P25(1.75%)，中等=P25~P75，大目标>=P75(26.99%)，细长=长宽比>P75(3.51)，多缺陷=连通域>1，无缺陷=缺陷面积=0。

**分析要点**：
- 小目标组（67 张）mIoU 显著低于中等/大目标组，验证了"小目标缺陷难分割"的核心卖点
- Base-B 在小目标组 mIoU=0.5141，A2MS-DefectNet-B 为 0.4084（-0.1057），STDC1-Seg 为 0.4549
- 细长缺陷组（67 张）mIoU 也明显低于中等目标组，支撑"细长缺陷边界模糊"的论点
- 详细 CSV：`experiments/results/small_object_group_results.csv`，汇总：`experiments/results/small_object_group_summary.md`

### 定性可视化分析（2026-05-18，Leather 测试集）

- 脚本：`tools/make_qualitative_figures.py`
- 输出目录：`figures/qualitative_results/`
- 共生成 14 张对比图，覆盖 6 类典型场景：

| 类别 | 样本数 | 文件名前缀 | 说明 |
|---|---|---|---|
| 小目标 | 3 | small_target_idx*.png | 面积占比 < P25 的微小缺陷 |
| 细长缺陷 | 3 | elongated_idx*.png | 长宽比 > P75 的条状/线状缺陷 |
| 多缺陷 | 3 | multi_defect_idx*.png | 同一图像多个缺陷连通域 |
| 纹理混淆 | 1 | texture_confusion_idx*.png | 无缺陷但模型预测 >5% 误报 |
| 漏检对比 | 1 | base_miss_a2ms_hit_idx*.png | Base-B 召回率 <0.5 但 A2MS >0.7 |
| 边界改善 | 3 | base_boundary_break_idx*.png | A2MS IoU 显著优于 Base-B |

- 样本索引：`figures/qualitative_results/sample_index.json`
- 每张图包含：原图 / Ground Truth / Base-B / A2MS-DefectNet-B / STDC1-Seg 五列对比
- 注意：CJK 字体缺失导致中文标题显示为方框，不影响图像内容展示

### DMC-Net 复现实验（2026-05-18，准备阶段）

| 日期 | 数据集 | 模型 | 状态 | 是否已核验 | 是否可写入论文 | 备注 |
|---|---|---|---|---|---|---|
| 2026-05-18 | NEU-Seg | DMC-Net | ❌ 无法复现 | 否 | 否 | 模型定义文件缺失 |

**关键问题**：
- 仓库地址：https://github.com/Michaelzyb/DMC-Net
- `models/net_factory.py` 引用 `from models.total_supvised.dmcnet import *`，但 `dmcnet.py` 文件不存在
- GitHub Issue #2 ("Where is DMC-Net's code?") 确认此问题，至今未解决
- 同样缺失的模型文件：`hdrnet.py`、`swin_unet.py`

**仓库中可用的替代模型**（可直接用于 NEU-Seg 对比实验）：

| 模型 | 文件 | 状态 |
|---|---|---|
| EDRNet | `models/total_supvised/edrnet.py` | ✓ 可用 |
| BiSeNet | `models/total_supvised/bisenet.py` | ✓ 可用 |
| U-Net | `models/total_supvised/u_net.py` | ✓ 可用 |
| DeepLabV3 | `models/total_supvised/deeplabv3.py` | ✓ 可用 |
| SegFormer | `models/total_supvised/seg_former.py` | ✓ 可用 |
| PGA-Net | `models/total_supvised/pga_net.py` | ✓ 可用 |
| TopFormer | `models/total_supvised/topformer.py` | ✓ 可用 |

**NEU-Seg 数据集兼容性**：DMC-Net 仓库已内置 NEU-Seg 支持（benchmark='neuseg'，4类，224x224），数据加载逻辑完整。

**建议行动**：
1. 联系作者获取 DMC-Net 模型定义（1287293308@qq.com）
2. 或使用仓库中已有的 EDRNet 等模型作为替代对比方法
3. 详细设置笔记：`experiments/dmcnet/dmcnet_setup_notes.md`

### 新训练结果（2026-05-22，A100，60000 iters）

> 以下结果来自训练日志中 best mIoU 验证点（非独立评估脚本），标注为"训练日志提取，待独立评估核验"。
> 权重文件位于 `new/model_savePath/`，不提交到 GitHub。

#### NEU-Seg 数据集（200×200，4类，测试集 840 张）

| 日期 | 数据集 | 模型 | 主干 | 训练脚本 | 配置文件 | 权重路径 | mIoU | background | crazing | inclusion | patches | 是否已核验 | 是否可写入论文 | 备注 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| 2026-05-22 | NEU-Seg | Base-B | ResNet-50 | `new/train_neu_resnet_detailloss.py` | `dsmonet_resnet_detailloss.yml` | `dsmonet_resnet_pascal_0110_neu_120000.pkl` | 0.9047 | 0.9846 | 0.8330 | 0.9268 | 0.8744 | 否 | 否 | 训练日志提取，待独立评估核验；best at iter 50500 |
| 2026-05-22 | NEU-Seg | A2MS-DefectNet-B | ResNet-50 | `new/train_neu_resnet_detailloss.py` | `dsmonet_resnet_detailloss.yml` | `dsmonet_resnet_pascal_0109_neu_eSE_adapt_datailloss.pkl` | 0.7131 | 0.9505 | 0.5391 | 0.8302 | 0.5324 | 否 | 否 | 训练日志提取；best at iter 42000；⚠️ mIoU 显著低于历史值 91.3，crazing/patches IoU 仅 0.53，需排查训练配置 |
| 2026-05-22 | NEU-Seg | DDRNet23slim | DDRNet-Internal | `new/train_neu_ddr.py` | `config_neu_ddr.yml` | `ddr_pascal_0119_neu_ddr_nolabels_23.pkl` | 0.8843 | 0.9819 | 0.8055 | 0.9255 | 0.8245 | 否 | 否 | 训练日志提取，待独立评估核验；best at iter 56000 |
| 2026-05-22 | NEU-Seg | Base-S | ResNet-18 | `new/train_neu_base_s.py` | `config_neu_base_s.yml` | `dsmonet_resnet_pascal_neu_base_s.pkl` | 0.8837 | 0.9819 | 0.8048 | 0.9256 | 0.8225 | 否 | 否 | 训练日志提取，待独立评估核验；best at iter 52500 |
| 2026-05-22 | NEU-Seg | A2MS-DefectNet-S | ResNet-18 | `new/train_neu_a2ms_s.py` | `config_neu_a2ms_s.yml` | `dsmonet_resnet_pascal_neu_a2ms_s.pkl` | 0.8858 | 0.9819 | 0.8059 | 0.9257 | 0.8297 | 否 | 否 | 训练日志提取，待独立评估核验；best at iter 55000 |

#### Leather 数据集（768×768，8类，验证集 234 张）

| 日期 | 数据集 | 模型 | 主干 | 训练脚本 | 配置文件 | 权重路径 | mIoU | background | open_wound | scratch | brand_mark | hole | skin_disease | rotten_surface | wart | 是否已核验 | 是否可写入论文 | 备注 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 2026-05-22 | Leather | A2MS-DefectNet-B | ResNet-50 | `new/train_pige_a2ms_b.py` | `config_pige_a2ms_b.yml` | `dsmonet_resnet_pascal_augnew_60000.pkl` | 0.8560 | 0.9918 | 0.6950 | 0.6383 | 0.8969 | 0.9876 | 0.9671 | 0.7351 | 0.9364 | 否 | 否 | 训练日志提取，待独立评估核验；best at iter 51500 |
| 2026-05-22 | Leather | A2MS-DefectNet-S | ResNet-18 | `new/train_pige_a2ms_s.py` | `config_pige_a2ms_s.yml` | `dsmonet_resnet_pascal_pige_a2ms_s.pkl` | 0.8472 | 0.9909 | 0.6483 | 0.6280 | 0.8787 | 0.9874 | 0.9592 | 0.7535 | 0.9315 | 否 | 否 | 训练日志提取，待独立评估核验；best at iter 54500 |
| 2026-05-22 | Leather | PIDNet-S | PIDNet-Internal | `new/train_pige_pid.py` | `config_pige_pid.yml` | `fcn_pascal_pige_pid_s.pkl` | 0.8406 | 0.9889 | 0.6341 | 0.5665 | 0.8360 | 0.9863 | 0.9511 | 0.8476 | 0.9143 | 否 | 否 | 训练日志提取，待独立评估核验；best at iter 37000 |
| 2026-05-22 | Leather | Base-S | ResNet-18 | `new/train_pige_base_s.py` | `config_pige_base_s.yml` | `dsmonet_resnet_pascal_pige_dsmor18_0228_80000.pkl` | 0.8397 | 0.9898 | 0.6585 | 0.5975 | 0.8689 | 0.9844 | 0.9508 | 0.7441 | 0.9233 | 否 | 否 | 训练日志提取，待独立评估核验；best at iter 53000 |
| 2026-05-22 | Leather | DDRNet23slim | DDRNet-Internal | `new/train_pige_ddr.py` | `config_pige_ddr.yml` | `ddr_pascal_pige_ddr23s.pkl` | 0.8317 | 0.9889 | 0.6814 | 0.5286 | 0.8495 | 0.9847 | 0.9501 | 0.7599 | 0.9104 | 否 | 否 | 训练日志提取，待独立评估核验；best at iter 24500 |
| 2026-05-22 | Leather | Base-B | ResNet-50 | `new/train_pige_base_b.py` | `config_pige_base_b.yml` | `dsmonet_resnet_pascal_pige_dsmor50_0220ksh.pkl` | 0.8278 | 0.9896 | 0.6171 | 0.5815 | 0.8826 | 0.9845 | 0.9541 | 0.6893 | 0.9240 | 否 | 否 | 训练日志提取，待独立评估核验；best at iter 57000 |

### 关键发现与异常

1. **A2MS-B NEU-Seg mIoU=0.7131 vs 历史 91.3**：差距巨大（-20个百分点）。crazing IoU=0.5391、patches IoU=0.5324 远低于其他模型（~0.80-0.83）。可能原因：
   - 训练配置与历史不同（detail_aggregate_loss 权重、学习率调度等）
   - 使用了不同的权重文件（新训练 vs 历史 `dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl`）
   - 需要对比历史训练配置排查

2. **Leather 数据集各模型 mIoU 集中在 0.83-0.86**：A2MS-B 最高（0.8560），Base-B 最低（0.8278）。scratch 和 open_wound 是所有模型最难的类别（IoU 0.53-0.70）。

3. **NEU-Seg 数据集**：Base-B 最高（0.9047），A2MS-S/DDRNet/Base-S/A2MS-S 在 0.88-0.89 区间，A2MS-B 异常偏低。

### 复杂度统计失败/跳过记录

| 模型 | 原因 | 需人工确认项 |
|---|---|---|
| PP-LiteSeg-T | model_dsmo_rs50_ppliteseg.py 的 backbone_out_chs 硬编码为 [512,1024,2048]，仅支持 ResNet-50 | 需确认 PP-LiteSeg-T 的正确代码入口或模型文件 |
| BiSeNetV2-L | 服务器上未找到 BiSeNetV2 代码文件 | 需从外部引入 BiSeNetV2 代码 |
| Sub-region UNet | Forward 失败：tensor 尺寸不匹配（pixelshuffle_invert 与 200x200 输入不兼容） | 需确认 Sub-region UNet 的正确输入格式和 inchannel 参数 |

## 历史结果（待核验）

> 以下结果来自硕士论文或历史日志，未经当前服务器核验。标注为"历史结果，待核验"。

| 日期 | 数据集 | 模型 | 主干 | 训练脚本 | 测试脚本 | 配置文件 | 权重路径 | Params/M | FLOPs/G | 模型大小/MB | mIoU | FPS | 类别级 IoU | 是否已核验 | 是否可写入论文 | 备注 |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| 历史 | NEU-Seg | A2MS-DefectNet-B | ResNet-50 | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 29.3 (历史) | - | - | 91.3 | 127.9 | - | 否 | 否 | 历史结果，待核验；RTX 3090 |
| 历史 | NEU-Seg | A2MS-DefectNet-S | ResNet-18 | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 14.0 (历史) | - | - | 89.7 | 196.5 | - | 否 | 否 | 历史结果，待核验；RTX 3090 |
| 历史 | NEU-Seg | Base-B | ResNet-50 | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 29.3 (历史) | - | - | 90.2 | 115.6 | - | 否 | 否 | 历史结果，待核验；RTX 3090 |
| 历史 | NEU-Seg | Base-S | ResNet-18 | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 14.0 (历史) | - | - | 88.4 | 178.5 | - | 否 | 否 | 历史结果，待核验；RTX 3090 |
| 历史 | Leather | A2MS-DefectNet-B | ResNet-50 | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 29.3 (历史) | - | - | 91.0 | 71.6 | - | 否 | 否 | 历史结果，待核验；RTX 3090 |
| 历史 | Leather | A2MS-DefectNet-S | ResNet-18 | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 14.0 (历史) | - | - | 89.7 | 188.7 | - | 否 | 否 | 历史结果，待核验；RTX 3090 |
| 历史 | Leather | Base-B | ResNet-50 | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 29.3 (历史) | - | - | 89.2 | 67.3 | - | 否 | 否 | 历史结果，待核验；RTX 3090 |
| 历史 | Leather | Base-S | ResNet-18 | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 14.0 (历史) | - | - | 88.1 | 167.2 | - | 否 | 否 | 历史结果，待核验；RTX 3090 |
