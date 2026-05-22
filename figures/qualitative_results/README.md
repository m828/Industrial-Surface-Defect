# 定性可视化结果
- 日期: 2026-05-18
- 数据集: leather
- 模型: Base-B, A2MS-DefectNet-B, STDC1-Seg

## 小目标缺陷

| 样本ID | 文件名 | 原因 |
|---|---|---|
| 1 | small_target_idx0001.png | 面积占比=0.0064 |
| 9 | small_target_idx0009.png | 面积占比=0.0061 |
| 13 | small_target_idx0013.png | 面积占比=0.0015 |

## 细长划痕

| 样本ID | 文件名 | 原因 |
|---|---|---|
| 5 | elongated_idx0005.png | 长宽比=7.4 |
| 14 | elongated_idx0014.png | 长宽比=4.9 |
| 16 | elongated_idx0016.png | 长宽比=3.6 |

## 多缺陷样本

| 样本ID | 文件名 | 原因 |
|---|---|---|
| 1 | multi_defect_idx0001.png | 连通域=2 |
| 2 | multi_defect_idx0002.png | 连通域=2 |
| 5 | multi_defect_idx0005.png | 连通域=3 |

## 背景纹理干扰

| 样本ID | 文件名 | 原因 |
|---|---|---|
| 117 | texture_confusion_idx0117.png | 无缺陷但Base-B误检率=49.93% |

## Base-B漏检_A2MS检出

| 样本ID | 文件名 | 原因 |
|---|---|---|
| 177 | base_miss_a2ms_hit_idx0177.png | Base-B Recall=0.49, A2MS Recall=0.91 |

## Base-B边界断裂_A2MS完整

| 样本ID | 文件名 | 原因 |
|---|---|---|
| 13 | base_boundary_break_idx0013.png | Base IoU=0.00, A2MS IoU=0.15 |
| 32 | base_boundary_break_idx0032.png | Base IoU=0.28, A2MS IoU=0.53 |
| 98 | base_boundary_break_idx0098.png | Base IoU=0.56, A2MS IoU=0.76 |

