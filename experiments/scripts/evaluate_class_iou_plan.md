# 类别级 IoU 评估计划

## 目标

输出 A2MS-DefectNet-S/B、Base-S/Base-B 及关键 baseline 在 NEU-Seg 和皮革数据集上各类别的 IoU、Dice、Recall、Precision。

## 前置条件

- [ ] 确认 NEU-Seg 数据集已部署到服务器（当前未找到 `train_neu.txt`/`test_neu.txt`）
- [x] Leather 数据集已确认：`new (copy)/dataset/pige/`（train=1637, val=234, test=467）
- [ ] 确认各模型对应的训练权重路径
- [ ] 确认测试脚本是否输出类别级 IoU

## NEU-Seg 类别（4 类含背景）

| 类别 ID | 类名 |
|---|---|
| 0 | background |
| 1 | inclusion（夹杂物） |
| 2 | patch（补丁） |
| 3 | scratch（划痕） |

## Leather 类别（8 类含背景）

| 类别 ID | 类名 |
|---|---|
| 0 | background |
| 1 | open_wound（开创伤） |
| 2 | scratch（刺刮伤） |
| 3 | brand_mark（烙印） |
| 4 | hole（破洞） |
| 5 | skin_disease（皮肤藓） |
| 6 | rotten_surface（烂面） |
| 7 | wart（刺猴） |

## 待评估模型

### 主模型

| 模型 | 数据集 | 测试脚本 | 权重路径 | 状态 |
|---|---|---|---|---|
| A2MS-DefectNet-B | NEU-Seg | 需人工确认 | 需人工确认 | 待执行 |
| A2MS-DefectNet-S | NEU-Seg | 需人工确认 | 需人工确认 | 待执行 |
| Base-B | NEU-Seg | 需人工确认 | 需人工确认 | 待执行 |
| Base-S | NEU-Seg | 需人工确认 | 需人工确认 | 待执行 |
| A2MS-DefectNet-B | Leather | 需人工确认 | 需人工确认 | 待执行 |
| A2MS-DefectNet-S | Leather | 需人工确认 | 需人工确认 | 待执行 |
| Base-B | Leather | 需人工确认 | 需人工确认 | 待执行 |
| Base-S | Leather | 需人工确认 | 需人工确认 | 待执行 |

### Baseline

| 模型 | 数据集 | 测试脚本 | 权重路径 | 状态 |
|---|---|---|---|---|
| BiSeNetV1 | NEU-Seg / Leather | 需人工确认 | 需人工确认 | 待执行 |
| DDRNet23slim | NEU-Seg / Leather | 需人工确认 | 需人工确认 | 待执行 |
| STDC2-Seg | NEU-Seg / Leather | 需人工确认 | 需人工确认 | 待执行 |
| Sub-region UNet | NEU-Seg / Leather | 需人工确认 | 需人工确认 | 待执行 |
| PIDNet-S | NEU-Seg / Leather | 需人工确认 | 需人工确认 | 待执行 |

## 评估指标

每个类别计算：
- **IoU** = TP / (TP + FP + FN)
- **Dice** = 2*TP / (2*TP + FP + FN)
- **Recall** = TP / (TP + FN)
- **Precision** = TP / (TP + FP)

## 输出格式

### CSV 格式

```
model,dataset,class_name,iou,dice,recall,precision,script,weight_path,date,verified
```

### new_results_pending.md 格式

在"类别级 IoU"列填写：`类名1:iou1;类名2:iou2;...`

## 评估脚本需求

需要编写或修改测试脚本以输出：
1. 混淆矩阵
2. 各类别 TP/FP/FN
3. 各类别 IoU/Dice/Recall/Precision

候选脚本基础：
- `new/test_neu_dsmonet.py`
- `new/test_pige_resnet.py`

## 注意事项

- 测试时使用 `model.eval()` 和 `torch.no_grad()`
- 输入尺寸需与训练时一致（NEU-Seg: 200x200, Leather: 768x768）
- 需要确认测试集 txt 文件路径
- 结果先写入 `new_results_pending.md`，核验后再进入 `fixed_existing_results.md`
