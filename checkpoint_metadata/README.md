# Checkpoint 元信息

> 仅保存 checkpoint **元信息**（迭代、数据集、seed、mIoU、sha256 等）；**权重文件（*.pkl）不上传**。

## 目录

| 目录 | 模型 | 覆盖数据集 |
|---|---|---|
| `DSMONet-B/` | 细节—语义互优化基础架构（ResNet-50） | NEU-Seg 240k 固定终点 + Leather val-best |
| `A2MS-DSMONet-B/` | DSMONet-B + AAM + 辅助细节监督 | 同上 |

## 论文结果对应关系

| 论文条目 | checkpoint | 实际 iter | mIoU（统一复评） |
|---|---|---:|---:|
| 表1/表7 DSMONet-B (NEU) | `base_b_s1337_long240k_codex1_iter240000.pkl` | 240000 | 91.24% |
| 表1/表7 A2MS-DSMONet-B (NEU) | `a2_full_s1337_long240k_codex1_iter240000.pkl` | 240000 | 91.45% |
| 表4/表5 DSMONet-B (Leather) | `dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl` | 78000（val-best） | 89.91% |
| 表4/表5 A2MS-DSMONet-B (Leather) | `dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl` | 130000（val-best） | 90.97% |

完整溯源（含历史 val 分数、模型文件 sha256、评价清单 sha256）见各目录下 `checkpoint_info.json` 与 `experiments/audit/`。
