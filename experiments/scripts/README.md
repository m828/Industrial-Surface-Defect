# 服务器脚本用途梳理

服务器代码路径：`/workspace/Industrial Surface Defect/`

当前本地环境未能读取该服务器路径；以下内容基于用户提供的目录结构与脚本命名规则整理。凡未实际打开文件核验的条目均标记为“需人工确认”，不得直接作为论文复现实验描述。

## 顶层目录

| 目录 | 预期用途 | 状态 |
|---|---|---|
| `new/` | DSMONet、A2MS-DefectNet、NEU-Seg/皮革训练测试脚本及多种基线脚本 | 需人工确认 |
| `new (copy)/` | 可能为服务器代码备份或复制版本 | 需人工确认 |
| `subregion unet/` | Sub-region UNet 相关代码或对比实验代码 | 需人工确认 |

## 脚本命名规则

| 脚本或模式 | 预期用途 | 备注 |
|---|---|---|
| `train_neu_*.py` | NEU-Seg 训练脚本 | 需人工确认具体模型、输入尺寸和训练策略 |
| `test_neu_*.py` | NEU-Seg 测试脚本 | 需人工确认是否输出 mIoU、FPS、类别 IoU |
| `train_pige_*.py` | 皮革数据集训练脚本 | `pige` 命名需人工确认是否对应 leather/pigskin 数据目录 |
| `test_pige_*.py` | 皮革数据集测试脚本 | 需人工确认测试集划分和权重路径 |
| `model_dsmo_*.py` | 细节—语义互优化相关模型 | 需人工确认对应 Base-S/Base-B 或 A2MS-DefectNet 版本 |
| `model_dsmo_rs50_eSE_adapt_*.py` | A2MS-DefectNet-B 相关模型候选实现 | 需人工确认最终论文版本使用的具体文件 |
| `model_dsmo_512_eSE.py` | A2MS-DefectNet-S 相关模型候选实现 | 需人工确认是否为最终 S 版实现 |
| `detail_loss.py` | 边缘细节损失相关实现 | 需核对与论文公式和 STDCNet 边缘监督来源一致性 |
| `pid.py` | PIDNet 相关模型或工具脚本 | 需人工确认复现实验入口 |
| `train_neu_pid.py` | PIDNet 在 NEU-Seg 上的训练脚本 | 需人工确认 |
| `train_pige_pid.py` | PIDNet 在皮革数据集上的训练脚本 | 需人工确认 |
| `stdcnet.py` | STDC 相关模型实现 | 需人工确认 |
| `train_pige_stdc.py` | STDC 在皮革数据集上的训练脚本 | 需人工确认 |

## 后续核验清单

1. 在服务器上执行 `find /workspace/Industrial\ Surface\ Defect -maxdepth 3 -type f -name '*.py'` 生成脚本清单；
2. 对每个进入论文实验表的脚本记录训练入口、测试入口、配置、输入尺寸、权重路径和输出表格；
3. 对不确定脚本保留“需人工确认”，不写入论文方法或实验复现细节。
