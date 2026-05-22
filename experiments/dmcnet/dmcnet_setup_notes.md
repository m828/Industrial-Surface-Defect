# DMC-Net 复现实验设置笔记

## 1. 仓库信息

| 项目 | 内容 |
|---|---|
| 仓库地址 | https://github.com/Michaelzyb/DMC-Net |
| 分支 | master |
| 最后提交 | 9b95974 (2025-01-16) |
| 作者 | Yubo Zheng (1287293308@qq.com) |
| 本地路径 | `third_party/DMC-Net/` |

## 2. 依赖环境

```
Python >= 3.6
PyTorch >= 1.1.0
Albumentations
tqdm
tensorboardX
cv2 (opencv-python)
numpy
scikit-learn
matplotlib
```

## 3. 支持的数据集

DMC-Net 内置支持以下数据集（通过 `benchmark` 参数选择）：

| 数据集 | benchmark 名称 | 输入尺寸 | 类别数 | 说明 |
|---|---|---|---|---|
| **NEU-SEG** | `neuseg` | 224x224 | 4 | 钢材表面缺陷（背景+3类缺陷） |
| KolektorSDD | `KolektorSDD` | - | 2 | - |
| KolektorSDD2 | `KolektorSDD2` | 512x256 | 2 | - |
| MT | `MT` | 256x256 | 4 | 磁瓦缺陷 |
| Carpet | `carpet` | 256x256 | 2 | - |
| Hazelnut | `hazelnut` | 256x256 | 2 | - |
| CrackForest | `CrackForest` | 320x480 | 2 | - |
| CDD | `CDD` | 384x544 | 2 | - |
| RSDD1 | `RSDD1` | 800x160 | 2 | - |
| DAGM1-10 | `DAGM1`~`DAGM10` | 256x256 | 2 | - |

## 4. NEU-Seg 数据集格式

DMC-Net 的 NEU-Seg 数据加载逻辑（`data/dataloaders.py:get_neuseg()`）：

```python
# 数据组织方式
root_path/
├── train/
│   ├── images/
│   │   ├── *.jpg
│   └── annotations/
│       ├── *.png
└── test/
    ├── images/
    │   ├── *.jpg
    └── annotations/
        ├── *.png
```

**关键逻辑**：
- 扫描 `root_path` 下所有文件
- `train` 目录下的 `images/*.jpg` 作为训练集
- `test` 目录下的 `images/*.jpg` 作为测试集
- 标签路径：将 `.jpg` 替换为 `.png`，`images` 替换为 `annotations`
- 训练集按 80/20 划分训练/验证（`random_state=69`）
- 标签二值化：`mask > 128 → 1, else → 0`

**当前 NEU-Seg 数据集状态**：
- 服务器路径：`/workspace/Industrial Surface Defect/new (copy)/dataset/NEUSeg/`
- 测试集列表：`dataset/test_neu.txt`（840 张，格式：`annotation_path image_path`）
- **问题**：DMC-Net 期望的数据目录结构与当前 NEU-Seg 数据集结构不完全一致
  - 当前 NEU-Seg 使用 `train_neu.txt` 和 `test_neu.txt` 列表文件
  - DMC-Net 期望按 `train/images/*.jpg` 和 `test/images/*.jpg` 目录结构组织

## 5. 支持的模型

| 模型 | 模型名 | 文件 | 状态 |
|---|---|---|---|
| **DMC-Net** | `dmcnet` | `models/total_supvised/dmcnet.py` | **❌ 缺失** |
| U-Net | `u_net` | `models/total_supvised/u_net.py` | ✓ |
| SegNet | `seg_net` | `models/total_supvised/seg_net.py` | ✓ |
| BiSeNet | `bisenet` | `models/total_supvised/bisenet.py` | ✓ |
| DeepLabV3 | `deeplabv3` | `models/total_supvised/deeplabv3.py` | ✓ |
| EDRNet | `edrnet` | `models/total_supvised/edrnet.py` | ✓ |
| SegFormer | `seg_former` | `models/total_supvised/seg_former.py` | ✓ |
| PGA-Net | `pga_net` | `models/total_supvised/pga_net.py` | ✓ |
| TopFormer | `topformer` | `models/total_supvised/topformer.py` | ✓ |
| HDR-Net | `hdrnet` | `models/total_supvised/hdrnet.py` | **❌ 缺失** |
| SwinUNet | `swin_unet` | `models/total_supvised/swin_unet.py` | **❌ 缺失** |

### ⚠️ 关键问题：DMC-Net 模型定义文件缺失

`models/net_factory.py` 中引用了 `from models.total_supvised.dmcnet import *`，但 `models/total_supvised/` 目录下不存在 `dmcnet.py` 文件。

同样缺失的还有 `hdrnet.py` 和 `swin_unet.py`。

**这意味着无法直接运行 DMC-Net 模型的训练和测试。**

## 6. 官方训练命令

```bash
cd third_party/DMC-Net
python main.py --model dmcnet --benchmark neuseg --dataset_root_path <数据集根目录> --epochs 100 --batch_size 4 --base_lr 0.001
```

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | `dmcnet` | 模型名称 |
| `--benchmark` | `neuseg` | 数据集名称 |
| `--dataset_root_path` | `''` | 数据集根目录 |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 4 | 批次大小 |
| `--base_lr` | 0.001 | 学习率 |
| `--log_path` | `logs_test` | 日志路径 |
| `--checkpoint` | `''` | 预训练权重路径 |
| `--mode` | `total-sup` | 训练模式（仅支持 `total-sup`） |

## 7. 官方测试命令

DMC-Net 的测试在训练完成后自动执行（`basenetwork.py:final_test()`）：

```python
# 训练完成后自动调用
self.final_test()  # 加载最佳模型，在测试集上评估
```

测试输出：
- Test mDice
- Test mIoU
- Test pixel accuracy

## 8. 训练配置

| 配置项 | 值 |
|---|---|
| 优化器 | Adam (lr=0.001, betas=(0.9, 0.999)) |
| 学习率调度 | CosineAnnealingLR (T_max=epochs*0.8, eta_min=1e-5) |
| 损失函数 | 0.5 * (CE_loss + Dice_loss) |
| 类别权重 | NEU-Seg: [0.6, 0.6, 0.6, 0.6] |
| 累积步数 | 2 |
| 数据增强 | HorizontalFlip(p=0.3), VerticalFlip(p=0.3), Blur(p=0.3), GaussNoise(p=0.3) |
| 评估指标 | mIoU, mDice (基于 confusion matrix 累积) |
| 模型保存 | 基于验证集 mDice 最优 |

## 9. 评估指标对比

DMC-Net 使用的评估指标与本文一致：

| 指标 | DMC-Net 实现 | 本文实现 |
|---|---|---|
| mIoU | `TotalDiceIou.get_mIoU()` | `runningScore` confusion matrix |
| mDice | `TotalDiceIou.get_mdice()` | 类似实现 |
| Pixel Accuracy | `TotalDiceIou.get_pixes_accuracy()` | - |

**注意**：DMC-Net 的 mIoU 计算从 class 1 开始（跳过背景），与本文一致。

## 10. 需要适配的地方

### 10.1 数据集格式适配（优先级：高）

当前 NEU-Seg 数据集使用 txt 列表文件，DMC-Net 期望目录结构扫描。

**方案**：
1. 创建符号链接或数据预处理脚本，将现有数据转换为 DMC-Net 期望的目录结构
2. 或修改 `data/dataloaders.py` 中的 `get_neuseg()` 函数，使其支持 txt 列表文件

### 10.2 模型定义文件（优先级：关键）

**必须解决**：DMC-Net 模型定义文件 `dmcnet.py` 缺失。

**可能的解决方案**：
1. 联系作者获取完整代码（1287293308@qq.com）
2. 根据论文描述重新实现 DMC-Net 架构
3. 检查是否有其他分支或 release 包含完整代码
4. 使用仓库中已有的其他模型（如 EDRNet、BiSeNet）作为替代对比

### 10.3 类别数和标签映射

NEU-Seg：4 类（背景 + 3 类缺陷），已正确配置：
```python
# basenetwork.py
elif benchmark == 'neuseg':
    self.num_classes = 4
    weight = torch.tensor([0.6, 0.6, 0.6, 0.6], device=self.device)
```

### 10.4 输入尺寸

NEU-Seg 输入尺寸为 224x224（DMC-Net 默认），与本文使用的 200x200 不一致。
需要确认是否需要调整为 200x200 以保持一致性。

## 11. 是否能直接用于 NEU-Seg

**否**，原因：
1. DMC-Net 模型定义文件缺失
2. 数据集目录结构需要适配
3. 输入尺寸需要确认（224 vs 200）

## 12. 是否支持多类别分割

**是**，DMC-Net 通过 `class_num` 参数支持任意类别数的语义分割。

## 13. 下一步行动

1. **紧急**：联系作者获取 DMC-Net 模型定义文件
2. **备选**：根据论文重新实现 DMC-Net 架构
3. **临时**：使用仓库中已有的其他模型（EDRNet、BiSeNet 等）进行对比实验
4. 数据集格式转换脚本开发
5. 输入尺寸一致性确认

## 14. 仓库中可用的替代模型

如果 DMC-Net 无法复现，以下模型可直接使用：

| 模型 | 文件 | 可用性 |
|---|---|---|
| EDRNet | `edrnet.py` | ✓ 可直接使用 |
| BiSeNet | `bisenet.py` | ✓ 可直接使用 |
| U-Net | `u_net.py` | ✓ 可直接使用 |
| DeepLabV3 | `deeplabv3.py` | ✓ 可直接使用 |
| SegFormer | `seg_former.py` | ✓ 可直接使用 |

这些模型可以使用相同的训练流程（`BaseLineTrain`）在 NEU-Seg 上进行训练和评估。
