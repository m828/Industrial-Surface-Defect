# LETNet / SeaFormer 复现实验设置笔记

> 创建日期：2026-05-19
> 目标：评估 LETNet 和 SeaFormer 作为轻量实时语义分割对比方法的可行性

---

## 1. LETNet

### 1.1 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | Lightweight Real-Time Semantic Segmentation Network With Efficient Transformer and CNN |
| 期刊 | IEEE Transactions on Intelligent Transportation Systems, 2023 (IF: 9.551) |
| arXiv | https://arxiv.org/abs/2302.10484 |
| 仓库 | https://github.com/XU-GITHUB-curry/LETNet |
| 本地路径 | `third_party/LETNet/` |

### 1.2 模型结构

- **架构**：CNN (DABModule) + Efficient Transformer (TransBlock)
- **编码器**：3 级下采样（stride=2），通道 32→64→128→32
- **Transformer**：在最深层特征上应用 Efficient Attention（3×3 patch, dim=288）
- **解码器**：3 级上采样 + Long Connection 跳跃连接
- **构造函数**：`LETNet(classes=19, block_1=3, block_2=12, block_3=12, block_4=3, block_5=3, block_6=3)`

### 1.3 复杂度统计（实际测量，A100 GPU）

| 指标 | 值 | 备注 |
|---|---|---|
| Params | 0.95M | 极轻量 |
| FLOPs | 1.21G | 输入 208×208 |
| Model Size | 5.04MB | - |
| FPS (208×208) | 23.5 | A100 GPU, batch=1 |

> **注意**：FPS 显著低于预期（论文报告 Cityscapes 512×1024 下较高速度），可能因为 208×208 输入下 batch size=1 时 GPU 利用率低，或 PyTorch 2.7 下效率不如 PyTorch 1.8。

### 1.4 环境依赖

| 依赖 | README 要求 | 实际兼容性 |
|---|---|---|
| Python | 3.6.4 | 当前 3.11，可运行 |
| PyTorch | 1.8.0+cu101 | 当前 2.7.1+cu126，可运行 |
| CUDA | 10.2 | 当前 12.6，可运行 |
| torchsummary | 未列出 | 需安装（已安装） |
| opencv-python | 未列出 | 训练/测试需要 |

### 1.5 发现的问题

#### 问题 1：解码器通道维度不匹配（代码 Bug）

**位置**：`Network/model/LETNet.py` 第 409 行

```python
output4 = self.upsample_1(output4 + self.LC3(output3))
```

- `DAB_Block_4` 输出 32 通道（`DABModule(32)` 保持通道数）
- `LC3 = LongConnection(32, 16, 3)` 输出 16 通道
- `output4(32ch) + LC3(output3)(16ch)` → 维度不匹配，运行时崩溃

**修复**：将 `LC3` 改为 `LongConnection(32, 32, 3)`，使输出通道匹配 `DAB_Block_4`。

#### 问题 2：200×200 输入导致空间维度不匹配

**原因**：编码器 4 次 stride-2 下采样（16× 降采样），200/16=12.5 不是整数。
- 200→100→50→25→12，解码器上采样 12→24 与跳跃连接 25 不匹配。

**修复**：将输入填充到 16 的倍数（如 208×208 或 224×224）。

#### 问题 3：数据集名称硬编码

**位置**：`Network/train.py` 第 126-142 行、`Network/test.py` 第 176-182 行、`Network/dataset/dataset_builder.py` 第 18-26 行

- 仅支持 `cityscapes` 和 `camvid`，其他名称抛出 `NotImplementedError`
- 需添加 NEU-Seg 分支

#### 问题 4：语法错误

**位置**：`Network/dataset/dataset_builder.py` 第 2 行

```python
import pickle ，  # 中文逗号
```

#### 问题 5：无预训练权重

仓库未提供任何预训练权重文件。

### 1.6 数据集适配方案

NEU-Seg 数据集格式（txt 列表，每行 `mask_path img_path`）与 LETNet 期望的格式（txt 列表，每行 `img_path mask_path`）基本一致，但：
1. 顺序相反（LETNet 先 img 后 mask，NEU-Seg 先 mask 后 img）
2. 路径前缀不同（LETNet 使用 `./dataset/<name>/`，NEU-Seg 使用 `dataset/`）
3. 需要创建 `Network/dataset/neuseg.py` 或修改现有数据集类

### 1.7 预估改动量

| 改动 | 文件 | 复杂度 |
|---|---|---|
| 修复 LC3 通道 | `Network/model/LETNet.py` | 1 行 |
| 修复中文逗号 | `Network/dataset/dataset_builder.py` | 1 行 |
| 添加 NEU-Seg 数据集分支 | `train.py`, `test.py`, `dataset_builder.py` | ~30 行 |
| 创建 NEU-Seg 数据集类 | `Network/dataset/neuseg.py` | ~50 行 |
| 输入填充 200→208 | 数据集或训练脚本 | ~10 行 |
| **总计** | | ~90 行改动 |

---

## 2. SeaFormer

### 2.1 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | SeaFormer: Squeeze-enhanced Axial Transformer for Mobile Semantic Segmentation |
| 会议 | ICLR 2023 |
| arXiv | https://arxiv.org/abs/2301.13156 |
| 仓库 | https://github.com/fudan-zvg/SeaFormer |
| 本地路径 | `third_party/SeaFormer/` |

### 2.2 模型变体

| 变体 | Params (M) | FLOPs (G) | ADE20K mIoU |
|---|---|---|---|
| SeaFormer-Tiny | 1.7 | 0.6 | 35.0-36.5 |
| SeaFormer-Small | 4.0 | 1.1 | 38.9-39.5 |
| SeaFormer-Base | 8.6 | 1.8 | 41.2-41.9 |
| SeaFormer-Large | 14.0 | 6.5 | 43.0-43.8 |

### 2.3 框架依赖（关键阻塞）

| 依赖 | 要求 | 当前环境 | 兼容性 |
|---|---|---|---|
| PyTorch | 1.5+ | 2.7.1+cu126 | ❌ |
| mmcv-full | 1.3.1-1.4.0 | 未安装 | ❌ 无法安装 |
| mmseg (bundled) | v0.19.0 | 未安装 | ❌ 依赖 mmcv |

**核心问题**：mmcv-full 1.3-1.5 不支持 PyTorch 2.x。其 C++/CUDA 扩展（`mmcv.ops`）在 PyTorch 2.7 下无法编译。迁移至 mmcv 2.x + mmengine 需要重写约 30 处 `mmcv.*` 导入，属于非平滑移植。

### 2.4 适配评估

| 项目 | 评估 |
|---|---|
| num_classes 修改 | 简单（配置文件 1 行） |
| 输入尺寸 200×200 | 支持（无硬编码分辨率） |
| 数据集格式 | MMSeg CustomDataset（img_dir/ann_dir 结构） |
| 预训练权重 | 提供 ImageNet-1K 分类预训练 + ADE20K 分割权重 |
| 环境兼容性 | ❌ 严重冲突 |

### 2.5 可选方案

1. **降级 PyTorch**：风险极高，会破坏当前所有已训练模型的环境
2. **移植到 mmcv2/mmengine**：工作量大（~100+ 行改动），且可能引入新问题
3. **Docker 隔离环境**：需要额外 GPU 资源和时间
4. **使用论文结果**：不满足"必须在本文数据集上训练/测试"的要求

---

## 3. 优先级建议

| 方法 | 可行性 | 改动量 | 风险 | 建议 |
|---|---|---|---|---|
| LETNet | ✓ 可行 | ~90 行 | 低 | 优先实施 |
| SeaFormer | ❌ 不可行 | 100+ 行 | 高（环境冲突） | 放弃复现，引用论文结果 |
