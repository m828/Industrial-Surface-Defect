# PIDNet 代码审计报告

## 1. 当前项目中 PIDNet 相关文件

| 文件路径 | 类型 | 状态 | 说明 |
|---|---|---|---|
| `new/pid.py` | 模型定义 | ✅ 完整 | PIDNet-S/M/L 模型定义（CVPR 2023） |
| `new/train_neu_pid.py` | 训练脚本 | ✅ 可用 | NEU-Seg 训练脚本（4类，200x200） |
| `new/train_pige_pid.py` | 训练脚本 | ✅ 可用 | Leather 训练脚本（8类，768x768） |
| `new/ddr.yml` | 配置文件 | ⚠️ 需确认 | arch='ddr'，但实际训练 PIDNet |
| `subregion unet/pid.yml` | 配置文件 | ⚠️ 需确认 | arch='pid'，train_iters=180000 |
| `subregion unet/pidnet.py` | 模型定义 | ❌ 空文件 | 0 字节 |
| `subregion unet/train_neu_pid.py` | 训练脚本 | ❌ 空文件 | 0 字节 |
| `subregion unet/datagenerator_neu.py` | 数据加载 | ✅ 可用 | NEU-Seg 数据加载器 |

## 2. 模型定义文件

**文件**: `new/pid.py`

**模型架构**:
- PIDNet-S: `m=2, n=3, planes=32, ppm_planes=96, head_planes=128`
- PIDNet-M: `m=2, n=3, planes=64, ppm_planes=96, head_planes=128`
- PIDNet-L: `m=3, n=4, planes=64, ppm_planes=112, head_planes=256`

**关键组件**:
- I Branch (Image): 主干特征提取
- P Branch (Parsing): 语义分割分支
- D Branch (Detail): 边界细节分支
- DAPPM/PAPPM: 多尺度上下文聚合
- PagFM: 注意力融合模块
- Light_Bag/Bag: 边界感知融合

**输出**:
- 训练模式: `[x_extra_p, x_, x_extra_d]` (P分支输出、主输出、D分支输出)
- 推理模式: `x_` (仅主输出)

## 3. 训练脚本

### 3.1 NEU-Seg 训练脚本 (`new/train_neu_pid.py`)

| 配置项 | 值 |
|---|---|
| 数据集 | NEU-Seg |
| 类别数 | 4 |
| 输入尺寸 | 200x200 |
| 训练数据 | `./dataset/train_neu.txt` |
| 验证数据 | `./dataset/test_neu.txt` |
| 模型 | `PIDNet(m=2, n=3, num_classes=4, planes=32, ppm_planes=96, head_planes=128, augment=True)` |
| 损失函数 | `get_loss_function_new(cfg)` (OHEM CrossEntropy) |
| 权重保存 | `model_savePaths/{arch}_{dataset}_0119_neu_pid.pkl` |

**注意**: 脚本使用 `ddr.yml` 配置文件，但模型实例化为 PIDNet-S。

### 3.2 Leather 训练脚本 (`new/train_pige_pid.py`)

| 配置项 | 值 |
|---|---|
| 数据集 | Leather (pige) |
| 类别数 | 8 |
| 输入尺寸 | 768x768 |
| 训练数据 | `./dataset/pige/train.txt` |
| 验证数据 | `./dataset/pige/val.txt` |
| 模型 | `PIDNet(m=2, n=3, num_classes=8, planes=32, ppm_planes=96, head_planes=128, augment=True)` |
| 损失函数 | OhemCrossEntropy |
| 权重保存 | `model_savePath/{arch}_{dataset}_pige_pid_s.pkl` |

## 4. 测试脚本

**未找到独立测试脚本**。训练脚本中包含验证逻辑（`val_interval` 触发），但无独立测试脚本。

需要创建测试脚本以：
1. 加载训练好的权重
2. 在测试集上评估 mIoU、类别级 IoU
3. 测量 FPS

## 5. 配置参数

### ddr.yml (NEU-Seg 训练使用)

```yaml
model:
    arch: ddr
training:
    train_iters: 60000
    batch_size: 16
    val_interval: 1000
    n_workers: 16
    optimizer:
        name: 'adam'
        lr: 1.0e-4
        weight_decay: 0.000002
    loss:
        - name: 'ohem_cross_entropy'
        - name: 'bce_with_logits_loss'
        - name: 'ohem_cross_entropy'
    lr_schedule: (未指定)
    resume: model_savePath/dsmonet_resnet50_neu_128_180000_9145.pkl
```

### pid.yml (subregion unet 目录)

```yaml
model:
    arch: pid
training:
    train_iters: 180000
    batch_size: 16
    val_interval: 1000
    optimizer:
        name: 'adam'
        lr: 1.0e-4
        weight_decay: 0.000002
    loss:
        - name: 'ohem_cross_entropy'
        - name: 'ohem_cross_entropy'
        - name: 'bondaryloss'
    resume: model_savePath/pid_neu_8788.pkl
```

## 6. 数据集支持

### NEU-Seg 支持

| 项目 | 状态 |
|---|---|
| 数据加载器 | ✅ `datagenerator_neu.py` |
| 训练脚本 | ✅ `train_neu_pid.py` |
| 配置文件 | ✅ `ddr.yml` (需确认) |
| 类别数 | 4 (背景 + 3类缺陷) |
| 输入尺寸 | 200x200 |
| 数据格式 | txt 文件 (annotation_path image_path) |

### Leather 数据集支持

| 项目 | 状态 |
|---|---|
| 数据加载器 | ✅ `datagenerator_yachi.py` |
| 训练脚本 | ✅ `train_pige_pid.py` |
| 配置文件 | ⚠️ 需确认 |
| 类别数 | 8 (背景 + 7类缺陷) |
| 输入尺寸 | 768x768 |
| 数据格式 | txt 文件 (image_path label_path) |

## 7. 已有权重

**未找到任何 PIDNet 权重文件**。

检查路径：
- `new (copy)/model_savePath/` - 无 PIDNet 权重
- `subregion unet/model_savePath/` - 未找到
- `pid.yml` 中引用 `model_savePath/pid_neu_8788.pkl` - 文件不存在

## 8. 缺失内容

| 缺失项 | 优先级 | 说明 |
|---|---|---|
| 独立测试脚本 | 高 | 需创建用于评估 mIoU、FPS 等 |
| 训练好的权重 | 高 | 需要从头训练或获取历史权重 |
| 配置文件确认 | 中 | `ddr.yml` 的 arch='ddr' 与实际 PIDNet 不一致 |
| Leather 配置文件 | 中 | train_pige_pid.py 未指定配置文件 |
| 复杂度统计 | 高 | 需要统计 Params、FLOPs、FPS |

## 9. 建议运行命令

### 9.1 复杂度统计 (PIDNet-S)

```bash
cd "/workspace/Industrial Surface Defect/new (copy)"
/opt/conda/bin/python -c "
import sys
sys.path.insert(0, '.')
from pid import PIDNet
import torch

# PIDNet-S
model = PIDNet(m=2, n=3, num_classes=4, planes=32, ppm_planes=96, head_planes=128, augment=False)
model.eval()

# Params
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'Params: {params:.2f}M')

# FLOPs
from thop import profile
input = torch.randn(1, 3, 200, 200)
flops, _ = profile(model, inputs=(input,), verbose=False)
print(f'FLOPs: {flops/1e9:.2f}G')

# FPS
import time
model.cuda()
input = torch.randn(1, 3, 200, 200).cuda()
with torch.no_grad():
    for _ in range(50):
        model(input)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(200):
        model(input)
    torch.cuda.synchronize()
    fps = 200 / (time.time() - start)
print(f'FPS: {fps:.1f}')
"
```

### 9.2 NEU-Seg 训练

```bash
cd "/workspace/Industrial Surface Defect/new (copy)"
/opt/conda/bin/python train_neu_pid.py --config ddr.yml
```

**注意**: 需要确认：
1. `ddr.yml` 中的 `resume` 路径是否正确
2. 是否需要修改 `arch` 为 'pid'
3. 训练迭代次数是否足够（60000 vs 180000）

### 9.3 Leather 训练

```bash
cd "/workspace/Industrial Surface Defect/new (copy)"
/opt/conda/bin/python train_pige_pid.py --config pige.yml
```

## 10. 风险评估

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| 配置文件不匹配 | 训练可能失败 | 先验证配置，必要时创建新配置 |
| 无预训练权重 | 需要从头训练 | 训练时间较长，建议先跑 NEU-Seg |
| 数据加载路径 | 可能找不到数据 | 确认 dataset/ 目录结构 |
| 损失函数依赖 | 可能导入失败 | 检查 tools/loss 模块 |

## 11. 结论

PIDNet 代码**基本完整**，具备训练和测试条件：
- ✅ 模型定义完整（PIDNet-S/M/L）
- ✅ 训练脚本存在（NEU-Seg 和 Leather）
- ✅ 数据加载器存在
- ⚠️ 配置文件需要确认/调整
- ❌ 无已有权重，需要从头训练
- ❌ 无独立测试脚本，需要创建

**建议优先级**：
1. 先运行复杂度统计（无需训练）
2. 创建独立测试脚本
3. 训练 PIDNet-S on NEU-Seg（迭代次数可适当减少）
4. 评估并记录结果
