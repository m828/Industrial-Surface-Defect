# 模型权重与训练脚本缺失分析

> 日期：2026-05-19
> 数据集：NEU-Seg（4类，200×200）+ Leather/Pige（8类，768×768）

---

## 一、已有权重总览

| 模型 | NEU-Seg 权重 | Leather 权重 | 来源 |
|---|---|---|---|
| **Sub-region UNet** | ✅ `finalModel_best_newmodel_neu.pkl` (68MB) | ❌ | 历史 |
| **U-Net** | ✅ `unet_pascal_augnew_neu.pkl` (63MB) | ❌ | 历史 |
| **PSPNet** | ✅ `fcn_pascal_neu_psp0121.pkl` (563MB) | ❌ | 历史（待确认） |
| **PIDNet-S** | ✅ `pid_neu_0119_neu_pid.pkl` (89MB) | ❌ | **本次训练** |
| **Base-B** | ❌ | ❌ | 需训练 |
| **Base-S** | ❌ | ❌ | 需训练 |
| **A2MS-DefectNet-B** | ❌ | ❌ | 需训练 |
| **A2MS-DefectNet-S** | ❌ | ❌ | 需训练 |
| **DDRNet23slim** | ❌ | ❌ | 需训练 |
| **STDC1-Seg** | ❌ | ✅ `sdtdcnet_pige_stdc2_pige.pkl` (158MB) | 历史 |
| **BiSeNetV1-L** | ❌ | ❌ | 需训练 |
| **FCN** | ❌ | ❌ | 需训练 |
| **DeepLabV3+** | ❌ | ❌ | 需训练 |
| **ENet** | ❌ | ❌ | 需训练 |

---

## 二、训练脚本可用性

### NEU-Seg 训练脚本

| 模型 | 训练脚本 | 模型定义文件 | 状态 |
|---|---|---|---|
| **Base-B** | `new/train_neu_resnet.py` | `model_dsmo_rs50.py` | ✅ 可用（model=DSMONet+resnet50） |
| **Base-S** | ❌ 无专用脚本 | `model_dsmo_rs18.py` | ⚠️ 需从 `train_pige_dsmor18.py` 适配 |
| **A2MS-DefectNet-B** | `new/train_neu_resnet_detailloss.py` | `model_dsmo_rs50_eSE_adapt_detailloss.py` | ✅ 可用 |
| **A2MS-DefectNet-S** | ❌ 无专用脚本 | `model_dsmo_rs18_eSE_adapt_detailloss_822.py` | ⚠️ 需创建 |
| **DDRNet23slim** | `new/train_neu_ddr.py` | `model_ddr.py` | ⚠️ 需确认（model 行被注释） |
| **PIDNet-S** | `new (copy)/train_neu_pid.py` | `pid.py` | ✅ 已完成 |
| **Sub-region UNet** | `new/train_neu.py` | `model_p.py` | ✅ 已完成 |
| **U-Net** | `new/train_neu.py` | `model_unet.py` | ✅ 已完成 |
| **STDC1-Seg** | ❌ 无 NEU 脚本 | `stdcnet.py` + `stdc.py` | ⚠️ 需创建 |
| **BiSeNetV1-L** | ❌ 无 NEU 脚本 | `stdc.py` (BiSeNet) | ⚠️ 需创建 |
| **FCN** | ❌ 无 NEU 脚本 | `fcn.py` | ⚠️ 需创建 |
| **DeepLabV3+** | ❌ 无 NEU 脚本 | `model/deeplabv3.py` | ⚠️ 需创建 |
| **ENet** | ❌ 无 NEU 脚本 | `model/enet.py` | ⚠️ 需创建 |

### Leather (Pige) 训练脚本

| 模型 | 训练脚本 | 模型定义文件 | 状态 |
|---|---|---|---|
| **Base-B** | `new (copy)/train_pige_dsmor50.py` | `model_dsmo_rs50.py` | ✅ 可用 |
| **Base-S** | `new (copy)/train_pige_dsmor18.py` | `model_dsmo_rs18.py` | ✅ 可用 |
| **A2MS-DefectNet-B** | `new (copy)/train_pige_resnet.py` | `model_dsmo822.py` | ✅ 可用（需确认是否 eSE+detailloss） |
| **A2MS-DefectNet-S** | ❌ 无专用脚本 | `model_dsmo_rs18_eSE_adapt_detailloss_822.py` | ⚠️ 需创建 |
| **DDRNet23slim** | `new (copy)/train_pige_ddr.py` | `model_ddr.py` | ✅ 可用 |
| **PIDNet-S** | `new/train_pige_pid.py` | `pid.py` | ✅ 可用 |
| **Sub-region UNet** | `new (copy)/train_pige.py` | `model_p.py` | ✅ 可用 |
| **U-Net** | `new (copy)/train_pige.py` | `model_unet.py` | ✅ 可用（注释状态） |
| **STDC1-Seg** | `new/train_pige_stdc.py` | `stdcnet.py` + `stdc.py` | ✅ 可用 |
| **BiSeNetV1-L** | `new/train_pige_stdc.py` | `stdc.py` (BiSeNet+STDC) | ✅ 可用 |
| **FCN** | `new/train_pige_fcn.py` | `fcn.py` | ✅ 可用 |
| **DeepLabV3+** | `new/train_pige_deeplabv3.py` | `model/deeplabv3.py` | ✅ 可用 |
| **ENet** | `new/train_pige_enet.py` | `model/enet.py` | ✅ 可用 |

---

## 三、核心缺失（必须训练的模型）

### P0 — 论文主实验必需

| # | 模型 | 数据集 | 缺失原因 | 训练脚本 | 预估时间 |
|---|---|---|---|---|---|
| 1 | **Base-B** | NEU-Seg | 无权重 | `new/train_neu_resnet.py` | ~40 min |
| 2 | **Base-B** | Leather | 无权重 | `new (copy)/train_pige_dsmor50.py` | ~2 hr |
| 3 | **A2MS-DefectNet-B** | NEU-Seg | 无权重 | `new/train_neu_resnet_detailloss.py` | ~40 min |
| 4 | **A2MS-DefectNet-B** | Leather | 无权重 | `new (copy)/train_pige_resnet.py` | ~2 hr |
| 5 | **Base-S** | NEU-Seg | 无权重+无脚本 | 需适配 | ~40 min |
| 6 | **Base-S** | Leather | 无权重 | `new (copy)/train_pige_dsmor18.py` | ~2 hr |
| 7 | **A2MS-DefectNet-S** | NEU-Seg | 无权重+无脚本 | 需创建 | ~40 min |
| 8 | **A2MS-DefectNet-S** | Leather | 无权重+无脚本 | 需创建 | ~2 hr |
| 9 | **DDRNet23slim** | NEU-Seg | 无权重 | `new/train_neu_ddr.py` | ~40 min |
| 10 | **DDRNet23slim** | Leather | 无权重 | `new (copy)/train_pige_ddr.py` | ~2 hr |
| 11 | **PIDNet-S** | Leather | 无权重 | `new/train_pige_pid.py` | ~1 hr |

### P1 — 补充对比方法

| # | 模型 | 数据集 | 缺失原因 | 训练脚本 | 预估时间 |
|---|---|---|---|---|---|
| 12 | **STDC1-Seg** | NEU-Seg | 无权重+无脚本 | 需创建 | ~40 min |
| 13 | **BiSeNetV1-L** | NEU-Seg | 无权重+无脚本 | 需创建 | ~40 min |
| 14 | **FCN** | NEU-Seg | 无权重+无脚本 | 需创建 | ~40 min |
| 15 | **DeepLabV3+** | NEU-Seg | 无权重+无脚本 | 需创建 | ~40 min |
| 16 | **ENet** | NEU-Seg | 无权重+无脚本 | 需创建 | ~40 min |

---

## 四、脚本适配工作量

### 需要创建/适配的脚本

| 脚本 | 基于 | 改动 | 工作量 |
|---|---|---|---|
| Base-S NEU-Seg | `train_pige_dsmor18.py` | 改数据集、类别数(8→4)、输入尺寸(768→200)、损失函数 | ~20 行 |
| A2MS-DefectNet-S NEU-Seg | `train_neu_resnet_detailloss.py` | 改 backbone(resnet50→resnet18)、模型文件 | ~10 行 |
| A2MS-DefectNet-S Leather | `train_pige_resnet.py` | 改模型文件(rs50→rs18_eSE) | ~10 行 |
| DDRNet23slim NEU-Seg | `train_neu_ddr.py` | 取消 model 行注释 | ~2 行 |
| STDC1-Seg NEU-Seg | `train_pige_stdc.py` | 改数据集、类别数、输入尺寸 | ~20 行 |
| BiSeNetV1-L NEU-Seg | `train_pige_stdc.py` | 同上 | ~20 行 |

### 通用适配要点

NEU-Seg 训练脚本通用改动：
- `n_classes`: 8 → 4
- `input_hw`: 768×768 → 200×200
- `dataset`: pige → neu
- 数据加载器：`datagenerator_yachi.py` → `datagenerator_neu.py`
- 配置文件：创建对应 `.yml` 或修改现有

---

## 五、建议训练计划

### 第一批（P0，~8 小时）

优先训练论文核心对比的 4 个模型（Base-S/B, A2MS-DefectNet-S/B）在两个数据集上的权重。

| 顺序 | 模型 | 数据集 | 预估时间 | 脚本状态 |
|---|---|---|---|---|
| 1 | Base-B | NEU-Seg | ~40 min | ✅ 脚本就绪 |
| 2 | A2MS-DefectNet-B | NEU-Seg | ~40 min | ✅ 脚本就绪 |
| 3 | DDRNet23slim | NEU-Seg | ~40 min | ⚠️ 需取消注释 |
| 4 | Base-S | NEU-Seg | ~40 min | ⚠️ 需适配 |
| 5 | A2MS-DefectNet-S | NEU-Seg | ~40 min | ⚠️ 需创建 |
| 6 | PIDNet-S | Leather | ~1 hr | ✅ 脚本就绪 |
| 7 | Base-B | Leather | ~2 hr | ✅ 脚本就绪 |
| 8 | A2MS-DefectNet-B | Leather | ~2 hr | ✅ 脚本就绪 |

### 第二批（P1，~4 小时）

补充对比方法和剩余 Leather 权重。

| 顺序 | 模型 | 数据集 | 预估时间 | 脚本状态 |
|---|---|---|---|---|
| 9 | Base-S | Leather | ~2 hr | ✅ 脚本就绪 |
| 10 | DDRNet23slim | Leather | ~2 hr | ✅ 脚本就绪 |

### 可选（P2）

| 模型 | 数据集 | 说明 |
|---|---|---|
| STDC1-Seg | NEU-Seg | 需创建脚本 |
| BiSeNetV1-L | NEU-Seg | 需创建脚本 |
| FCN/DeepLabV3+/ENet | NEU-Seg | 需创建脚本 |

---

## 六、并行训练可行性

- 当前服务器有 1 块 A100 GPU
- 训练脚本均为单 GPU 训练
- 无法并行训练，需按顺序执行
- 建议使用 `nohup` 或 `tmux` 后台运行

### 快速启动命令（第一批）

```bash
# 1. Base-B NEU-Seg (~40 min)
cd "/workspace/Industrial Surface Defect/new" && nohup /opt/conda/bin/python train_neu_resnet.py --cfg config/dsmonet_resnet.yml > runs/base_b_neu.log 2>&1 &

# 2. A2MS-DefectNet-B NEU-Seg (~40 min)
cd "/workspace/Industrial Surface Defect/new" && nohup /opt/conda/bin/python train_neu_resnet_detailloss.py --cfg dsmonet_resnet_detailloss.yml > runs/a2ms_b_neu.log 2>&1 &

# 3. DDRNet23slim NEU-Seg (~40 min)
cd "/workspace/Industrial Surface Defect/new" && nohup /opt/conda/bin/python train_neu_ddr.py --cfg ddr.yml > runs/ddr_neu.log 2>&1 &

# 4. PIDNet-S Leather (~1 hr)
cd "/workspace/Industrial Surface Defect/new" && nohup /opt/conda/bin/python train_pige_pid.py --cfg pid_pige.yml > runs/pid_leather.log 2>&1 &
```

> 注意：以上命令需逐一执行，不能并行。具体配置文件和参数需根据脚本实际内容确认。
