# 服务器代码审计报告

> 本文档记录 `/workspace/Industrial Surface Defect/` 下三个历史代码目录的结构、模型文件、训练/测试脚本、配置文件、数据集和权重文件审计结果。所有推断均标注为"需人工确认"。

## 1. 目录概览

| 目录 | 用途推断 | 模型文件数 | 训练脚本数 | 测试脚本数 | 配置文件数 | 权重文件数 | 数据集 |
|---|---|---|---|---|---|---|---|
| `new/` | 最完整历史版本，模型变体和训练脚本最多 | 29 | 36 | 15 | 14 | 4 | 无（dataset/ 不存在） |
| `new (copy)/` | new 的整理版本，结构更清晰 | 30 | 14 | 6 | 15 | 12 | Leather (pige) |
| `subregion unet/` | 早期原型，Sub-region UNet 相关 | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 |

**注意**：`new/dataset/` 目录不存在，NEU-Seg 数据集的 `train_neu.txt` 和 `test_neu.txt` 未在服务器上找到。Leather 数据集仅存在于 `new (copy)/dataset/pige/`。

## 2. 模型文件审计

### 2.1 `new/` 目录模型文件（29 个）

| 文件名 | 关键组件 | 推断用途 | 状态 |
|---|---|---|---|
| `model_dsmo.py` | SELayer, UAFM, DAPPM, SqueezeBodyEdge | 基础 DSMONet | 需人工确认 |
| `model_dsmo822.py` | SELayer, Light_Bag, UAFM, DAPPM, SqueezeBodyEdge | 变体（含 Light_Bag） | 需人工确认 |
| `model_dsmo822_detailloss.py` | SELayer, Light_Bag, detailloss, UAFM, DAPPM, SqueezeBodyEdge | 变体（含 detailloss） | 需人工确认 |
| `model_dsmo_512.py` | SELayer, UAFM, DAPPM, SqueezeBodyEdge | **候选 Base-S**（STDCNet backbone, chs=[128,256,512]） | 需人工确认 |
| `model_dsmo_512_822.py` | SELayer, Light_Bag, UAFM, DAPPM, SqueezeBodyEdge | 变体 | 需人工确认 |
| `model_dsmo_512_csfcn.py` | PSPModule, LocalAttenModule, CFC_CRB | CFC 变体 | 需人工确认 |
| `model_dsmo_512_csfcn1.py` ~ `model_dsmo_512_csfcn5.py` | 各种 CFC/PSP 组合 | CFC 系列变体 | 需人工确认 |
| `model_dsmo_512_eSE.py` | **eSE**, SELayer, UAFM, DAPPM, SqueezeBodyEdge | **候选 A2MS-DefectNet-S**（STDCNet backbone） | 需人工确认 |
| `model_dsmo_ddr_s.py` | DAPPM（本地定义）, UAFM | DDRNet 变体 | 需人工确认 |
| `model_dsmo_new906.py` | SELayer, Light_Bag, UAFM, DAPPM, SqueezeBodyEdge | 变体 | 需人工确认 |
| `model_dsmo_rs50.py` | SELayer, UAFM, DAPPM, SqueezeBodyEdge | **候选 Base-B**（ResNet-50 backbone） | 需人工确认 |
| `model_dsmo_rs50_csfcn.py` | SELayer, UAFM, DAPPM, SqueezeBodyEdge | ResNet-50 CFC 变体 | 需人工确认 |
| `model_dsmo_rs50_csfcn2.py` | SELayer, UAFM, DAPPM, SqueezeBodyEdge | ResNet-50 CFC 变体2 | 需人工确认 |
| `model_dsmo_rs50_csfcn_yuan.py` | PSPModule, LocalAttenModule, CFC_CRB | ResNet-50 原始 CFC | 需人工确认 |
| `model_dsmo_rs50_detailloss.py` | SELayer, detailloss, UAFM, DAPPM, SqueezeBodyEdge | ResNet-50 + detailloss | 需人工确认 |
| `model_dsmo_rs50_eSE.py` | **eSE**, SELayer, UAFM, DAPPM, SqueezeBodyEdge | ResNet-50 + eSE | 需人工确认 |
| `model_dsmo_rs50_eSE_adapt.py` | **eSE**, **AdaptiveChannelWeight**, SELayer, UAFM, DAPPM, SqueezeBodyEdge | ResNet-50 + eSE + adapt | 需人工确认 |
| `model_dsmo_rs50_eSE_adapt_cfc.py` | eSE, AdaptiveChannelWeight, CFC_CRB | ResNet-50 + eSE + adapt + CFC | 需人工确认 |
| `model_dsmo_rs50_eSE_adapt_detailloss.py` | **eSE**, **AdaptiveChannelWeight**, **detailloss**, SELayer, UAFM, DAPPM, SqueezeBodyEdge | **候选 A2MS-DefectNet-B**（ResNet-50） | 需人工确认 |
| `model_dsmo_rs50_eSE_adapts.py` | eSE, AdaptiveChannelWeight, SELayer, UAFM, DAPPM, SqueezeBodyEdge | adapt 变体 | 需人工确认 |
| `model_dsmo_rs50_eSE_s.py` | eSE, SELayer, UAFM, DAPPM, SqueezeBodyEdge | eSE 变体 | 需人工确认 |
| `model_dsmo_rs50_gateatt.py` | GateAttention, GA_Decoder, SELayer, UAFM, DAPPM, SqueezeBodyEdge | GateAttention 变体 | 需人工确认 |
| `model_dsmo_rs50_ksh.py` | SELayer, UAFM, DAPPM, SqueezeBodyEdge | KSH 变体 | 需人工确认 |
| `model_dsmo_rs50_ppliteseg.py` | SELayer, UAFM, SqueezeBodyEdge（无 DAPPM） | PP-LiteSeg 变体 | 需人工确认 |
| `model_dsmo_rs50s.py` | SELayer, UAFM, DAPPM, SqueezeBodyEdge | ResNet-50 精简版 | 需人工确认 |

### 2.2 `new (copy)/` 独有模型文件（不在 `new/` 中）

| 文件名 | 关键组件 | 推断用途 | 状态 |
|---|---|---|---|
| `model_dsmo_r50_eSE_adapt_detailloss.py` | eSE, AdaptiveChannelWeight, detailloss | A2MS-DefectNet-B 变体 | 需人工确认 |
| `model_dsmo_rs18.py` | SELayer, UAFM, DAPPM, SqueezeBodyEdge | **候选 Base-S**（ResNet-18） | 需人工确认 |
| `model_dsmo_rs18_eSE_adapt_detailloss_822.py` | eSE, AdaptiveChannelWeight, Light_Bag | **候选 A2MS-DefectNet-S**（ResNet-18） | 需人工确认 |
| `model_dsmo_rs50_4_baseline.py` | eSE, AdaptiveChannelWeight, Light_Bag | baseline 变体 | 需人工确认 |
| `model_dsmo_rs50_eSE_adapt_detailloss_822.py` | eSE, AdaptiveChannelWeight, Light_Bag | A2MS-DefectNet-B + Light_Bag | 需人工确认 |
| `model_dsmo_rs50_loss1.py` | SELayer, UAFM, DAPPM, SqueezeBodyEdge | loss 变体 | 需人工确认 |
| `model_dsmo_rs50_nose.py` | SELayer, UAFM, DAPPM, SqueezeBodyEdge | 无 SE 变体 | 需人工确认 |

## 3. 训练脚本审计

### 3.1 `new/` 目录训练脚本

| 脚本 | 数据集 | 模型推断 | 配置文件 | 需人工确认项 |
|---|---|---|---|---|
| `train_neu.py` | NEU-Seg | 需人工确认 | 需人工确认 | 未检查模型导入 |
| `train_neu_ddr.py` | NEU-Seg | DDRNet | 需人工确认 | 未检查 |
| `train_neu_ddr_dsmo.py` | NEU-Seg | DDRNet + DSMONet | 需人工确认 | 未检查 |
| `train_neu_dsmonet.py` | NEU-Seg | DSMONet | 需人工确认 | 未检查 |
| `train_neu_pid.py` | NEU-Seg | PIDNet | 需人工确认 | 未检查 |
| `train_neu_resnet.py` | NEU-Seg | DSMONet + ResNet（含 detailloss 监督） | `dsmonet_resnet.yml` | 已检查，使用 `model_dsmo_rs50` |
| `train_neu_resnet2.py` ~ `train_neu_resnet5.py` | NEU-Seg | ResNet 变体 | 需人工确认 | 未检查 |
| `train_neu_resnet_.py` | NEU-Seg | ResNet | 需人工确认 | 未检查 |
| `train_neu_resnet_101.py` | NEU-Seg | ResNet-101 | 需人工确认 | 未检查 |
| `train_neu_resnet_detailloss.py` | NEU-Seg | ResNet + detailloss | 需人工确认 | 未检查 |
| `train_neu_resnet_detailloss_2.py` | NEU-Seg | ResNet + detailloss 变体 | 需人工确认 | 未检查 |
| `train_neu_resnet_eSEs.py` | NEU-Seg | ResNet + eSE | 需人工确认 | 未检查 |
| `train_neu_resnet_lossw.py` | NEU-Seg | ResNet + loss 加权 | 需人工确认 | 未检查 |
| `train_neu_resnet_s.py` | NEU-Seg | ResNet-S（轻量版） | 需人工确认 | 未检查 |
| `train_neus.py` | NEU-Seg | DSMONet（使用 `dataset/train_neu.txt`） | 需人工确认 | 已检查脚本结构 |
| `train_pige.py` | Leather | 需人工确认 | 需人工确认 | 未检查 |
| `train_pige_ddr.py` | Leather | DDRNet | 需人工确认 | 未检查 |
| `train_pige_deeplabv3.py` | Leather | DeepLabV3 | 需人工确认 | 未检查 |
| `train_pige_dsmor50.py` | Leather | DSMONet + ResNet-50 | `config/dsmonet_resnet_pige.yml` | 已检查，使用 `model_dsmo_rs50` |
| `train_pige_enet.py` | Leather | ENet | 需人工确认 | 未检查 |
| `train_pige_fcn.py` | Leather | FCN | 需人工确认 | 未检查 |
| `train_pige_hrnet.py` | Leather | HRNet | 需人工确认 | 未检查 |
| `train_pige_pid.py` | Leather | PIDNet | 需人工确认 | 未检查 |
| `train_pige_psp.py` | Leather | PSPNet | 需人工确认 | 未检查 |
| `train_pige_resnet.py` | Leather | ResNet | 需人工确认 | 未检查 |
| `train_pige_resnet3.py` | Leather | ResNet 变体 | 需人工确认 | 未检查 |
| `train_pige_resnet_loss.py` | Leather | ResNet + loss | 需人工确认 | 未检查 |
| `train_pige_resnet_loss904.py` | Leather | ResNet + loss（0904 版） | 需人工确认 | 未检查 |
| `train_pige_stdc.py` | Leather | STDC | 需人工确认 | 未检查 |
| `train_pige_unet.py` | Leather | U-Net | 需人工确认 | 未检查 |
| `train_yachi.py` | Yachi | 需人工确认 | 需人工确认 | 未检查 |
| `train_yachi_res_dsmonet.py` | Yachi | DSMONet + ResNet | 需人工确认 | 未检查 |

### 3.2 `new (copy)/` 目录训练脚本

| 脚本 | 数据集 | 模型推断 | 需人工确认项 |
|---|---|---|---|
| `train_neu.py` | NEU-Seg | 需人工确认 | 未检查 |
| `train_neus.py` | NEU-Seg | 需人工确认 | 未检查 |
| `train_pige.py` | Leather | 需人工确认 | 未检查 |
| `train_pige_4_0228.py` | Leather | 需人工确认 | 未检查 |
| `train_pige_ddr.py` | Leather | DDRNet | 未检查 |
| `train_pige_dsmor18.py` | Leather | DSMONet + ResNet-18 | 未检查 |
| `train_pige_dsmor50.py` | Leather | DSMONet + ResNet-50 | 未检查 |
| `train_pige_resnet.py` | Leather | ResNet | 未检查 |
| `train_pige_resnet3.py` | Leather | ResNet 变体 | 未检查 |
| `train_pige_resnet_loss.py` | Leather | ResNet + loss | 未检查 |
| `train_pige_resnet_loss904.py` | Leather | ResNet + loss（0904 版） | 未检查 |
| `train_yachi.py` | Yachi | 需人工确认 | 未检查 |
| `train_yachi_res_dsmonet.py` | Yachi | DSMONet + ResNet | 未检查 |

## 4. 测试脚本审计

### 4.1 `new/` 目录测试脚本

| 脚本 | 数据集 | 需人工确认项 |
|---|---|---|
| `test_neu.py` | NEU-Seg | 是否输出类别级 IoU |
| `test_neu_400.py` | NEU-Seg | 400x400 输入？ |
| `test_neu_dsmonet.py` | NEU-Seg | 使用哪个模型文件 |
| `test_neu_dsmonet2.py` | NEU-Seg | 使用哪个模型文件 |
| `test_neu_dsmonet_0118.py` | NEU-Seg | 0118 日期变体 |
| `test_neu_dsmonet_aug.py` | NEU-Seg | 含数据增强 |
| `test_neu_dsmonet_detailloss.py` | NEU-Seg | 含 detailloss |
| `test_neu_other.py` | NEU-Seg | 其他模型 |
| `test_neu_resnet_s.py` | NEU-Seg | ResNet-S |
| `test_neus.py` | NEU-Seg | 需人工确认 |
| `test_pige.py` | Leather | 需人工确认 |
| `test_pige_fcn_unet.py` | Leather | FCN/U-Net |
| `test_pige_resnet.py` | Leather | ResNet |
| `test_pige_resnet_aug.py` | Leather | ResNet + 增强 |
| `test_yachi.py` | Yachi | 需人工确认 |

### 4.2 `new (copy)/` 目录测试脚本

| 脚本 | 数据集 | 需人工确认项 |
|---|---|---|
| `test_pige.py` | Leather | 需人工确认 |
| `test_pige_4.py` | Leather | 4 类？ |
| `test_pige_fcn_unet.py` | Leather | FCN/U-Net |
| `test_pige_resnet.py` | Leather | ResNet |
| `test_pige_resnet_aug.py` | Leather | ResNet + 增强 |
| `test_yachi.py` | Yachi | 需人工确认 |

## 5. 配置文件审计

### 5.1 `new/` 目录配置文件（14 个）

| 配置文件 | 模型 | 数据集 | 训练迭代 | batch_size | 输入尺寸 | 需人工确认项 |
|---|---|---|---|---|---|---|
| `ddr.yml` | DDRNet | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 未检查 |
| `dsmonet.yml` | DSMONet | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 未检查 |
| `dsmonet_res101.yml` | ResNet-101 | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 未检查 |
| `dsmonet_res_ddr.yml` | DDRNet + DSMONet | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 未检查 |
| `dsmonet_resnet.yml` | DSMONet + ResNet | pascal（占位） | 120000 | 16 | same | 已检查 |
| `dsmonet_resnet822.yml` | DSMONet + ResNet (822) | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 未检查 |
| `dsmonet_resnet_detailloss.yml` | ResNet + detailloss | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 未检查 |
| `dsmonet_resnet_grid.yml` | ResNet + grid | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 未检查 |
| `dsmonet_resnet_pige.yml` | ResNet + pige | pascal（占位） | 60000 | 12 | same | 已检查 |
| `dsmonet_resnet_pige_loss.yml` | ResNet + pige + loss | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 未检查 |
| `dsmonet_resnet_random.yml` | ResNet + random | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 未检查 |
| `dsmonet_resnet_yacchi.yml` | ResNet + yachi | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 未检查 |
| `dsmonet_resnets.yml` | ResNet-S | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 未检查 |
| `unet_pascal.yml` | U-Net | 需人工确认 | 需人工确认 | 需人工确认 | 需人工确认 | 未检查 |

### 5.2 `new (copy)/` 独有配置文件

| 配置文件 | 推断用途 | 需人工确认项 |
|---|---|---|
| `dsmonet_resnet_detailloss_neu.yml` | NEU-Seg + detailloss | 未检查 |

**注意**：所有 yml 中 `data.dataset` 字段为 `pascal`（占位），实际数据路径通过训练脚本中的 `DataGenerator(txtpath=...)` 指定。

## 6. 数据集审计

| 数据集 | 位置 | 图像数 | 标签数 | train.txt | val.txt | test.txt | 状态 |
|---|---|---|---|---|---|---|---|
| Leather (pige) | `new (copy)/dataset/pige/` | 2341 | 2341 | 1637 行 | 234 行 | 467 行 | **可直接使用** |
| NEU-Seg | 未找到 | - | - | 不存在 | - | 不存在 | **需部署** |
| Yachi | 未确认 | - | - | - | - | - | 需确认 |

## 7. 运行环境审计

| 项目 | 当前服务器状态 |
|---|---|
| GPU | NVIDIA A100-PCIE-40GB (40GB) |
| PyTorch | 2.7.1+cu126（conda 环境 `/opt/conda/bin/python`） |
| Python | 3.11（conda）/ 3.10（系统） |
| 关键依赖 | thop（FLOPs 计算）, tensorboardX, opencv-python, pillow |
| 历史训练环境 | PyTorch 1.10, CUDA 11.2, cuDNN 8.2, Python 3.9.7（RTX 3090） |

**注意**：当前 PyTorch 版本 (2.7.1) 与历史训练环境 (1.10) 不同，加载旧权重时可能需要适配。

## 8. 关键待确认问题汇总

1. **A2MS-DefectNet-S 对应哪个文件？** `new/model_dsmo_512_eSE.py`（STDCNet backbone）还是 `new (copy)/model_dsmo_rs18_eSE_adapt_detailloss_822.py`（ResNet-18 backbone）？
2. **Base-S 对应哪个文件？** `new/model_dsmo_512.py`（STDCNet）还是 `new (copy)/model_dsmo_rs18.py`（ResNet-18）？
3. **NEU-Seg 数据集在哪里？** `train_neu.txt` 和 `test_neu.txt` 未在服务器上找到。
4. **论文中 FPS 测试的硬件和条件？** 历史结果在 RTX 3090 上测得，当前服务器为 A100。
5. **哪些权重文件对应哪些模型和数据集？** 需要逐个加载验证。
6. **`new/` 和 `new (copy)/` 中同名文件是否完全一致？** 需要 diff 比较。
