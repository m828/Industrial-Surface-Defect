# 实验入口映射表

> 本文件记录论文模型名称与服务器代码文件的对应关系。所有映射均标注为"需人工确认"，直到在服务器上实际运行验证。

## 1. 主模型映射

| 论文名称 | 最可能对应的代码文件 | 关键组件 | backbone_indices | backbone_out_chs | 需人工确认项 |
|---|---|---|---|---|---|
| A2MS-DefectNet-B | `new/model_dsmo_rs50_eSE_adapt_detailloss.py` | eSE + AdaptiveChannelWeight + detailloss + 4路联合监督 | [0,1,2,3,4] | [512,1024,2048] | 是否为论文最终版本；是否使用 `new (copy)/` 中的同名文件 |
| A2MS-DefectNet-S | `new (copy)/model_dsmo_rs18_eSE_adapt_detailloss_822.py` | eSE + AdaptiveChannelWeight + Light_Bag + detailloss | 需人工确认 | 需人工确认 | `new/` 中无对应文件；backbone 为 ResNet-18 还是 STDCNet 需确认 |
| A2MS-DefectNet-S（备选） | `new/model_dsmo_512_eSE.py` | eSE + SELayer + UAFM + DAPPM + SqueezeBodyEdge | [1,2,3,4] | [128,256,512] | backbone 为 STDCNet1446（非 ResNet-18），需确认是否对应论文 S 版 |
| Base-B | `new/model_dsmo_rs50.py` | SELayer（标准SE）+ UAFM + DAPPM + SqueezeBodyEdge，无 eSE、无 detailloss | [0,1,2,3,4] | [512,1024,2048] | 需确认是否使用 `new (copy)/` 中的同名文件 |
| Base-S | `new (copy)/model_dsmo_rs18.py` | SELayer（标准SE）+ UAFM + DAPPM + SqueezeBodyEdge，无 eSE、无 detailloss | 需人工确认 | 需人工确认 | `new/` 中无对应 `model_dsmo_rs18.py`；需确认 backbone 通道数 |
| Base-S（备选） | `new/model_dsmo_512.py` | SELayer（标准SE）+ UAFM + DAPPM + SqueezeBodyEdge | [1,2,3,4] | [128,256,512] | backbone 为 STDCNet1446（非 ResNet-18），需确认 |

## 2. 论文术语与代码术语对应

| 论文术语 | 代码中的对应实现 | 所在文件 | 需人工确认项 |
|---|---|---|---|
| ELMM（高效轻量映射模块） | `bot_fine` (ConvBNRelus) + `conv_up` (ConvTranspose2d 序列) + `laplacian` (Laplacian) | 各 `model_dsmo_*.py` | 论文公式与代码实现是否完全一致 |
| AAM（自适应注意力模块） | `AdaptiveChannelWeight` + `eSE` 的组合 | 含 `eSE_adapt` 的模型文件 | ACW 和 eSE 的组合方式是否与论文公式一致 |
| ACW（自适应通道权重） | `class AdaptiveChannelWeight` | 含 `adapt` 的模型文件 | 需人工确认 |
| eSE（高效挤压激励） | `class eSE` | 含 `eSE` 的模型文件 | 与标准 SE 的区别需人工确认 |
| 细节—语义互优化机制 | `SqueezeBodyEdge` + `UAFM` + `edge_fusion` + `arm2` | 各 `model_dsmo_*.py` | 双向交互的具体实现需人工确认 |
| 边缘细节损失 (ledge) | `detail_loss.py` 中的边缘损失实现 | `new/detail_loss.py` | 与 STDCNet 边缘监督的一致性需确认 |
| 掩码感知损失 (lmask) | 二值前景—背景监督（binarymask） | `datagenerator_neu.py` 中 `binarymask` | 与 Sub-region UNet mask-aware loss 的一致性需确认 |
| DAPPM | `class DAPPM` | 各 `model_dsmo_*.py`（从 tools 导入）或 `model_dsmo_ddr_s.py`（本地定义） | 需确认导入来源 |
| UAFM | `class UAFM` | 各 `model_dsmo_*.py`（从 tools 导入） | 需确认导入来源 |
| SqueezeBodyEdge | `class SqueezeBodyEdge` | 各 `model_dsmo_*.py`（从 tools 导入） | 需确认导入来源 |

## 3. Baseline 模型映射

| 论文 Baseline | 代码文件 | 状态 |
|---|---|---|
| BiSeNetV1 | `new/model_dsmo_rs50_csfcn_yuan.py`（含 PSPModule/LocalAttenModule/CFC_CRB）| 需人工确认是否为 BiSeNetV1 实现 |
| BiSeNetV2 | 未找到对应文件 | 需从外部引入或自建 |
| STDC1-Seg / STDC2-Seg | `new/stdcnet.py` + `new/train_pige_stdc.py` | 需人工确认 STDC1/STDC2 区分 |
| DDRNet23slim | `new/model_dsmo_ddr_s.py` + `new/train_pige_ddr.py` | 需人工确认是否为 DDRNet23-slim |
| PP-LiteSeg-T / PP-LiteSeg-B | `new/model_dsmo_rs50_ppliteseg.py` | 需人工确认 T/B 版本区分 |
| Sub-region UNet | `subregion unet/` 目录 | 需人工确认入口脚本 |
| U-Net | `new/model_unet.py` + `new/unet.py` | 可直接使用 |
| FCN-8s | `new/fcn.py` | 可直接使用 |
| DeepLabV3+ | `new (copy)/model/deeplabv3.py` + `new/train_pige_deeplabv3.py` | 需迁移脚本 |
| PSPNet | `new (copy)/model/pspnet.py` + `new/train_pige_psp.py` | 需迁移脚本 |
| ENet | `new (copy)/model/enet.py` + `new/train_pige_enet.py` | 需迁移脚本 |
| HRNet | `new/train_pige_hrnet.py`（模型文件需人工确认） | 需确认模型定义位置 |
| PIDNet | `new/pid.py` + `new/train_neu_pid.py` + `new/train_pige_pid.py` | 需人工确认 |
| DMC-Net | 未找到对应文件 | 需从外部引入 |
| SFNet | 未找到对应文件 | 需从外部引入 |
| FDSNet | 未找到对应文件 | 需从外部引入 |
| PGA-Net | 未找到对应文件 | 需从外部引入 |

## 4. 训练脚本映射

| 数据集 | 论文模型 | 最可能训练脚本 | 配置文件 | 需人工确认项 |
|---|---|---|---|---|
| NEU-Seg | A2MS-DefectNet-B | `new/train_neu_resnet_detailloss.py` 或 `new/train_neu_resnet_detailloss_2.py` | `new/dsmonet_resnet_detailloss.yml` | 具体使用哪个脚本和配置 |
| NEU-Seg | A2MS-DefectNet-S | `new/train_neu_resnet_eSEs.py` | 需人工确认 | 脚本是否对应 S 版模型 |
| NEU-Seg | Base-B | `new/train_neu_resnet.py` | `new/dsmonet_resnet.yml` | 需确认模型导入行 |
| NEU-Seg | Base-S | `new/train_neu_resnet_s.py` | 需人工确认 | 需确认 |
| Leather | A2MS-DefectNet-B | `new/train_pige_resnet_loss904.py` 或 `new (copy)/train_pige_resnet_loss904.py` | `new (copy)/config/dsmonet_resnet_pige_loss.yml` | 需确认 |
| Leather | A2MS-DefectNet-S | `new (copy)/train_pige_dsmor18.py` | 需人工确认 | 需确认 |
| Leather | Base-B | `new/train_pige_resnet.py` 或 `new (copy)/train_pige_resnet.py` | `new (copy)/config/dsmonet_resnet_pige.yml` | 需确认 |
| Leather | Base-S | `new (copy)/train_pige_resnet3.py` | 需人工确认 | 需确认 |

## 5. 测试脚本映射

| 数据集 | 最可能测试脚本 | 需人工确认项 |
|---|---|---|
| NEU-Seg | `new/test_neu_dsmonet.py` 或 `new/test_neu_dsmonet2.py` 或 `new/test_neu_dsmonet_detailloss.py` | 哪个脚本输出类别级 IoU |
| Leather | `new/test_pige_resnet.py` 或 `new (copy)/test_pige_resnet.py` 或 `new/test_pige_resnet_aug.py` | 哪个脚本输出类别级 IoU |

## 6. 已有权重文件

| 权重文件路径 | 大小 | 对应模型（推测） | 需人工确认项 |
|---|---|---|---|
| `new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl` | ~337MB | A2MS-DefectNet-B (Leather) | 需确认训练迭代数和数据集 |
| `new/trainedfile/finalModel_best_newmodel_neu.pkl` | ~68MB | Base-S 或 A2MS-DefectNet-S (NEU-Seg) | 需确认具体模型 |
| `new/trainedfile/finalModel_best_newmodel_usb.pkl` | ~68MB | 需人工确认 | USB 数据集权重 |
| `new/trainedfile/unet_pascal_augnew_neu.pkl` | ~64MB | U-Net (NEU-Seg) | 需确认 |
| `new (copy)/model_savePath/dsmonet_resnet_pascal_0120_pige_eSE_adapt.pkl` | 需确认 | A2MS-DefectNet-B 变体 (Leather) | 需确认 |
| `new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0126.pkl` | 需确认 | 需人工确认 | 需确认 |
| `new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl` | 需确认 | 需人工确认 | 80000 iter |
| `new (copy)/model_savePath/sdtdcnet_pige_stdc2_pige.pkl` | 需确认 | STDC2-Seg (Leather) | 需确认 |
| `new (copy)/tests/fcn_pascal_neu_psp0121.pkl` | 需确认 | PSPNet (NEU-Seg) | 需确认 |
| `new (copy)/tests/fcn_pascal_neu_unet.pkl` | 需确认 | U-Net (NEU-Seg) | 需确认 |

## 7. 数据集状态

| 数据集 | 图像尺寸 | 类别数 | 训练集 | 验证集 | 测试集 | 数据位置 | 状态 |
|---|---|---|---|---|---|---|---|
| NEU-Seg | 200x200 | 4 (含背景) | 需确认 | 需确认 | 需确认 | 未找到 `train_neu.txt`/`test_neu.txt` | **需部署数据集文件** |
| Leather (pige) | 768x768 | 8 (含背景) | 1637 | 234 | 467 | `new (copy)/dataset/pige/` | 可直接使用 |
| Yachi | 320x640 | 2 (含背景) | 需确认 | 需确认 | 需确认 | 未确认目录 | 需确认 |
