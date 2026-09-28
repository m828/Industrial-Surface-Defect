# A2MS-DefectNet-B NEU-Seg 候选权重扫评估报告

> 生成日期：2026-05-22
> 模型定义：`new/model_dsmo_rs50_eSE_adapt_detailloss.py` + `resnet50()`
> 评估协议：NEU-Seg eval_protocol_lock.md (test_neu.txt, 840 张, 200×200, 4 类)

---

## 一、候选权重搜索

搜索范围：
- `/workspace/Industrial Surface Defect/new/` (含 model_savePath/)
- `/workspace/Industrial Surface Defect/new (copy)/` (含 model_savePath/, model_savePaths/)
- `/workspace/Industrial Surface Defect/subregion unet/` (含 trainedfile/)

搜索关键词：eSE, adapt, detailloss, detail, neu, rs50, dsmor50, dsmonet, a2ms

### 搜索结果

共找到 16 个与 NEU-Seg 相关的权重文件，但只有 **1 个** 是 A2MS-DefectNet-B 候选：

| # | 权重路径 | 文件大小 | 可能模型 | 可能数据集 | 是否 A2MS-B NEU |
|---|---:|---|---|---|---|
| 1 | `new/model_savePath/dsmonet_resnet_pascal_0109_neu_eSE_adapt_datailloss.pkl` | 336 MB | A2MS-B | NEU-Seg | ✅ **唯一候选** |
| 2 | `new/model_savePath/dsmonet_resnet_pascal_0110_neu_120000.pkl` | 337 MB | Base-B | NEU-Seg | ❌ 无 eSE/adapt |
| 3 | `new/model_savePath/dsmonet_resnet_pascal_neu_a2ms_s.pkl` | 161 MB | A2MS-S | NEU-Seg | ❌ Small (RS18) |
| 4 | `new/model_savePath/dsmonet_resnet_pascal_neu_base_s.pkl` | 161 MB | Base-S | NEU-Seg | ❌ Small (RS18) |
| 5 | `new/model_savePath/ddr_pascal_0119_neu_ddr_nolabels_23.pkl` | 233 MB | DDRNet | NEU-Seg | ❌ 不同架构 |
| 6 | `new/trainedfile/finalModel_best_newmodel_neu.pkl` | 68 MB | Sub-region UNet | NEU-Seg | ❌ 不同架构 |
| 7 | `new/trainedfile/unet_pascal_augnew_neu.pkl` | 68 MB | U-Net | NEU-Seg | ❌ 不同架构 |
| 8 | `new (copy)/model_savePaths/pid_neu_0119_neu_pid.pkl` | 89 MB | PIDNet-S | NEU-Seg | ❌ 不同架构 |
| 9 | `new (copy)/tests/fcn_pascal_neu_psp0121.pkl` | 562 MB | PSPNet/FCN | NEU-Seg | ❌ 不同架构 |
| 10 | `new (copy)/tests/fcn_pascal_neu_unet.pkl` | 355 MB | U-Net | NEU-Seg | ❌ 不同架构 |
| 11 | `new (copy)/trainedfile/finalModel_best_newmodel_neu.pkl` | 68 MB | Sub-region UNet | NEU-Seg | ❌ 不同架构 |
| 12 | `new (copy)/trainedfile/unet_pascal_augnew_neu.pkl` | 68 MB | U-Net | NEU-Seg | ❌ 不同架构 |
| 13 | `subregion unet/trainedfile/finalModel_best_newmodel_neu.pkl` | 68 MB | Sub-region UNet | NEU-Seg | ❌ 不同架构 |
| — | `new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl` | 338 MB | A2MS-B | **Leather** | ❌ Pige 数据集 |
| — | `new (copy)/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl` | 338 MB | A2MS-B | **Leather** | ❌ Pige 数据集 |
| — | `new (copy)/model_savePath/dsmonet_resnet_pascal_0120_pige_eSE_adapt.pkl` | 337 MB | A2MS-B variant | **Leather** | ❌ Pige 数据集 |

---

## 二、唯一候选权重详情

| 属性 | 值 |
|---|---|
| 文件路径 | `/workspace/Industrial Surface Defect/new/model_savePath/dsmonet_resnet_pascal_0109_neu_eSE_adapt_datailloss.pkl` |
| 文件大小 | 336 MB (352,321,140 bytes) |
| 命名解析 | dsmonet_resnet_pascal → DSMONet + ResNet backbone; 0109 → 1月9日; neu → NEU-Seg; eSE_adapt_datailloss → eSE + AdaptiveChannelWeight + DetailLoss (typo: "datailloss" for "detailloss") |
| 对应模型文件 | `new/model_dsmo_rs50_eSE_adapt_detailloss.py` |
| 对应训练脚本 | `new/train_neu_resnet_detailloss.py` |
| 对应配置文件 | `new/dsmonet_resnet_detailloss.yml` |
| 训练日志 | `new/runs_other/train_all/a2ms_b_neu.log` |
| 训练日志 mIoU | 0.7131 (best at iter 42000/60000) |
| 训练日志 per-class IoU | bg=0.9505, crazing=0.5391, inclusion=0.8302, patches=0.5324 |

---

## 三、权重加载测试

```
[需人工执行] — 确认 conda 环境可运行后执行以下测试
```

```bash
cd "/workspace/Industrial Surface Defect/new"
source /opt/conda/etc/profile.d/conda.sh && conda activate base

python -c "
import torch
import sys
sys.path.insert(0, '.')
from model_dsmo_rs50_eSE_adapt_detailloss import DSMONet
from model_resnet import resnet50

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DSMONet(num_classes=4, backbone=resnet50()).to(device)
print(f'Model created. Params: {sum(p.numel() for p in model.parameters()):,}')

# 加载权重
weight_path = 'model_savePath/dsmonet_resnet_pascal_0109_neu_eSE_adapt_datailloss.pkl'
checkpoint = torch.load(weight_path, map_location=device)

# 检查 checkpoint 结构
print(f'Checkpoint keys: {list(checkpoint.keys())}')
if 'model_state' in checkpoint:
    state = checkpoint['model_state']
    print(f'Epoch: {checkpoint.get(\"epoch\", \"N/A\")}')
    print(f'Best IoU: {checkpoint.get(\"best_iou\", \"N/A\")}')
    # 检查 state_dict 匹配
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    print(f'Missing keys: {len(missing)}')
    print(f'Unexpected keys: {len(unexpected)}')
    if missing: print(f'First 5 missing: {list(missing)[:5]}')
    if unexpected: print(f'First 5 unexpected: {list(unexpected)[:5]}')
    
    # 尝试加载
    model.load_state_dict(state)
    print('Weight loading: SUCCESS')
else:
    print(f'Unknown checkpoint format. Keys: {list(checkpoint.keys())[:10]}')
"
```

---

## 四、候选权重评估结果

### 权重 #1: dsmonet_resnet_pascal_0109_neu_eSE_adapt_datailloss.pkl

| 指标 | 值 | 来源 |
|---|---|---|
| 是否可加载 | [需人工确认] | 待执行 |
| mIoU (训练日志) | 0.7131 | `runs_other/train_all/a2ms_b_neu.log`, iter 42000 |
| per-class IoU (训练日志) | bg=0.9505, crazing=0.5391, inclusion=0.8302, patches=0.5324 | 同上 |
| 是否接近历史 91.3 | ❌ 否，差距 -19.99pp | — |
| 能否作为历史主结果候选 | ❌ 不可 | — |

### 评估结论

**候选权重扫评估结果：无法找到接近历史 91.3 的 NEU-Seg 权重。**

- 搜索范围覆盖全部 3 个代码目录
- 仅发现 1 个 A2MS-B NEU-Seg 候选权重
- 该权重的训练日志 mIoU = 71.31，远低于历史 91.3
- 无其他历史 A2MS-B NEU 权重存在于当前服务器

### 判定

```
❌ 必须重训 A2MS-DefectNet-B NEU-Seg
```

历史 91.3 的权重文件不在当前服务器上。当前唯一的候选权重 (0109) 训练异常 (mIoU=71.31, crazing IoU=0.5391)。在确认训练配置无误之前，不应使用此权重。

---

## 五、输出文件

| 文件 | 路径 | 状态 |
|---|---|---|
| 候选权重清单 | `experiments/results/a2ms_b_neu_candidate_weights.md` | ✅ 已生成（本文件） |
| 权重扫评估 CSV | `experiments/results/a2ms_b_neu_weight_sweep.csv` | [待] 需执行评估后填入 |
| 权重扫评估摘要 | `experiments/results/a2ms_b_neu_weight_sweep_summary.md` | [待] 同上 |

### a2ms_b_neu_weight_sweep.csv 模板

```csv
weight_path,file_size_mb,model_file,loadable,epoch,best_iou_in_ckpt,mIoU_eval,bg_iou,crazing_iou,inclusion_iou,patches_iou,close_to_91.3,notes
new/model_savePath/dsmonet_resnet_pascal_0109_neu_eSE_adapt_datailloss.pkl,336,model_dsmo_rs50_eSE_adapt_detailloss.py,TBD,TBD,TBD,0.7131,0.9505,0.5391,0.8302,0.5324,false,"唯一候选权重;训练日志提取;crazing 异常低"
```
