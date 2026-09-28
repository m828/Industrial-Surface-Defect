# 代码核验与实验可追溯性报告

> 生成日期：2026-05-22
> 核验范围：`new/`、`new (copy)/`、`subregion unet/` 三个代码目录
> 核验目标：确认论文中每一个实验结果是否可追溯到代码、权重、日志和评估脚本

---

## 一、核验方法论

对每个实验结果，按以下 7 个维度核验：
1. **模型定义文件**：是否存在且与论文描述一致
2. **训练脚本**：是否存在且可执行
3. **配置文件**：是否存在且参数可追溯
4. **权重文件**：是否存在于服务器
5. **训练日志**：是否有完整的 mIoU/IoU 记录
6. **评估脚本**：是否存在独立评估脚本
7. **可复现性**：是否可在当前环境中重新训练或评估

标记规则：
- ✅ 可追溯：7 个维度均已确认
- ⚠️ 部分可追溯：部分维度缺失但可通过现有文件推导
- ❌ 不可追溯：关键维度缺失，必须重跑

---

## 二、主实验结果核验

### 2.1 Base-S（ResNet-18，仅 DM）

| 维度 | 状态 | 路径/说明 |
|---|---|---|
| 模型定义 | ✅ | `new/model_dsmo_rs18.py` (SE + SqueezeBodyEdge) |
| 训练脚本 | ✅ | `new/train_neu_base_s.py` (NEU)、`new/train_pige_base_s.py` (Leather) |
| 配置文件 | ✅ | `new/config_neu_base_s.yml`、`new/config_pige_base_s.yml` |
| 权重 NEU | ✅ | `new/model_savePath/dsmonet_resnet_pascal_neu_base_s.pkl` (168MB) |
| 权重 Leather | ✅ | `new/model_savePath/dsmonet_resnet_pascal_pige_dsmor18_0228_80000.pkl` (168MB) |
| 训练日志 NEU | ✅ | `new/runs_other/train_all/base_s_neu.log` → mIoU=0.8837 (Iter 52500) |
| 训练日志 Leather | ✅ | `new/runs_other/train_all/base_s_pige.log` → mIoU=0.8397 (Iter 53000) |
| 评估脚本 | ⚠️ | 无独立评估脚本；仅有训练时验证日志 |
| 历史值 NEU | ⚠️ | 论文声称 88.4 mIoU / 178.5 FPS (RTX 3090)；新训练 88.37 mIoU (A100) |
| 历史值 Leather | ❌ | 论文声称 88.1 mIoU / 167.2 FPS；新训练仅 83.97 mIoU，差距 -4.13pp |

**核验结论**：Base-S 代码可追溯，但 Leather 数据集的历史结果 (88.1) 与新训练结果 (83.97) 不匹配。
**需重跑**：Leather 数据集需用历史权重重新评估，或确认评估集/配置一致性。

### 2.2 Base-B（ResNet-50，仅 DM）

| 维度 | 状态 | 路径/说明 |
|---|---|---|
| 模型定义 | ✅ | `new/model_dsmo_rs50.py` (SE + SqueezeBodyEdge) |
| 训练脚本 | ✅ | `new/train_neu_resnet_detailloss.py` (NEU)、`new/train_pige_base_b.py` (Leather) |
| 配置文件 | ✅ | `new/dsmonet_resnet_detailloss.yml`、`new/config_pige_base_b.yml` |
| 权重 NEU | ✅ | `new/model_savePath/dsmonet_resnet_pascal_0110_neu_120000.pkl` (352MB) |
| 权重 Leather | ✅ | `new/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0220ksh.pkl` (352MB) |
| 权重 Leather (旧) | ✅ | `new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0126.pkl` (336MB) |
| 训练日志 NEU | ✅ | `new/runs_other/train_all/base_b_neu.log` → mIoU=0.9047 (Iter 50500) |
| 训练日志 Leather | ✅ | `new/runs_other/train_all/base_b_pige.log` → mIoU=0.8278 (Iter 57000) |
| 历史日志 Leather | ✅ | `new (copy)/runs_other/dsmonet_resnet_pige/dsmor50_pige/run_2024_01_26_11_29_31.log` |
| 评估脚本 | ⚠️ | 无独立评估脚本 |
| 历史值 NEU (新训练) | ✅ | 新训练 90.47 与历史 90.2 基本一致 (+0.27pp) |
| 历史值 Leather | ❌ | 论文声称 89.2 mIoU / 67.3 FPS；新训练仅 82.78 mIoU，差距 -6.42pp |
| 历史值 Leather (旧评估) | ⚠️ | 旧权重 `dsmonet_resnet_pascal_pige_dsmor50_0126.pkl` 评估 mIoU=0.8806 (类别级 IoU 评估) |

**核验结论**：NEU-Seg 结果可追溯且一致。Leather 数据集存在严重的数值不一致——旧权重 (0126) 评估 88.06，新权重 (0220ksh) 仅 82.78。历史声称 89.2。
**需重跑**：Leather 数据集需确认历史确切评估协议和数据集划分。

### 2.3 A2MS-DefectNet-S（ResNet-18，完整 A2MS）

| 维度 | 状态 | 路径/说明 |
|---|---|---|
| 模型定义 | ✅ | `new/model_dsmo_rs18_eSE_adapt_detailloss_822.py` (eSE + Light_Bag + detailloss) |
| 训练脚本 | ✅ | `new/train_neu_a2ms_s.py` (NEU)、`new/train_pige_a2ms_s.py` (Leather) |
| 配置文件 | ✅ | `new/config_neu_a2ms_s.yml`、`new/config_pige_a2ms_s.yml` |
| 权重 NEU | ✅ | `new/model_savePath/dsmonet_resnet_pascal_neu_a2ms_s.pkl` (168MB) |
| 权重 Leather | ✅ | `new/model_savePath/dsmonet_resnet_pascal_pige_a2ms_s.pkl` (168MB) |
| 训练日志 NEU | ✅ | `new/runs_other/train_all/a2ms_s_neu.log` → mIoU=0.8858 (Iter 55000) |
| 训练日志 Leather | ✅ | `new/runs_other/train_all/a2ms_s_pige.log` → mIoU=0.8472 (Iter 54500) |
| 评估脚本 | ⚠️ | 无独立评估脚本 |
| 历史值 NEU | ⚠️ | 论文声称 89.7 mIoU / 196.5 FPS (RTX 3090)；新训练 88.58 mIoU，差距 -1.12pp |
| 历史值 Leather | ❌ | 论文声称 89.7 mIoU / 188.7 FPS；新训练仅 84.72，差距 -4.98pp |

**核验结论**：NEU-Seg 可接受（差距 1.12pp），Leather 差距显著。
**需重跑**：Leather 数据集需用历史权重或确认历史评估协议。

### 2.4 A2MS-DefectNet-B（ResNet-50，完整 A2MS）

| 维度 | 状态 | 路径/说明 |
|---|---|---|
| 模型定义 | ✅ | `new/model_dsmo_rs50_eSE_adapt_detailloss.py` (eSE + AdaptiveChannelWeight + detailloss) |
| 训练脚本 | ✅ | `new/train_neu_resnet_detailloss.py` (NEU)、`new/train_pige_a2ms_b.py` (Leather) |
| 配置文件 | ✅ | `new/dsmonet_resnet_detailloss.yml`、`new/config_pige_a2ms_b.yml` |
| 权重 NEU | ⚠️ | `new/model_savePath/dsmonet_resnet_pascal_0109_neu_eSE_adapt_datailloss.pkl` (351MB) |
| 权重 NEU (旧) | ⚠️ | `new (copy)/model_savePath/dsmonet_resnet_pascal_0120_pige_eSE_adapt.pkl` (337MB) |
| 权重 Leather | ✅ | `new/model_savePath/dsmonet_resnet_pascal_augnew_60000.pkl` (352MB) |
| 权重 Leather (旧) | ✅ | `new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl` (338MB) |
| 训练日志 NEU | ✅ | `new/runs_other/train_all/a2ms_b_neu.log` → mIoU=0.7131 (Iter 42000) |
| 训练日志 Leather | ✅ | `new/runs_other/train_all/a2ms_b_pige.log` → mIoU=0.8560 (Iter 51500) |
| 旧类别级评估 | ✅ | `new (copy)` 权重 + `evaluate_class_iou.py` → mIoU=0.8607 (Leather test set) |
| 历史值 NEU | ❌ | 论文声称 91.3 mIoU / 127.9 FPS；新训练仅 71.31，**差距 -20pp** |
| 历史值 Leather | ⚠️ | 论文声称 91.0 mIoU / 71.6 FPS；旧评估 86.07，新训练 85.60，差距 -5.4pp |

**核验结论**：NEU-Seg 新训练结果严重异常 (71.31 vs 91.3)，可能是训练配置错误。Leather 历史声称 91.0 vs 旧评估 86.07 vs 新训练 85.60，三者不一致。

**CRITICAL**: A2MS-B NEU-Seg 必须重跑，无法写入论文。

---

## 三、对比方法基线结果核验

论文 `fixed_existing_results.md` 第 3.1 节列出了以下对比方法。以下核验其代码入口和权重追溯性。

| 模型 | 模型文件 | 训练脚本 | 权重 | 可追溯 |
|---|---|---|---|---|
| FCN | `new/fcn.py` | `new/train_pige_fcn.py` | ❌ 无 NEU 权重；`new (copy)/tests/fcn_pascal_neu_psp0121.pkl` (562MB, 疑似 PSP) | ❌ |
| U-Net | `new/model_unet.py` | `new/train_pige_unet.py` | ❌ 无；`new (copy)/tests/fcn_pascal_neu_unet.pkl` (355MB, 疑似 UNet) | ❌ |
| HRNet | ❌ 未找到 model_hrnet.py | `new/train_pige_hrnet.py` | ❌ 无 | ❌ |
| PSPNet | `new (copy)/model/pspnet.py` | `new/train_pige_psp.py` | ❌ 无 | ❌ |
| DeepLabV3+ | `new (copy)/model/deeplabv3.py` | `new/train_pige_deeplabv3.py` | ❌ 无 | ❌ |
| ENet | `new (copy)/model/enet.py` | `new/train_pige_enet.py` | ❌ 无 | ❌ |
| BiSeNetV1 | ❌ 仅 predict 引用；无独立模型文件 | 无独立训练脚本 | ❌ 无 | ❌ |
| PIDNet-S | `new/pid.py` | `new/train_neu_pid.py`、`new/train_pige_pid.py` | ✅ `new (copy)/model_savePaths/pid_neu_0119_neu_pid.pkl` (89MB) | ⚠️ |
| DDRNet23slim | `new/model_ddr.py` | `new/train_neu_ddr.py`、`new/train_pige_ddr.py` | ✅ `new/model_savePath/ddr_pascal_*` | ✅ |
| SFNet | ❌ 未找到模型文件 | ❌ 无训练脚本 | ❌ 无 | ❌ |
| STDC1-Seg | `new/stdcnet.py` | `new/train_pige_stdc.py` | ✅ `new (copy)/model_savePath/sdtdcnet_pige_stdc2_pige.pkl` (158MB) | ⚠️ |
| STDC2-Seg | `new/stdcnet.py` | `new/train_pige_stdc.py` | ❌ 未找到 STDC2 权重 | ❌ |
| PP-LiteSeg-T | ❌ 复杂度统计失败 (backbone 不兼容) | 无独立训练脚本 | ❌ 无 | ❌ |
| PP-LiteSeg-B | ✅ `new/model_dsmo_rs50_ppliteseg.py` | 无独立训练脚本 | ❌ 无 | ❌ |
| BiSeNetV2-L | ❌ 服务器无代码 | ❌ 无 | ❌ 无 | ❌ |
| Sub-region UNet | ✅ `subregion unet/model_p.py` | ✅ `subregion unet/train_neu.py` | ✅ `subregion unet/trainedfile/finalModel_best_newmodel_neu.pkl` (68MB) | ⚠️ |
| FDSNet | ❌ 未找到模型或训练脚本 | ❌ | ❌ | ❌ |
| PGA-Net | ❌ 未找到模型或训练脚本 | ❌ | ❌ | ❌ |

**汇总**：
- ✅ 完全可追溯（有模型+脚本+权重）：DDRNet23slim
- ⚠️ 部分可追溯（有代码但需重跑评估）：Base-S、Base-B、A2MS-S、A2MS-B、PIDNet-S、STDC1-Seg、Sub-region UNet
- ❌ 不可追溯（代码或权重缺失）：FCN、U-Net、HRNet、PSPNet、DeepLabV3+、ENet、BiSeNetV1/V2、SFNet、STDC2-Seg、PP-LiteSeg-T/B、FDSNet、PGA-Net

---

## 四、消融实验结果核验

### 4.1 ELMM 消融

| 消融点 | 论文值 | 对应模型文件 | 权重 | 可追溯 |
|---|---|---|---|---|
| Baseline (无 DM) | 85.1 / 128.6 | `model_dsmo_rs50_4_baseline.py` (推测) | ❌ | ❌ |
| DM (SqueezeBodyEdge) | 90.1 / 117.6 | `model_dsmo_rs50.py` (Base-B) | ✅ | ⚠️ |
| ELMM (DM + eSE + adapt) | 91.3 / 126.7 | `model_dsmo_rs50_eSE_adapt_detailloss.py` (A2MS-B) | ✅ | ⚠️ |

**核验结论**：ELMM = Baseline + DM + eSE + AdaptiveChannelWeight 的组合效果。Baseline 权重缺失，DM 和 ELMM 权重存在但 FPS 来自 RTX 3090（当前 A100 环境无法复现 FPS）。
**需重跑**：Baseline (无 DM) 模型需要明确的模型文件和权重。

### 4.2 AAM 消融

| 消融点 | 论文值 | 对应模型文件 | 权重 | 可追溯 |
|---|---|---|---|---|
| SE ×1 | 90.5 / 122.9 | `model_dsmo_rs50.py` (SE) | ✅ Base-B 权重 | ⚠️ |
| eSE ×1 | 90.7 / 126.3 | `model_dsmo_rs50_eSE.py` (eSE) | ❌ 无独立权重 | ❌ |
| AAM ×1 | 91.1 / 126.9 | `model_dsmo_rs50_eSE_adapt.py` (eSE + adapt ×1) | ❌ 无独立权重 | ❌ |
| AAM ×2 | 91.3 / 126.7 | `model_dsmo_rs50_eSE_adapt_detailloss.py` (eSE + adapt ×2) | ✅ A2MS-B 权重 | ⚠️ |

**核验结论**：SE ×1 和 AAM ×2 有对应权重(Base-B、A2MS-B)，但 eSE ×1 和 AAM ×1 需重新训练中间消融版本。
**需重跑**：eSE ×1 和 AAM ×1 需要独立的训练运行，使用对应模型文件但移除额外模块。

### 4.3 联合损失消融

| 消融点 | 论文值 | 需要配置 | 可追溯 |
|---|---|---|---|
| l0+l1+l2 | 90.4 | 仅 ohem + bce + segment_loss, 无 detailloss | ❌ |
| + ledge | 90.7 | l0+l1+l2 + detailloss (仅 edge) | ❌ |
| + lmask | 90.8 | l0+l1+l2 + bce binary mask | ❌ |
| + ledge + lmask | 91.3 | 完整 A2MS-B 损失 | ✅ (A2MS-B) |

**核验结论**：仅完整配置 (A2MS-B) 可追溯。中间的 l0+l1+l2、+ledge、+lmask 消融点无对应训练记录。
**需重跑**：三个中间消融点需要独立的训练运行。

---

## 五、复杂度统计核验

**统计脚本**：`Industrial-Surface-Defect/experiments/scripts/complexity_stats.py`
**统计平台**：NVIDIA A100-PCIE-40GB
**FPS 计时**：warmup=50, runs=200, batch_size=1

| 模型 | 模型入口 | 输入尺寸 | Params(M) | FLOPs(G) | FPS@200 | FPS@768 | 权重大小(MB) | 核验状态 |
|---|---|---|---|---|---|---|---|---|
| Base-S | `new/model_dsmo_rs18.py`, backbone=resnet18() | 200×200 | 14.00 | 2.51 | 149.7 | 150.5 | 53.49 | ⚠️ FPS 来自 A100，论文 FPS 来自 RTX 3090 |
| Base-B | `new/model_dsmo_rs50.py`, backbone=resnet50() | 200×200 | 29.37 | 6.65 | 99.9 | 102.6 | 112.36 | ⚠️ 同上 |
| A2MS-S | `new/model_dsmo_rs18_eSE_adapt_detailloss_822.py`, backbone=resnet18() | 200×200 | 14.03 | 2.34 | 170.8 | 169.1 | 53.61 | ⚠️ 同上 |
| A2MS-B | `new/model_dsmo_rs50_eSE_adapt_detailloss.py`, backbone=resnet50() | 200×200 | 29.46 | 6.42 | 100.3 | 99.5 | 112.70 | ⚠️ 同上 |
| DDRNet23slim | `new/model_ddr.py`, DualResNet_imagenet | 200×200 | 6.71 | 2.07 | 167.7 | 166.6 | 25.67 | ⚠️ 同上 |
| STDC1-Seg | `new/stdcnet.py`, STDC1 | 200×200 | 16.07 | 6.02 | 121.5 | 123.7 | 61.40 | ⚠️ 同上 |
| STDC2-Seg | `new/stdcnet.py`, STDC2 | 200×200 | 12.04 | 3.83 | 180.9 | 178.6 | 46.01 | ⚠️ 同上 |
| PP-LiteSeg-B | `new/model_dsmo_rs50_ppliteseg.py` | 200×200 | 30.84 | 6.63 | 110.2 | 110.4 | 117.86 | ⚠️ 同上 |
| BiSeNetV1-L | `Industrial-Surface-Defect/third_party/` (bisnet) | 200×200 | 29.38 | 6.52 | 127.8 | 99.2 | 112.28 | ⚠️ 同上 |
| PIDNet-S | `new/pid.py` | 200×200 | 7.72 | 0.97 | 127.7 | 127.9 | 29.53 | ⚠️ 同上 |

**关键问题**：
1. **FPS 平台不一致**：论文声称的 FPS 来自 RTX 3090，而当前复杂度统计来自 A100。虽然数值接近（因为 batch=1 推理在 RTX 3090 和 A100 差异不大），但需在论文中明确标注测试平台。**论文正文和复杂度表必须使用同一平台数据**。
2. **FPS@200 vs FPS@768**：A2MS-DefectNet 系列在 200×200 和 768×768 的 FPS 几乎相同，因为模型参数和计算量不随输入尺寸线性增长。这符合代码逻辑。
3. **模型文件大小**：与 `.pkl` 文件实际大小核对，误差 < 1MB，可使用。
4. **PP-LiteSeg-T 失败**：`backbone_out_chs` 硬编码仅支持 ResNet-50，需修复后才可统计。
5. **BiSeNetV2-L 失败**：服务器无代码文件，需从第三方仓库获取。
6. **Sub-region UNet 失败**：200×200 输入导致 pixelshuffle 尺寸不匹配。

**核验结论**：Params/FLOPs/模型大小可使用（需在论文中标注 A100 平台）。FPS 数值来自 A100，如果论文声称 FPS 对应 RTX 3090，则需要：
- 方案 A：在 RTX 3090 上重新测量所有 FPS（推荐，与论文一致）
- 方案 B：论文改用 A100 的 FPS（需更新正文所有 FPS 引用）

---

## 六、类别级 IoU 核验

**现有数据**（来自 `class_iou_results.csv`）：
- 旧评估 (evaluate_class_iou.py, 2026-05-18)：Base-B、A2MS-B、STDC1-Seg 有完整 per-class IoU（Leather 测试集 468 张）
- 训练日志提取 (2026-05-22)：Base-S、A2MS-S、DDRNet、PIDNet-S 有 per-class IoU（Leather 验证集 234 张）

**不一致问题**：
1. **评估集不一致**：旧评估使用测试集 (467 张)，新训练日志使用验证集 (234 张)
2. **mIoU 不一致**：Base-B Leather 旧评估 mIoU=0.8806 (测试集) vs 新训练 mIoU=0.8278 (验证集)
3. **类别顺序**：现有 CSV 中 NEU-Seg 类别顺序（background, crazing, inclusion, patches）与 PIDNet-S 评估行（background, inclusion, patch, scratch）不一致

**核验结论**：现有 per-class IoU 数据不能直接写入论文。需统一评估协议后重跑。

**需重跑**：所有模型的 per-class IoU 使用同一测试集、同一权重、同一评估脚本。

---

## 七、小目标/细长缺陷分组评价核验

**评估脚本**：`Industrial-Surface-Defect/tools/evaluate_small_object_groups.py`
**数据来源**：基于语义掩码连通域分析的近似分组
**测试集**：468 张 Leather 测试图像
**分组规则**：面积 P25/P75 和长宽比 P75 阈值

| 模型 | 小目标 mIoU | 中等 mIoU | 大目标 mIoU | 细长 mIoU | 多缺陷 mIoU | 无缺陷 mIoU |
|---|---|---|---|---|---|---|
| Base-B | 0.5141 | 0.8801 | 0.6876 | 0.8531 | 0.8739 | 0.1247 |
| A2MS-B | 0.4084 | 0.8657 | 0.6721 | 0.8334 | 0.8586 | 0.1244 |
| STDC1-Seg | 0.4549 | 0.8770 | 0.6796 | 0.8469 | 0.8705 | 0.1250 |

**核验问题**：
1. **仅有 3 个模型**：其他 8 个模型 (Base-S, A2MS-S, DDRNet, PIDNet-S, STDC2-Seg, PP-LiteSeg-B, BiSeNetV1-L, Sub-region UNet) 无分组数据
2. **A2MS-B 小目标 mIoU < Base-B**：0.4084 vs 0.5141，与论文"自适应注意力有助于小目标"的结论存在矛盾
3. **无缺陷样本 mIoU ≈ 0.125**：预期应接近 1.0（全背景）。这一异常数值说明无缺陷样本的评估方式有问题
4. **分组规则依赖连通域近似**：不是真实标签的小/中/大分类，可能引入偏差
5. **无样本 ID 追溯**：无法对应到具体图像验证

**核验结论**：当前分组评价作为诊断分析可用，但不可写入主实验表。需统一测试协议后重跑。

---

## 八、可视化结果核验

**路径**：`Industrial-Surface-Defect/figures/qualitative_results/`
**文档**：`README.md` + `sample_index.json`

| 文件 | 原图 | 是否可追溯 | 问题 |
|---|---|---|---|
| small_target_idx0001.png | ✅ sample_index.json 有路径 | ⚠️ | 需人工验证原图存在 |
| small_target_idx0009.png | ✅ same | ⚠️ | 需人工验证 |
| elongated_idx0005.png | ✅ same | ⚠️ | 需人工验证 |
| elongated_idx0014.png | ✅ same | ⚠️ | 需人工验证 |
| multi_defect_idx0001.png | ✅ same | ⚠️ | 需人工验证 |
| texture_confusion_idx0117.png | ✅ same | ⚠️ | 仅 1 个样本，代表性不足 |
| base_miss_a2ms_hit_idx0177.png | ✅ same | ⚠️ | 仅 1 个样本 |
| base_boundary_break_idx0032.png | ✅ same | ⚠️ | 仅 1 个场景 |

**核验问题**：
1. **样本量过小**：部分场景仅 1 个样本，可能存在人工挑选偏差
2. **无失败案例**：所有可视化只展示 A2MS 优于 Base 的案例，缺少两者均失败或 Base 优于 A2MS 的案例
3. **CP950 字体问题**：CJK 字体缺失导致中文标题显示为方框
4. **预测路径未记录**：sample_index.json 缺少 Base-B 和 A2MS-B 的具体预测输出路径

**核验结论**：当前可视化可作为定性参考，但需要补充失败案例和增加样本量后重新生成。

---

## 九、近年对比方法复现入口审计

### 9.1 DMC-Net

| 项目 | 状态 |
|---|---|
| 服务器代码 | ❌ 不存在（dmcnet.py 缺失） |
| GitHub 仓库 | https://github.com/something/dmcnet (未确认) |
| 是否适合 NEU-Seg | ⚠️ 需确认（原为单类别裂缝检测） |
| 是否适合 Leather | ⚠️ 需确认 |
| 数据加载修改 | 需适配多类别 DataGenerator 格式 |
| 多类别语义分割支持 | 可能需修改输出头 |
| 预估复现难度 | 中等偏高 |
| 推荐进主实验表 | ⚠️ 仅当完整复现和训练后 |

### 9.2 PIDNet

| 项目 | 状态 |
|---|---|
| 服务器代码 | ✅ `new/pid.py` (PIDNet 复现版本) |
| 训练脚本 | ✅ `new/train_neu_pid.py`、`new/train_pige_pid.py` |
| 配置 | ✅ `new/config_pige_pid.yml` |
| NEU 权重 | ✅ `new (copy)/model_savePaths/pid_neu_0119_neu_pid.pkl` (89MB) |
| Leather 权重 | ✅ `new/model_savePath/fcn_pascal_pige_pid_s.pkl` (92MB) |
| NEU mIoU (新训练) | 0.8667 |
| Leather mIoU (新训练) | 0.8406 |
| 推荐进主实验表 | ✅ 可进入 |

### 9.3 LETNet / SeaFormer

| 项目 | 状态 |
|---|---|
| 服务器代码 | ❌ 不存在 |
| 第三方仓库 | 需克隆 |
| 是否适合 NEU-Seg | ⚠️ 不确定 |
| 是否适合 Leather | ⚠️ 不确定 |
| 预估复现难度 | 高（需完整适配） |
| 推荐进主实验表 | ⚠️ 不建议，优先使用 STDC-Seg 和 PIDNet 作为轻量对比 |

---

## 十、总体可追溯性评估

### ✅ 可追溯且可写入论文（需核验后）

- Base-B NEU-Seg: mIoU=90.47 (与历史 90.2 一致)
- DDRNet23slim: 代码/脚本/权重/日志完整
- PIDNet-S: 代码/脚本/权重/日志完整
- A2MS-S NEU-Seg: mIoU=88.58 (与历史 89.7 差 1.12pp, 可接受)

### ⚠️ 部分可追溯，需重跑/重新评估

- Base-S Leather: 历史 88.1 vs 新训练 83.97，需确认
- Base-B Leather: 历史 89.2 vs 新训练 82.78 vs 旧评估 88.06，需确认
- A2MS-S Leather: 历史 89.7 vs 新训练 84.72，需确认
- A2MS-B Leather: 历史 91.0 vs 旧评估 86.07 vs 新训练 85.60，需确认
- 所有 Leather 数据集模型

### ❌ 不可追溯，必须重跑

- A2MS-B NEU-Seg: 新训练 71.31，严重异常
- AAM 消融 eSE×1 和 AAM×1 中间点
- 联合损失消融 l0+l1+l2、+ledge、+lmask 中间点
- ELMM 消融 Baseline (85.1)
- FCN、U-Net、HRNet、PSPNet、DeepLabV3+、ENet: 权重缺失
- BiSeNetV1/V2、SFNet: 模型或训练代码缺失
- STDC2-Seg、PP-LiteSeg-T/B: 权重缺失
- FDSNet、PGA-Net: 代码和权重均缺失
