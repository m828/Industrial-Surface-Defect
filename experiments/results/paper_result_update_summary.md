# 论文实验结果更新汇总

> 生成日期：2026-05-22
> 数据来源：11 个新训练模型的训练日志（best mIoU 验证点）
> 核验状态：所有新结果 verified=false，待人工核验

---

## 一、新训练结果总览

### NEU-Seg 数据集（200×200，4类，测试集 840 张）

| 模型 | 主干 | mIoU | background | crazing | inclusion | patches | 权重文件 |
|---|---|---:|---:|---:|---:|---:|---|
| **Base-B** | ResNet-50 | **0.9047** | 0.9846 | 0.8330 | 0.9268 | 0.8744 | dsmonet_resnet_pascal_0110_neu_120000.pkl |
| **A2MS-DefectNet-S** | ResNet-18 | **0.8858** | 0.9819 | 0.8059 | 0.9257 | 0.8297 | dsmonet_resnet_pascal_neu_a2ms_s.pkl |
| **DDRNet23slim** | DDRNet-Internal | **0.8843** | 0.9819 | 0.8055 | 0.9255 | 0.8245 | ddr_pascal_0119_neu_ddr_nolabels_23.pkl |
| **Base-S** | ResNet-18 | **0.8837** | 0.9819 | 0.8048 | 0.9256 | 0.8225 | dsmonet_resnet_pascal_neu_base_s.pkl |
| **A2MS-DefectNet-B** | ResNet-50 | **0.7131** | 0.9505 | 0.5391 | 0.8302 | 0.5324 | dsmonet_resnet_pascal_0109_neu_eSE_adapt_datailloss.pkl |

### Leather 数据集（768×768，8类，验证集 234 张）

| 模型 | 主干 | mIoU | background | open_wound | scratch | brand_mark | hole | skin_disease | rotten_surface | wart | 权重文件 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **A2MS-DefectNet-B** | ResNet-50 | **0.8560** | 0.9918 | 0.6950 | 0.6383 | 0.8969 | 0.9876 | 0.9671 | 0.7351 | 0.9364 | dsmonet_resnet_pascal_augnew_60000.pkl |
| **A2MS-DefectNet-S** | ResNet-18 | **0.8472** | 0.9909 | 0.6483 | 0.6280 | 0.8787 | 0.9874 | 0.9592 | 0.7535 | 0.9315 | dsmonet_resnet_pascal_pige_a2ms_s.pkl |
| **PIDNet-S** | PIDNet-Internal | **0.8406** | 0.9889 | 0.6341 | 0.5665 | 0.8360 | 0.9863 | 0.9511 | 0.8476 | 0.9143 | fcn_pascal_pige_pid_s.pkl |
| **Base-S** | ResNet-18 | **0.8397** | 0.9898 | 0.6585 | 0.5975 | 0.8689 | 0.9844 | 0.9508 | 0.7441 | 0.9233 | dsmonet_resnet_pascal_pige_dsmor18_0228_80000.pkl |
| **DDRNet23slim** | DDRNet-Internal | **0.8317** | 0.9889 | 0.6814 | 0.5286 | 0.8495 | 0.9847 | 0.9501 | 0.7599 | 0.9104 | ddr_pascal_pige_ddr23s.pkl |
| **Base-B** | ResNet-50 | **0.8278** | 0.9896 | 0.6171 | 0.5815 | 0.8826 | 0.9845 | 0.9541 | 0.6893 | 0.9240 | dsmonet_resnet_pascal_pige_dsmor50_0220ksh.pkl |

---

## 二、新结果 vs 历史结果对比

### NEU-Seg 数据集

| 模型 | 历史 mIoU | 新 mIoU | 差值 | 状态 |
|---|---:|---:|---:|---|
| Base-B | 90.2 | 90.47 | +0.27 | ✅ 基本一致，略优 |
| Base-S | 88.4 | 88.37 | -0.03 | ✅ 基本一致 |
| A2MS-DefectNet-S | 89.7 | 88.58 | -1.12 | ⚠️ 略低，需核验 |
| A2MS-DefectNet-B | 91.3 | 71.31 | **-19.99** | ❌ **严重异常，需排查** |

### Leather 数据集

| 模型 | 历史 mIoU | 新 mIoU | 差值 | 状态 |
|---|---:|---:|---:|---|
| Base-B | 89.2 | 82.78 | -6.42 | ⚠️ 显著低于历史 |
| Base-S | 88.1 | 83.97 | -4.13 | ⚠️ 显著低于历史 |
| A2MS-DefectNet-B | 91.0 | 85.60 | -5.40 | ⚠️ 显著低于历史 |
| A2MS-DefectNet-S | 89.7 | 84.72 | -4.98 | ⚠️ 显著低于历史 |

> **注意**：Leather 数据集新结果普遍低于历史值 4-6 个百分点。可能原因：
> 1. 历史训练使用了不同的训练配置（更长迭代次数、不同数据增强等）
> 2. 历史结果来自不同的权重文件（如 dsmonet_resnet_pascal_pige_dsmor50_0126.pkl vs dsmonet_resnet_pascal_pige_dsmor50_0220ksh.pkl）
> 3. 验证集/测试集划分可能不同（新结果基于 val.txt 234 张，历史可能基于 test.txt 467 张）

---

## 三、已更新文件清单

| 文件 | 更新内容 | 状态 |
|---|---|---|
| `new_results_pending.md` | 新增 11 条训练结果（NEU-Seg 5 条 + Leather 6 条），含 per-class IoU | ✅ 已更新 |
| `class_iou_results.csv` | 填充 Base-S/A2MS-S/DDRNet/PIDNet-S Leather 的 N/A 值；新增 5 个 NEU-Seg 模型条目 | ✅ 已更新 |
| `complexity_results.csv` | 无新数据（已有全部模型的复杂度统计） | ℹ️ 无需更新 |
| `small_object_group_results.csv` | 无新数据（仅 Base-B/A2MS-B/STDC1-Seg 有分组评价） | ℹ️ 无需更新 |
| `fixed_existing_results.md` | 添加状态更新说明（新结果待核验） | ✅ 已更新 |
| `paper_result_update_summary.md` | 本文件 | ✅ 已生成 |

---

## 四、论文可用性评估

### 4.1 可直接用于论文的结果（需核验后写入 fixed_existing_results.md）

**NEU-Seg 主实验表**：
- Base-B: 90.47 ✅
- Base-S: 88.37 ✅
- DDRNet23slim: 88.43 ✅（新训练，历史无数据）
- A2MS-DefectNet-S: 88.58 ⚠️（低于历史 89.7，需确认）

**Leather 主实验表**：
- 所有 Leather 新结果均低于历史值，建议优先使用历史结果（来自 `fixed_existing_results.md`）
- 如果需要用新结果替换，需先确认训练配置和评估集一致性

### 4.2 需要排查的结果

| 模型 | 数据集 | 问题 | 建议行动 |
|---|---|---|---|
| A2MS-DefectNet-B | NEU-Seg | mIoU=0.7131 vs 历史 91.3，差距 -20pp | 1. 检查训练配置（detail_loss 权重、学习率）<br>2. 对比历史权重 `dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl`<br>3. 可能需要重新训练 |
| 所有 Leather 模型 | Leather | 新结果普遍低 4-6pp | 1. 确认评估集是否与历史一致<br>2. 确认训练迭代次数是否足够<br>3. 考虑使用历史结果 |

### 4.3 补充数据需求

| 数据项 | 当前状态 | 需要的行动 |
|---|---|---|
| NEU-Seg per-class IoU (PIDNet-S) | 已有（来自独立评估脚本） | 无需更新 |
| NEU-Seg per-class IoU (其他模型) | 已从训练日志提取 | 需用独立评估脚本核验 |
| Leather per-class IoU (所有新模型) | 已从训练日志提取 | 需用独立评估脚本核验 |
| 小目标分组评价 (新模型) | 仅有 Base-B/A2MS-B/STDC1-Seg | 运行 `tools/evaluate_small_object_groups.py` |
| Leather 测试集结果 | 当前为验证集 (234 张) | 在测试集 (467 张) 上重新评估 |

---

## 五、论文表格/图表更新建议

### 5.1 主实验表（Table X）

**NEU-Seg 数据集**：可用新结果更新 Base-B (90.47)、Base-S (88.37)、DDRNet23slim (88.43)。A2MS-DefectNet-S (88.58) 与历史 (89.7) 差异较小，可标注待核验。A2MS-DefectNet-B 暂用历史值 (91.3)。

**Leather 数据集**：建议暂用历史结果（来自 `fixed_existing_results.md`），新结果作为补充参考。

### 5.2 Per-class IoU 表

新增 NEU-Seg 4 类 IoU 数据（background, crazing, inclusion, patches），可支撑"各类别性能分析"段落。

### 5.3 复杂度表

已有完整数据（Params, FLOPs, FPS），无需更新。

### 5.4 消融实验表

使用 `fixed_existing_results.md` 中的历史消融数据，无需更新。

### 5.5 定性分析图

已有 14 张对比图（`figures/qualitative_results/`），无需更新。

---

## 六、下一步行动建议

1. **优先排查 A2MS-B NEU-Seg 异常**：mIoU 0.7131 vs 91.3 差距过大，可能是训练配置错误
2. **确认 Leather 评估一致性**：新结果基于 val.txt (234 张)，历史结果可能基于 test.txt (467 张)
3. **运行独立评估脚本核验**：用 `tools/evaluate_class_iou.py` 对所有新权重重新评估
4. **运行小目标分组评价**：对新模型运行 `tools/evaluate_small_object_groups.py`
5. **核验后更新 fixed_existing_results.md**：将 verified=true 的结果写入
6. **论文正文更新**：待核验完成后，由 DeepScientist 根据 fixed_existing_results.md 更新论文

---

## 七、权重文件清单

所有权重文件位于 `new/model_savePath/`，不提交到 GitHub：

| 文件名 | 大小 | 对应模型 | 数据集 |
|---|---|---|---|
| dsmonet_resnet_pascal_0110_neu_120000.pkl | 352MB | Base-B | NEU-Seg |
| dsmonet_resnet_pascal_0109_neu_eSE_adapt_datailloss.pkl | 351MB | A2MS-DefectNet-B | NEU-Seg |
| ddr_pascal_0119_neu_ddr_nolabels_23.pkl | 243MB | DDRNet23slim | NEU-Seg |
| dsmonet_resnet_pascal_neu_base_s.pkl | 167MB | Base-S | NEU-Seg |
| dsmonet_resnet_pascal_neu_a2ms_s.pkl | 168MB | A2MS-DefectNet-S | NEU-Seg |
| fcn_pascal_pige_pid_s.pkl | 92MB | PIDNet-S | Leather |
| dsmonet_resnet_pascal_pige_dsmor50_0220ksh.pkl | 352MB | Base-B | Leather |
| dsmonet_resnet_pascal_pige_dsmor18_0228_80000.pkl | 168MB | Base-S | Leather |
| dsmonet_resnet_pascal_augnew_60000.pkl | 352MB | A2MS-DefectNet-B | Leather |
| ddr_pascal_pige_ddr23s.pkl | 243MB | DDRNet23slim | Leather |
| dsmonet_resnet_pascal_pige_a2ms_s.pkl | 168MB | A2MS-DefectNet-S | Leather |
