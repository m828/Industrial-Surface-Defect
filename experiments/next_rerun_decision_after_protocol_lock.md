# 协议锁定后下一步执行决策

> 生成日期：2026-05-22
> 决策依据：`eval_protocol_lock.md` + `a2ms_b_neu_candidate_weights.md` + `leather_unified_eval_manifest.md` + `fps_platform_decision.md`
> 状态：待人工确认后执行

---

## 一、A2MS-DefectNet-B NEU-Seg 判决

### 判决结果

```
❌ 必须重训
```

### 证据链

| 证据 | 值 |
|---|---|
| 服务器候选权重数量 | 1 个 |
| 候选权重路径 | `new/model_savePath/dsmonet_resnet_pascal_0109_neu_eSE_adapt_datailloss.pkl` |
| 候选权重 best_iou_in_ckpt | 0.7131 |
| 候选权重 crazing IoU | 0.5391 |
| 候选权重 patches IoU | 0.5324 |
| 历史声称值 | 91.3 |
| 差距 | -19.99 pp |
| 历史权重是否存在 | ❌ 不在当前服务器 |

### 原因分析（不排除）

可能原因包括但不限于：
1. 训练配置与历史不一致（detailloss 权重、学习率调度等）
2. 类别平衡问题（crazing 占比仅 2.49%）
3. 随机种子导致不收敛
4. 历史权重在 RTX 3090 上训练且未传输至当前服务器

**不做单一归因**。需通过重新训练来验证。

---

## 二、Leather 数据集判决

### 判决结果

```
⚠️ 不需要重训，但需要统一评估后确认哪些权重可写入论文
```

### 证据链

| 模型 | 推荐权重 | best_iou_in_ckpt | 接近历史？ |
|---|---|---|---|
| Base-B | `new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl` | 0.8930 | ✅ 接近 89.2 |
| A2MS-B | `new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl` | 0.9091 | ✅ 接近 91.0 |
| Base-S | `new/model_savePath/dsmonet_resnet_pascal_pige_dsmor18_0228_80000.pkl` | 0.8397 (训练日志) | ❌ 低于 88.1 |
| A2MS-S | `new/model_savePath/dsmonet_resnet_pascal_pige_a2ms_s.pkl` | 0.8472 (训练日志) | ❌ 低于 89.7 |

### 关键发现

1. **Base-B 和 A2MS-B 有历史权重接近论文声称值**：
   - Base-B 0127: best_iou=0.8930 → 与历史 89.2 差距仅 +0.10pp
   - A2MS-B 160000: best_iou=0.9091 → 与历史 91.0 差距 -0.09pp
   - 这两个权重的 best_iou_in_ckpt 是在验证集（非测试集）上的值
   - 需在 test.txt (467 张) 上独立评估后确认最终 mIoU

2. **Base-S 和 A2MS-S 无历史权重**：
   - 仅有 A100 新训练权重，mIoU 明显低于历史值
   - 这可能是 A100 vs RTX 3090 训练差异，或配置差异
   - 需要重训来匹配历史值

3. **augnew 系列权重不是 A2MS-B**：
   - 使用 `model_dsmo822.py`（不同架构，Light_Bag 替代 SqueezeBodyEdge）
   - 不纳入 A2MS-B 主实验

### 可立即执行的评估

以下权重的评估不需要重训，只需要运行独立的测试集评估：

| 优先级 | 模型 | 权重 | 预期 mIoU |
|---|---|---|---|
| 🔴 立即 | Base-B | 0127_80000 | ~89.3 |
| 🔴 立即 | A2MS-B | 160000 | ~90.9 |
| 🟡 同步 | Base-B | 0126 (备选) | ~87.2 |
| 🟡 同步 | Base-B | 0220ksh (新训) | ~82.8 |
| 🟡 同步 | Base-S | 0228_80000 (新训) | ~84.0 |
| 🟡 同步 | A2MS-S | a2ms_s (新训) | ~84.7 |
| 🟡 同步 | DDRNet | ddr23s (新训) | ~83.2 |
| 🟡 同步 | PIDNet-S | pid_s (新训) | ~84.1 |
| 🟢 后续 | STDC1-Seg | stdc2_pige | 待确认 |

---

## 三、FPS 判决

### 判决结果

```
推荐方案 B：统一使用 A100 作为 FPS 平台
```

详见 `fps_platform_decision.md`。

### 不需要重测 FPS

当前 `complexity_results.csv` 中的 A100 FPS 数据可直接使用（Params/FLOPs/FPS 均已在 A100 上统一测量）。

### 需要执行

1. 如果选择方案 B：更新 `fixed_existing_results.md` 硬件标注，替换所有 FPS 为 A100 值
2. 如果选择方案 A：寻找 RTX 3090 并重测 FPS

---

## 四、历史结果保留/废弃决策

### 可以保留的历史结果

| 结果 | 原因 |
|---|---|
| NEU-Seg Base-B: 90.2 mIoU | 新训练 90.47 一致，可用 |
| NEU-Seg Base-S: 88.4 mIoU | 新训练 88.37 一致，可用 |
| NEU-Seg DDRNet: 86.9 mIoU | 新训练 88.43 更高，更新为新高值 |
| Leather Base-B: ~89.2 mIoU | 0127 权重 best_iou=89.3 可支撑 |
| Leather A2MS-B: ~91.0 mIoU | 160000 权重 best_iou=90.9 可支撑 |
| ELMM 消融整体结论 | ELMM (91.3) > DM (90.1) > Baseline (85.1) 定性结论正确 |
| AAM 消融整体结论 | 定性结论正确（AAM×2 ≥ AAM×1 > eSE > SE） |

### 必须废弃的结果

| 结果 | 原因 |
|---|---|
| NEU-Seg A2MS-B: 91.3 mIoU | 当前权重 mIoU=71.31，历史权重不存在 |
| Leather Base-S: 88.1 mIoU | 当前权重 mIoU=83.97，历史权重不存在 |
| Leather A2MS-S: 89.7 mIoU | 当前权重 mIoU=84.72，历史权重不存在 |
| 所有对比方法 FPS | 来自 RTX 3090，与 A100 不可比 |
| FCN/U-Net/HRNet/PSPNet/DeeplabV3+/ENet 全部 | 无权重文件，无法验证 |
| Per-class IoU 全部 | 来源混杂，需统一评估后重建 |
| 小目标分组评价全部 | 方法论问题（无缺陷组 mIoU=0.125） |

### 暂挂（待进一步确认）

| 结果 | 待确认事项 |
|---|---|
| AAM 消融 eSE×1 (90.7) | 权重 `0120_eSE_adapt` 可加载为 "eSE + adapt"，但不完全等于 eSE×1 |
| AAM 消融 AAM×1 (91.1) | 无独立权重 |
| 联合损失消融中间点 | 无独立训练记录 |
| SFNet/FDSNet/PGA-Net 引用值 | 已在论文中标注为"来自原论文"，暂不处理 |

---

## 五、论文表格状态

### 可立即填入（A 类结果）

| 表格 | 可填入内容 |
|---|---|
| NEU-Seg 主实验表 | Base-B: 90.47, Base-S: 88.37, A2MS-S: 88.58, DDRNet: 88.43, PIDNet-S: 86.67 |
| 复杂度表 | Params/FLOPs/模型大小 全部（FPS 需确认平台后填入） |
| ELMM 消融（定性） | 可保留定性结论 |
| AAM 消融（定性） | 可保留定性结论 |

### 必须暂时清空/标注 TODO

| 表格 | 清空/TODO 内容 |
|---|---|
| NEU-Seg 主实验表 | A2MS-B 行 → 标注 TODO |
| Leather 主实验表 | **全部行** → 标注 TODO（等待统一评估） |
| 消融实验表 | 所有定量数值 → 标注 "待验证" |
| 类别级 IoU 表 | **全部** → 标注 TODO（等待统一评估） |
| 小目标分组评价表 | **全部** → 标注 "诊断分析，待重跑" |

---

## 六、下一步执行命令

### 步骤 1（立即可执行）：Leather 统一评估

```bash
cd "/workspace/Industrial Surface Defect/Industrial-Surface-Defect"
source /opt/conda/etc/profile.d/conda.sh && conda activate base

# [需人工确认 evaluate_class_iou.py 的实际参数名称]
# python tools/evaluate_class_iou.py --help
```

### 步骤 2（确认后执行）：A2MS-B NEU 重训

```bash
cd "/workspace/Industrial Surface Defect/new"
source /opt/conda/etc/profile.d/conda.sh && conda activate base

# 在重训前先排查配置：
# 1. 检查 dsmonet_resnet_detailloss.yml 中的 loss 配置
# 2. 对比 train_neu_resnet_detailloss.py 的损失权重
# 3. 确认 detailloss head 是否正确输出

# 然后重训：
# python train_neu_resnet_detailloss.py \
#     --config dsmonet_resnet_detailloss.yml \
#     2>&1 | tee runs_other/a2ms_b_neu_retrain_$(date +%Y%m%d).log
```

### 步骤 3（决策后执行）：FPS 平台统一

- 如果方案 A (RTX 3090): 在 RTX 3090 上运行 `complexity_stats.py`
- 如果方案 B (A100): 更新 `fixed_existing_results.md` 中的 FPS 值和 GPU 标注

### 步骤 4（前 3 步完成后执行）：更新所有结果文件

- 更新 `class_iou_results.csv`
- 更新 `fixed_existing_results.md`
- 更新 `new_results_pending.md`
- 更新论文表格

---

## 七、仍需人工确认的问题

| # | 问题 | 影响范围 |
|---|---|---|
| Q1 | `tools/evaluate_class_iou.py` 的确切参数和 model_type 取值？ | Leather 统一评估能否执行 |
| Q2 | RTX 3090 是否可用？ | FPS 平台决策 |
| Q3 | `dsmonet_resnet_detailloss.yml` 的配置是否与历史 A2MS-B NEU 训练一致？ | A2MS-B NEU 重训方向 |
| Q4 | `new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl` 是否来自原硕士论文？ | Base-B Leather 历史结果有效性 |
| Q5 | `new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl` 是否来自原硕士论文？ | A2MS-B Leather 历史结果有效性 |
| Q6 | 论文审稿人是否接受更换 GPU 平台标注？ | FPS 方案选择 |
| Q7 | STDC1-Seg 的权重 `sdtdcnet_pige_stdc2_pige.pkl` 文件名含 stdc2，是否确认为 STDC1？ | STDC1-Seg 模型评估 |
| Q8 | 历史 NEU-Seg A2MS-B 91.3 的权重是否可能在离线存储中？ | A2MS-B NEU 是否需要重训 |
