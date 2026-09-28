# 240k 长训练冻结协议（Protocol L1，预注册）

冻结时间：2026-09-22，训练启动前
状态：**预注册冻结文档**——本文件先于任何训练与任何 test 评价生成；后续判读必须严格按本文件执行。

---

## 1. 本轮问题

- **Q1**：修复后的完整 A2MS-DSMONet-B（A2 Full）在统一 840 张 NEU 测试协议、固定 240k 终点下，能否达到 **mIoU ≥ 0.910000**？
- **Q2**：在完全相同的 240k 条件下，A2 Full 相对 Base-B 的终点差异 ΔLong = A2 − Base 方向如何？

## 2. 模型（仅两个，代码已锁定）

| 标识 | 模型文件 | SHA256 | 结构 |
|---|---|---|---|
| L-B | `new/model_dsmo_rs50.py` | `fab316c57518e8273ba676199ccff16500ad60ae89a77c2843585f3ce0c184cf` | Base-B：SELayer，无 eSE/ACW，3 路训练输出，无 detail supervision |
| L-AD | `new/model_dsmo_rs50_eSE_adapt_detailloss_fixed.py` | `a0e6fd52b00162ebd16805c473a59983cbd34c2308da115cd170ba1710bb376a` | A2 Full：eSE+AdaptiveChannelWeight，`head_seg1 = out`（fixed），4 路输出，detail head + detail_aggregate_loss |

锁定要求：训练前记录 SHA256，训练结束后复算，必须一致（before == after）。

## 3. Seed（冻结理由）

`seed = 1337`（顶层 `seed` 键，已被代码真实读取）。

1. 1337 是本项目 canonical seed；2. historical 复现链使用该 seed；3. fixed A2MS 早期诊断以 1337 为主；4. **选择 1337 不是因为它对 A2 最有利**——10k 阶段 A2 相对 Base 的优势在 seed2026（+0.010283）反而大于 1337（+0.008169），故不存在挑最好 seed 的动机；5. 两模型使用同一 matched seed。

## 4. 训练协议（Protocol L1：Historical-like）

| 项 | 值 |
|---|---|
| 数据 | NEU-Seg train 3630 / test 840，4 类含 background，`new (copy)/dataset/train_neu.txt`（SHA256 `498b038f…`）/`test_neu.txt`（SHA256 `2c45b319…`） |
| 输入 | 200×200 |
| train transform | Compose([Transforms_PIL(200,200), ToTensor])（与历史及全部既有控制实验一致，无增删） |
| test transform | TestRescale + ToTensor，无 ImageNet Normalize |
| optimizer | Adam，lr = 1e-4，weight_decay = 2e-6 |
| scheduler | **常数 LR = 1e-4，iter 0 → 240000 全程不变**（config `lr_schedule: null` → `ConstantLR`，与产出历史 0.913–0.916 的真实 recipe 一致；禁止 cosine） |
| train_iters | **240000，固定**（不因任何结果提前停止/延长/缩短） |
| batch_size | 16 |
| 初始化 | from scratch（禁止 historical/10k checkpoint、ImageNet、previous A2 checkpoint） |
| Base-B loss | [ohem_ce, bce_with_logits, ohem_ce]，weights [10, 1, 3]，3 路输出 |
| A2 loss | [ohem_ce, bce_with_logits, ohem_ce, detail_aggregate_loss]，weights [10, 1, 3, 3]，4 路输出 |

## 5. 运行方式

- 单张 A100-PCIE-40GB 上**顺序执行**：先 Base-B 240k，完成后自动接 A2 Full 240k（同一 wrapper 脚本串联）；
- 启动时 GPU 状态（记录在 manifest）：另有一无关 nnUNet 任务占用约 7.3GB / 87% util；本训练显存需求 ~4GB，显存充裕；计算争用只影响 wall time，不影响训练动力学与 seed 确定性，两模型在相同背景条件下顺序运行，环境可比；
- 报告记录 GPU、是否独占、其他进程、wall time。

## 6. Run 目录（非覆盖）

```text
repro_runs/long240k_seed1337/
    base_b/   {code, config, logs, checkpoints, analysis, manifest.json}
    a2_full/  {code, config, logs, checkpoints, analysis, manifest.json}
```

禁止写入 `new/model_savePath/`；禁止覆盖任何历史文件。训练脚本内置防覆盖守卫（checkpoint 已存在则报错）。

## 7. Checkpoint 计划（全部由 iteration 预注册，与 test 无关）

- 永久保存点：`iter 10000 / 30000 / 60000 / 120000 / 180000 / 240000`（180k 因历史 91.x 出现在 165k 之后而增设）；
- rolling recovery：`latest_recovery.pkl`，每 5000 iter 原子覆盖（tmp+rename），仅用于意外中断恢复，**不是 best、不是评价点、不是比较点**；
- 每个 checkpoint 含 model/optimizer/scheduler/iter/RNG 状态。

## 8. 中断恢复规则

仅允许从 `latest_recovery.pkl` 恢复：恢复 model/optimizer/scheduler/iteration/RNG（dataloader 顺序为近似恢复，须记录 incident）。不得从零重跑后只保留较好结果。仅 loss 持续 NaN、optimizer state 损坏、数据读取不可恢复错误、GPU/进程异常允许提前终止；禁止因"mIoU 可能不好"终止。

## 9. Test 纪律（本轮最关键规则）

1. **0–240000 iter 训练期间禁止任何 test 评价**：训练脚本不构建 test loader、无 832 monitor、无 test-driven best、无 early stopping；训练阶段只记录 training loss、各 loss component、LR、gradient norm、iteration、wall time；
2. 中间 checkpoint 的 840 评价在两个 240k 训练**全部结束之后**才允许；
3. 评价顺序：**E1** 先评价两个 iter240000 终点 → **E2** 冻结写入 `long240k_endpoint_results.json`（含 SHA256，写入后不得重挑 checkpoint）→ **E3** 之后才回顾评价 10k/30k/60k/120k/180k，标记为 **descriptive trajectory**，不得用于替换终点/选 best/宣称最佳/决定论文 checkpoint。

## 10. 正式评价与 91% 定义

统一脚本（`unified_eval_840_ms.py`）：全 840 张、batch1、TestRescale+ToTensor、无 Normalize、`pred=outputs[0]`、4 类、含 background；输出 mIoU、per-class IoU/Dice/Precision/Recall、confusion matrix。

**"达到 91%" ≡ 统一 840 终点原始 mIoU ≥ 0.910000**（非 832 monitor、非中间 best、非 subset、非四舍五入）。

## 11. 训练过程记录

- 每 100 iter：total loss + 分量 loss（Base-B: comp0/1/2；A2: comp0/1/2/detail）+ LR + wall time → 训练日志 + tensorboard + `analysis/train_metrics.jsonl`；
- GradAudit（只读，不改训练行为）iter 1/100/1000/10000/60000/120000/180000/240000：Base-B 记录 arm2/SELayer/edge_fusion/seg_heads；A2 记录 arm2/eSE/ACW/edge_fusion/seg_heads/detail_head。

## 12. 结果判读规则（冻结）

- **Scenario A**：A2_240k ≥ 0.91 → fixed A2MS-DSMONet-B 在统一 840 终点协议下达到 91%；
- **Scenario B**：两者均 ≥ 0.91 → 只描述 ΔLong，单 seed 不得宣称 A2 显著优；
- **Scenario C**：Base ≥ 0.91、A2 < 0.91 → 历史 Base-like 长训能力获支持，Full A2MS 未达同水平，A2 路线需重新评估；
- **Scenario D**：A2 ≥ 0.91、Base < 0.91 → A2 在该 seed 统一协议下达 91%、Base 未达，强正向信号但仍需多 seed；
- **Scenario E**：两者均 < 0.91 → 当前协议/环境下历史 91% 未复现；先分析终点值、轨迹、口径差异、环境、train loss，**不立即调参**。

结论限制：单 matched seed，A2 > Base 也只能表述为 "A2 achieved a higher numerical endpoint mIoU for seed 1337"，不得使用 "consistently"。

历史比较口径声明：历史 0.913–0.916 来自 832-monitor + test-driven best + 旧环境；本轮为 unified840 fixed endpoint + 无 test 选择 + 新环境。报告必须区分 "historical reference" 与 "current unified endpoint protocol"，不得写"完全复现历史 0.9159"。

## 13. 结果后行动（冻结）

- A2 ≥ 0.91：**停止并报告**——不自动启动 seed2026/3407、不启动 L2（cosine 240k）、不做其他实验；
- A2 < 0.91：**停止并报告**——不调 LR/scheduler/loss weight、不延长、不换 seed、不加 pretrained。

## 14. 交付文件（`Industrial-Surface-Defect/experiments/audit/`）

`long240k_protocol_frozen.md`（本文件）、`long240k_base_b_manifest.json`、`long240k_a2_full_manifest.json`、`long240k_endpoint_results.md`、`long240k_endpoint_results.json`、`long240k_descriptive_trajectory.md`、`long240k_classwise_analysis.md`、`long240k_final_decision.md`。JSON 均须通过 `python -m json.tool`。

## 15. 禁止事项汇总

不修改 AAM/detail loss/loss weight/forward/optimizer/lr；不用 cosine/pretrained；不改 augmentation/dataset/input/class 数；不排除 background；不换/不加 seed；训练中不评 test；不按 test 存 best；不 early stop；不改 240k 预算；不改论文与 `fixed_existing_results.md`；不自动启动 L2/第二 seed；不因结果差重训。
