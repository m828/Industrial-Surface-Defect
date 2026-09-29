# Baseline 仓库审计报告 (repository_audit)

> 日期：2026-09-29
> 审计方式：逐文件代码审计 + 只读实证（import/前向验证）；三份详细审计由独立子任务完成，行号证据可回溯。
> 协议基线：NEU-Seg（train 3630 / test 840，200×200，4 类含背景，240k 固定终点，无 test 选择，seed 1337）；Leather（1638/235/468，768×768，8 类，val235 选择）。

## 一、已有 third_party 审计

| 方法 | 来源 | 任务 | 适合NEU | 适合Leather | 状态 |
|---|---|---|---|---|---|
| DMC-Net | github.com/Michaelzyb/DMC-Net @9b95974（快照，无.git） | 工业缺陷分割 | **否** | 否 | ❌ 快照不完整：`models/total_supvised/dmcnet.py` 模型本体缺失（另有 4 处断链 import）；只能跑 U-Net/BiSeNet 等 8 个非 DMC 模型。**DMC-Net 本体无法复现，不纳入，除非获取完整源码** |
| LETNet | github.com/XU-GITHUB-curry/LETNet @faaa065 | 轻量实时分割 | 是（需适配） | 是（需适配） | ⚠️ 可修复。3 个 1 行 blocker（ESPNet_v2 断链 import、transformer 导包路径、LC3 通道 bug）+ NEU 数据集适配 ~70 行；200×200 需 pad 至 208（16 整除约束）；from scratch 原生支持（无预训练依赖） |
| SeaFormer | github.com/fudan-zvg/SeaFormer @9718d05 | 轻量 Transformer 分割 | 否（本环境） | 否（本环境） | ❌ 环境级阻塞：自带 mmseg 0.19 硬性要求 mmcv 1.3.13–1.5.0，与 torch 2.7.1 / Python 3.11 不兼容；移植成本高。**不纳入本环境实验**；如需引用仅能以文献值标注协议差异 |

### DMC-Net 补充（对论文的潜在影响）

- 其内置 `benchmark='neuseg'` 支持 4 类 NEU-Seg，但 val 为 train 随机 20%（random_state=69），mIoU **不含背景**且缺失类记 1.0 —— 即便补全代码，其原生口径也与本文统一协议不同，必须统一用 `tools/evaluate_class_iou.py` 重评。
- 数据扫描机制（子串匹配）与我们现有 NEUSeg 目录兼容，这一点已实证。

### LETNet 补充

- 评价口径坑：`metric.py` 的 `jaccard()` 对 TP=0 类别剔除出平均（虚高 mIoU）；`test.py --best` 存在 test 集选优模式。**统一协议下禁用 `--best`，最终数字一律由统一评价脚本产生**。
- 无官方预训练权重，天然 from scratch，与本文协议兼容。

## 二、内部 baseline 链审计（STDC / PIDNet / DDRNet）

| 方法 | 内部实现 | 参数量（实测） | NEU 训练脚本 | 统一评价工具支持 | 状态 |
|---|---|---|---|---|---|
| STDC-Seg（STDCNet1446） | `new (copy)/stdc.py` `BiSeNet('STDCNet1446', N)` | 16.07M | ❌ 不存在（`train_neus.py` 训的是 DSMONet 解码器变体 16.28M，与 Leather 88.24 权重不兼容，**不可用**） | ✅ 已注册（stdc→build_stdc1） | ⚠️ 需新建协议合规 NEU 训练脚本 |
| PIDNet-S | `new/pid.py` ≡ `new (copy)/pid.py` | 7.72M | ⚠️ 两版：`new/train_neu_pid.py` 有死循环 bug（`:145` 计数器遮蔽，禁用）；`new (copy)/train_neu_pid.py` 为修复版但含 test 监控+选优 | ✅ 已注册 | ⚠️ 需协议化改造（60k→240k、删 test 监控、cosine→恒定 lr） |
| DDRNet23-slim | `new/model_ddr.py` `DualResNet_imagenet` | **20.30M（实测）** | ✅ `new/train_neu_ddr.py` 可用但含 test 监控+选优 | ✅ 已注册 | ⚠️ 需协议化改造（同上） |

### 重要事实澄清（真实性记录）

1. **内部 "DDRNet23-slim" 实为 20.30M 参数**（planes=64, head_planes=128），接近官方 DDRNet-23 量级；`model_ddr.py:357` 被注释的 planes=32 变体（5.73M）才是官方 23-slim 量级。`experiments/results/complexity_results_a100_final_summary.md` 中 "DDRNet23slim 6.71M" 与实际训练/评估产物（244MB checkpoint、成功加载记录）矛盾，疑为旧测量口径问题；**论文若引用参数量需重测确认**。
2. **Leather 88.24（STDC）权重的训练脚本已丢失**：当前代码树只有其 test/predict 引用；NEU 侧必须新建训练脚本且使用 `BiSeNet('STDCNet1446', 4)` 才能与 Leather 结果和评估注册表兼容。
3. 三条 NEU 旧链均以 `test_neu.txt` 做训练中监控 + test mIoU 选优（被禁用的原因），协议化改造的核心是删除验证/选优块并改固定终点保存。
4. 旧链保存路径会覆盖历史 checkpoint——协议版脚本一律使用独立 `checkpoint_dir`。

## 三、官方仓库拉取记录

| 方法 | 仓库 | 说明 |
|---|---|---|
| BiSeNetV2 | github.com/CoinCheung/BiSeNet | 官方实现（含 V1/V2/V3） |
| STDC-Seg | github.com/MichaelFan01/STDC-Seg | 官方实现 |
| PIDNet | github.com/XuyangBai/PIDNet | 官方实现（CVPR2023） |
| DDRNet | github.com/ydhongHIT/DDRNet | 官方实现（488 stars） |

拉取状态与 commit hash：见 `results/repo_clone_log.md`（网络波动，含重试）。

> 说明：官方仓库为 Cityscapes 导向，主要作结构参照与来源核对；**实际训练仍优先走内部链**（与本项目数据管线、统一评价工具、历史 Leather 结果同源），以保证口径一致。是否切换为官方实现训练，待 smoke test 与训练计划评审后决定。

## 四、纳入论文建议（本阶段结论）

| 方法 | 纳入建议 | 原因 |
|---|---|---|
| STDC-Seg（内部 BiSeNet-1446） | ✅ 优先 | 与 Leather 88.24 同源；评价工具已支持；仅需新建 NEU 协议训练脚本 |
| PIDNet-S（内部） | ✅ 优先 | 修复版脚本存在；评价工具已支持；协议化改造量小 |
| DDRNet23-slim（内部） | ✅ 优先（需先解决参数量口径） | 同上；参数量以实测 20.30M 重测为准 |
| LETNet | ✅ 备选 | 修复成本低（3×1行 + 70行适配）；轻量 Transformer 混合路线，与本文方法形成结构对照 |
| DMC-Net | ❌ 暂不纳入 | 快照缺模型本体，无法复现；待获取完整源码再评估 |
| SeaFormer | ❌ 不纳入（本环境） | mmcv 1.x 环境冲突；若引用仅作文献值并标注协议差异 |
| BiSeNetV2 | ⏸ 视情况 | 官方仓库可作参照；内部链无现成实现，新增成本中等 |
