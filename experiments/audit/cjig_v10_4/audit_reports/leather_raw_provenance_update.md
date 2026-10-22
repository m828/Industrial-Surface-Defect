# CJIG v10.4 — Leather 原始来源补档（只读核验）

日期：2026-10-22
范围：Leather（Pige）历史主结果 89.91 / 90.97、60k 固定终点控制链 87.21 / 84.99、图4选例。
本文件只基于服务器现有文件，不含任何重新训练或重新推理。

---

## 1. 证据总览与等级

| 链路 | 证据项 | 状态 |
|---|---|---|
| DSMONet-B 历史训练 | 完整训练日志 + checkpoint + 统一468评价 | 闭合 |
| A2MS-DSMONet-B 历史训练 | checkpoint 元数据 + 统一468评价；**完整训练日志缺失** | 部分闭合 |
| 60k 固定终点控制链 | 配置 + 训练 metrics + LR 曲线 + 统一468评价 | 闭合（无原始预测缓存） |
| 图4 六例 | manifest + 生成脚本 + 一致性门 | manifest 级闭合；268 张逐图原始预测缓存缺失 |

## 2. DSMONet-B（Base，89.91）— 训练日志链闭合

- checkpoint：`new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl`
  - SHA-256：`a96c78ea8ba95f792cb1619ad593722e2961801bf158fe330b45b6ef6bf9abe3`
  - 内嵌 epoch 字段：78000（文件名 80000 为链目标，实际保存点为 val-best 截获 78000）
- 训练日志：`new (copy)/runs_other/dsmonet_resnet_pige/dsmor50_pige/run_2024_01_27_14_05_37.log`
  - SHA-256：`4b4a25ad7e8a136b45febd07e935098a5fc0669d3af87a5cc2cdeb359c21366e`，817 行
  - 日志记录 Adam、constant LR（"Using No LR Scheduling"）、3 项 loss（ohem CE + BCE + ohem CE）
  - 第 742–745 行：`Iter [78000/80000]` 之后随即记录 val235 `Mean IoU : 0.8929607161637454`
  - 该值与 checkpoint 内嵌 best_iou 逐位一致，亦与 manifest `hist_0127_baseb_78k_reverify.json` 中 `historical_val235_best_miou` 一致
  - 全文最大值即 0.8929607161637454（已 grep 全部 Mean IoU 行确认 78k 为 val 轨迹峰值）
- 统一评价：test468 mIoU `0.89912`（evals/hist_0127_baseb_78k_reverify_class_iou.csv，含 8×8 混淆矩阵）

**结论：Base 历史链从训练日志 → val-best 选择 → checkpoint → 统一468评价全部对上，来源等级可提升为"已核验"。**

## 3. A2MS-DSMONet-B（90.97）— 训练日志缺失，其余链闭合

- checkpoint：`new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl`
  - SHA-256：`336c5a4c2e5c1def67e34dc5061c5ed384829c8fdf318ce67b64f04e1a54f01f`
  - 内嵌 epoch：130000（文件名 160000 为链目标，实际为 val-best 截获 130000；`checkpoint_metadata/A2MS-DSMONet-B/checkpoint_info.json` 已记录该不一致）
  - 内嵌 val235 best：`0.9090842792341525`
- 统一评价（fixed-forward 模型 `new/model_dsmo_rs50_eSE_adapt_detailloss_fixed.py`，SHA `a0e6fd52…376a`，strict load 成功）：test468 mIoU `0.909713`
- **完整训练日志未找到。** 本轮在 `new (copy)/runs`、`new (copy)/runs_other`、`new/runs`、`new/runs_other` 全量枚举 2024 年代 pige 日志，并对整个项目 grep `0.9090842`：命中仅为 manifest / checkpoint_metadata / 既有审计文档，无任何训练日志。
- 排除性核对：
  - `new (copy)/runs/dsmonet_resnet_detailloss/pige_4_0228/` 三段日志（2024-02-28 → 03-01，0→80k→110k→160k）虽含 4 项 loss（含 detail_aggregate_loss），但 checkpoint 家族为 `pige_4_0228*.pkl` / `pige_4_0229_160000.pkl`，即既有审计已排除的 822/Light_Bag 变体（822 协议 0.917625），**不是**论文 A2MS 结构的训练日志。
  - `new (copy)/runs/dsmonet_resnet_pige/eSE_adapt/run_2024_01_20_16_34_38.log` 仅 60k、3 项 loss（无 detail_aggregate_loss），best 0.8695，排除。
  - `new (copy)/runs_other/dsmonet_resnet_pige/dsmor50_pige/run_2024_02_20_20_36_52.log`（80k，3 loss，best 0.8837）与 `run_2024_01_26_11_29_31.log`（60k，3 loss，best 0.8723）、`dsmor50_pige_nose/run_2024_03_03`（60k，3 loss）均为 Base 家族变体，排除。

**结论：A2MS 主评价 90.97 的来源等级维持"checkpoint + 统一评价已核验，训练过程日志缺失"。不能进一步升级；但也不存在与既有审计相冲突的新证据。**

## 4. 60k 固定终点控制链（87.21 / 84.99）— 协议复核

- 配置：
  - `repro_runs/leather_60k_seed1337/base_b/logs/base_b_s1337_leather60k_codex1/config.yml`
  - `repro_runs/leather_60k_seed1337/a2_full/logs/a2_full_s1337_leather60k_codex1/config.yml`
  - 共同：60k、batch 8、Adam lr=1e-4、wd=2e-6、cosine annealing（T_max=48000，eta_min=1e-6）、768×768、8 类、无 test/val loader、固定终点评价；Base loss 权重 [10,1,3]，A2 [10,1,3,3]
- 实际 LR（`analysis/train_metrics.jsonl`，各 600 条）：首条 ≈9.999896e-5，末条 ≈1.549592e-5，min=1e-6，max≈1e-4，与 cosine 配置一致
- **文档不一致（需记录，不影响结果）**：训练脚本日志措辞写 "expects constant LR"，但配置与实测 LR 均为 cosine annealing。属脚本注释/日志措辞与真实调度不符。
- 统一468评价：Base `0.872137`，A2 `0.849932`，delta `−0.022205`；rotten_surface 类 delta `−0.124551`
- `experiments/audit/leather_paper_final_results.json` 已声明 60k 结果为 archived-only，论文主结果为历史 val-best 链（89.91/90.97）
- 同样不存在 60k 链的 468 张原始预测 mask 缓存

**两条链路方向相反（历史链 A2>Base，60k 链 A2<Base）是既有事实；本轮未发现可归因于单一因素的证据，不作单因素推断。**

## 5. 图4（Leather 定性图）选例核验

- manifest：`repro_runs/leather_historical_audit/final_viz/final_viz_manifest.json`（SHA `dc26d413…4152`）
  - `n_defect_bearing=268, n_improved=211, n_worsened=45, n_tie=12`
  - 六例：worsened `210`（Δ=−0.5331）、`129`（Δ=−0.3984）；improved `396`（Δ=+0.4223）、`19`（Δ=+0.4216）；near_zero `77`、`102`（Δ=0.0）
  - 六张 panel PNG 全部存在于 `final_viz/`
- 生成脚本 `leather_viz_final_v6.py`：
  - 测试清单 `leather_valid_test_list.txt` 468 行（SHA `f9ac23b3…f95c`，与两份历史评价 manifest 一致）
  - 选择规则固定：缺陷阳性子集按逐图 8 类 mIoU delta 排序取 top2/bottom2/最近零 2 例（脚本第 92–94 行）
  - 内置一致性门：聚合 mIoU 必须与审计值 0.899120/0.909713 相差 <2e-4 才继续（第 71 行 assert）
  - **预测仅在内存中计算，未保存 468 张逐图预测 mask**
- 因本轮禁止重新推理，268 张的 211/45/12 分布只能确认为 manifest 记录 + 确定性脚本 + 一致性门；**不能声称完成全量原始 mask 级复核**。

## 6. 本轮仍未取得的项目（明确缺失）

1. A2MS 历史 130k 完整训练日志（含逐 1000 迭代的 val235 轨迹）。
2. Leather 468 张两模型原始预测 mask 缓存（任何链路：历史、60k、基线）。
3. 60k 链训练日志措辞与真实 LR 调度的一致性修正（属代码注释问题，未修改）。
4. 图4 六例以外的 262 张缺陷阳性样本逐图指标表（manifest 只保存了六例与聚合计数）。

## 7. 来源等级结论

| 结果 | 等级 | 依据 |
|---|---|---|
| Leather Base 89.91 | VERIFIED（日志级） | 训练日志 val-best 逐位闭合 + checkpoint + 统一468评价 |
| Leather A2MS 90.97 | PARTIALLY VERIFIED | checkpoint 元数据 + fixed-forward 统一468评价 + strict load；训练日志缺失 |
| 60k 控制链 87.21/84.99 | VERIFIED（配置/日志级） | 配置 + LR 实测 + 统一468评价；无预测缓存 |
| 图4 选例 | PARTIALLY VERIFIED | manifest + 确定性脚本 + 一致性门；缺全量预测缓存 |
