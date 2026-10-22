# CJIG v10.4 — 最终科学证据状态报告

日期：2026-10-22
任务：服务器原始预测、GT 与统计生成链的只读核验。
范围声明：本轮未训练、未重新推理、未重选 checkpoint、未修改任何实验结果/论文/Git 状态。所有复算均基于服务器已有的预测 mask PNG 与 GT PNG。所有输出位于仓库外 `/workspace/cjig_v10_4_readonly_audit/`。

前置说明：v10.3 任务书 `docs/paper/cjig_v10_3_server_readonly_audit_task.md` 在服务器不存在（已在项目两处候选路径确认缺失），本轮按 v10.4 提示词执行。因此凡涉及"v10.3 原脚本逐行一致"的表述均不可作出。

---

## 一、四类最终状态

| 维度 | 状态 | 一句话依据 |
|---|---|---|
| NEU MASK EVIDENCE | **VERIFIED** | 840×3 组 mask 身份完全闭合，mIoU 从 mask 复算与冻结值逐位一致 |
| STATISTICAL CONSISTENCY | **PARTIALLY VERIFIED** | 点值/空例规则/A/B/C/RQ1/RQ2 均从原始实现与 mask 复算一致；Bootstrap 区间按声明设置独立复算一致，但 v10.3 原统计脚本未找到，不能声称逐行验证；Dice paired 字段确认存在写出回退缺陷 |
| LEATHER PROVENANCE | **PARTIALLY VERIFIED** | Base 日志级闭合；A2MS 训练日志缺失；468 张原始预测缓存不存在；60k 控制链配置/日志级闭合 |
| SCIENTIFIC GENERALIZABILITY | **EXPLORATORY ONLY** | 现有证据为单种子、固定权重、测试集内分析；RQ1/RQ2 属统计分解而非机制因果；不足以支持独立验证或泛化强表述 |

---

## 二、十个必答问题

### 1. 840张原始预测和GT是否真正找到？
**是。** `repro_runs/mechanism_eval_240k/` 下 `gt/`、`base_predictions/`、`a2_predictions/` 各 840 个 200×200 PNG；sample_id 三组完全一致、顺序与 test list（SHA `2c45b319…8f06`）一致；标签值均在 {0,1,2,3}；缓存 GT 与源标注 840/840 逐像素一致。缓存脚本 `cache_predictions.py:69` 与统一评价 `tools/evaluate_class_iou.py:571-575` 均取 `outputs[0]` argmax，无归一化/后处理差异。

### 2. mIoU是否从mask成功复算？
**是。** 独立混淆矩阵复算：DSMONet-B `0.9124083744680408`、A2MS-DSMONet-B `0.9145442177671137`，与冻结值 0.912408 / 0.914544 完全一致。原始 `paired_case_metrics.csv`（2520 行）与独立复算全部共有列差异 ≤1e-10。

### 3. 表3的三套统计目标与区间是否得到原始实现验证？
**点值：是；区间：独立复算层面是、原脚本层面否。**
- 裂纹对称带 IoU@2：A=−0.00452784（n=371/370）、B=−0.00650628（n=368）、C=−0.00657778（n=364），与 v10.3 报告值逐位一致；样本集合存于 `estimand_abc_sample_sets.json`。
- 空例规则经合成样本测试：GT空+预测空→NA；单方空→0；全图 mask 因 contour/band 双空→NA。
- Bootstrap：按声明设置（5000 次、seed20261018、按图像重抽样、两模型共用索引、百分位 95%）独立复算，72 行全部区间无 empty replicate；抽样索引 SHA `c454e06b…c876`。但 v10.3 的 seed20261018 原始统计脚本未在服务器找到（旧 `analyze_mechanism.py` 用 seed 20260924 且按类别子集重抽样），故只能声称"按声明设置复算一致"，不能声称原脚本逐行验证。

### 4. Dice字段问题是否找到根因？
**是。** `compute_mechanism_metrics.py` 写出 `base_dice`/`a2_dice` 但未写 `delta_dice`；`analyze_mechanism.py:87-90` 在尺度分组循环中检测不到 `delta_dice` 时回退使用 `delta_iou`，导致旧 `scale_stratified_results.json` 九个类别×面积组的 Dice paired 字段与 IoU paired 字段完全相同。属字段写出回退缺陷，非 mask 或公式错误。正确值已复算于 `scale_dice_corrected_recalculation.csv`（例：裂纹 Small Dice Δ=−0.010739，旧错误字段 −0.012401；斑块 Small Dice Δ=+0.005660，旧错误字段 +0.007328）。建议修正字段，不涉及任何实验结果改动。

### 5. RQ1聚合与病例平均反向的数学贡献是否复算完成？
**是。** 裂纹 GT 阳性集合（n=364）：聚合 IoU Δ=+0.003192，逐病例均差 Δ=−0.007218。精确分解：固定 Base union 权重的质量项 −0.002668，union 重加权项 +0.005860，合计 +0.003192（闭合到 1e-15）；聚合与逐病例之差 +0.010410。即方向反转由"等权逐病例平均"与"union 加权聚合"的估计目标差异造成，union 重加权项大于质量项并扭转符号。面积组贡献：裂纹 Large 组贡献聚合 Δ 的 90.6%（其中 union 重加权 +0.00398 为主）。同向性：区域 IoU 与 BIoU@2 严格同向 319/364（含共同并列 320/364）。现有证据不支持"普遍的区域—边界权衡"：斑块聚合 +0.004527 与逐病例 +0.003932 同向，夹杂物基本为零。

### 6. RQ2面积与复杂度字段是否从GT复现？
**是。** 三类 P33/P67 阈值从 GT 原 mask 复算与冻结值逐位一致（裂纹 0.027324/0.06185775；夹杂 0.1064235/0.18496475；斑块 0.037379/0.08505）。裂纹 Small IoU Δ=−0.012401、斑块 Small Δ=+0.007328 均复现。复杂度按原始实现（一次腐蚀 contour 像素数/√area，非几何周长）：裂纹 high 组（n=121）IoU Δ=−0.017411、BIoU@2 Δ=−0.013643、BF1@2 Δ=−0.014785，且高复杂度在三个粗面积档全部为负（Small −0.0242 / Medium −0.0127 / Large −0.0181）。探索性混杂：面积—复杂度相关 r=0.0098（可忽略），复杂度—连通组件数 r=0.676；连通组件分析为新增探索性内容，不得写作预注册分析。

### 7. 图3、图4选例是否追溯到原始mask？
**图3：是。** 18 条候选记录、17 个唯一 ID；六例（000653/000722/000058/000802/000722/000439）由 `assemble_fig34_v6.py:39-46` 硬编码，复算 ΔBF1@2 与记录最大误差为 0；panel 文件存在。
**图4：部分。** manifest 记录 268/211/45/12 与六例（210/129/396/19/77/102），panel 文件存在，生成脚本带一致性门；但 Leather 无 468 张原始预测缓存，本轮禁止重新推理，故不能完成全量 mask 级复核。

### 8. Leather来源是否有实质补齐？
**部分补齐。** Base：训练日志第 745 行 val235 `0.8929607161637454` 紧随 `Iter [78000/80000]`，与 checkpoint 内嵌值逐位一致，日志链闭合。A2MS：训练日志确认缺失（全项目 grep 0.9090842 无日志命中；三段 detailloss 日志属 822 变体链，eSE_adapt 日志仅 60k 3-loss，均排除）。60k 控制链：配置与实测 LR（cosine，T_max=48000）闭合，但训练脚本"expects constant LR"措辞与真实调度不符，已记录。两条 Leather 链方向相反是既有事实，无单一因素归因证据。

### 9. 哪些v10.3候选发现应保留、修正或撤回？
- **保留**：NEU 91.24/91.45 主结果；A/B/C 三套统计目标点值与区间解读；裂纹"聚合正、逐病例负"的分解结论；RQ2 面积/复杂度分层结论；图3。
- **修正**：`scale_stratified_results.json` 九个 Dice paired 字段（用 `scale_dice_corrected_recalculation.csv` 替换）；任何把 v10.3 区间表述为"原脚本已验证"的说法应降级为"按声明设置独立复算一致"；60k 链相关文档应注明脚本日志措辞与 cosine 调度不符。
- **撤回/降级**：任何"Leather 268 张已完成原始预测级核验"的表述（只有 manifest 级）；任何"普遍区域—边界权衡"表述（斑块反例）；任何从现有证据出发的泛化/机制因果强表述。

### 10. 下一步是否有必要开展独立权重验证？
**有必要，但本轮未执行。** 若论文需要支持泛化或机制层面结论，需要：独立种子/独立训练的多权重评价、A/B/C 机制对照消融（v10 已设计未执行）、以及 Leather 468 张预测缓存的重建式核验（需重新推理，超出本轮只读范围）。继续只读统计已不能进一步提升证据等级。

---

## 三、输出清单（全部位于 /workspace/cjig_v10_4_readonly_audit/）

`source_inventory.md`、`source_manifest.json`、`source_manifest_supplemental.json`、`neu_mask_identity_verification.md`、`neu_pixel_metric_reproduction.csv`、`neu_confusion_matrices.json`、`neu_identity_checks.json`、`recomputed_case_metrics.csv`、`recomputed_per_image_miou.csv`、`paired_case_reproduction_diff.json`、`boundary_metric_implementation_audit.md`、`boundary_empty_rule_tests.json`、`boundary_named_sample_audit.json`、`estimand_abc_pixel_verification.csv`、`estimand_abc_sample_sets.json`、`bootstrap_reproduction_summary.csv`、`bootstrap_runtime_metadata.json`、`scale_dice_root_cause_audit.md`、`scale_dice_corrected_recalculation.csv`、`rq1_pixel_decomposition.md`、`rq1_decomposition.csv`、`rq1_size_group_contributions.csv`、`rq2_raw_mask_condition_audit.md`、`rq2_area_reproduction.csv`、`rq2_complexity_reproduction.csv`、`rq2_exploratory_confounders.csv`、`figure_case_source_audit.md`、`figure3_candidate_trace.csv`、`figure3_actual_panels.json`、`neu_pixel_audit_summary.json`、`leather_raw_provenance_update.md`、`final_scientific_evidence_status.md`（本文件）。

未生成项及原因：Leather 逐图指标表（无原始预测缓存）；A2MS 训练日志摘录（日志不存在）。
