# CJIG v10.4 核心证据归档

归档日期：2026-10-22
任务：CJIG v10.4 服务器原始预测、GT 与统计生成链只读核验的核心证据版本化归档。
生成方式：`audit_reports/scripts/build_archive.sh` 从服务器原始路径逐字节复制（`cp -p`），全部 56 个文本文件经 SHA-256 与来源逐一比对一致（见 `archive_integrity_verification.json`）；2520 张 mask 经完整解包逐文件哈希复核（见 `neu_masks/mask_archive_verification.json`）。
本归档不修改任何历史实验结果；`audit_reports/scale_dice_corrected_recalculation.csv` 是**新复算的修正文件**，不是原始实验输出，不替代历史 `scale_stratified_results.json`。

## 核验状态总表

| 维度 | 状态 |
|---|---|
| NEU MASK EVIDENCE | VERIFIED |
| STATISTICAL CONSISTENCY | PARTIALLY VERIFIED |
| LEATHER PROVENANCE | PARTIALLY VERIFIED |
| SCIENTIFIC GENERALIZABILITY | EXPLORATORY ONLY |

依据见 `audit_reports/final_scientific_evidence_status.md`。

## 目录结构与文件清单

### `audit_reports/` — v10.4 审计交付（NEU 为主，含跨数据集最终状态）

| 文件 | 字节 | SHA-256 | 作用 / 来源 |
|---|---:|---|---|
| final_scientific_evidence_status.md | 9269 | edc52127aeb0c3abe9565056b064138724022af8f7a20c61212812bddb2c3182 | 最终证据状态报告（四类状态 + 十问十答）；v10.4 复算 |
| leather_raw_provenance_update.md | 7491 | b9cde97278b99342033c935a29725eb106ede33fa58cb651fd8c2ac52327df46 | Leather 来源补档；v10.4 只读核验 |
| neu_mask_identity_verification.md | 844 | c2cd57ce88f8f1a750b1ce6208d6b3d39a4b53354341cd6e2bbd677275169829 | 840×3 mask 身份核验结论 |
| neu_identity_checks.json | 2620 | 7108c619de3e78d83042a791a70019e77c82cb87ba3cc030a904e6e20e9bd6ad | 身份核验明细（数量/顺序/尺寸/标签域） |
| neu_pixel_metric_reproduction.csv | 1476 | c0520443a2205ea76126559e936c0c8065a3a06ccc4d5c4e1dfb462f46438a5c | mIoU 从 mask 独立复算（0.912408/0.914544） |
| neu_confusion_matrices.json | 1024 | 6b1a19ad4a706bbaf669af1bf5e2facbb1baa18142f18e94361f1c7f0375e731 | 两模型 4×4 混淆矩阵 |
| neu_pixel_audit_summary.json | 3619 | abef313214d8b0562362b1d3b7f9aafe2e09264b0ea61731e01a9a41184608ec | NEU 审计关键数值汇总 |
| recomputed_case_metrics.csv | 812023 | fb4583034e6db58887fb5ad7ce66b8a04adcffba6a282903b5e18465725c8bc5 | 2520 行逐样本逐类指标独立复算 |
| recomputed_per_image_miou.csv | 55992 | e0a3c5518f9e5aed8f4b657d6a1b91b56e8cac28006ac0df0203bdc16e17780f | 840 张逐图 mIoU 复算 |
| paired_case_reproduction_diff.json | 4866 | cbb83e147949cfc788104a85cd98e097a4077330a9c721523e13b2206a07156b | 与原始 paired_case_metrics.csv 的差异比对（≤1e-10） |
| boundary_metric_implementation_audit.md | 1236 | 5acdaa7726a8bc2d4f69796d7c9203df44340d042c8ff7e0858c2060bed98ded | 对称边界带 / BF1 原始实现审计 |
| boundary_empty_rule_tests.json | 1908 | 6d92d9abda1aebddaeb3f7a356f4f076859114db275a1f42c6bb30248b79135f | 空标注四种组合规则合成测试 |
| boundary_named_sample_audit.json | 2012 | 7517645b4a406da2254ed03c5d87f6410837ddea922b00dd83305dadbf6923c9 | 指定单方有效样本（000510 等 7 例）核验 |
| estimand_abc_pixel_verification.csv | 8715 | 3f8cfe8ad6f20529a7945f728c91fcd72d9cd5a43e1890b33f239e604b5855b2 | A/B/C 三套统计目标逐指标点值复算 |
| estimand_abc_sample_sets.json | 560502 | 0e3800ff713104b9609b7c571016962ee759f0f8cce263980cd60620f1af8080 | A/B/C 各估计目标的完整样本 ID 集合 |
| bootstrap_reproduction_summary.csv | 19688 | ef369faa743dfb6e7f197d53353acbc241dbab1133b16923db1852a5117c11d6 | 72 行 Bootstrap 区间独立复算（5000 次、seed20261018、按图重抽样） |
| bootstrap_runtime_metadata.json | 383 | 2286de263dc789906d1e8ca38ffeb3665c2be3a1937ee19a9b76717e9752f472 | NumPy/RNG 版本与抽样索引 SHA |
| scale_dice_root_cause_audit.md | 2367 | dcc022a17f482effb78f992116e6bb6dc8c1a6d01294e5e244197ad194c065e2 | Dice 字段缺陷根因（delta_dice 缺失→回退 delta_iou） |
| scale_dice_corrected_recalculation.csv | 1893 | 24014149e31bfc99f2aea8835ecd8840b1e4bc489de9cf916f33dfa05619c118 | **修正文件**：正确的 Dice paired 复算（非原始实验输出） |
| rq1_pixel_decomposition.md | 2800 | d1045db4bb3116988acbf09f5249cdf1bbd37b4922af7dc3e8c29b7ed7e6635b | RQ1 聚合 vs 逐病例反向的数学分解说明 |
| rq1_decomposition.csv | 1317 | a2e2611cb9793ef68430ca7ac28e643e6eb1765309076d66bee83f84c3a5b338 | RQ1 分解数值（质量项/union 重加权项） |
| rq1_size_group_contributions.csv | 1286 | 8c6fbbdf7ff6c6448c649da6fed3c5efffbcdce3e075d11460da3da8e631f3ca | RQ1 按面积组的贡献分解 |
| rq2_raw_mask_condition_audit.md | 3212 | 2c52dba6c7413cc2e21dce737aec84e4e920fba38f3ad07d460612686ac462af | RQ2 面积/复杂度/混杂核验说明 |
| rq2_area_reproduction.csv | 1437 | af42b0e8bb354731edd36666a598cfe4238c0f2a73e44a00a3875cc634adc05d | 面积 P33/P67 阈值与分组指标复算 |
| rq2_complexity_reproduction.csv | 665 | 9966edbd5c497d0e1e010b58ce59eba56345fd306a1e5d07a5358ade80a3685d | 裂纹复杂度分组指标复算 |
| rq2_exploratory_confounders.csv | 507 | 4c2b69c22839b0699cc3852ec06fe8040ff342d3ecd32c1cc9d6cb7a6eb21b59 | 探索性混杂分析（连通组件等，非预注册） |
| figure_case_source_audit.md | 457 | c7301d13fb14ae249f5e27e4f80ec84258236949a7021b0bdbf49fd1aaf66d0d | 图3/图4 选例追溯结论 |
| figure3_candidate_trace.csv | 1326 | 383ced7840b7be1b8624530ba8f300715f8e457467de3e20da26f088135e58bb | 图3 18 条候选记录复算（误差 0） |
| figure3_actual_panels.json | 1881 | 352613123999571ac71b96f2c76c2c790630d45132c22f1ad25fb210190c6888 | 图3 实际入图六例与复算指标 |
| source_inventory.md | 1623 | 16964b140a9d08fd7d39fa2f979ad5030f73a11c4a8c9619089117a07f3a6d8a | 证据盘点说明（2626 条） |
| source_manifest.json | 1389381 | 5edda954445f6cce6c9fcba753b0944db80c0dbf4596165ff8e418f8c67a7ccd | 主证据 manifest（2626 条含 SHA） |
| source_manifest_supplemental.json | 5780 | 0db771698645c6188eab28eec386a61bd96ad32d1fe0348231948179e5060588 | 补充 manifest（Leather 深搜 16 条 + 缺失结论） |
| scripts/build_inventory.py | 7419 | 76699d1fadd4c10bf7d58807b0e28f94f616e2ad4e77691a4b1488279f859310 | 证据盘点脚本 |
| scripts/run_neu_pixel_audit.py | 41321 | 998565f7769fd6bf6001abfb52c593e9d65304d3e9316ba3b0d6287be72b2274 | NEU 像素级复算主脚本（只读 mask，不推理） |
| scripts/build_archive.sh | 7771 | e9890acedfa7681754485036e0f61b807c03310dc80fce592f54fc0c22ecee4f | 本归档构建脚本 |

### `neu_masks/` — NEU 840×3 原始 mask（打包）

| 文件 | 字节 | SHA-256 | 作用 |
|---|---:|---|---|
| neu_test840_masks.tar.gz | 1563571 | 768d53cbce4c1d3600caf3b607e4f020cdab3a8b941cc219c7b3ca05cffc45c5 | 2520 张 200×200 PNG：gt/、base_predictions/、a2_predictions/ 各 840 |
| mask_sha256_manifest.csv | 233806 | ab2e4bf39bf505fefbd969968caac7e8284b3d375c08c5c58d3c4ecf07269f3f | 2520 行逐文件 relative_path/sha256/bytes（按原始目录计算） |
| mask_archive_verification.json | 635 | c2d6cf577ebef4868776c526212dbe84f1d23d0f1319b968443988a534a8f006 | 解包逐文件复核结果：2520/2520 PASS |

- mask 来源：`/workspace/Industrial Surface Defect/repro_runs/mechanism_eval_240k/{gt,base_predictions,a2_predictions}/`
- 身份：840 个 sample_id 三组一致、顺序同 `experiments/protocol/splits/neu_test.txt`（SHA `2c45b319…8f06`）；标签值 {0,1,2,3}；GT 与源标注 840/840 逐像素一致；预测为 `outputs[0]` argmax 直接保存，无后处理。
- 校验方法：解包后对每个 PNG 重新计算 SHA-256 并与 `mask_sha256_manifest.csv` 比对；`mask_archive_verification.json` 记录本次构建时的复核结果。
- 许可说明：GT 为 NEU-Seg 公开研究数据集（Song & Yan, NEU）衍生的 200×200 整型标签图，预测为本项目模型输出。服务器端数据集目录未附显式许可文件；该数据集由作者公开发布供研究使用，本仓库此前已公开发布数据划分清单与含原图的可视化图件。归档目的为结果可复现；如权利方提出异议，可移除本目录。

### `leather/` — Leather 来源证据

| 文件 | 字节 | SHA-256 | 作用 / 来源 |
|---|---:|---|---|
| logs/run_2024_01_27_14_05_37.log | 58478 | 4b4a25ad7e8a136b45febd07e935098a5fc0669d3af87a5cc2cdeb359c21366e | DSMONet-B 历史训练日志；第 745 行 val235 0.8929607161637454 紧随 Iter[78000/80000]；源 `new (copy)/runs_other/dsmonet_resnet_pige/dsmor50_pige/` |
| checkpoint_metadata/DSMONet-B_checkpoint_info.json | 2506 | c87574f32d046b1464bc8b8316fd6209d6a71b5264d35001c1ef3a633bcca5cf | Base checkpoint 元数据（iter 78000、SHA、strict load） |
| checkpoint_metadata/A2MS-DSMONet-B_checkpoint_info.json | 2597 | 5b7191e25364b9711c630f3883165552b339f2b189f03e12c5e5847dba8c1745 | A2MS checkpoint 元数据（文件名 160000 vs 实际 130000 已记录） |
| evals/hist_0127_baseb_78k_reverify_class_iou.csv | 3697 | 84ee26b6503061e62b242206e91e235df58508829b15807839a009fb1af1c4db | Base 统一 test468 评价（mIoU 0.89912，含 8×8 混淆矩阵） |
| evals/hist_a2_160k_fixedfwd_class_iou.csv | 4016 | f40175c43d625973d7e0b9f930519621c7a8e73464ee71619b4c7f0c5a126aa5 | A2MS fixed-forward 统一 test468 评价（mIoU 0.909713） |
| manifests/hist_0127_baseb_78k_reverify.json | 1693 | ba3b71d00ba7780ce45f903298213274639e3ce7f22c5388aa1c0cfd660fee99 | Base 历史复核 manifest |
| manifests/hist_a2_160k_fixedfwd.json | 1736 | ca8737aecb7d93717ed4f8b6e8c1ebd7ecf20196f407021ee95f0fc5fa390cb3 | A2MS 历史复核 manifest |
| leather_paper_final_results.json | 3980 | 3656e34afa0ddce6a78de076e82cc2298d5850c9eb2716bc7be1e14abbd5ec31 | 论文最终 Leather 结果声明（89.91/90.97；60k 为 archived-only） |
| control_60k/base_b_config.yml | 1611 | d1c3434e80e1efaf4466ad57b3d066a45ee6e9122bd54fc697247ee7730991f7 | 60k Base 配置（cosine T_max=48000） |
| control_60k/a2_full_config.yml | 1680 | d2609c4b6438e6e75f94c664c28c31488fa94641e5ddf54f597ac1562d485f65 | 60k A2MS 配置 |
| control_60k/base_b_run_2026_09_24_15_39_05.log | 86631 | fb7d332fcf355babdefec18068f8b62795b6cd6872e8d1dc2be81cf41fd3a94b | 60k Base 训练日志；第 23 行 "CosineAnnealingLR … (Protocol L1 expects constant LR)" 措辞不一致证据 |
| control_60k/a2_full_run_2026_09_25_04_23_08.log | 93050 | d39c6fe375fc0f1c00916750a0429d4187d83e227a6ef3f2c18b89dea8bdd850 | 60k A2MS 训练日志 |
| control_60k/base_b_train_metrics.jsonl | 111098 | 6a7192628504e22ff266729946143c73c742d053229fe8f3f6d99fb2b1ea1d16 | 60k Base 逐 100 迭代指标（含实际 LR 曲线） |
| control_60k/a2_full_train_metrics.jsonl | 117180 | e5e579dac789c5c535ed86707938fd13ca876e1bfc0b0945456c318d1a4fb867 | 60k A2MS 逐 100 迭代指标 |
| control_60k/base_b_train_summary.json | 874 | 9c1828f9886187d959c57f6119df48134c05ad34ce04c6a0123a92f9bbd1943a | 60k Base 训练摘要 |
| control_60k/a2_full_train_summary.json | 884 | c7a597bb81feaa5b2c7817888ffb23e6e7aa35b810b3de933dc69b6ddda14483 | 60k A2MS 训练摘要 |
| control_60k/base_b_unified468_iter60000.csv | 4319 | 4cad1bc4341faf23ca795e0a06c88a4025e4afba98fba97b5acce29084b5251c | 60k Base 固定终点统一468评价（0.872137） |
| control_60k/a2_full_unified468_iter60000.csv | 4727 | f26c81f3bd5ba221bd5c2e5f0a3e49cbc09571c97d949c0642ca95fdd818c026 | 60k A2MS 固定终点统一468评价（0.849932） |
| figure4/final_viz_manifest.json | 2721 | dc26d413d6ba5948e3b8917511f3a553f1375ff8e68c0638c4aa3aafed1b4152 | 图4 manifest：268 缺陷阳性、211/45/12、六例 ID |
| figure4/leather_viz_final_v6.py | 6210 | 2ad77a44a9c3ce38197706ca96441bef82761c10f9dfd55e64597dba141910d7 | 图4 生成脚本（内存推理 + 一致性门，不保存预测） |
| figure4/assemble_fig34_v6.py | 2792 | 00d0c62d0203275951e2be638c541a986060167e86bd5dcf39f2a786423adc2e | 图3/图4 拼装脚本（图3 六例硬编码于 39-46 行） |

### 顶层

| 文件 | 字节 | SHA-256 | 作用 |
|---|---:|---|---|
| archive_integrity_verification.json | 17136 | 291b230957dc568c077f9a8674f0cc21a1af33ecd5f0dd95a848775cf75c20a5 | 56 个复制文件与来源 SHA-256 逐一比对：全部一致 |

## 可用性分级

- **可直接用于论文**：`neu_pixel_metric_reproduction.csv`（mIoU 复算）、`estimand_abc_pixel_verification.csv`、`bootstrap_reproduction_summary.csv`、`rq1/rq2` 各复算表、Leather 评价 CSV 与 manifest、60k 配置与统一468 CSV。
- **复算/修正（须标注性质）**：`scale_dice_corrected_recalculation.csv` 为对历史写出缺陷的修正复算；引用时须说明其替代旧 `scale_stratified_results.json` 中错误的 Dice paired 字段。
- **探索性（不得写作预注册分析）**：`rq2_exploratory_confounders.csv`（连通组件相关分析）。
- **部分核验（引用时须带限制语）**：Leather 图4 选例（manifest 级，无全量预测缓存）；A2MS Leather 90.97（checkpoint+统一评价已核验，训练日志缺失）；Bootstrap 区间（按声明设置独立复算一致，v10.3 原统计脚本未找到）。

## 未闭环缺口（如实记录）

1. A2MS Leather 130k 完整训练日志缺失（全项目检索 0.9090842 无日志命中；三段 detailloss 日志属 822 变体链，已排除，不得用作替代）。
2. Leather 468 张原始预测 mask 缓存不存在（任何链路），图4 268 分布不能做到原始 mask 级复核。
3. v10.3 任务书与 seed20261018 原统计脚本未在服务器找到。
4. 60k 训练日志 "expects constant LR" 措辞与实际 cosine 调度不一致（证据：`control_60k/base_b_run_2026_09_24_15_39_05.log` 第 23 行），未修改代码。

## 归档边界

本目录不含：模型 checkpoint（*.pkl）、原始图像数据集、tfevents、A2MS 缺失日志的替代品。checkpoint 仅记录元数据与 SHA-256。
