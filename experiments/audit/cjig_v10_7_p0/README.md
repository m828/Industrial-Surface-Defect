# CJIG v10.7-P0：NEU-Seg 数据谱系与划分独立性审计

## 审计对象与边界

- 对象：当前实际训练/评价链中的 train 3630、test 840，共 4470 条记录；唯一键为 `(split, sample_id)`。
- 实际数据：`/workspace/Industrial Surface Defect/new (copy)/dataset/NEUSeg`。正式 Base/A2 240k 运行副本的 `code/dataset` 均解析到其父目录。
- 上游参照：[DHW-Master/NEU_Seg@da8cebed](https://github.com/DHW-Master/NEU_Seg/tree/da8cebedb292beca4d101db8c442d5706fac3bfe)。当前 4470 张图像和 4470 张 GT 与该固定版本的对应文件逐字节一致，双方解码像素也一致。
- 本轮只做 CPU 文件读取、解码、哈希、图像统计和谱系审计。没有训练、模型推理、checkpoint 选择、重标注、重划或论文修改。
- 近重复阈值是在发现异常后回顾性选择的，不是盲法预注册。检索完整性和阈值内复核完成，不等于证明了所有未知变换或真实采集关系。

## 最终状态

| 项目 | 状态 | 限定 |
|---|---|---|
| ACTUAL TRAINING INPUT IDENTITY | **VERIFIED** | 限定于实际路径、清单、当前源文件和固定上游身份；没有逐 batch 历史采样记录或增强张量重放。 |
| FULL DATASET TRANSFORM SCREEN | **COMPLETE** | 全 4470 记录的文件/像素/D4 筛查完成；既定全对检索产生 33145 个候选并全部精核；两项独立复核均 PASS。 |
| CROSS-SPLIT SOURCE INDEPENDENCE | **COMPROMISED** | 至少一对跨 split 图像精确 D4 等价；不能称当前 train/test 为独立图像划分。 |
| ORIGINAL ACQUISITION PROVENANCE | **PARTIAL** | 上游固定版本已闭合；原始采集 ID、制作与划分顺序未闭合。 |
| MANUSCRIPT SCIENTIFIC READINESS | **HOLD FOR SOURCE EVIDENCE** | 冻结数值可限定为固定清单内部结果，不能据此宣称 840 个独立采集图像上的泛化。 |
| GITHUB DELIVERY | 以实际 commit 和 `git ls-remote` 回执为准 | 本文及文件生成本身不代表已提交或已推送。 |

## 主要结果

| 证据项 | 结果 | 解释 |
|---|---:|---|
| 图像原始文件字节相同 | 4 对、8 条记录 | train/train 3 对，test/test 1 对，跨 split 0 对。 |
| 图像解码像素相同 | 4 对、8 条记录 | 与上述四对一致；每对 GT 均相差 580 像素。 |
| GT 原始文件/未变换像素相同 | 0 对 | 4470 个未变换 GT 均唯一。 |
| 图像 D4 精确关系 | 5 对 | 包含 4 个 r0 恒等关系；非恒等变换仅 1 对。 |
| GT D4 精确关系 | 5220 对、870 个规范组 | train/train 3510，test/test 0，跨 split 1710；GT-only 不能单独合并图像来源家族。 |
| 跨 split 图像精确变换 | 1 对 | `test:000717` 旋转 180° 后与 `train:004097` 逐像素相同；GT 不同。 |
| 全对 Hamming 预筛 | 9988215 个非自配对 | image 预筛 445473、GT 预筛 2894526、并集 3035263。 |
| 正式近重复候选 | 33145 对 | 两遍全部精核，剩余 `CAP_NOT_REVIEWED=0`；不含未复核候选。 |
| 高置信候选记录对 | 4481 对 | 含 1 个精确 r180 图像对和 4480 个达到全部图像高相似条件的近似对；不是已证实采集泄漏数。 |
| 高置信跨 split 候选 | 1160 对、413 个 test 记录 | 仅用于候选敏感性范围，不得改写成 1160 对已证实泄漏。 |
| 高置信候选图 | 1920 个组件（含单例） | 非单例 1107；跨 split 非单例 413；连接 4485 条。 |
| 原始采集 ID / 有直接采集证明的多记录家族 | 0 / 0 | 真实采集家族总数不可知，不是零。 |

高置信候选图的 4485 条连接与 4481 个高置信候选记录对是不同统计口径：前者还连接四个原始数字身份对。低阈值及 GT-only 关系不用于传递合并图像家族。`cap_resolution/source_family_summary.json` 中的候选覆盖比例也不能称为泄漏比例。

### 八个重点记录

`actual_priority_training_input_identity.csv` 给出完整路径、清单行、文件 SHA、像素 SHA 和上游匹配。四个原始同图对分别形成数字身份二人组，不是一个八人采集证明家族。最初六组跨 split 旋转 GT 关系全部检出；图像 Pearson 约 0.9943、但 SSIM 约 0.910，未达到既定 0.95 高置信阈值，故保留为 `IMAGE_NEAR_ONLY_SOURCE_UNRESOLVED`。

### 新确认的精确图像跨 split 关系

`test:000717 --r180--> train:004097`：图像逐像素相同；同方向 GT 相差 4682 像素、像素一致率 0.88295、非零前景 IoU=0.49709989258861437。该关系在当前文件和指定上游固定版本中均存在，足以否定“图像变换独立”，但不证明相同 GT 或具体制作顺序。

## 权威文件层级

1. **`cap_resolution/`：最终权威近重复与来源图结果。** 第一遍精核 20810 对，第二遍仅完成既定 12335 个 backlog；无新增 pair ID、无剩余未复核、无阈值变化。
2. **根目录的精确身份、D4、正例、配置和报告。** `full_image_identity_manifest.csv`、`d4_transform_duplicate_pairs.csv` 等不受原 LSH 缺陷影响；`cap_resolution/` 保留了相应同字节副本。
3. **`exhaustive_hamming/`：第一遍部分精核结果。** 它不是独立的最终完整近重复结果；不得继续引用其中 413 个跨 split 家族等第一遍统计代替最终统计。
4. **根目录原 LSH 近重复/来源图文件及 `overflow_resolution/`：非权威诊断材料。** 不进入 Git 候选清单。具体排除项和原因见 `publication_scan.json`、`original_lsh_run_defect_notice.json` 和 `server_private_artifact_manifest.json`；原件未删除。

原 LSH 的 503436 行只有 500000 个逻辑 pair，3436 个反向重复来自未排序 D4 种子索引。该缺陷不影响文件、像素或 D4 精确结果，但原近重复计数与来源汇总不可用于论文。

## 必需交付文件索引

所有文件都属于本 P0 审计，不产生新的模型实验。逐文件 SHA-256 统一见 `SHA256SUMS.txt`，避免在正文重复维护另一份哈希表。

| 文件 | 作用与来源 |
|---|---|
| `README.md` | 本索引、权威层级、证据边界与校验方法。 |
| `actual_training_input_identity.md` | 实际训练/评价路径、八个重点记录和固定上游身份闭环；来源为运行副本、DataLoader、清单、日志及本轮 SHA。 |
| `full_image_identity_manifest.csv` | 4470 条全量文件、解码像素、D4 规范哈希、格式、标签域、上游对应与清单位置；不含原始像素。 |
| `exact_duplicate_summary.md` | 原始文件/像素精确重复及全量 D4 结果。 |
| `d4_transform_duplicate_pairs.csv` | 图像 5 对、GT 5220 对精确变换关系及实际方向。 |
| `source_family_graph_summary.md` | 数字身份、精确图像 D4、高置信候选和采集证明四个层级的解释。 |
| `split_independence_audit.md` | 划分独立性判定及其对评价的约束。 |
| `data_preparation_provenance.md` | 上游提交历史、本地制备记录和五种形成顺序假设的证据分级。 |
| `manuscript_scientific_impact.md` | 对固定 mIoU、原 Bootstrap、RQ1/RQ2 和未来训练的限定；本轮不改论文。 |
| `remaining_source_gaps.md` | 未解决的采集、类别语义、许可和历史运行缺口，以及 NOT RUN。 |
| `SHA256SUMS.txt` | 所有 Git 候选文件的 SHA-256（标准清单不包含自身）。 |
| `cap_resolution/near_duplicate_candidates.csv` | 任务要求的权威近重复候选表；33145 个唯一无序记录对，全部 `FULLY_REVIEWED`，保存实际比较方向。 |

## 其他公开证据文件

| 文件或目录 | 作用与来源 |
|---|---|
| `cap_resolution/key_findings.json` | 从最终 CSV/JSON 派生的只读汇总，不是新实验结果。 |
| `cap_resolution/near_duplicate_summary.json` | 候选类别、跨 split 计数、全对预筛计数和两遍完成证明。 |
| `cap_resolution/source_family_{edges,membership}.csv` | 分级来源边与 3×4470 条成员记录；GT-only 不合并图像家族。 |
| `cap_resolution/source_family_summary.json` | 各层组件、覆盖范围及跨 split 统计。 |
| `cap_resolution/gt_d4_canonical_groups.csv` | 870 个 GT 规范组及组内图像证据；不是采集家族表。 |
| `cap_resolution/priority_rotation_pair_metrics.csv` | 最初六组旋转 GT 对应图像/GT 的完整复核指标。 |
| `cap_resolution/cross_split_exact_image_pair_metrics.csv` | 新确认 r180 图像对的对齐指标。 |
| `cap_resolution/additional_cross_split_candidates.csv` | 高置信跨 split 候选明细，不是已证实泄漏清单。 |
| `actual_priority_training_input_identity.csv`、`priority_sample_identity.csv` | 八个重点记录的完整身份及上游对应。 |
| `exact_duplicate_pairs.csv`、`exact_duplicate_summary.json` | 原始文件与未变换像素关系的明细和汇总。 |
| `d4_duplicate_summary.json` | 精确 D4 的分 split 汇总。 |
| `positive_control_verification.json`、`detector_self_tests.json` | 四个同图对、六组旋转 GT、D4 方向和合成 JPEG+旋转检查。 |
| `upstream_identity_summary.json` | 固定上游 commit、归档 SHA 和 8940 文件匹配汇总。 |
| `upstream_history_evidence.json` | 两次上游提交、固定树文件清单和许可/语义缺口。 |
| `external_provenance_leads.json` | 后续论文及作者代码库线索；二手材料不升级为原始证明。 |
| `cap_resolution/source_temporal_metadata.json` | 源文件 mtime 辅助证据；不是密码学时间证明。 |
| `pre_audit_environment.json` | 审计前仓库、HEAD、remote、既有工作区状态和数据路径。 |
| `analysis_config.json`、`exhaustive_hamming/analysis_config.json`、`cap_resolution/analysis_config.json` | 回顾性检索、阈值、上限与 backlog 完成规则。 |
| `cap_resolution/independent_verification_report.json` | **PASS**；37 项检查，独立复算源文件/像素/D4、33145 个候选精核指标和来源组件。 |
| `cap_resolution/retrieval_screen_verification.json` | **PASS**；7 项检查，独立复算全对方向 Hamming、缩略图闸门、候选集合、描述子和方向。 |
| `cap_resolution/execution_record.json`、`cap_resolution/run_environment.json` | 实际执行参数、Git 状态、耗时和库版本；`COMPLETED_CAP_RESOLUTION` 不替代独立 PASS。 |
| `analysis_attempts.json` | 失败、停止、部分覆盖、最终运行和验证尝试，避免只保留成功路径。 |
| `original_lsh_run_defect_notice.json` | 原 LSH 缺陷、未受影响输出和最终权威层级。 |
| `server_private_artifact_manifest.json` | 私有归档、描述子、完整日志和排除文件的路径、大小、SHA 与未上传状态。 |
| `publication_scan.json` | 公开性、凭据、邮箱、原始数据和文件大小扫描，以及排除清单。 |
| `git_commit_candidate_list.txt` | 显式 Git 暂存清单；不使用 `git add .` 或 `git add -A`。 |

`cap_resolution/` 中同名的身份、精确关系和正例文件是与父级精确结果一致的副本。各文件的实测 SHA 以 `SHA256SUMS.txt` 为准。

## 实际分析脚本

| 脚本 | 用途 |
|---|---|
| `scripts/run_neu_source_audit.py` | 公共数据读取、身份、D4、候选精核、来源图及发布函数；原始 LSH 流程不是最终近重复依据。 |
| `scripts/legacy_lsh_defect_version.py` | 保留缺陷版本供追溯；不要用于重新生成结论。 |
| `scripts/compute_oriented_hamming.py` | 全记录对、八方向 Hamming 矩阵；描述子仅保存在服务器私有目录。 |
| `scripts/run_exhaustive_source_audit.py` | 全对预筛、缩略图闸门及第一遍精核。 |
| `scripts/complete_candidate_backlog.py` | 用 8 个 CPU worker 完成既定 12335 个 backlog，不新增候选或改变阈值。 |
| `scripts/verify_neu_source_audit.py` | 独立重读源文件并复算精确关系、候选全尺寸指标和来源组件。 |
| `scripts/verify_retrieval_screen.py` | 独立复算检索矩阵、闸门、候选集合和描述子；失败尝试已保留，不跳过失败检查。 |
| `scripts/derive_final_findings.py` | 生成重点关系、GT 规范组和最终汇总。 |
| `scripts/render_summary_reports.py` | 在独立精核 PASS 后生成汇总报告。 |
| `scripts/finalize_publication.py` | 强制两项独立复核 PASS，生成排除清单、公开扫描、暂存清单和 SHA。 |

完整执行日志和含描述子的 NPZ 不上传；位置、大小与 SHA 见 `server_private_artifact_manifest.json`。公开脚本中的 `/workspace` 路径用于服务器复现，不含个人主目录、凭据或邮箱。

## 校验与复现边界

在服务器本目录执行：

```bash
sha256sum -c SHA256SUMS.txt
```

在 Windows 本地同步后，可用 `Get-FileHash -Algorithm SHA256` 逐文件核对清单；**本地同步及校验不属于本轮服务器执行结果，必须在本地实际完成后单独报告。**

清单验证的是公开元数据交付完整性，不需要重新分发原始数据。重新运行像素级分析则仍需原始服务器数据及固定上游私有副本；本包不包含原图、GT、LabelMe、checkpoint 或上游数据归档，也没有把缺少这些材料的环境描述成可完整重跑。

## 公开性与许可

- 公开候选仅为非敏感元数据、统计、报告、脚本和哈希；没有原始图像字节、GT 像素数组或模型文件。
- 固定上游没有 LICENSE 或整数类别映射；GT 再分发授权未建立，类别语义没有新增原始证据。不得凭外形把 `class_id1` 命名为 scratches。
- 服务器原始文件和完整日志保持原状；排除项未通过删除、Git 改写或变更仓库可见性来规避。
- Git 仓库：`https://github.com/m828/Industrial-Surface-Defect.git`；分支 `main`；审计前 HEAD 为 `0426b978ef76cd082a060874b3eff2a16184ce6d`。只暂存明确清单，保留其他历史工作区状态。

## 论文与后续实验约束

冻结的 Base 0.9124083744680408 和 A2 0.9145442177671137 可继续作为固定上游、固定清单、固定权重下的内部结果，不得改为独立采集泛化结论。原 840 记录 Bootstrap 不是来源簇或跨 seed 区间。RQ1/RQ2 的固定 mask 数学事实仍可复算，但其独立性解释需要来源簇敏感性分析。

若目标是独立泛化，需要采集家族级划分或独立外部评价；删除若干测试记录不能自动消除训练曝光风险。当前证据足以暂停“在当前 split 上增加多种子来解决独立性”，但不足以设计已证明采集独立的新训练。

**NOT RUN：**训练、模型推理、checkpoint 加载/选择、新预测指标、新记录或来源簇 Bootstrap、删样本敏感性分析、重标注、重划、机制消融及论文修改。
