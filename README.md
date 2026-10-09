# Industrial Surface Defect Segmentation with A2MS-DSMONet

本项目用于整理工业表面缺陷实时语义分割论文、实验代码、实验结果和投稿材料。当前目标期刊为《中国图象图形学报》；若后续投稿受阻，备选期刊为《计算机工程与应用》。

## 当前论文方法

论文采用两层方法体系：

1. **DSMONet**：细节—语义相互优化基础架构，通过深层语义约束浅层细节、优化细节反向补偿深层语义，实现双向互优化；
2. **A2MS-DSMONet**：在 DSMONet 基础上面向工业表面缺陷（低占比、弱纹理、多尺度、形态差异显著）引入自适应语义增强模块（AAM = eSE + ACW）与辅助细节监督（仅训练阶段）。

命名规则（v6 起锁定）：

- DSMONet / DSMONet-B：基础架构及其 ResNet-50 主干模型（NEU-Seg 91.24% / Leather 89.91% mIoU）；
- A2MS-DSMONet / A2MS-DSMONet-B：工业缺陷增强完整模型（NEU-Seg 91.45% / Leather 90.97% mIoU）；
- 历史整理期别名 Base-B、A2MS-DefectNet 已废止，见 `docs/paper/model_naming_audit_v6.md`。

当前投稿候选稿：`docs/paper/paper_draft_cjig_v6_2_submission_candidate.md`（CJIG v6.2，论文主线已冻结，实验数字锁定，不再修改）。

## 数据集

- NEU-Seg 带钢表面缺陷公开数据集（3630 训练 / 840 测试，200×200，4 类含背景）；
- 皮革表面缺陷数据集（1638 训练 / 235 验证 / 468 测试，768×768，7 缺陷类 + 背景）。

数据不上传；划分清单见 `experiments/protocol/splits/`。

## Baseline 对比实验（CJIG v7 准备）

为论文实验章节补充的三级对比，全部在统一协议下完成：

- **工业缺陷方法**：FDSNet、LETNet；
- **通用实时语义分割**：STDC-Seg、PIDNet-S、DDRNet23-slim；
- **本文方法**：DSMONet-B、A2MS-DSMONet-B。

统一训练/评价协议（所有 baseline 与本文模型一致）：NEU-Seg train 3630 / test 840；4 类含背景；from scratch；seed 1337；Adam（lr 1e-4 恒定、wd 2e-6）；batch 16；240k 固定终点评价；训练全程不使用验证/测试集、无 test-based 模型选择；统一 `tools/evaluate_class_iou.py` 评价。

资产位置（`experiments/baseline_comparison/`）：

- `results/final_baseline_table_v2.md`：7 模型对比总表（mIoU/逐类 IoU/Params/MACs；FPS 待统一复测）；
- `results/{fdsnet,letnet,stdc,pidnet,ddrnet}_results.json` + 对应 `_audit.md`：单模型结果与审计（checkpoint sha256、配置、评价口径）；
- `results/fps_rebenchmark_plan.md`：FPS 统一复测计划（GPU 空闲时执行后回填）；
- `industrial_baseline_audit.md`：DMC-Net 不可复现的证据链与工业 baseline 选型记录；
- `configs/`、`scripts/`：各 baseline 的 240k 训练配置与脚本。

## 目录说明

- `docs/paper/`：论文稿（v2–v6.2，当前投稿候选稿为 v6.2）、图件（Fig1–4，SVG/PDF/600dpi PNG）、修订日志、claim 审查、方法谱系说明、命名审计、投稿检查清单；
- `docs/journal/`：期刊材料（CJIG 体例排版模板 `CJIG_template.doc`、版权转让声明及保密审查证明 `CJIG_copyright.pdf`）；
- `docs/literature/`：相关工作调研与参考文献核验；
- `experiments/audit/`：全部审计档案（NEU 历史结果溯源、Leather checkpoint 取证、统一复评 manifest、边界/尺度/形态分析结果 JSON）；
- `experiments/results/`：类别 IoU、复杂度、小目标分组等结果表与判定记录；
- `experiments/configs/`：论文结果对应的四个正式训练配置（NEU/Leather × DSMONet-B/A2MS-DSMONet-B）；
- `experiments/baseline_comparison/`：CJIG v7 基线对比实验（工业缺陷 FDSNet/LETNet + 通用实时 STDC/PIDNet/DDRNet 的 240k 配置、脚本、结果 JSON 与逐模型审计），详见上文"Baseline 对比实验"一节；
- `experiments/protocol/`：评价协议锁定记录与数据划分清单（`splits/`）；
- `checkpoint_metadata/`：论文 checkpoint 元信息（iter/seed/sha256/mIoU），**不含权重文件**；
- `tools/`：统一评价与复杂度测量脚本（`evaluate_class_iou.py`、`measure_complexity.py` 等）；
- `src/`：遗留代码说明。

## 实验资产与复现指引

**实验协议**

- NEU-Seg：from scratch、240k 迭代固定终点、Adam（lr 1e-4 恒定、weight decay 2e-6）、batch 16、seed 1337；训练中不使用验证/测试集做模型选择；统一 840 张测试图像评价一次，mIoU 含背景（TestRescale + ToTensor，无 ImageNet Normalize）。
- Leather：历史训练链（Adam lr 1e-4 恒定、batch 12、每 1000 迭代在 val235 监控并按验证集最优保存）；最终仅在 468 张独立测试集评价一次。

**论文表格数据出处**

| 论文条目 | 数据文件 |
|---|---|
| 表1/表7 NEU 主结果 | `experiments/audit/long240k_endpoint_results.json`、`neu_current_base_9124_manifest.json` |
| 表2 类别/边界差异 | `experiments/audit/boundary_quality_results.json`、`paired_case_analysis.json` |
| 表3 尺度分层 | `experiments/audit/scale_stratified_results.json` |
| 表4/表5 Leather 主结果与逐类 | `experiments/audit/leather_paper_final_results.json` |
| 表6 10k 消融 | `experiments/audit/a2ms_multiseed_10k_results.md` 及各 seed manifest |
| 表8 复杂度/实时性 | `experiments/audit/efficiency_benchmark_results.json`、`leather_efficiency_benchmark_historical_ckpt.json` |
| 图3/图4 选例 | `docs/paper/figures/fig3_case_selection.json`、`fig3_viz_manifest_v6.json`、`fig4_final_viz_manifest.json` |

**如何复现评价**（需自行准备数据与权重，本仓库不提供）：

1. 按 `experiments/protocol/splits/` 的清单组织数据目录（注意 NEU 与 Leather 清单字段顺序相反）；
2. 按 `experiments/configs/` 对应配置训练或获取 checkpoint；
3. 使用 `tools/evaluate_class_iou.py`（strict 加载）在对应测试清单上评价；A2MS 模型须显式指定 fixed 模型文件，命令示例见 `experiments/scripts/evaluate_class_iou_command_examples.md`；
4. 评价结果应与 `checkpoint_metadata/` 中记录的 sha256/mIoU 对齐。

## 重要说明

- 本仓库是论文与代码的主版本库（CJIG 投稿准备阶段唯一工作版本）；服务器仅用于训练实验，不作为论文修改源，baseline 等训练实验在服务器执行后按协议回传结果汇总；
- GitHub 仅保存：代码、配置、论文、审计文件与小型结果文件；
- 训练权重、数据集、大规模训练日志不提交到 GitHub（`.gitignore` 已覆盖 `*.pkl/*.pth/*.pt/*.ckpt`、`data/`、`dataset/`、`repro_runs/`、`runs/` 等）；
- 权重建议使用 Git LFS、Release、网盘或服务器路径管理；
- 所有新实验结果必须先写入 `experiments/results/new_results_pending.md`，核验后再进入论文；
- 不允许将口头结果、未确认日志或未核验表格直接写入摘要、结论或主实验表。
