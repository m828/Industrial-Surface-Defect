# 仓库同步差异扫描报告 (missing_files_report)

> 日期：2026-09-28
> 范围：服务器项目工作区 vs GitHub 仓库 `m828/Industrial-Surface-Defect`（main，同步基线 commit b88bd53）
> 原则：只补充科研工程元数据、实验结果、配置、审计证据；不上传权重/数据集/大日志/临时文件。

## 一、扫描结论概览

上一轮推送（b88bd53）已覆盖：`docs/paper/` 全部论文稿与图件、`experiments/audit/`（68 个审计文件，含任务点名的 long240k_endpoint_results.json、boundary_quality_results.json、scale_stratified_results.json、morphology_exploratory_results.json、paired_case_analysis.json、leather_results.json、leather_efficiency_benchmark.json 等）、`experiments/results/`、`experiments/protocol/` 决策文档、`tools/` 5 个脚本。

本轮扫描发现的**真实缺口**：数据划分清单、正式训练配置、checkpoint 元信息、图 3/图 4 选例索引。

## 二、差异清单

| 文件/目录 | 是否建议上传 | 原因 |
|---|---|---|
| `new (copy)/dataset/train_neu.txt` / `test_neu.txt` | **是** → `experiments/protocol/splits/` | NEU 3630/840 划分是论文协议的一部分；仅含相对路径与 ID，无图像 |
| `new (copy)/dataset/pige/{train,val,test}.txt` | **是** → `experiments/protocol/splits/` | Leather 1638/235/468 划分同上 |
| `repro_runs/long240k_seed1337/*/config/config.yml` | **是** → `experiments/configs/`（脱敏后） | NEU 240k 正式训练配置（optimizer/lr/batch/iter/seed/loss 权重）；原始绝对路径已改占位 |
| `new (copy)/dsmonet_resnet_pige.yml`、`new/dsmonet_resnet_pige_loss.yml` | **是** → `experiments/configs/`（整理存档） | Leather 历史训练链配置（batch 12、val_interval 1000、λ_d=1 等论文依据） |
| checkpoint sha256/iter/seed/mIoU 元信息 | **是** → `checkpoint_metadata/` | 权重不上传，但元信息是论文溯源关键 |
| `repro_runs/mechanism_eval_240k/case_selection.json`、`viz_manifest_v6.json`、`leather_historical_audit/final_viz/final_viz_manifest.json` | **是** → `docs/paper/figures/` | 图 3/图 4 固定选例规则与面板索引（纯 JSON，与既有 Fig2 manifest 同位置） |
| `repro_runs/**/checkpoints/*.pkl` | **否** | 训练权重（>100MB 单文件），.gitignore 排除 |
| `repro_runs/**/*.log`、runs/、wandb/ | **否** | 大规模日志，.gitignore 排除 |
| `repro_runs/**` 预测 PNG / 可视化面板原图 | **否** | 体积大且为中间产物；最终图件已在 `docs/paper/figures/`，索引 JSON 已收录 |
| `new/`、`new (copy)/`、`subregion unet/` 模型与训练 .py | **本轮否** | 服务器工作目录代码，未整理；模型身份已由 `checkpoint_metadata/` 中 sha256 锁定，仓库 `src/` 仅留存档说明。如后续需要代码级复现，可单独整理后上传 |
| `experiments/retrain_a2ms_b_neu/`（5.3GB） | **否** | 已排除实验链的训练产物（checkpoint/日志）；其结论文档已在 `experiments/audit/` |
| `third_party/LETNet` | **否** | 第三方代码，非本项目产物 |
| 数据集图像（NEUSeg/pige images、annotations） | **否** | 原始数据不上传 |

## 三、本轮实际新增（详见 docs/repository_sync_report.md）

1. `experiments/protocol/splits/`：5 个划分清单 + README（含类别映射与 sha256 校验）；
2. `experiments/configs/`：4 个正式训练配置（NEU/Leather × DSMONet-B/A2MS-DSMONet-B）；
3. `checkpoint_metadata/`：两模型 checkpoint 元信息 + README；
4. `docs/paper/figures/`：fig3/fig4 选例与面板索引 JSON ×3；
5. `README.md`：更新方法命名与新增实验资产/复现指引章节；
6. `.gitignore`：补 `images/`、`imgs/`、`tensorboard/`、`.cache/`。
