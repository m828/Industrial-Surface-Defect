# 仓库同步报告 (repository_sync_report)

> 日期：2026-09-28
> 仓库：https://github.com/m828/Industrial-Surface-Defect
> 提交：`a3b7578` — "Add reproducibility metadata, experiment results and configs"
> 性质：科研工程资产补充同步。**未改动论文正文、未改动任何实验数字、未重新训练、未改变已有目录结构。**

## 一、新增文件列表与用途

本轮新增 17 个文件、更新 2 个文件（19 files changed, +7559/−29）：

### 数据划分（experiments/protocol/splits/）

| 文件 | 用途 | 影响论文复现？ |
|---|---|---|
| `neu_train.txt`（3630 行） | NEU-Seg 训练集清单 | 是：训练集定义 |
| `neu_test.txt`（840 行） | NEU-Seg 测试集清单（sha256 与审计 manifest 一致） | 是：表1/2/3/6/7 的评价集定义 |
| `leather_train.txt` / `leather_val.txt` / `leather_test.txt`（1638/235/468 行） | Leather 三划分清单 | 是：表4/表5 的划分与验证集选择依据 |
| `README.md`（splits） | 格式说明、类别映射、sha256 校验、与 leather_valid_test_list.txt 的关系（内容相同，仅行尾符差异） | 防止误用 |

### 训练配置（experiments/configs/）

| 文件 | 用途 | 影响论文复现？ |
|---|---|---|
| `neu_dsmonet_b_240k.yaml` | NEU DSMONet-B 正式配置（240k、Adam 1e-4 恒定、batch16、seed1337、ω=[10,1,3]） | 是：表1/表7 的 91.24% |
| `neu_a2ms_dsmonet_b_240k.yaml` | NEU A2MS-DSMONet-B 正式配置（λ_d=3） | 是：表1/表7 的 91.45% |
| `leather_dsmonet_b.yaml` | Leather DSMONet-B 历史链配置存档（batch12、val_interval=1000） | 是：表4/表5 的 89.91% |
| `leather_a2ms_dsmonet_b.yaml` | Leather A2MS-DSMONet-B 历史链配置存档（detail_aggregate_loss，λ_d=1） | 是：表4/表5 的 90.97% |

配置均基于服务器真实运行/历史配置文件整理，绝对路径已替换为相对占位。

### Checkpoint 元信息（checkpoint_metadata/）

| 文件 | 用途 | 影响论文复现？ |
|---|---|---|
| `DSMONet-B/checkpoint_info.json` | NEU 240k ckpt（sha256 46c5b83c…）与 Leather ckpt（a96c78ea…，实际 iter 78000）元信息 | 是：权重身份与结果对齐 |
| `A2MS-DSMONet-B/checkpoint_info.json` | NEU 240k ckpt（5beaaa1c…）与 Leather ckpt（336c5a4c…，实际 iter 130000）元信息 | 是 |
| `README.md` | 论文表格与 checkpoint 的对应关系总表 | 导航 |

### 图件索引（docs/paper/figures/）

| 文件 | 用途 | 影响论文复现？ |
|---|---|---|
| `fig3_case_selection.json` | 图 3 固定选例规则输出（裂纹/斑块 × 改善/退化/相当） | 是：图 3 可复核 |
| `fig3_viz_manifest_v6.json` | 图 3 面板索引（18 面板，v6 命名） | 是 |
| `fig4_final_viz_manifest.json` | 图 4 选例（268 张含缺陷样本排序，top2/bottom2/near-zero2）与一致性门记录 | 是 |

以上 JSON 中的服务器绝对路径已脱敏为 `<workspace>/` 相对形式，文件名与 ID 不变。

### 其他

| 文件 | 用途 |
|---|---|
| `README.md`（仓库根，更新） | 方法命名更正为 DSMONet/A2MS-DSMONet 两层体系；新增"实验资产与复现指引"章节（数据集说明、实验协议、表格数据出处、评价复现步骤） |
| `.gitignore`（更新） | 增补 `images/`、`imgs/`、`tensorboard/`、`.cache/` |
| `docs/missing_files_report.md`（新增） | 差异扫描报告（含不建议上传项及原因） |

## 二、已在仓库、本轮无需重复上传的内容

- `experiments/audit/`（68 文件）：long240k_endpoint_results.json、boundary_quality_results.json、scale_stratified_results.json、morphology_exploratory_results.json、paired_case_analysis.json、leather_results.json、leather_efficiency_benchmark.json、neu_current_base_9124_manifest.json 等全部审计证据（commit b88bd53 已上传）；
- `experiments/results/`：类别 IoU、复杂度、小目标分组等结果；
- `docs/paper/`：v2–v6.1 全部论文稿与 Fig1–4 终版图件；
- `tools/`：统一评价与复杂度脚本。

## 三、未上传内容及原因

| 内容 | 原因 |
|---|---|
| 训练权重（*.pkl，含 4 个论文 checkpoint） | 单文件 >100MB，.gitignore 排除；sha256 元信息已存档于 `checkpoint_metadata/` |
| 数据集图像与标注 | 原始数据不上传；划分清单已收录 |
| `repro_runs/**` 训练日志、预测 PNG、中间面板 | 体积大/中间产物；结论性 JSON 与最终图件已收录 |
| `experiments/retrain_a2ms_b_neu/`（5.3GB 训练产物） | 已排除实验链产物；其结论文档在 `experiments/audit/` |
| 服务器工作目录训练代码（`new/`、`new (copy)/`、`subregion unet/`） | 未整理代码，本轮不在同步范围；模型身份由 sha256 锁定 |
| `third_party/LETNet` | 第三方代码 |

## 四、上传前检查结果

| 检查项 | 结果 |
|---|---|
| 数据集混入 | 无（仅清单 txt） |
| 权重混入 | 无 |
| 大文件 | 无 >1MB 文件（远低于 100MB 上限） |
| 隐私信息 | 无 |
| 新增文件绝对服务器路径 | 无（已脱敏；历史审计文件中的路径为既有内容，未改动） |
| `git diff --stat` | 19 files changed, +7559/−29 |

## 五、推送状态

- 本地提交：`a3b7578`（资产同步）+ `67c184f`（本报告），均在 b88bd53 之上；
- 推送：**已完成**。期间两次网络瞬断（TLS/SSL 超时），重试后成功；经 GitHub API 核验 origin/main = `67c184f`，新增文件全部可访问。
