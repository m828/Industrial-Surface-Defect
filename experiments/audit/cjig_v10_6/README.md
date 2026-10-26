# CJIG v10.6-P1 核心研究管理证据包

日期：2026-10-26
范围：NEU-Seg 原始标签溯源、DSMONet-B / A2MS-DSMONet-B 独立权重与预测资产盘点、240k 复用门禁、GPU 成本只读调查、RQ1/RQ2 最小验证预注册与投资决策。
执行边界：全程只读取证与文档生成；**未训练、未推理、未选择新 checkpoint、未修改既有论文与原始实验**。

## 最终状态

| 维度 | 状态 |
|---|---|
| CLASS SOURCE CHAIN | **PARTIAL**（整数编码链 VERIFIED；上游语义命名 MISSING） |
| E0 10K ASSETS | **INCOMPLETE**（权重+聚合评价已核验；840 逐图预测缓存全部缺失；10k≠240k 协议） |
| E1 240K ASSETS | **INCOMPLETE**（1337 一对 AVAILABLE_AND_VERIFIED；2026/3407 四运行 MISSING） |
| INDEPENDENT VALIDATION | **READY FOR AUTHORIZATION**（限定表述下非必需） |
| seed1337 复用门禁 | **PASS**（附实际更新 239,999 的保存点偏移记录） |

## 文件清单

| 文件 | 内容 |
|---|---|
| `server_class_semantics_trace.md` | 类别语义溯源：数据版本、转换链、重叠/重复检查、命名冲突、结论 |
| `class_mapping_source_inventory.csv` | 类别证据逐项清单（路径/SHA/证明力/状态） |
| `class_data_integrity_checks.json` | split 闭合、300 同名 ID 内容不同、跨 split 0 内容重复、split 内重复图像对（仅计数/文件名/哈希） |
| `independent_weight_inventory.csv` | 18 个权重对象的实测核验表（实际 SHA、配置、评价、缓存、状态） |
| `weight_provenance_gate.md` | 复用门禁逐项判定；239,999 更新偏移的代码级证据；E0/E1 可用性边界 |
| `candidate_prediction_cache_inventory.csv` | 29 项预测/评价缓存清单（240k 已核验；10k 逐图缓存 MISSING） |
| `weight_asset_verification_details.json` | 每个对象的原始核验明细（配置、metrics、日志路径、manifest 对照） |
| `e0_checkpoint_sha_crosscheck.json` | 10k 全部 checkpoint 实测 SHA 与 manifest 任意登记值交叉核对 |
| `gpu_cost_evidence.md` | 真实墙钟、Time/Image、争用分析、校准值与缺口；不使用 FPS 推算 |
| `validation_preregistration.md` | RQ1/RQ2 冻结分析、反证条件、E1 四运行条件方案（未执行） |
| `experiment_go_nogo_decision.md` | 方案 0–3 投资决策与建议 |
| `scripts/verify_weight_assets.py` | 权重/配置/缓存实测核验脚本 |
| `scripts/emit_inventories.py` | 由实测数据生成两份 CSV 的脚本 |
| `SHA256SUMS.txt` | 本目录全部文件哈希 |
| `git_commit_candidate_list.txt` | 本次 Git 提交候选路径清单（实际按此暂存） |

## 关键新事实（相对 v10.5 已知信息的增量）

1. 整数标签链闭合且训练/评价同源；**上游类别语义命名在服务器无原始证据**，结论只能以 class_id 表述。
2. 训练/测试目录存在 300 个同名 ID，但**内容全部不同**；全部 4470 图像与标注跨 split **0 内容重复**；split 内存在 4 对同图不同标注的上游重复样本（含 test 000301/000305）。
3. **240k 与 10k 保存点标签均为实际更新次数+1**：240k 实际更新 239,999、10k 实际 9,999；两模型同入口同偏移，未来运行须沿用同一脚本。
4. 10k 全部运行**无逐图预测缓存**；240k 仅 1337 一对具有完整身份链与缓存；2026/3407 的 240k 运行不存在。
5. Base 240k 墙钟 20.89 h 受 GPU 争用污染；A2 后段干净区段约 0.066 s/iter（约 4.4–4.6 h/运行的单侧校准）；不给未校准总预算。

## 发布内容审查（PUBLIC 仓库）

- 不含 GT/原图/标注 JSON 本体；不含 checkpoint；不含 tfevents/大型日志；不含凭据或个人信息。
- `class_data_integrity_checks.json` 仅含计数、文件名与内容哈希，不构成原始数据再分发。
- 文档中的绝对服务器路径与既有公开审计一致；科学结论均标注事实/推断/缺失等级，无未经核验的新实验数字。
- 本包不表示 GT 再分发许可已建立，也不构成数据发布包。
