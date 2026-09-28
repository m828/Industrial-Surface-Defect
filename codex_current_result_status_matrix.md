# 当前结果状态矩阵

> 生成日期：2026-06-15  
> 原则：只记录当前文件和服务器观测到的状态，不把 val/best_iou 当作 test 结果，不补造缺失数值。  
> 说明：Leather/Pige 的协议样本数存在 467 vs 468 差异；矩阵中 Leather 统一测试结果来自当前 `pige/test.txt` 实际 468 行评估摘要。

| 结果项 | 数据集 | 模型 | 当前数值 | 来源 | 是否可追溯 | 是否可写论文 | 处理建议 |
|---|---|---|---|---|---|---|---|
| Base-S NEU | NEU-Seg | Base-S | mIoU=0.8837；历史 88.4 基本一致 | `codex_experiment_verification_report.md`、`paper_result_safety_decision.md`；日志 `new/runs_other/train_all/base_s_neu.log` | 是，权重和日志存在 | 可作为相对可信结果；建议统一评估脚本再复核 | 保留为 A 类候选；后续用统一脚本重评并记录命令 |
| Base-B NEU | NEU-Seg | Base-B | mIoU=0.9047；历史 90.2 基本一致 | `codex_experiment_verification_report.md`、`paper_result_safety_decision.md`；日志 `new/runs_other/train_all/base_b_neu.log` | 是，权重和日志存在 | 可作为相对可信主结果 | 保留为 A 类候选；后续统一脚本重评 |
| A2MS-S NEU | NEU-Seg | A2MS-DefectNet-S | mIoU=0.8858；低于历史 89.7 约 1.12pp | `codex_experiment_verification_report.md`、`paper_result_safety_decision.md`；日志 `new/runs_other/train_all/a2ms_s_neu.log` | 是，权重和日志存在 | 谨慎使用；不宜直接写历史 89.7 | 统一评估脚本复核；必要时重训或说明波动 |
| A2MS-B NEU | NEU-Seg | A2MS-DefectNet-B | mIoU=0.7131；bg=0.9505, crazing=0.5391, inclusion=0.8302, patches=0.5324 | `a2ms_b_neu_weight_sweep.csv`、`a2ms_b_neu_weight_sweep_summary.md` | 当前异常权重可追溯；历史 91.3 权重不可追溯 | 否 | 第一优先级：找回历史 91.3 权重或重训；不可写入论文主结果 |
| Base-S Leather | Leather/Pige | Base-S | 统一 test mIoU=0.8586；训练日志 val/best_iou=0.8397 | `leather_unified_eval_all_summary.md`、`codex_experiment_verification_report.md` | 统一 test 结果可追溯到当前评估摘要；需确认 468/467 协议 | 待确认；不能写历史 88.1 | 确认 test 样本数后保留实际 test 值；不支撑原历史结论 |
| Base-B Leather | Leather/Pige | Base-B | 0.8991(0127)；0.8806(0126)；新训 val/best_iou=0.8278 | `leather_unified_eval_all_summary.md`、`leather_unified_eval_test_summary.md`、`leather_unified_eval_manifest.md` | 是，但权重版本需固定 | 可写实际统一 test 值，但需人工确认协议和权重选择；不能混用 0126/0127 | 优先确认采用 0127 还是 0126；记录权重路径和评估命令 |
| A2MS-S Leather | Leather/Pige | A2MS-DefectNet-S | 统一 test mIoU=0.8690；训练日志 val/best_iou=0.8472 | `leather_unified_eval_all_summary.md`、`codex_experiment_verification_report.md` | 部分可追溯 | 待确认；不能写历史 89.7 | 统一协议确认后可作为实际结果；若论文需体现优势需进一步重训/分析 |
| A2MS-B Leather | Leather/Pige | A2MS-DefectNet-B | 统一 test mIoU=0.8607；checkpoint best_iou=0.9091(val) | `leather_unified_eval_test_summary.md`、`leather_unified_eval_all_summary.md`、`leather_unified_eval_manifest.md` | test 结果可追溯；历史 91.0 不可按 test 复现 | 否，不能作为“优于 Base-B”的主结论；可写入困难分析/泛化分析需人工确认 | 不把 0.9091 写成 test；排查权重、val/test 分布、类别顺序、预处理 |
| STDC1 Leather | Leather/Pige | STDC1-Seg | 统一 test mIoU=0.8824 | `leather_unified_eval_test_summary.md`、`leather_unified_eval_all_summary.md` | 部分可追溯；权重文件名含 `stdc2` 但脚本按 STDC1 加载 | 待确认 | 人工确认 `sdtdcnet_pige_stdc2_pige.pkl` 是否确为 STDC1/STDCNet1446 |
| DDRNet Leather | Leather/Pige | DDRNet23slim | 训练日志/val 候选约 0.8317；统一 test 未成功 | `leather_unified_eval_manifest.md`、`class_iou_summary.md` 错误记录 | 训练日志可追溯；统一 test 不可用 | 否 | 修复 `evaluate_class_iou.py` 中 DDRNet loader，用 `model_ddr.py` 正确加载后重评 |
| PIDNet Leather | Leather/Pige | PIDNet-S | 统一 test mIoU=0.8620；训练日志候选约 0.8406 | `leather_unified_eval_all_summary.md`、`codex_experiment_verification_report.md` | 部分可追溯 | 待确认 | 统一协议确认后可作为实际结果；补充完整命令和权重路径 |
| Params/FLOPs/FPS | NEU/Leather | Base/A2MS/DDR/STDC/PID 等 | Params/FLOPs 较稳定；A100 FPS 05-26 主模型：Base-S 127.3/119.0, Base-B 87.8/74.6, A2MS-S 130.8/122.4, A2MS-B 87.5/92.7 | `complexity_results.csv`、`complexity_results_a100_final_summary.md`、`fps_platform_decision.md` | Params/FLOPs 可追溯；FPS 需平台决策和复测稳定性确认 | Params/FLOPs 可写；FPS 仅在统一 A100 并标注条件后可写 | 第一优先级：统一 A100 平台，重跑全模型 FPS，避免混用 RTX 3090 历史 FPS |
| 类别级 IoU | NEU/Leather | 多模型 | 当前文件混合 Leather/NEU、test/val；`class_iou_summary.md` 标题与结果不一致 | `class_iou_results.csv`、`class_iou_summary.md`、`codex_experiment_verification_report.md` | 否，当前口径混杂 | 否 | 修复评估脚本后，按统一协议全量重跑并重建 CSV/summary |
| 小目标/细长缺陷评价 | Leather/Pige | Base-B/A2MS-B/STDC1 | 小目标：Base-B 0.5141, A2MS-B 0.4084, STDC1 0.4549；无缺陷组约 0.125 | `small_object_group_results.csv`、`small_object_group_summary.md` | 当前脚本和结果可追溯，但存在已知评估 bug | 否 | 修复无缺陷组逻辑，保存样本 ID，扩展到所有模型后重跑 |
| 可视化结果 | Leather/Pige | Base-B/A2MS-B/STDC1 | 已有 small/elongated/multi/texture/base_miss/boundary 样本 | `figures/qualitative_results/README.md`、`sample_index.json` | 部分可追溯；缺预测路径 | 可作为内部参考；不宜作为最终图 | 重新生成，补失败案例、Base 优于 A2MS 案例、预测路径和样本选择规则 |

## 汇总判断

### 当前相对可用

- NEU Base-S、Base-B：与历史基本一致，可作为可信候选。
- A2MS-S NEU：可追溯但略低于历史，建议谨慎写入或统一评估复核。
- Leather Base-B(0127)：当前统一 test mIoU=0.8991，接近历史 89.2，但需先解决 467/468 协议差异并固定权重版本。
- Params/FLOPs/模型大小：比 FPS 更稳定，可优先保留。

### 当前不可用

- A2MS-B NEU 历史 91.3。
- A2MS-B Leather 历史 91.0 作为 test 结果。
- val/best_iou 直接写入论文 test 表。
- 类别级 IoU、小目标/细长缺陷评价、当前可视化作为主结论。
- 未核验 baseline 的历史主表结果。

### 必须重跑或重评

- A2MS-B NEU：重训或找回历史权重。
- Leather：Base-B、A2MS-B、Base-S、A2MS-S、STDC、DDRNet、PIDNet 统一 test 评估；其中 DDRNet 必须先修 loader。
- A100 FPS：全模型统一测量并标注平台。
- 类别级 IoU、小目标/细长缺陷评价、可视化。
