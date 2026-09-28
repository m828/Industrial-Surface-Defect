# CJIG v2 图表清单（paper_draft_cjig_v2.md)

全部表格按三线表排版；图题、表题中英文对照。数字来源均为 `experiments/audit/` 冻结结果。

## 表格

| 表号 | 标题 | 内容 | 数据来源 | 状态 |
|---|---|---|---|---|
| 表 1 | NEU-Seg 数据集长训练终点结果 | Base-B / A2MS-DSMONet-B：mIoU、四类 IoU、FPS、Params | `long240k_endpoint_results.json` + `efficiency_benchmark_results.json` | ✅ 完整 |
| 表 2 | 缺陷类别级性能差异（配对比较） | 三类缺陷：ΔIoU（聚集）、ΔBIoU@2px、ΔBF1@2px | `boundary_quality_results.json` / `paired_case_analysis.json` | ✅ 完整 |
| 表 3 | 缺陷尺度分层 IoU 差异 | 三类 × 小/中/大（P33/P67 分层）ΔIoU | `scale_stratified_results.json` | ✅ 完整 |
| 表 4 | 不同数据集实验结果 | NEU-Seg 两行（真实）+ Leather 两行（占位） | NEU 行为冻结值；Leather 行 **[待填入皮革实验结果]，复核前不得填数** | ⚠️ 含占位 |
| 表 5 | 模块组合消融（10k，3 种子 mean±SD） | Base / +AAM / +细节监督 / 联合：85.22±0.41 / 85.70±0.43 / 85.44±0.19 / 85.75±0.54 | `a2ms_2x2_factorial_analysis.md` | ✅ 完整 |
| 表 6 | 长训练终点验证（240k） | Base-B 91.24 / A2MS 91.45 | `long240k_endpoint_results.json` | ✅ 完整（含"未做 240k 全消融"脚注） |
| 表 7 | 复杂度与实时性对比 | Params / MACs / 延迟均值 / P95 / FPS / 峰值显存 | `efficiency_benchmark_results.json`（v2 交错测量） | ✅ 完整（含共享 GPU 脚注） |

## 图件

| 图号 | 标题 | 内容 | 素材 | 状态 |
|---|---|---|---|---|
| 图 1 | A2MS-DSMONet 网络结构示意图 | 主干 + DAPPM + SqueezeBodyEdge + UAFM×2 + AAM(eSE/ACW) + 训练态细节监督分支 | 需按 `new/model_dsmo_rs50_eSE_adapt_detailloss_fixed.py` 重绘 | ⛔ 待重绘（不得含 ELMM；300 dpi/矢量） |
| 图 2 | 主实验数据集样例 | NEU-Seg 三类缺陷 + 皮革数据集样例 | 数据集原图 | ⛔ 待排版（300 dpi） |
| 图 3 | NEU-Seg 可视化分割结果 | 裂纹/斑块 × {改善 top3、退化 bottom3、近零 3}：原图/GT/双预测/双误差图 | `repro_runs/mechanism_eval_240k/visualizations/`（18 张已生成，选例规则见 `case_selection.json`） | ✅ 素材已备，需按版式合成导出 300 dpi |

## 可选补充（不进正文亦可）

- 训练收敛轨迹（10k/30k/60k/120k/180k/240k 六点统一840评价）：`long240k_descriptive_trajectory.md`。若放入，须标注"固定迭代 checkpoint 的描述性评价，未用于模型选择"。
- 逐病例配对指标（2520 行）：`paired_case_metrics.csv`，适合作为附录或开放数据说明。

## 投稿前图表演示检查单

1. 表 4 皮革行在复核完成前保持占位，不得在摘要/正文任何位置引用皮革数字；
2. 表 1/表 7 的 FPS 保留共享 GPU 测量脚注；
3. 表 5 注明 10k 固定预算与 3 种子口径，表 6 注明单 seed 与固定终点；
4. 图 1 重绘后与 fixed 模型 forward 图逐模块核对（AAM=eSE+ACW、细节监督分支仅训练态）；
5. 图 3 图题注明选例规则（按 ΔBF1@2px 排序取改善/退化/近零各 3 例）。
