# CJIG v4 图表清单 (cjig_table_list_v4)

## 图

| 图号 | 内容 | 状态 |
|---|---|---|
| 图 1 | A2MS-DSMONet 网络结构（Fig1_A2MS_DSMONet.svg/pdf/png） | 已定稿（无 ELMM、detail 仅训练态、head_seg1=out） |
| 图 2 | 数据集示例（NEU-Seg + 皮革） | **占位待绘制** |
| 图 3 | NEU-Seg 可视化（固定规则选例：crazing/patches 各改善/退化/近零 3 例） | 素材在 `repro_runs/mechanism_eval_240k/` |
| 图 4 | 皮革可视化（正式历史 checkpoint，固定规则：268 张含缺陷图按 ΔmIoU 排序取改善/退化/近零各 2 例） | **v4 已重新生成**：`repro_runs/leather_historical_audit/final_viz/`（6 张 + final_viz_manifest.json，一致性门控通过） |

## 表

| 表号 | 内容 | 关键数字 |
|---|---|---|
| 表 1 | NEU-Seg 固定终点主结果 | Base 91.24 / A2 91.45，FPS 38.16/37.37 |
| 表 2 | NEU 类别级差异（聚集 IoU/BIoU@2px/BF1@2px） | crazing +0.32/−0.45/−0.60 等 |
| 表 3 | NEU 尺度分层 ΔIoU | patches 小 +0.73 等 |
| 表 4 | **Leather 主结果（val-best 选择，test468）** | Base 89.91 / A2 **90.97** |
| 表 5 | **Leather 逐类 IoU（新增）** | 6 类正向、wart −3.75 pt |
| 表 6 | 10k 3-seed 消融 | 85.22/85.70/85.44/85.75 |
| 表 7 | 240k 长训终点 | 91.24/91.45 |
| 表 8 | 效率（200²）+ 皮革 768² FPS 句 | 29.37M/29.46M，64.68/62.83 FPS(768²) |

编号连续性：图 1–4、表 1–8，正文引用已同步检查。
