# CJIG v4 修订日志 (cjig_revision_log_v4)

> 文件：`docs/paper/paper_draft_cjig_v4_leather_recovered.md`（由 v3 定点更新，v3 保留不覆盖）
> 日期：2026-09-28
> 驱动事件：Leather 历史 91% forensic audit 完成，状态 HISTORICAL RESULT RECOVERED（`experiments/audit/leather_*`）。

## 一、结果替换（核心变更）

| 位置 | v3 | v4 | 依据 |
|---|---|---|---|
| 中文摘要结果段 | Leather 无数值，仅"考察适用性" | 补 90.97% vs 89.91%（+1.06 pt），注明"验证集选择 + 独立468测试集" | leather_paper_final_results.json |
| 英文摘要 Result | 同上 | 同步加入 validation-selected / independent 468-image test 表述 | 同上 |
| 2.4 节 | 60k 固定终点协议，Base 87.21 / A2 84.99（负结果） | 全部重写：历史 val-best 链统一 test468 复评，Base 89.91 / A2 90.97 | 审计 E2/E5 |
| 表 4 | NEU+Leather 合并表（含旧 87.21/84.99） | 拆分为 Leather 独立表：Model/选择方式/测试集/mIoU/Params | 协议不同不可并表 |
| 表 5（新增） | 无 | Leather 逐类 IoU（Base/A2/Δ，8 行含 mIoU 行） | manifest per-class |
| 图 4 | 基于 60k 复现权重 | 基于正式历史 checkpoint（Base 0127@78k / A2 160k@130k）重新推理，固定选例规则，含一致性门控 | final_viz_manifest.json |
| 讨论 3.1 Leather 段 | "A2 整体低于 Base 2.22 pt" | 重写：+1.06 pt 正向总体结果 + 刺猴类回退 + 类型依赖主线 | 审计 |
| 3.2 局限性 | 含"皮革整体低于基础网络" | 重写 6 条：单 seed、协议差异、形态依赖、共享 GPU FPS、两种材质、主流对比待核验 | — |
| 结论 | 含皮革负结果表述 | 重写：NEU +0.21 / Leather +1.06，形态依赖性收尾 | — |

## 二、协议表述变更

- 2.1 评价协议明确区分：NEU-Seg = 固定 240k 终点 + test840 终评一次；Leather = 历史链 val235 选 checkpoint + test468 独立终评一次。两者均不用 test 做模型选择。
- 2.1 训练设置补 Leather 历史链：Adam、lr 1e-4 恒定、batch≈12、val235 每 1000 iter 监控、val-best 保存；并注明 Base/A2 checkpoint 实际保存点为 78k/约130k，文件名 80000/160000 为计划目标值。
- 1.5 损失函数补注：皮革历史链 λ_d=1（NEU 为 3）。

## 三、图号与表号

- 新增图 2 占位（数据集示例），修复 v3 图号从图 1 直跳图 3 的空缺；图序：图1 网络结构 / 图2 数据集示例 / 图3 NEU 可视化 / 图4 皮革可视化。
- 表号因新增 Leather 逐类表顺延：表1 NEU 主结果 / 表2 类别差异 / 表3 尺度分层 / 表4 Leather 主结果 / 表5 Leather 逐类 / 表6 消融 / 表7 长训终点 / 表8 效率。正文引用已同步。

## 四、删除内容（退入审计档案）

- 87.21 / 84.99 / −2.22 pt / rotten_surface −12.46 pt / "烂面退缩" 等 60k 固定终点协议的负结果表述。
- 这些内容仍保留在 `experiments/audit/leather_results.json`、`leather_experiment_report.md`、`leather_historical_vs_current_training_diff.md`，供可重复性与审稿回复使用。

## 五、效率数据

- 表 8（NEU 200²）沿用机制评价实测（38.16/37.37 FPS，共享环境交错测量）。
- 皮革 768² 效率改用正式历史 checkpoint 重测（独占 A100）：Base 64.68 FPS / A2 62.83 FPS，params 29.37M/29.46M，MACs 持平，峰值显存 547MB 相同（`experiments/audit/leather_efficiency_benchmark_historical_ckpt.json`）。

## 六、未改动部分

- 2.2/2.3 NEU 主结果与机制分析数值（仅 2.2 引导句按约束微调，删除"四个类别均不低于"的强表述）。
- 图 1 网络结构图、参考文献 12 条、作者/单位/基金占位（仍待补）。
- 主流方法对比仍标注"待协议核验后补充"——投稿前必须完成或删除该声明（未决事项）。

## 七、遗留事项

1. 主流方法对比表：未完成协议核验，正文保留待补声明（投稿前必须了结）。
2. 英文摘要未达约 1000 词要求（当前约 300 词）。
3. 作者/单位/基金/中图分类号占位。
4. Zuo H、Li Q 完整作者列表。
5. 图 2 数据集示例图尚未绘制（仅占位与图件要求）。
