# CJIG v2 Claim 审查表（paper_draft_cjig_v2.md)

审查日期：2026-09-24。证据来源：`experiments/audit/`（long240k 冻结终点、10k 2×2 三种子消融、机制评价、效率 benchmark v2）。

## A. 禁用表述扫描（已对全文执行 grep）

`显著提升 / 大幅提高 / 全面优于 / 所有类别 / 普遍提升 / 有效解决 / 明显改善边界 / 小目标优势 / 边界增强 / 全面提升 / 普遍增强` → **v2 全文中 0 命中**。

保留的合规措辞：差异化增强、类型相关改善、精度—实时平衡、工程应用潜力、应用潜力。

## B. 逐条数字核验

| # | v2 位置 | 表述 | 审计依据 | 状态 |
|---|---|---|---|---|
| C1 | 摘要/2.2/表1 | A2MS-DSMONet-B 91.45% mIoU | long240k endpoint：0.914544（统一840固定终点） | ✅ supported |
| C2 | 摘要/2.2/表1 | Base-B 91.24% | 0.912408 | ✅ supported |
| C3 | 摘要/2.2 | +0.21 个百分点 | Δ=+0.002136 | ✅ supported（未用"显著"） |
| C4 | 表1 | 四类 IoU 均不低于 Base-B（98.64/84.47/93.55/89.16 vs 98.62/84.15/93.49/88.71） | 冻结 per-class IoU | ✅ supported |
| C5 | 摘要/2.6/表7 | 37.37 FPS vs 38.16 FPS；延迟 26.76/26.21 ms；P95 45.43/42.42 ms | efficiency_benchmark v2（交错测量） | ✅ supported（共享 GPU 已加注） |
| C6 | 摘要/2.6 | 参数量 +0.31%（29.46M vs 29.37M） | thop/torch 统计：+90,175 | ✅ supported |
| C7 | 2.6 | MACs 6.648G 持平 | thop 推理图；细节/辅助头仅训练态 | ✅ supported（工具报告口径已注明） |
| C8 | 摘要/2.3 | patches 边界 IoU 提高约 0.5~0.9 pt（1/2/3px） | macro +0.55/+0.54/+0.55；aggregate +0.74/+0.85/+0.86；配对 bootstrap CI 均排除 0 | ✅ supported（病例级口径） |
| C9 | 2.3/表2 | crazing 聚集 IoU +0.32 pt；BF1@2px −0.60 pt；BIoU@2px −0.45 pt | paired_case_analysis | ✅ supported（双向如实报告） |
| C10 | 2.3/表2 | inclusion 变化较小（+0.06/+0.32/+0.04） | 同上，CI 多含 0 | ✅ supported |
| C11 | 2.3/表3 | 尺度分层：裂纹 −1.24/−0.50/−0.44；夹杂 +0.28/−0.24/−0.05；斑块 +0.73/−0.04/+0.49 | scale_stratified_results.json | ✅ supported |
| C12 | 2.3 | 高复杂度裂纹组 ΔIoU −1.74 pt | morphology_exploratory_results.json（exploratory，已在正文标注探索性） | ✅ supported（exploratory 限定语保留） |
| C13 | 2.5/表5 | 10k 消融四格：85.22±0.41 / 85.70±0.43 / 85.44±0.19 / 85.75±0.54 | 2×2 三种子 matched 实验 | ✅ supported |
| C14 | 2.5 正文 | "AAM 3 种子中 2 个正向（平均 +0.48pt）；细节监督 3/3 正向（平均 +0.23pt）；联合后方向存在种子间差异" | ΔAAM=+0.004777(2/3)；ΔD_noA=+0.002258(3/3)；ΔD_withA=+0.000499(1/3)；交互 −0.001759 方向不一致 | ✅ supported（未写"均稳定提升"） |
| C15 | 2.5/表6 | 240k 终点 91.24 / 91.45 | long240k_endpoint_results.json | ✅ supported（AAM-only/Detail-only 未跑 240k 已在注中声明） |
| C16 | 摘要/2.4 | 皮革数据集：定性句 + 全占位符 | 无审计数据 | ✅ 合规（未引用任何旧数字；占位符 [待填入皮革实验结果]） |
| C17 | 3.2 | 局限性 5 条（单 seed、单材质、共享 GPU、裂纹折中、对比待核验） | 与实验实际一致 | ✅ supported |
| C18 | 2.7 | 可视化预注册选例规则 | repro_runs/mechanism_eval_240k/visualizations/（18 panels + case_selection.json） | ✅ supported（素材存在） |

## C. 残留风险与投稿前必办

| # | 事项 | 风险 | 处置要求 |
|---|---|---|---|
| S1 | 皮革数据集实验 | 当前全部占位；若补入必须按 NEU-Seg 同协议（固定终点、统一测试、同效率测量）复核 | 复核完成前不得填入任何数字 |
| S2 | 主流方法对比表 | v1 的 16 行对比数字未核验来源与协议 | 仅保留可核验来源的行，并标注"文献值，协议/硬件不同，仅供量级参考"；否则删除该表 |
| S3 | 参考文献 | 全部"作者待核验" | 投稿前逐条核验作者/题名/来源/DOI，未核验不得列入 |
| S4 | 英文摘要 | 当前为骨架版（约 350 词） | 按体例扩展至约 1000 词并人工校订 |
| S5 | 图件 | 图 1 网络结构图需按 fixed 模型重绘（不得含 ELMM）；图 3 可视化用预注册选例素材导出 300 dpi | 图 1 结构必须与 `model_dsmo_rs50_eSE_adapt_detailloss_fixed.py` 一致 |
| S6 | 单 seed 表述 | 240k 终点为单 seed 比较 | 全文任何位置不得升级为"稳定优于"；3.2 已声明 |
| S7 |  bootstrap CI | 病例级 CI ≠ 种子级稳定性 | 若正文引用 CI，须保留该限定（当前正文仅 2.3 节使用一次，已限定"病例级配对自助法"） |
| S8 | FPS 绝对值 | 共享 GPU 下测得，空闲环境更高 | 2.6 已加脚注；若审稿人质疑，按交错测量方法说明回应 |

## D. claim—证据映射（主要 claim 终态）

| Claim | Evidence | Status |
|---|---|---|
| 精度—实时平衡（91.45% / 37.37 FPS / +0.31% 参数） | 冻结 240k 终点 + 效率 v2 | supported（单 seed 限定） |
| 固定终点评价下较 Base-B 精度更高（+0.21 pt） | long240k endpoint | supported（禁"显著"） |
| 斑块类边界质量改善约 0.5~0.9 pt | 3 容差 × 2 口径 CI 一致 | supported（病例级） |
| 裂纹类区域完整性改善、边界存在折中 | 聚集 +0.32 pt / BF1 CI 负向 / 复杂度逆趋势 | supported |
| 两模块联合取得最佳终点性能 | 10k 四格均值 Full 最高 + 240k Full>Base | supported（禁"均稳定提升"） |
| 跨材质应用潜力（皮革） | 无（待复核） | placeholder——仅定性句，无数字 |
