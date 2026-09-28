# CJIG v4 claim 审计 (cjig_claim_audit_v4)

> 对象：`docs/paper/paper_draft_cjig_v4_leather_recovered.md`
> 日期：2026-09-28
> 证据基线：NEU 240k 冻结终点（`long240k_endpoint_results.json`）；机制评价（`experiments/audit/mechanism_*`）；Leather 审计（`leather_paper_final_results.json` 等）。

## 一、强表述扫描

| 关键词 | 命中 | 处理 |
|---|---|---|
| 显著提升/大幅提高/全面优于/普遍提升 | 0 | — |
| 泛化能力（作为 claim） | 0（仅引言"泛化能力不足"用于传统方法背景） | 保留，非本文 claim |
| 所有类别均提升 | 0（明确写"6 类正向、刺猴回退"） | — |
| 跨域/鲁棒 | 0 | — |
| 稳定优于 | 0 | — |

## 二、数字一致性核验

| Claim | 文中值 | 证据值 | 判定 |
|---|---|---|---|
| NEU A2 mIoU | 91.45% | 0.914544 | ✓ |
| NEU Base mIoU | 91.24% | 0.912408 | ✓ |
| NEU Δ | +0.21 pt | +0.002136 | ✓ |
| NEU FPS | 37.37 / 38.16 | 同 | ✓ |
| 参数量增加 | 0.31% | +0.307% | ✓ |
| patches BIoU@1/2/3px 提升 | 0.55/0.54/0.55 pt（macro），0.74/0.85/0.86（聚集） | 机制评价表 | ✓ |
| crazing 高复杂度组 ΔIoU | −1.74 pt | 机制评价 | ✓ |
| 尺度分层表 3 数值 | 与机制评价一致 | ✓ | ✓ |
| Leather Base mIoU | 89.91% | 0.899120（test468 复评） | ✓ |
| Leather A2 mIoU | 90.97% | 0.909713（test468 复评） | ✓ |
| Leather Δ | +1.06 pt | +0.010593 | ✓ |
| Leather 逐类 | 表 5 全列 | manifest per-class（含 wart −3.75 pt 回退） | ✓ |
| Leather 268/211/45/12 | 图 4 案例统计 | final_viz_manifest.json | ✓ |
| Leather FPS | 64.68 / 62.83 | leather_efficiency_benchmark_historical_ckpt.json | ✓ |
| 10k 消融 4 组 mean±SD | 85.22±0.41 / 85.70±0.43 / 85.44±0.19 / 85.75±0.54 | 多 seed 审计 | ✓ |

## 三、协议表述核验

| 表述 | 核验 |
|---|---|
| "NEU 固定终点、不用 val/test 选择" | ✓ 与训练纪律一致 |
| "Leather 验证集选择 checkpoint、468 独立测试集最终评价一次" | ✓ 历史脚本仅加载 val.txt；本轮对 val 选出的 ckpt 在 test468 各评一次 |
| "两种协议均不使用测试集进行模型选择" | ✓ |
| "batch size 约 12"、"A2 实际约 130k" | ✓ 以"历史训练配置记录表明"限定，未把 STRONG INFERENCE 写成绝对事实 |
| 文件名 80000/160000 vs 实际 78k/130k | ✓ 正文已显式说明 |

## 四、claim 分级

- **Claim A（事实结果）**：NEU 91.45/91.24；Leather 90.97/89.91。✓ 可写。
- **Claim B（相对比较）**：NEU +0.21 pt、Leather +1.06 pt。✓ 可写"提高"，禁止"显著"。文中已遵守。
- **Claim C（机制）**：patches 区域+边界改善；crazing 区域—边界权衡；inclusion 稳定；Leather 刺猴回退。✓ 均有审计证据。

## 五、结论

v4 正文无超出证据的强 claim；Leather 新旧数字已完成切换且无残留（见全文 grep 记录：87.21/84.99/2.22/12.46 命中 0）。claim 审计 **PASS**。
