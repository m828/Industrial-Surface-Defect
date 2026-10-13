# Leather Baseline Extension 结果指针（CJIG v8 候选素材）

> 日期：2026-10-13。完整证据链在仓库外：`repro_runs/leather_baseline_extension/`（README、配置、脚本、日志、逐类 CSV/JSON、效率、审计）。
> 本文件仅作指针与头条数字登记；不修改论文正文，不改变任何冻结数字。

## 头条数字（Leather test468，8 类含背景，60k 固定终点，seed 1337）

| 模型 | 类别 | mIoU | Params/M | FPS(768²,FP32,b1) |
|---|---|---:|---:|---:|
| U-Net | Classical | 82.42 | 31.04 | 28.26 |
| DeepLabv3+ | Classical | 87.60 | 59.34 | 52.63 |
| FDSNet | Industrial defect | 78.19 | 2.60 | 176.52 |

同协议（60k 固定终点）参照，存档值：DSMONet-B 87.21 / A2MS-DSMONet-B 84.99。

## 口径警示（整合论文前必读）

- 论文 Leather 主表数字（89.91/90.97/88.24/86.20/84.21）来自**历史 val235 监控 + val-best** checkpoint 的 test468 复评；
- 本批三模型为**固定终点、无验证监控**协议；
- 两者并列时必须加表注说明选择协议差异；同口径直接比较应使用 60k 存档值。

## 证据文件

- 审计报告：`repro_runs/leather_baseline_extension/results/leather_baseline_extension_audit.md`（库内副本：`experiments/baseline_comparison/leather_extension/leather_baseline_extension_audit.md`）
- 汇总 JSON：`…/results/leather_baseline_extension_results.json`（库内副本：`experiments/baseline_comparison/leather_extension/leather_baseline_extension_results.json`）
- 效率：`…/results/leather_extension_efficiency_768.json`（数值已并入汇总 JSON 的 FPS_768 / latency_mean_ms_768 / Params_M 字段）
- 评价交叉验证：DeepLabv3+/FDSNet 官方工具 vs 复用驱动 |ΔmIoU| < 5e-7（通过）；U-Net 经该已验证驱动评价。
- checkpoint / 训练日志 / 逐类 CSV 仅存服务器 `repro_runs/leather_baseline_extension/`，按纪律不入库。
