# NEU-Seg 240k Baseline 结果汇总（pending 用户确认，未进论文）

生成时间：2026-09-30
协议：NEU-Seg train3630 / test840，200×200，4 类含 background；from scratch；seed 1337；Adam lr=1e-4 恒定；wd=2e-6；batch 16；240k 固定终点；训练全程无 val/test，无 test-based 模型选择。
评价：`tools/evaluate_class_iou.py`（strict 加载，840 全量，TestRescale+ToTensor，无 ImageNet Normalize）。
效率：A100-PCIE-40GB、batch1、FP32、warmup100/measure1000（共享环境相对测量）。

## 主结果

| 模型 | mIoU | crazing | inclusion | patches | background | Params | MACs | FPS |
|---|---|---|---|---|---|---|---|---|
| DDRNet23-slim* | 89.24% | 82.03 | 93.44 | 83.17 | 98.33 | 20.30M | 2.90G | 138.77 |
| STDC-Seg | 88.16% | 80.59 | 92.54 | 81.33 | 98.17 | 16.07M | 6.02G | 91.52 |
| PIDNet-S | 88.82% | 81.44 | 92.84 | 82.70 | 98.28 | 7.72M | 0.97G | 108.87 |
| DSMONet-B（冻结） | **91.24%** | — | — | — | — | 29.37M | 6.65G | 38.16 |
| A2MS-DSMONet-B（冻结） | **91.45%** | — | — | — | — | 29.46M | 6.65G | 37.37 |

\* 内部"DDRNet23-slim"实测 20.30M，与历史表 6.71M 冲突，引用前以实测为准并核对命名。
冻结行数字未改动；baseline 三行为本轮新测，pending 确认。

## 文件索引

- results JSON：`results/{ddrnet,stdc,pidnet}_results.json`
- 逐模型审计：`results/{ddrnet,stdc,pidnet}_audit.md`
- checkpoint：`repro_runs/baseline_neu_240k/{ddrnet,stdc,pidnet}/checkpoints/*iter240000.pkl`
- 训练日志：`repro_runs/baseline_neu_240k/<model>/logs/train_console.log`
- 评价日志：`repro_runs/baseline_neu_240k/<model>/results/eval.log`
- 效率数据：`repro_runs/baseline_neu_240k/<model>/results/bench.json`

## 过程事故与修复（均不影响最终数字）

1. `run_all.sh` 评价阶段 `--data-root` 少传一层 `/dataset`，三模型首次评价均 FileNotFoundError。已用正确根（`new (copy)/dataset`，即评价工具 NEU 默认）手动重跑；失败日志留档为 `results/eval_fail_dataroot_bug.log`。
2. `bench_baseline.py` 初版 thop profile 在 FPS 测量之前，thop 0.1.1 对共享叶子模块模型残留失效 hook（STDC bench 崩溃）。已修复为 FPS 先行 + profile 后清理 hook/buffer；三个 bench 均为修复后干净值。
