# Baseline FPS 统一复测计划（CJIG v7 准备）

生成：2026-10-09
状态：**计划已冻结，暂不执行**——等 GPU 空闲窗口统一复测，复测前最终表 FPS 列保持"待复测"。

## 1. 目的

现有 FPS 数据来自不同时间、不同 GPU 负载背景，口径不一致：

| 数据来源 | 口径问题 | 处置 |
|---|---|---|
| 各模型 `*_results.json` 的 `fps` 字段（训练后顺序测量） | 测量时 GPU 存在其他训练任务，曾出现 A2MS 快于 DSMONet 的污染反序（已作废） | 仅 audit |
| 论文历史值（DSMONet-B 38.16 / A2MS 37.37） | 历史测量环境与协议与 baseline 不同批 | 仅 audit，是否刷新待用户决策 |
| `fps_idle/interleaved_fps.json`（2026-10-09 交错式） | 协议合规，但仍在共享 GPU 上测得 | 作为**参考值**（见 §4），复测确认前不进最终表 |

统一复测后，7 个模型同一批出数，FPS 列才可进入 `final_baseline_table_v2.md` 与论文表。

## 2. 待测模型（7 个）

| 模型 | checkpoint（240k 终点） | 评价注册别名 |
|---|---|---|
| FDSNet | `repro_runs/baseline_neu_240k_industrial/fdsnet/checkpoints/*iter240000.pkl` | `fds` |
| LETNet | `repro_runs/baseline_neu_240k_industrial/letnet/checkpoints/neu_letnet_240k_iter240000.pkl` | `let`（208 pad wrapper） |
| STDC-Seg | `repro_runs/baseline_neu_240k/stdc/checkpoints/*iter240000.pkl` | `stdc` |
| PIDNet-S | `repro_runs/baseline_neu_240k/pidnet/checkpoints/*iter240000.pkl` | `pid` |
| DDRNet23-slim | `repro_runs/baseline_neu_240k/ddrnet/checkpoints/*iter240000.pkl` | `ddr` |
| DSMONet-B | `repro_runs/long240k_seed1337/base_b/checkpoints/base_b_s1337_long240k_codex1_iter240000.pkl` | `base` |
| A2MS-DSMONet-B | `repro_runs/long240k_seed1337/a2_full/checkpoints/a2_full_s1337_long240k_codex1_iter240000.pkl` | `a2`（**必须** `--model-file new/model_dsmo_rs50_eSE_adapt_detailloss_fixed.py`） |

## 3. 统一协议（冻结）

- 硬件：A100-PCIE-40GB，**尽量 GPU 空闲**（无其他训练任务；执行前先 `nvidia-smi` 确认）
- batch = 1，FP32，`model.eval()`，`torch.no_grad()`
- 输入 200×200（LETNet 经 208×208 padding wrapper，与其评价口径一致）
- warmup ≥ 100 次迭代
- 每段测量前后 `torch.cuda.synchronize()`
- **交错式测量**：7 模型逐段轮转，每模型 10 段 × 100 次迭代；取各段均值的**中位数**换算 FPS（对共享环境残留抖动稳健）
- 记录：mean / median / P95 latency、FPS、逐段明细、测量时 GPU 占用快照
- 脚本：`repro_runs/baseline_neu_240k/bench_interleaved.py`（2026-10-09 已用同一脚本产出参考值，复测直接重跑即可）

## 4. 现有参考值（2026-10-09 交错式 10×100，仅 audit，不进最终表）

| 模型 | 段均值中位数 (ms) | FPS（参考） |
|---|---|---|
| FDSNet | 4.86 | 205.57 |
| DDRNet23-slim | 5.95 | 168.11 |
| PIDNet-S | 8.09 | 123.67 |
| STDC-Seg | 8.59 | 116.40 |
| DSMONet-B | 11.06 | 90.43 |
| A2MS-DSMONet-B | 10.96 | 91.26 |
| LETNet | 45.84 | 21.82 |

参考值解读（复测时需复核的点）：

1. DSMONet-B 与 A2MS-DSMONet-B 差异 ≈0.9%，在交错段噪声内（两者 MACs 相同、参数仅差 0.31%）——预期复测结论仍为"推理速度基本相当"；
2. LETNet ~22 FPS 显著低于其官方报告值，10 段全部 ≈45.8ms、段内稳定，初步判断为第三方实现在 PyTorch 2.7.1 下 TransBlock patch（unfold/fold）操作效率低，非共享污染；复测时若 GPU 完全空闲仍 ≈22 FPS，则在论文表注中如实说明"第三方实现、本环境实测"；
3. FDSNet 段 4/段 8 有轻微抬升（6.78/5.11ms vs 其余 ≈4.86ms），复测时空闲环境应消失。

## 5. 产出与回填

复测完成后：

1. 新 JSON 存 `results/fps_idle/interleaved_fps_final.json`（不覆盖现有参考值）；
2. 回填 `final_baseline_table_v2.md` 的 FPS 列；
3. 论文 FPS 列是否整体刷新（涉及 DSMONet-B/A2MS 冻结值 38.16/37.37 的口径说明）由用户拍板，本计划不预设结论。

## 6. 禁止事项

- 不引用任何模型官方论文 FPS 值；
- 不用顺序（非交错）测量值；
- 不在 GPU 有明显其他负载时测量并采信；
- 不修改任何 mIoU / Params / MACs 冻结数字。
