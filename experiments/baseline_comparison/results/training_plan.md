# Baseline 训练计划与成本估计 (training_plan)

> 日期：2026-09-29
> 状态：**计划，未执行**。本阶段（v1）只完成审计/适配/smoke，未进行任何大规模训练。
> 协议基线：见 `experiments/eval_protocol_lock.md` 与 `experiments/protocol/`；NEU = 240k 固定终点 / 无 test 选择 / seed 1337 / batch16 / Adam 1e-4 恒定 / from scratch。

## 一、已就绪的协议化训练入口（smoke 验证通过）

| 方法 | 训练脚本 | 正式配置 | smoke |
|---|---|---|---|
| STDC-Seg（BiSeNet+STDCNet1446，16.07M） | `scripts/train_neu_stdc_bisenet_240k.py` | `configs/neu_stdc_bisenet_240k.yaml` | ✅ 1 epoch 通过 |
| PIDNet-S（7.72M） | `scripts/train_neu_pidnet_240k.py` | `configs/neu_pidnet_240k.yaml` | ✅ |
| DDRNet23-slim（内部实现，实测 20.30M） | `scripts/train_neu_ddrnet_240k.py` | `configs/neu_ddrnet_240k.yaml` | ✅ |
| LETNet（0.95M） | third_party/LETNet 原生 train.py（已修复+适配，见 `results/letnet_adaptation_log.md`） | 原生 recipe | ✅（原生 recipe 口径） |

## 二、预计训练成本（实测稳态 s/iter 外推，A100-40GB）

### NEU-Seg（200×200，batch16，240k iters）

| 方法 | 实测 s/iter（smoke） | 预估时长 | 历史锚点复核 |
|---|---:|---:|---|
| DDRNet23-slim | 0.055 | **≈3.6 h** | 旧链 60k≈53 min（含 120 次 test 监控），量级一致 |
| STDC-Seg | 0.068 | **≈4.5 h** | — |
| PIDNet-S | 0.084 | **≈5.6 h** | 旧链 60k≈47 min（含 test 监控），量级一致 |
| LETNet（208×208） | 0.27 | **≈18 h**（若协议化） | — |
| 参考：DSMONet-B | 0.30（240k 实测） | 20h54m（已完成） | repro_runs/long240k_seed1337 |

NEU 三方法合计 ≈ **14 h**（串行）；加 LETNet ≈ 32 h。

### Leather（768×768；历史链口径 batch12，val235 监控）

参考实测锚点（batch8/60k，含 val）：PIDNet-S 2h38m、DDRNet 2h51m；外推 batch12：

| 方法 | 80k 预估 | 160k 预估 |
|---|---:|---:|
| PIDNet-S | ≈4–4.7 h | ≈8–9.5 h |
| DDRNet23-slim | ≈4.5–6 h | ≈9–12 h |
| STDC-Seg | ≈8–9 h | ≈16–18 h（估计值，无本机日志） |

注意：Leather 侧 baseline 已有统一协议结果（STDC 88.24 / PIDNet-S 86.20 / DDRNet 84.21，test468），**是否需要重训取决于论文对 NEU 主表的设计，不是本阶段任务**。

## 三、推荐训练顺序（正式阶段）

1. **DDRNet23-slim NEU 240k**（最快、脚本最成熟）→ 验证整条协议链产出；
2. **STDC-Seg NEU 240k**（新建脚本，需首轮重点观察 loss 曲线是否正常）；
3. **PIDNet-S NEU 240k**；
4. 每完成一个立即用 `tools/evaluate_class_iou.py` 统一评价 test840（含背景），结果写入 `results/`（先进入 pending 状态，核验后再动用）；
5. LETNet 是否协议化训练（恒定 1e-4 / 240k / 去训练期 val）还是保留原生 recipe 并在论文中注明差异，**需用户决策后再排期**；
6. Leather 侧暂不新增训练。

## 四、正式开跑前检查单

- [ ] 正式配置的 `checkpoint_dir` 改为独立正式目录（当前默认指向 `repro_runs/baseline_smoke/checkpoints/`，smoke 占位）；
- [ ] DDRNet 参数量口径确认（实测 20.30M vs 旧表 6.71M，需在论文引用前以统一脚本重测）；
- [ ] STDC 首跑前确认多路 loss 权重设置（主输出 1.0 + 辅助路权重）并记录到配置；
- [ ] 确认 240k 终点 checkpoint 命名与 `checkpoint_metadata/` 登记格式一致；
- [ ] 每方法训练完成后更新 `results/` 与审计链。
