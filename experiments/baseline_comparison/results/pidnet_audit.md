# PIDNet-S NEU-Seg 240k 基线审计

状态：结果已产出，**pending 用户确认，未进论文**。

## 1. Checkpoint

| 项 | 值 |
|---|---|
| 路径 | `repro_runs/baseline_neu_240k/pidnet/checkpoints/neu_pidnet_240k_iter240000.pkl` |
| sha256 | `b40e3ab97cbc039b58d00721deff8e144211e227535be3217fc632c970418c53` |
| 实际迭代 | 240000（固定终点，无 best 选择） |

## 2. 训练配置（`configs/neu_pidnet_240k.yaml`）

- 模型：PIDNet-S（内部实现 `new/pid.py`；旧版 `new/train_neu_pid.py` 有死循环 bug 已禁用，本训练用 `new (copy)/` 修复链路的协议化版本）
- 数据：NEU-Seg train3630，200×200，4 类含背景
- from scratch；seed 1337；Adam lr=1e-4 恒定；wd=2e-6；batch 16
- 训练全程无 val/test；仅保存 240k 终点 checkpoint
- 训练墙钟：17257.4 s（约 4.79 h，共享 A100）
- 训练入口：`experiments/baseline_comparison/scripts/train_neu_pidnet_240k.py`

## 3. 评价结果（统一840口径）

- 工具：`tools/evaluate_class_iou.py`（strict 加载）
- test_samples=840；含 background；TestRescale+ToTensor；无 ImageNet Normalize
- **mIoU = 0.888158（88.82%）**
- 类别 IoU：background 0.982813 / crazing 0.814390 / inclusion 0.928415 / patches 0.827015
- pixel accuracy = 0.984618
- 评价日志：`repro_runs/baseline_neu_240k/pidnet/results/eval.log`

## 4. 参数量 / 计算量 / 速度（`results/bench.json`）

| 指标 | 值 |
|---|---|
| Params | 7,717,065（7.72M，与官方 PIDNet-S ≈7.6M 量级一致） |
| MACs | 0.969G @200×200 |
| FPS | 108.87（mean 9.19ms，P95 11.57ms） |
| 峰值显存 | 131 MB |

测量协议：A100-PCIE-40GB、batch1、FP32、eval+no_grad、warmup100/measure1000。PIDNet 的 bench 在三个模型训练全部结束后测量，GPU 无并行训练负载。

## 5. 与论文表格字段对应

| 论文字段 | 本审计来源 |
|---|---|
| Method | PIDNet-S |
| mIoU | 88.82% |
| crazing/inclusion/patches IoU | 81.44 / 92.84 / 82.70 |
| Params | 7.72M（实测） |
| FPS | 108.87 |

## 6. 过程事故记录（不影响结果有效性）

orchestrator 首次评价失败：`run_all.sh` 误传 `--data-root "new (copy)"`（正确为 `new (copy)/dataset`）。已修正手动重跑；失败日志保留于 `results/eval_fail_dataroot_bug.log`。bench 使用修复后脚本（FPS 先行 + thop hook 清理），一次通过。
