# STDC-Seg (BiSeNet + STDCNet1446) NEU-Seg 240k 基线审计

状态：结果已产出，**pending 用户确认，未进论文**。

## 1. Checkpoint

| 项 | 值 |
|---|---|
| 路径 | `repro_runs/baseline_neu_240k/stdc/checkpoints/neu_stdc_bisenet_240k_iter240000.pkl` |
| sha256 | `faa0665f29d8e33e17fea39aeaceb7e7080be03dcb3abd94ff6030963b2670b4` |
| 实际迭代 | 240000（固定终点，无 best 选择） |

## 2. 训练配置（`configs/neu_stdc_bisenet_240k.yaml`）

- 模型：STDC-Seg（内部实现 `new (copy)/stdc.py`，BiSeNet 结构 + STDCNet1446 主干）
  - 注：Leather 历史 88.24 权重的训练脚本已丢失；本 NEU 重训使用 `BiSeNet('STDCNet1446', 4)`，与统一评价注册别名 `stdc` 一致
- 数据：NEU-Seg train3630，200×200，4 类含背景
- from scratch；seed 1337；Adam lr=1e-4 恒定；wd=2e-6；batch 16
- 损失：主 CE + aux16 CE + aux32 CE + boundary BCE（各 1.0，沿用项目训练链习惯）
- 训练全程无 val/test；仅保存 240k 终点 checkpoint
- 训练墙钟：16215.9 s（约 4.50 h，共享 A100）
- 训练入口：`experiments/baseline_comparison/scripts/train_neu_stdc_bisenet_240k.py`

## 3. 评价结果（统一840口径）

- 工具：`tools/evaluate_class_iou.py`（strict 加载）
- test_samples=840；含 background；TestRescale+ToTensor；无 ImageNet Normalize
- **mIoU = 0.881555（88.16%）**
- 类别 IoU：background 0.981672 / crazing 0.805886 / inclusion 0.925397 / patches 0.813264
- pixel accuracy = 0.983614
- 评价日志：`repro_runs/baseline_neu_240k/stdc/results/eval.log`

## 4. 参数量 / 计算量 / 速度（`results/bench.json`）

| 指标 | 值 |
|---|---|
| Params | 16,073,632（16.07M） |
| MACs | 6.017G @200×200 |
| FPS | 91.52（mean 10.93ms，P95 16.13ms，std 3.11ms） |
| 峰值显存 | 242 MB |

测量协议：A100-PCIE-40GB、batch1、FP32、eval+no_grad、warmup100/measure1000；**共享 GPU**（与 PIDNet 训练并行，std 偏大与此相关），相对值有效。

## 5. 与论文表格字段对应

| 论文字段 | 本审计来源 |
|---|---|
| Method | STDC-Seg |
| mIoU | 88.16% |
| crazing/inclusion/patches IoU | 80.59 / 92.54 / 81.33 |
| Params | 16.07M（实测） |
| FPS | 91.52（共享 A100 相对测量） |

## 6. 过程事故记录（不影响结果有效性）

1. orchestrator 首次评价失败：同 DDRNet（`--data-root` 传错），已修正重跑；失败日志保留于 `results/eval_fail_dataroot_bug.log`。
2. 首次 bench 失败：thop 0.1.1 在 STDC（含共享叶子模块）上 profile 后残留失效 hook，warmup 阶段触发 `AttributeError: 'Conv2d' object has no attribute 'total_ops'`。`bench_baseline.py` 已修复为 FPS 先行 + profile 后清理 hook/buffer，本次 bench 为修复后干净重测。
