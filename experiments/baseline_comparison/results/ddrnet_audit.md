# DDRNet23-slim (内部实现) NEU-Seg 240k 基线审计

状态：结果已产出，**pending 用户确认，未进论文**。

## 1. Checkpoint

| 项 | 值 |
|---|---|
| 路径 | `repro_runs/baseline_neu_240k/ddrnet/checkpoints/neu_ddrnet_240k_iter240000.pkl` |
| sha256 | `c2ce05e8b83f1303ff1122c1d1c78b249ac902d69906dd4c39df2d146cf4a8d5` |
| 实际迭代 | 240000（固定终点，无 best 选择） |

## 2. 训练配置（`configs/neu_ddrnet_240k.yaml`）

- 模型：DDRNet（内部实现 `new/model_ddr.py`，DualResNet_imagenet 主干）
- 数据：NEU-Seg train3630（`new (copy)/dataset/train_neu.txt`），200×200，4 类含背景
- from scratch；seed 1337；Adam lr=1e-4 恒定（无 schedule）；wd=2e-6；batch 16
- 损失：OHEM CE + BCE（权重 1.0/0.4，沿用项目训练链习惯）
- 训练全程无 val/test；仅保存 240k 终点 checkpoint
- 训练墙钟：18519.7 s（约 5.14 h，共享 A100）
- 训练入口：`experiments/baseline_comparison/scripts/train_neu_ddrnet_240k.py`

## 3. 评价结果（统一840口径）

- 工具：`tools/evaluate_class_iou.py`（strict 加载）
- test_samples=840；含 background；TestRescale+ToTensor；无 ImageNet Normalize
- **mIoU = 0.892410（89.24%）**
- 类别 IoU：background 0.983286 / crazing 0.820255 / inclusion 0.934433 / patches 0.831668
- pixel accuracy = 0.985184
- 评价日志：`repro_runs/baseline_neu_240k/ddrnet/results/eval.log`

## 4. 参数量 / 计算量 / 速度（`results/bench.json`）

| 指标 | 值 |
|---|---|
| Params | 20,295,496（20.30M） |
| MACs | 2.905G @200×200 |
| FPS | 138.77（mean 7.21ms，P95 8.96ms） |
| 峰值显存 | 343 MB |

测量协议：A100-PCIE-40GB、batch1、FP32、eval+no_grad、warmup100/measure1000；**共享 GPU**（与 PIDNet 训练并行），相对值有效，绝对值受共享环境影响。

**注意**：实测参数量 20.30M 与历史表格中"DDRNet23-slim 6.71M"不一致（此前审计已发现）。内部实现的"DDRNet23-slim"实际为 DualResNet_imagenet 变体。论文引用前必须统一以本次实测为准并核对命名。

## 5. 与论文表格字段对应

| 论文字段 | 本审计来源 |
|---|---|
| Method | DDRNet23-slim（命名待用户确认，见上） |
| mIoU | 89.24% |
| crazing/inclusion/patches IoU | 82.03 / 93.44 / 83.17 |
| Params | 20.30M（实测） |
| FPS | 138.77（共享 A100 相对测量） |

## 6. 过程事故记录（不影响结果有效性）

1. orchestrator 首次评价失败：`run_all.sh` 误传 `--data-root "new (copy)"`（正确应为 `new (copy)/dataset`，即 `evaluate_class_iou.py` 的 NEU 默认根）。已用手动命令修正重跑；首次失败日志保留于 `results/eval_fail_dataroot_bug.log`。
2. `bench_baseline.py` 初版先 thop 后 FPS，thop 0.1.1 在共享叶子模块模型上残留失效 hook。已修复为 FPS 先行 + 清理 hook，DDRNet 本次 bench 为重测后的干净值。
