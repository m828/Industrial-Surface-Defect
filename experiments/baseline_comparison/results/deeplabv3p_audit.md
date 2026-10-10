# DeepLabv3+ NEU-Seg 240k 基线审计

状态：结果已产出，**pending 用户确认，未进论文**。

## 1. Checkpoint

| 项 | 值 |
|---|---|
| 路径 | `repro_runs/baseline_neu_240k_classical/deeplabv3p/checkpoints/neu_deeplabv3p_240k_iter240000.pkl` |
| sha256 | `5e9192b095fba07b0aec7092153039341002158abb1de91316c4a1c7f965112b` |
| 实际迭代 | 240000（固定终点，无 best 选择） |

## 2. 训练配置（`configs/neu_deeplabv3p_240k.yaml` + `scripts/train_neu_deeplabv3p_240k.py`）

- 模型：DeepLabv3+（`new (copy)/model/deeplabv3.py` 的 `DeepLab` 类：ASPP + decoder 的真 v3+ 结构，即历史 `train_pige_deeplabv3.py` 链所用模型）
- backbone=ResNet-50，output_stride=16，`pretrained=False`（from scratch，无 ImageNet 权重下载）
- 输入：原生 200×200（forward 末端 interpolate 回原尺寸，无 padding）
- 数据：NEU-Seg train3630，4 类含背景
- seed 1337；Adam lr=1e-4 恒定；wd=2e-6；batch 16；loss = plain CE
- 训练全程无 val/test；仅保存 240k 终点 checkpoint
- 训练墙钟：约 12.7 h（共享 A100，约 0.19 s/iter，与另外两个 baseline 并发）

## 3. 评价结果（统一840口径，2026-10-10）

- 工具：`tools/evaluate_class_iou.py`（strict 加载，注册别名 `deeplabv3p`）
- test_samples=840；含 background；TestRescale+ToTensor；无 ImageNet Normalize
- **mIoU = 0.909480（90.95%）**
- 类别 IoU：background 0.985519 / crazing 0.840130 / inclusion 0.930767 / patches 0.881504
- pixel accuracy = 0.987159

## 4. 参数量 / 计算量 / 速度

| 指标 | 值 |
|---|---|
| Params | 59,339,940（59.34M） |
| MACs | 14.29G @200×200（thop 实测） |
| FPS | 待统一复测（`fps_rebenchmark_plan.md`；不在训练负载下测量） |

## 5. 与论文表格字段对应

| 论文字段 | 本审计来源 |
|---|---|
| Method | DeepLabv3+ |
| Category | Classical |
| Backbone | ResNet-50 (OS=16) |
| mIoU | 90.95% |
| crazing/inclusion/patches IoU | 84.01 / 93.08 / 88.15 |
| Params | 59.34M（实测） |
| MACs | 14.29G（实测） |
| FPS | 待统一复测 |

## 6. 结果解读备注（供论文讨论，不改数字）

DeepLabv3+（59.34M 参数、14.29G MACs）90.95% 是目前最强 classical 参照：高于全部实时/工业 baseline，低于 DSMONet-B（91.24%，29.37M/6.65G）。该结果如实记录——它恰好支持"重型经典模型精度接近但代价翻倍，本文方法以一半参数量和实时速度取得更高精度"的叙事，无需任何修饰。
