# U-Net NEU-Seg 240k 基线审计

状态：结果已产出，**pending 用户确认，未进论文**。

## 1. Checkpoint

| 项 | 值 |
|---|---|
| 路径 | `repro_runs/baseline_neu_240k_classical/unet/checkpoints/neu_unet_240k_iter240000.pkl` |
| sha256 | `27d5eb6542e0f1bc712eb452ef24d466bcc98c49e11b9e78fc3394b99cfaac9e` |
| 实际迭代 | 240000（固定终点，无 best 选择） |

## 2. 训练配置（`configs/neu_unet_240k.yaml` + `scripts/train_neu_unet_240k.py`）

- 模型：U-Net（`new/model_unet.py`，本仓库历史实现），canonical 全尺寸：feature_scale=1（filters 64–1024）、is_deconv=True、BatchNorm——经典 Ronneberger 2015 范式，非裁剪版
- 输入：200×200 pad 至 208×208（图像补 0、标注补 255）——200 不满足 /16 整除，且 unetUp 的负偏移 F.pad 在 200 处会 shape mismatch（24 vs 25）；评价时输出 crop 回 200×200（`_PadCropWrapper(208)`，与 LETNet 同例）
- 数据：NEU-Seg train3630，4 类含背景
- seed 1337；Adam lr=1e-4 恒定；wd=2e-6；batch 16；loss = plain CE（ignore 255 pad 区）
- 训练全程无 val/test；仅保存 240k 终点 checkpoint
- 训练墙钟：约 14.5 h（共享 A100，约 0.13–0.21 s/iter，与另外两个 baseline 并发）

## 3. 评价结果（统一840口径，2026-10-10）

- 工具：`tools/evaluate_class_iou.py`（strict 加载，注册别名 `unet`）
- test_samples=840；含 background；TestRescale+ToTensor；无 ImageNet Normalize
- **mIoU = 0.906176（90.62%）**
- 类别 IoU：background 0.985270 / crazing 0.830625 / inclusion 0.931481 / patches 0.877328
- pixel accuracy = 0.986822

## 4. 参数量 / 计算量 / 速度

| 指标 | 值 |
|---|---|
| Params | 31,039,876（31.04M） |
| MACs | 36.10G @200×200 输入（thop 实测，经 208 pad wrapper，内部按 208×208 前向） |
| FPS | 待统一复测（`fps_rebenchmark_plan.md`；不在训练负载下测量） |

## 5. 与论文表格字段对应

| 论文字段 | 本审计来源 |
|---|---|
| Method | U-Net |
| Category | Classical |
| Backbone | 五级编解码（VGG 式） |
| mIoU | 90.62% |
| crazing/inclusion/patches IoU | 83.06 / 93.15 / 87.73 |
| Params | 31.04M（实测） |
| MACs | 36.10G（实测） |
| FPS | 待统一复测 |

## 6. 备注

- 输入 padding 表注与 LETNet/BiSeNetV2 统一口径："U-Net 按 /16 整除约束 padding 至 208×208，其余评价协议完全一致"。
- 90.62% 为 strong classical 参照（与 DeepLabv3+ 90.95% 同属经典重型档），如实记录；两者均以明显高于本文模型的 MACs（36.10G / 14.29G vs 6.65G）取得仍低于 DSMONet-B 的精度。
