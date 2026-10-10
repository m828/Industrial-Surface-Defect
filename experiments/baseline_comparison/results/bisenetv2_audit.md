# BiSeNetV2 NEU-Seg 240k 基线审计

状态：结果已产出，**pending 用户确认，未进论文**。

## 1. Checkpoint

| 项 | 值 |
|---|---|
| 路径 | `repro_runs/baseline_neu_240k_classical/bisenetv2/checkpoints/neu_bisenetv2_240k_iter240000.pkl` |
| sha256 | `e0f66982b9ae3812b55bb5355da170492327ceb0d38864abbb39e1128636958d` |
| 实际迭代 | 240000（固定终点，无 best 选择） |

## 2. 训练配置（`configs/neu_bisenetv2_240k.yaml` + `scripts/train_neu_bisenetv2_240k.py`）

- 模型：BiSeNetV2（`third_party/BiSeNetV2` CoinCheung/BiSeNet 官方 @6b4b67a，`lib/models/bisenetv2.py`，aux_mode=train）
- **from scratch 保障**：上游 `__init__` 默认联网下载 backbone 预训练权重（`backbone_v2.pth`），已通过 `load_pretrain` 置空禁用
- 输入：200×200 pad 至 224×224（图像补 0、标注补 255）——BGALayer 结构约束要求 /32 分支 ×4 与 /8 detail 分支尺寸一致，200 不满足（24 vs 28），224 全整除；评价时输出 crop 回 200×200（`_PadCropWrapper(224)`，与 LETNet 208 同例）
- 数据：NEU-Seg train3630，4 类含背景
- seed 1337；Adam lr=1e-4 恒定；wd=2e-6；batch 16
- 损失（链习惯，对应上游 ohem_ce_loss 全部头）：OHEM-CE × 5 头各权重 1.0（thresh 0.7、min_kept 1e5、ignore 255 pad 区）
- 训练全程无 val/test；仅保存 240k 终点 checkpoint
- 训练墙钟：约 13.5 h（共享 A100，约 0.15 s/iter，与另外两个 baseline 并发）

## 3. 评价结果（统一840口径，2026-10-10）

- 工具：`tools/evaluate_class_iou.py`（strict 加载，注册别名 `bisenetv2`，aux_mode=train 保留辅助头以 strict 加载，消费主头输出）
- test_samples=840；含 background；TestRescale+ToTensor；无 ImageNet Normalize
- **mIoU = 0.887461（88.75%）**
- 类别 IoU：background 0.982675 / crazing 0.807185 / inclusion 0.930033 / patches 0.829952
- pixel accuracy = 0.984469

## 4. 参数量 / 计算量 / 速度

| 指标 | 值 |
|---|---|
| Params | 5,195,652（5.20M） |
| MACs | 3.42G @200×200 输入（thop 实测，经 224 pad wrapper、含 5 头——与 FDSNet aux=True 的测量口径一致；推理仅消费主头） |
| FPS | 待统一复测（`fps_rebenchmark_plan.md`；不在训练负载下测量） |

## 5. 与论文表格字段对应

| 论文字段 | 本审计来源 |
|---|---|
| Method | BiSeNetV2 |
| Category | Real-time |
| Backbone | 双分支（Detail+Semantic） |
| mIoU | 88.75% |
| crazing/inclusion/patches IoU | 80.72 / 93.00 / 83.00 |
| Params | 5.20M（实测） |
| MACs | 3.42G（实测，含辅助头） |
| FPS | 待统一复测 |

## 6. 备注

- 输入 padding 表注与 LETNet 统一口径："BiSeNetV2 按原模型结构约束 padding 至 224×224，其余评价协议完全一致"。
- 88.75% 落在实时档合理区间（STDC 88.16 / PIDNet 88.82 / DDRNet 89.24），与文献相对水平一致，无需解释性修饰。
