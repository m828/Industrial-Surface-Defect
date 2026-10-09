# LETNet NEU-Seg 240k 基线审计

状态：结果已产出，**pending 用户确认，未进论文**。

## 1. Checkpoint

| 项 | 值 |
|---|---|
| 路径 | `repro_runs/baseline_neu_240k_industrial/letnet/checkpoints/neu_letnet_240k_iter240000.pkl` |
| sha256 | `6ab2806a9cd5f6cf905afc64dc6037a98eb3893ff296aa27860c76d37cc8df56` |
| 实际迭代 | 240000（固定终点，无 best 选择） |

## 2. 训练配置（`configs/neu_letnet_240k.yaml` + `scripts/train_neu_letnet_240k.py`）

- 模型：LETNet（`third_party/LETNet` @faaa065 + 本地适配，见 `results/letnet_adaptation_log.md`：4 个上游 blocker 修复 + NEU 数据集适配）
- 输入：200×200 pad 至 208×208（图像补 0、标注补 255，满足 /16 整除）；评价时输出 crop 回 200×200
- 数据：NEU-Seg train3630，4 类含背景
- from scratch；seed 1337；Adam lr=1e-4 恒定；wd=2e-6；batch 16
- 损失（链习惯）：加权 CE，类别权重 [1.4646, 8.7251, 6.2481, 8.4495]（NeuSegTrainInform 统计自 train_neu.txt），ignore 255
- 训练全程无 val/test；仅保存 240k 终点 checkpoint
- 训练墙钟：约 14.5 h（共享 A100，约 0.19 s/iter，受共享负载影响明显）

## 3. 评价结果（统一840口径）

- 工具：`tools/evaluate_class_iou.py`（strict 加载，注册别名 `let`，200→208 pad / 208→200 crop wrapper）
- test_samples=840；含 background；TestRescale+ToTensor；无 ImageNet Normalize
- **mIoU = 0.863617（86.36%）**
- 类别 IoU：background 0.975905 / crazing 0.747656 / inclusion 0.896234 / patches 0.834674
- pixel accuracy = 0.978613

## 4. 参数量 / 计算量 / 速度

| 指标 | 值 |
|---|---|
| Params | 950,976（0.95M） |
| MACs | 1.207G @200×200（thop 实测；注意内部按 208×208 前向） |
| FPS | 21.82（交错式 10×100 复测；见下） |

**FPS 说明**：LETNet 在本环境（PyTorch 2.7.1）实测仅 ~22 FPS，远低于其论文报告值（1080Ti 上 >100 FPS）。10 个交错段全部 ≈45.8ms，排除共享污染；其 TransBlock 的 extract/reverse patch（unfold/fold 式）操作在当前框架版本下效率低是合理解释。如实记录，引用时建议注明"第三方实现、本环境实测"。

## 5. 与论文表格字段对应

| 论文字段 | 本审计来源 |
|---|---|
| Method | LETNet |
| Category | 轻量实时（非工业专用） |
| mIoU | 86.36% |
| crazing/inclusion/patches IoU | 74.77 / 89.62 / 83.47 |
| Params | 0.95M（实测） |
| FPS | 21.82 |

## 6. 过程事故记录（不影响结果有效性）

orchestrator 首次评价失败：checkpoint 存的是 LETNet 裸模型键（无 `inner.` 前缀），评价 wrapper 期望带前缀。已在 `_PadCropWrapper.load_state_dict` 加键前缀兼容（评价逻辑零改动），重跑通过。
