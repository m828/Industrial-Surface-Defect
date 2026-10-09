# FDSNet NEU-Seg 240k 基线审计

状态：结果已产出，**pending 用户确认，未进论文**。

## 1. Checkpoint

| 项 | 值 |
|---|---|
| 路径 | `repro_runs/baseline_neu_240k_industrial/fdsnet/checkpoints/neu_fdsnet_240k_iter240000.pkl` |
| sha256 | `942c8bb1ee7987a23df685eb975c3dc178116a2b05aef4b12d426ec45cbf3db2` |
| 实际迭代 | 240000（固定终点，无 best 选择） |

## 2. 训练配置（`configs/neu_fdsnet_240k.yaml` + `scripts/train_neu_fdsnet_240k.py`）

- 模型：FDSNet（ICASSP 2022 官方代码 `third_party/FDSNet` @ced4d0d，`num_classes=4, aux=True`）
- 上游修复（不影响语义）：① 官方仓库 `core/__init__` 断链（缺 `core.nn.jpu`）→ importlib 直接加载模型文件；② `gcblock.py` 依赖 mmcv 的两个初始化函数 → 注入等价初始化 shim（mmcv 未安装，与 SeaFormer 排除原因相同）
- 数据：NEU-Seg train3630，200×200，4 类含背景
- from scratch；seed 1337；Adam lr=1e-4 恒定；wd=2e-6；batch 16
- 损失（FDSNet 链习惯）：OHEM CE（主）+ BCE（边缘辅助）+ BCE（图像级语义辅助），权重 [1.0, 0.5, 0.5]；辅助目标在线生成（官方 AuxiliaryGT 生成脚本未发布）：边缘=多类标注形态学边界（3×3 膨胀≠腐蚀），语义=类别 1–3 存在向量
- 训练全程无 val/test；仅保存 240k 终点 checkpoint
- 训练墙钟：约 3.31 h（共享 A100，0.035 s/iter）

## 3. 评价结果（统一840口径）

- 工具：`tools/evaluate_class_iou.py`（strict 加载，注册别名 `fds`）
- test_samples=840；含 background；TestRescale+ToTensor；无 ImageNet Normalize
- **mIoU = 0.870843（87.08%）**
- 类别 IoU：background 0.979506 / crazing 0.786900 / inclusion 0.914578 / patches 0.802390
- pixel accuracy = 0.981737
- 参考：官方报告的 78.8 mIoU 对应其自有 trainval/test 划分与训练 recipe，与本文 3630/840 + 240k 统一协议口径不同，不直接可比

## 4. 参数量 / 计算量 / 速度（`results/bench.json`）

| 指标 | 值 |
|---|---|
| Params | 2,591,609（2.59M，含 aux 头） |
| MACs | 0.333G @200×200 |
| FPS | 205.57（交错式 10×100 复测，见 `results/fps_idle/interleaved_fps.json`） |
| 峰值显存 | 57 MB |

注：FPS 已与其余 6 模型统一为交错式 10×100 段测量（同批快照），段内稳定（±2% 内）。

## 5. 与论文表格字段对应

| 论文字段 | 本审计来源 |
|---|---|
| Method | FDSNet |
| Category | 工业表面缺陷专用 |
| mIoU | 87.08% |
| crazing/inclusion/patches IoU | 78.69 / 91.46 / 80.24 |
| Params | 2.59M（实测） |
| FPS | 205.57 |

## 6. 备注

- FDSNet 为本轮"DMC-Net 不可复现"后的工业缺陷替代 baseline（见 `industrial_baseline_audit.md`）。
- DMC-Net 官方仓库缺失模型文件的事实链不变，不虚构其结果。
