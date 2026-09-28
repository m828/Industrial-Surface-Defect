# 皮革真实工业数据集实验报告（CJIG v3 收口）

- 日期：2026-09-24/25
- 目的：为论文表 4 / 图 4 提供**可溯源**的皮革数据集结果（替换 v1 旧链未复核数字）
- 运行目录：`repro_runs/leather_60k_seed1337/`（全部产物：code/config/logs/checkpoints/analysis/viz）

## 1. 数据与协议

- 数据集：`new (copy)/dataset/pige/`（2341 张 768×768，8 类含背景：background + open_wound / scratch / brand_mark / hole / skin_disease / rotten_surface / wart）
- 划分：train.txt 1638 / val.txt 235 / test.txt 468（**注意：论文 v1/v2 草稿中"验证 468 / 测试 235"系对调错误，v3 已更正**；测试集使用锁定清单 `experiments/protocol/leather_valid_test_list.txt`，468 张）
- 训练协议（两模型完全一致）：60k 迭代、batch 8、Adam（lr 1e-4、wd 2e-6）、cosine_annealing（T_max 48000、eta_min 1e-6）、768×768、seed 1337、从头训练；训练中**不构建 test/val loader**，固定 60k 终点一次评价
- 评价协议（与 2026-06 皮革锁定协议及 NEU 纪律一致）：`tools/evaluate_class_iou.py --dataset leather`，batch=1，PIL resize + ToTensor，无 ImageNet 归一化，`pred=outputs[0]`，无 TTA/后处理/阈值调整，mIoU 含背景
- 效率测量：768×768、FP32、batch 1、预热 50 + 5×50 交错分段测量（消除共享 GPU 干扰）；thop 报告 MACs
- 环境：A100-PCIE-40GB、PyTorch 2.7.1+cu126、Python 3.11.13；训练期 GPU 有无关任务共享（仅影响 wall time）

## 2. 结果（固定 60k 终点，统一 468 张）

| 模型 | mIoU | FPS | 延迟均值 ms | Params | MACs | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| Base-B（`new/model_dsmo_rs50.py`） | **0.872137** | 35.84 | 27.90 | 29.3709 M | 86.83 G | 547 MB |
| A2MS-DSMONet-B（`..._fixed.py`） | 0.849932 | **37.11** | 26.94 | 29.4611 M | 86.83 G | 547 MB |

**Δ(A2 − Base) = −2.22 个百分点（mIoU）**；参数量 +0.31%；MACs 持平；FPS 差异 +3.5%（约 1 ms，处于共享环境测量噪声量级，判为基本持平）。

逐类 ΔIoU（pt）：open_wound −2.36、scratch −1.06、brand_mark −1.12、hole +0.07、skin_disease −0.07、**rotten_surface −12.46**、wart −0.72、background −0.06。

## 3. 如实结论

1. 在皮革真实工业数据集上，A2MS-DSMONet-B 保持可用精度（84.99% mIoU）与实时性（37.11 FPS），但**整体精度低于 Base-B 2.22 个百分点**，差异主要来自烂面类别的区域退缩（−12.46 pt）。
2. NEU-Seg 上观察到的类别差异化增益（patches 边界改善等）未直接迁移到皮革缺陷形态。
3. 论文表述边界：皮革实验支持"验证模型在不同工业材质和缺陷形态下的应用潜力"；**不支持**任何"跨域泛化能力/泛化提升/跨材质优于"表述。

## 4. 可追溯性

- checkpoint SHA256：Base-B `3467f3de…`、A2 `b90f9785…`（均为 iter060000 固定终点，非 test 选择）
- 模型文件 SHA256：`fab316c5…` / `a0e6fd52…`（与 NEU 240k 实验所用文件逐字节一致）
- 评价 CSV（含逐类 IoU/Dice/Precision/Recall 与混淆矩阵）：`base_b/analysis/unified468_iter60000.csv`、`a2_full/analysis/unified468_iter60000.csv`
- 训练 summary：两模型均 `resumed_from: null`、`test_during_training: none`；wall time 12.7h / 11.4h
- 机器可读汇总：`experiments/audit/leather_results.json`、`leather_efficiency_benchmark.json`（均通过 json.tool）

## 5. 图 4 素材（固定选例规则）

选例规则（预先固定，不人工择优）：在 468 张测试图中取**标注含缺陷的 268 张**，按逐图 8 类 mIoU 差值（A2−Base）排序，取改善最大 2 例（116、134）、退化最大 2 例（105、165）、差值近零 2 例（19、102）。
素材：`repro_runs/leather_60k_seed1337/viz/leather_{improved,worsened,near_zero}_*.png`（6 张，原图/GT/Base 预测/A2 预测四联），清单 `viz/leather_viz_manifest.json`。
定性观察：改善病例中 A2 对大面积缺陷覆盖更完整；退化病例中 A2 在区域性缺陷边界出现退缩——与逐类定量结果一致。

## 6. 插曲记录

无训练中断/恢复事件。烟测阶段曾产生 `*_smoke` 临时目录，已删除；正式 run 无覆盖任何历史文件。
