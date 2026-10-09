# Baseline 对比的论文表述建议（供正文/脚注参考，未写入正文）

## 1. 实现与协议表述

建议表述（可按期刊风格润色）：

> 本文对比的实时语义分割方法（PIDNet-S、STDC-Seg、DDRNet23-slim）均基于开源实现由本文统一重新训练与评价。所有方法采用相同的数据划分（NEU-Seg 训练集3630张、测试集840张）、输入尺寸（200×200）、训练预算（240000次迭代、固定终点评价）与评价指标（4类mIoU，含背景）。训练过程中不使用验证集或测试集进行模型选择，仅保存最终迭代 checkpoint 进行一次性测试集评价。参数量与计算量（MACs）为相同输入尺寸下的统一实测值；推理速度在相同硬件（NVIDIA A100）与相同测量协议（batch=1、FP32、预热100次、测量1000次）下实测。

要点：

1. baseline 均为**本文统一重训**（from scratch、同协议），不引用各方法原论文/原仓库在 NEU-Seg 上的任何历史数字；
2. **240k 固定终点**评价，与 DSMONet-B / A2MS-DSMONet-B 的协议完全一致；
3. **无 test 模型选择**（训练全程不加载测试集，无 best checkpoint 挑选）；
4. Params / MACs / FPS 全部**统一实测**，不引用官方报告值。

## 2. 需要向读者交代的口径说明

- DDRNet23-slim 的参数量为本文实现的实测值（20.30M），与原论文报告值存在差异，建议表格使用实测值并以脚注说明"参数量为本文复现实现的实测值"。
- FPS 为同机相对比较值；本机为共享环境，最终采用**交错式 10×100 段测量**（多模型逐段轮转、取段均值中位数）抑制干扰，数据见 `experiments/baseline_comparison/results/fps_idle/interleaved_fps.json`。若论文效率列引用 FPS，建议表注说明测量协议（batch=1、FP32、A100、交错段测量）。
- mIoU 包含背景类，与主结果口径一致（建议在表注中明确"含背景类的4类mIoU"）。

## 3. 不建议写入正文的内容

- 不写 DDR/STDC/PIDNet 在其原论文数据集（Cityscapes 等）上的精度，避免跨协议比较；
- 不使用"本文方法全面优于所有实时方法"式表述——当前证据为：DSMONet-B / A2MS-DSMONet-B 在 NEU-Seg 上 mIoU 高于三种统一复现的实时分割方法，代价是更大的参数量与更低的推理速度；
- 效率权衡建议如实表述：本文方法以约 29M 参数、37–38 FPS 换取 91%+ mIoU，满足实时性需求（>30 FPS）的同时精度更高。

## 4. 结果有效性自证材料（备审稿问询）

- 训练配置：`experiments/baseline_comparison/configs/neu_*_240k.yaml`
- 训练日志与终点 checkpoint：`repro_runs/baseline_neu_240k/<model>/`
- 统一评价输出：`repro_runs/baseline_neu_240k/<model>/results/eval.log`（含 test_samples=840、strict 加载记录）
- 逐模型审计：`experiments/baseline_comparison/results/<model>_audit.md`
- checkpoint sha256 见各 audit 文件
