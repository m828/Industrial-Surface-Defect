# 实验运行日志

> 每次实验运行后，在本文件追加一条记录。记录包括命令、环境、时间、结果路径和状态。

## 日志格式

每次实验记录包含以下字段：

| 字段 | 说明 |
|---|---|
| 日期 | YYYY-MM-DD HH:MM |
| 实验类型 | complexity / class_iou / small_object / qualitative / train / test / reproduce |
| 模型 | 论文统一命名 |
| 数据集 | NEU-Seg / Leather / Yachi |
| 运行命令 | 完整命令行 |
| 工作目录 | 执行时的 cwd |
| 设备 | GPU 型号、CUDA 版本 |
| Python 环境 | python 路径、PyTorch 版本 |
| 开始时间 | YYYY-MM-DD HH:MM:SS |
| 结束时间 | YYYY-MM-DD HH:MM:SS |
| 是否成功 | 是/否 |
| 日志路径 | stdout/stderr 保存路径 |
| 权重路径 | 生成或使用的权重路径 |
| 结果文件 | 输出的 CSV/MD 文件路径 |
| 备注 | 补充说明 |

---

## 实验记录

### 2026-05-18 — 复杂度统计（首次运行）

| 字段 | 值 |
|---|---|
| 日期 | 2026-05-18 |
| 实验类型 | complexity |
| 模型 | A2MS-DefectNet-S, A2MS-DefectNet-B, Base-S, Base-B |
| 数据集 | N/A（仅模型结构统计） |
| 运行命令 | `cd /workspace/Industrial\ Surface\ Defect/new && /opt/conda/bin/python /workspace/Industrial\ Surface\ Defect/Industrial-Surface-Defect/experiments/scripts/complexity_stats.py` |
| 工作目录 | `/workspace/Industrial Surface Defect/new` |
| 设备 | NVIDIA A100-PCIE-40GB, CUDA 12.6 |
| Python 环境 | `/opt/conda/bin/python`, PyTorch 2.7.1+cu126 |
| 开始时间 | 2026-05-18 |
| 结束时间 | 2026-05-18 |
| 是否成功 | 是 |
| 日志路径 | 终端输出（未保存到文件） |
| 权重路径 | N/A（随机初始化） |
| 结果文件 | `experiments/results/complexity_results.csv`, `experiments/results/new_results_pending.md` |
| 备注 | 使用随机初始化权重，FPS 为推理速度参考值。模型映射基于 `experiment_entry_mapping.md` 推断，需人工确认。 |

### 2026-05-18 — 复杂度统计（全模型扩展）

| 字段 | 值 |
|---|---|
| 日期 | 2026-05-18 |
| 实验类型 | complexity |
| 模型 | Base-S, Base-B, A2MS-DefectNet-S, A2MS-DefectNet-B, DDRNet23slim, STDC1-Seg, STDC2-Seg, PP-LiteSeg-T, PP-LiteSeg-B, BiSeNetV1-L, BiSeNetV2-L, Sub-region UNet, PIDNet-S |
| 数据集 | N/A（仅模型结构统计） |
| 运行命令 | `cd /workspace/Industrial\ Surface\ Defect/new && /opt/conda/bin/python /workspace/Industrial\ Surface\ Defect/Industrial-Surface-Defect/tools/measure_complexity.py` |
| 工作目录 | `/workspace/Industrial Surface Defect/new` |
| 设备 | NVIDIA A100-PCIE-40GB, CUDA 12.6 |
| Python 环境 | `/opt/conda/bin/python`, PyTorch 2.7.1+cu126 |
| 开始时间 | 2026-05-18 |
| 结束时间 | 2026-05-18 |
| 是否成功 | 部分成功（10/13 模型完成） |
| 日志路径 | 终端输出（未保存到文件） |
| 权重路径 | N/A（随机初始化） |
| 结果文件 | `tools/measure_complexity.py`, `experiments/results/complexity_results.csv`, `experiments/results/complexity_summary.md`, `experiments/results/complexity_errors.json` |
| 备注 | 10 个模型成功测量。3 个失败：PP-LiteSeg-T（backbone_out_chs 硬编码仅支持 ResNet-50）、BiSeNetV2-L（代码文件不存在）、Sub-region UNet（pixelshuffle 输入不兼容 200x200）。使用独立脚本 `tools/measure_complexity.py`，修复了 thop hooks 影响 FPS 测量的问题（FPS 先于 FLOPs 测量）。 |

### 2026-05-18 — 类别级 IoU 评估（Leather 数据集）

| 字段 | 值 |
|---|---|
| 日期 | 2026-05-18 |
| 实验类型 | class_iou |
| 模型 | Base-B, A2MS-DefectNet-B, STDC1-Seg（另有 7 个模型缺少权重，跳过） |
| 数据集 | Leather（皮革缺陷测试集，768x768，8 类，468 张） |
| 运行命令 | `cd "/workspace/Industrial Surface Defect/new (copy)" && /opt/conda/bin/python "/workspace/Industrial Surface Defect/Industrial-Surface-Defect/tools/evaluate_class_iou.py"` |
| 工作目录 | `/workspace/Industrial Surface Defect/new (copy)` |
| 设备 | NVIDIA A100-PCIE-40GB, CUDA 12.6 |
| Python 环境 | `/opt/conda/bin/python`, PyTorch 2.7.1+cu126 |
| 开始时间 | 2026-05-18 |
| 结束时间 | 2026-05-18 |
| 是否成功 | 是（3/10 模型完成评估，7 个缺少权重跳过） |
| 日志路径 | 终端输出（未保存到文件） |
| 权重路径 | Base-B: `model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0126.pkl`; A2MS-DefectNet-B: `dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl`; STDC1-Seg: `model_savePath/sdtdcnet_pige_stdc2_pige.pkl` |
| 结果文件 | `tools/evaluate_class_iou.py`, `experiments/results/class_iou_results.csv`, `experiments/results/class_iou_summary.md`, `experiments/results/class_iou_errors.json` |
| 备注 | 3 个模型成功评估。Base-B 权重 0126 匹配 model_dsmo_rs50.py（SELayer，无 detailloss）。STDC 权重文件名含"stdc2"但实际为 STDCNet1446 架构，已更正为 STDC1-Seg。7 个模型缺少 Leather 权重。 |

### 2026-05-18 — 小目标与细长缺陷分组评价（Leather 数据集）

| 字段 | 值 |
|---|---|
| 日期 | 2026-05-18 |
| 实验类型 | small_object |
| 模型 | Base-B, A2MS-DefectNet-B, STDC1-Seg |
| 数据集 | Leather（皮革缺陷测试集，768x768，8 类，468 张） |
| 运行命令 | `cd "/workspace/Industrial Surface Defect/new (copy)" && /opt/conda/bin/python "/workspace/Industrial Surface Defect/Industrial-Surface-Defect/tools/evaluate_small_object_groups.py" --dataset leather` |
| 工作目录 | `/workspace/Industrial Surface Defect/new (copy)` |
| 设备 | NVIDIA A100-PCIE-40GB, CUDA 12.6 |
| Python 环境 | `/opt/conda/bin/python`, PyTorch 2.7.1+cu126 |
| 开始时间 | 2026-05-18 |
| 结束时间 | 2026-05-18 |
| 是否成功 | 是（3/3 模型完成） |
| 日志路径 | 终端输出（未保存到文件） |
| 权重路径 | 同类别级 IoU 评估 |
| 结果文件 | `tools/evaluate_small_object_groups.py`, `experiments/results/small_object_group_results.csv`, `experiments/results/small_object_group_summary.md` |
| 备注 | 分组规则：小目标=面积占比<P25(1.75%)，中等=P25~P75，大目标>=P75(26.99%)，细长=长宽比>P75(3.51)，多缺陷=连通域>1，无缺陷=缺陷面积=0。使用 connected component analysis 近似实例数。 |

### 2026-05-18 — 定性可视化分析（Leather 数据集）

| 字段 | 值 |
|---|---|
| 日期 | 2026-05-18 |
| 实验类型 | qualitative |
| 模型 | Base-B, A2MS-DefectNet-B, STDC1-Seg |
| 数据集 | Leather（皮革缺陷测试集，768x768，8 类，468 张） |
| 运行命令 | `cd "/workspace/Industrial Surface Defect/new (copy)" && /opt/conda/bin/python "/workspace/Industrial Surface Defect/Industrial-Surface-Defect/tools/make_qualitative_figures.py" --dataset leather` |
| 工作目录 | `/workspace/Industrial Surface Defect/new (copy)` |
| 设备 | NVIDIA A100-PCIE-40GB, CUDA 12.6 |
| Python 环境 | `/opt/conda/bin/python`, PyTorch 2.7.1+cu126 |
| 开始时间 | 2026-05-18 |
| 结束时间 | 2026-05-18 |
| 是否成功 | 是（14 张图生成） |
| 日志路径 | 终端输出（未保存到文件） |
| 权重路径 | 同类别级 IoU 评估 |
| 结果文件 | `tools/make_qualitative_figures.py`, `figures/qualitative_results/*.png`, `figures/qualitative_results/sample_index.json`, `figures/qualitative_results/README.md` |
| 备注 | 6 类典型场景：小目标(3)、细长缺陷(3)、多缺陷(3)、纹理混淆(1)、漏检对比(1)、边界改善(3)。CJK 字体缺失导致中文标题显示为方框，不影响图像内容。 |

### 2026-05-18 — DMC-Net 复现实验准备（NEU-Seg 数据集）

| 字段 | 值 |
|---|---|
| 日期 | 2026-05-18 |
| 实验类型 | reproduce (准备阶段) |
| 模型 | DMC-Net |
| 数据集 | NEU-Seg |
| 运行命令 | `cd "/workspace/Industrial Surface Defect/Industrial-Surface-Defect/third_party/DMC-Net"` |
| 工作目录 | `third_party/DMC-Net/` |
| 设备 | N/A（仅代码检查） |
| Python 环境 | N/A |
| 开始时间 | 2026-05-18 |
| 结束时间 | 2026-05-18 |
| 是否成功 | 否（模型定义文件缺失） |
| 日志路径 | `experiments/dmcnet/dmcnet_setup_notes.md` |
| 权重路径 | N/A |
| 结果文件 | `experiments/dmcnet/dmcnet_setup_notes.md` |
| 备注 | **关键问题**：DMC-Net 模型定义文件 `dmcnet.py` 缺失。仓库 GitHub Issue #2 确认此问题。`net_factory.py` 引用 `from models.total_supvised.dmcnet import *` 但文件不存在。同样缺失的还有 `hdrnet.py` 和 `swin_unet.py`。NEU-Seg 数据集已内置支持（benchmark='neuseg'，4类，224x224），但无法运行。仓库中可用的替代模型：EDRNet、BiSeNet、U-Net、DeepLabV3、SegFormer、PGA-Net、TopFormer。 |

### 2026-05-18 — PIDNet 复杂度统计（PIDNet-S/M/L）

| 字段 | 值 |
|---|---|
| 日期 | 2026-05-18 |
| 实验类型 | complexity |
| 模型 | PIDNet-S, PIDNet-M, PIDNet-L |
| 数据集 | N/A（仅模型结构统计） |
| 运行命令 | `cd "/workspace/Industrial Surface Defect/new (copy)" && /opt/conda/bin/python -c "from pid import PIDNet; ..."` |
| 工作目录 | `/workspace/Industrial Surface Defect/new (copy)` |
| 设备 | NVIDIA A100-PCIE-40GB, CUDA 12.6 |
| Python 环境 | `/opt/conda/bin/python`, PyTorch 2.7.1+cu126 |
| 开始时间 | 2026-05-18 |
| 结束时间 | 2026-05-18 |
| 是否成功 | 是（3/3 模型完成） |
| 日志路径 | 终端输出（未保存到文件） |
| 权重路径 | N/A（随机初始化） |
| 结果文件 | `experiments/results/complexity_results.csv` |
| 备注 | PIDNet-S: 7.62M params, 0.97G FLOPs, 29.31MB, 125.1 FPS@200/126.1@768; PIDNet-M: 28.54M params, 3.63G FLOPs, 109.18MB, 122.6 FPS@200; PIDNet-L: 36.94M params, 5.56G FLOPs, 141.25MB, 107.1 FPS@200。模型定义来自 `new/pid.py`（CVPR 2023 PIDNet）。 |

### 2026-05-18 — PIDNet-S 训练（NEU-Seg 数据集）

| 字段 | 值 |
|---|---|
| 日期 | 2026-05-18 |
| 实验类型 | train |
| 模型 | PIDNet-S |
| 数据集 | NEU-Seg（钢材表面缺陷，200x200，4类，3630 训练 / 840 测试） |
| 运行命令 | `cd "/workspace/Industrial Surface Defect/new (copy)" && /opt/conda/bin/python train_neu_pid.py --cfg pid_neu.yml` |
| 工作目录 | `/workspace/Industrial Surface Defect/new (copy)` |
| 设备 | NVIDIA A100-PCIE-40GB, CUDA 12.6 |
| Python 环境 | `/opt/conda/bin/python`, PyTorch 2.7.1+cu126 |
| 开始时间 | 2026-05-18 22:10:53 |
| 结束时间 | 2026-05-18 ~22:58（约 47 分钟） |
| 是否成功 | 是 |
| 日志路径 | `runs_other/pid_neu/pid/run_2026_05_18_22_10_53.log` |
| 权重路径 | `model_savePaths/pid_neu_0119_neu_pid.pkl` |
| 结果文件 | `experiments/results/pidnet_neu_results.txt` |
| 备注 | 训练 60000 iters, batch_size=16, lr=1e-4, Adam, cosine_annealing (T_max=48000, eta_min=1e-6), OHEM cross-entropy loss, val_interval=500。Best mIoU=0.8667 at iter 58500。配置文件：`pid_neu.yml`。 |

### 2026-05-18 — PIDNet-S 评估（NEU-Seg 数据集）

| 字段 | 值 |
|---|---|
| 日期 | 2026-05-18 |
| 实验类型 | class_iou / test |
| 模型 | PIDNet-S |
| 数据集 | NEU-Seg（钢材表面缺陷，200x200，4类，840 测试） |
| 运行命令 | `cd "/workspace/Industrial Surface Defect/new (copy)" && /opt/conda/bin/python "/workspace/Industrial Surface Defect/Industrial-Surface-Defect/tools/evaluate_pidnet_neu.py" --weight model_savePaths/pid_neu_0119_neu_pid.pkl --test_txt dataset/test_neu.txt --dataset_root dataset/` |
| 工作目录 | `/workspace/Industrial Surface Defect/new (copy)` |
| 设备 | NVIDIA A100-PCIE-40GB, CUDA 12.6 |
| Python 环境 | `/opt/conda/bin/python`, PyTorch 2.7.1+cu126 |
| 开始时间 | 2026-05-18 |
| 结束时间 | 2026-05-18 |
| 是否成功 | 是 |
| 日志路径 | 终端输出（未保存到文件） |
| 权重路径 | `model_savePaths/pid_neu_0119_neu_pid.pkl` |
| 结果文件 | `tools/evaluate_pidnet_neu.py`, `experiments/results/pidnet_neu_results.txt`, `experiments/results/class_iou_results.csv`, `experiments/results/new_results_pending.md` |
| 备注 | mIoU=0.8667, Overall Acc=0.9674, Mean Acc=0.8848, FreqW Acc=0.9396。Per-class IoU: background=0.9786, inclusion=0.7768, patch=0.9111, scratch=0.8004。FPS (200x200)=125.1, Params=7.62M, FLOPs=0.97G, Model Size=29.31MB。 |

### 2026-05-19 — LETNet / SeaFormer 补充复现评估

| 字段 | 值 |
|---|---|
| 日期 | 2026-05-19 |
| 实验类型 | reproduce (评估阶段) |
| 模型 | LETNet, SeaFormer |
| 数据集 | NEU-Seg（评估适配可行性） |
| 运行命令 | `cd third_party/LETNet && python -c "from Network.model.LETNet import LETNet; ..."` |
| 工作目录 | `third_party/LETNet/`, `third_party/SeaFormer/` |
| 设备 | NVIDIA A100-PCIE-40GB, CUDA 12.6 |
| Python 环境 | `/opt/conda/bin/python`, PyTorch 2.7.1+cu126 |
| 开始时间 | 2026-05-19 |
| 结束时间 | 2026-05-19 |
| 是否成功 | 部分成功（LETNet 可运行但 FPS 低；SeaFormer 环境冲突） |
| 日志路径 | `experiments/letnet_or_seaformer/setup_notes.md`, `experiments/letnet_or_seaformer/reproducibility_assessment.md` |
| 权重路径 | N/A |
| 结果文件 | `experiments/letnet_or_seaformer/setup_notes.md`, `experiments/letnet_or_seaformer/reproducibility_assessment.md` |
| 备注 | **LETNet**：0.95M params, 1.21G FLOPs, 5.04MB, 23.5 FPS@208×208 (A100)。代码有 Bug（解码器 LC3 通道维度不匹配），需修改模型定义。输入需填充到 16 的倍数（200→208）。无预训练权重。FPS 过低，不适合作为实时分割对比基线。**SeaFormer**：依赖 mmcv-full 1.3-1.4，不兼容 PyTorch 2.x，迁移成本高（100+ 行改动）。两者均不进入主实验对比表，在相关工作中引用论文结果即可。 |
