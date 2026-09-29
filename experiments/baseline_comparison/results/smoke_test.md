# Baseline NEU Smoke Test 记录

- 日期： 2026-09-29
- GPU： NVIDIA A100-PCIE-40GB（共享卡，运行期间约有 12.5 GB 被其他任务占用）
- Python： `/opt/conda/bin/python`（torch 2.7.1+cu126）
- 数据： NEU-Seg，`new (copy)/dataset/`；train_neu.txt 3630 条 / test_neu.txt 840 条（每行 "标注 图像"）；200×200；标签值 {0,1,2,3}（4 类含背景）
- 协议： seed=1337、from scratch、batch16、Adam lr=1e-4 恒定（ConstantLR）、wd=2e-6；训练全程不构建 val/test loader、不做模型选择；仅在终点 iter 保存 checkpoint（含 `model_state` 与 `iter` 键）；smoke = 226 iters（1 epoch，drop_last）
- **以下所有指标均为 1-epoch smoke 的接口验证数字，不是论文结果。**

## 总览

| 模型 | 数据读取(226批) | forward | loss 有限且下降 | checkpoint | eval 接口 | 状态 |
|---|---|---|---|---|---|---|
| STDC-Seg (BiSeNet+STDCNet1446) | 通过 | 通过 | 通过 (3.69→1.66) | 通过 | 通过 (strict 加载, mIoU 0.3882*) | **通过** |
| PIDNet-S | 通过 | 通过 | 通过 (3.43→1.22) | 通过 | 通过 (strict 加载, mIoU 0.3425*) | **通过** |
| DDRNet (DualResNet_imagenet) | 通过 | 通过 | 通过 (3.16→1.12) | 通过 | 通过 (strict 加载, mIoU 0.5076*) | **通过** |
| LETNet (third_party 适配) | 通过 (3630/840) | 通过 (208×208) | 通过 (1.63→0.74) | 通过 (model_1.pth) | 通过 (临时脚本 20 图, mIoU 0.2634*) | **通过** |

\* 1-epoch 未收敛 smoke 的 mIoU 无任何意义，仅证明"checkpoint 可 strict 加载 + 统一评价工具/临时脚本可跑通"。

## 1. STDC-Seg（BiSeNet + STDCNet1446）

- 代码： 内部文件 `new (copy)/stdc.py`（+ 同目录 `stdcnet.py`）；训练脚本（新建）`experiments/baseline_comparison/scripts/train_neu_stdc_bisenet_240k.py`；配置 `experiments/baseline_comparison/configs/neu_stdc_bisenet_240k{,_smoke}.yaml`
- 构建： `BiSeNet('STDCNet1446', 4, use_boundary_8=True)`，train 返回 4 路 `(out, out16, out32, out_sp8)`（out_sp8 为 1 通道 1/8 分辨率边界图）
- loss： CE × 3（主+2 辅助，权重均 1.0）+ 边界 BCE-with-logits（out_sp8 vs 二值前景掩码 label1 下采样，权重 1.0）——BiSeNet 惯例的边界监督，权重见配置 `loss_weights: [1,1,1,1]`
- 输入/batch： 200×200 / 16
- 结果： 五项全部通过（见总览）。checkpoint `repro_runs/baseline_smoke/checkpoints/neu_stdc_bisenet_240k_iter000226.pkl`
- 问题： 无（首版脚本 scheduler.step() 在 optimizer.step() 之前触发 UserWarning，ConstantLR 下无实际影响，已修正调用顺序；已产出的 smoke checkpoint 不受影响）

## 2. PIDNet-S

- 代码： 内部文件 `new/pid.py`；训练脚本（以修复版 `new (copy)/train_neu_pid.py` 为底版协议化：删除全部 val/test 监控与 best 逻辑、cosine 改恒定 lr、独立 checkpoint_dir）`experiments/baseline_comparison/scripts/train_neu_pidnet_240k.py`；配置 `neu_pidnet_240k{,_smoke}.yaml`
- 构建： `PIDNet(m=2, n=3, num_classes=4, planes=32, ppm_planes=96, head_planes=128, augment=True)`，train 返回 3 路 `[x_extra_p, x_, x_extra_d]`
- loss： 2×ohem_cross_entropy（`get_loss_function_new` 从配置构建），权重 [1.0, 0.4]，监督 3 路中的前 2 路（与修复版旧链脚本的路由完全一致）
- 结果： 五项全部通过。checkpoint `repro_runs/baseline_smoke/checkpoints/neu_pidnet_240k_iter000226.pkl`
- 问题： 无（注意 `new/train_neu_pid.py` 的死循环 bug 版本未被使用）

## 3. DDRNet（DualResNet_imagenet）

- 代码： 内部文件 `new/model_ddr.py`；训练脚本（以 `new/train_neu_ddr.py` 为底版协议化）`experiments/baseline_comparison/scripts/train_neu_ddrnet_240k.py`；配置 `neu_ddrnet_240k{,_smoke}.yaml`
- 构建： `DualResNet_imagenet(num_classes=4)`（planes=64, spp_planes=128, head_planes=128, augment=True），train 返回 2 路 `[x_, x_extra]`（已是输入分辨率）
- loss： [ohem_cross_entropy, bce_with_logits_loss] × 权重 [1.0, 0.4]（同旧链）
- 结果： 五项全部通过。checkpoint `repro_runs/baseline_smoke/checkpoints/neu_ddrnet_240k_iter000226.pkl`
- 问题： 无（同 STDC 的 scheduler 调用顺序修正，无实际影响）

## 4. LETNet（third_party/LETNet 修复 + NEU 适配）

- 代码： `Industrial-Surface-Defect/third_party/LETNet/`，嵌套 git 仓库基线 commit `faaa065`；修复与适配清单见 `results/letnet_adaptation_log.md`
- 运行： LETNet 原生 train.py（非 240k 协议脚本）：`--model LETNet --dataset neuseg --max_epochs 1 --batch_size 16 --num_workers 4`，原生超参（Adam 5e-4 + warmpoly）
- 数据适配： 新增 `Network/dataset/neuseg.py`（列表列序交换、4 类、200→208 pad、ignore_label=255）+ `Network/dataset/neuseg/`（列表拷贝 + NEUSeg symlink）；inform pkl 自动生成（histogram range (0,3)）
- 结果： 五项全部通过。数据 3630/840；forward 输出 [N,4,208,208]；loss 1.631→0.744；checkpoint `repro_runs/baseline_smoke/letnet_checkpoint/neuseg/LETNetbs16gpu1_trainval/model_1.pth`；eval 接口用临时脚本 `repro_runs/baseline_smoke/code/eval_letnet_smoke_20img.py`（LETNet 未注册进统一评价工具，本阶段不改该工具）
- 问题： 4 个已知 blocker 按审计修复（中文逗号、ESPNet_v2 断链、transformer 相对 import、LC3 通道）；另发现 ESPNet_v2 分支残留引用（②的连带）改为显式 raise；train.py 需补 neuseg 的 loss/入口/保存 3 处分支（否则 NotImplementedError 或不存 checkpoint）。均已在适配日志逐条记录

## 240k 正式训练时长预估（实测 s/iter 推算）

实测条件： smoke 配置（n_workers=4）、共享 A100-40GB。正式配置为 n_workers=8；独占卡会更快。

| 模型 | 稳态 s/iter | 240000 iters 预估 |
|---|---|---|
| STDC-Seg | ≈0.068 | **≈4.5 h** |
| PIDNet-S | ≈0.084 | **≈5.6 h** |
| DDRNet | ≈0.055 | **≈3.6 h** |
| LETNet（原生 train.py，208×208） | ≈0.27 | ≈18 h（如需 240k 协议化运行；其原生 recipe 为 epoch 制，是否纳入 240k 协议待定） |

## 产物索引

- 训练脚本： `experiments/baseline_comparison/scripts/train_neu_{stdc_bisenet,pidnet,ddrnet}_240k.py`
- 配置： `experiments/baseline_comparison/configs/neu_{stdc_bisenet,pidnet,ddrnet}_240k{,_smoke}.yaml`
- 日志摘要： `experiments/baseline_comparison/logs/smoke_{stdc_bisenet,pidnet,ddrnet,letnet}.md`
- 大日志/指标/checkpoint/临时 eval 脚本： `repro_runs/baseline_smoke/{logs,results,checkpoints,code,letnet_checkpoint}/`
- LETNet 适配清单： `experiments/baseline_comparison/results/letnet_adaptation_log.md`
