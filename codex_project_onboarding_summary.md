# Codex 项目接手总览

> 生成日期：2026-06-15  
> 范围：只读接手盘点；未训练、未修改论文、未移动或覆盖任何权重。  
> 主仓库：`/workspace/Industrial Surface Defect/Industrial-Surface-Defect`

## 一、项目背景

本项目面向“工业表面缺陷实时语义分割”中文期刊论文与实验核验。GitHub 远程仓库为：

```text
https://github.com/m828/Industrial-Surface-Defect.git
```

服务器历史代码位于：

```text
/workspace/Industrial Surface Defect/
```

该路径下保留了三个历史代码目录：

| 目录 | 当前定位 | 接手判断 |
|---|---|---|
| `new/` | 最完整历史版本 | 主代码参考目录。包含 DSMONet/A2MS 多变体、eSE、AAM/gate attention、PPLiteSeg、CS-FCN、DDRNet、PIDNet、STDC、FCN、UNet、PSPNet、DeepLabV3、ENet 等脚本；`dataset` 是到 `new (copy)/dataset` 的 symlink。 |
| `new (copy)/` | 整理版本 | 结构更清晰，保留更多旧权重、`model/` 第三方模型和 `tests/` 旧测试文件；适合作为 Leather/NEU 统一评估与历史权重定位的主实验目录之一。 |
| `subregion unet/` | 早期原型 | 包含 `model_p.py` / `fianlModel`、Sub-region UNet、USB/Kaggle/NEU/Yachi 相关训练测试脚本；主要作为 Sub-region UNet 对比和历史代码参考。 |

## 二、论文目标与命名规则

当前目标期刊为《中国图象图形学报》；若投稿受阻，备选包括《计算机辅助设计与图形学学报》《计算机工程与应用》等。

正文模型命名规则应保持如下：

| 论文名称 | 含义 |
|---|---|
| `Base-S` | 仅采用细节-语义互优化机制的轻量基础版本，ResNet-18 方向。 |
| `Base-B` | 仅采用细节-语义互优化机制的增强基础版本，ResNet-50 方向。 |
| `A2MS-DefectNet-S` | 在 Base-S 基础上加入 ELMM、AAM 和联合监督约束的轻量版本。 |
| `A2MS-DefectNet-B` | 在 Base-B 基础上加入 ELMM、AAM 和联合监督约束的增强版本。 |
| `A2MS-DefectNet` | 论文正文主模型名。 |

接手边界：

- 不再把 `A2MS-DSMONet` 作为正文主模型名。
- `DSMONet` 只作为“细节-语义互优化机制”的基础代码/机制来源，不作为最终论文主模型名。
- DeepScientist 负责论文组织与写作；Codex 负责代码、协议、实验核验、实验重跑和结果记录。

## 三、已读取的关键文件

本轮优先读取并核对了下列文件，均已找到：

| 文件 | 位置 |
|---|---|
| `eval_protocol_lock.md` | `experiments/eval_protocol_lock.md` |
| `leather_unified_eval_manifest.md` | `experiments/leather_unified_eval_manifest.md` |
| `a2ms_b_neu_candidate_weights.md` | `experiments/results/a2ms_b_neu_candidate_weights.md` |
| `a2ms_b_neu_weight_sweep.csv` | `experiments/results/a2ms_b_neu_weight_sweep.csv` |
| `a2ms_b_neu_weight_sweep_summary.md` | `experiments/results/a2ms_b_neu_weight_sweep_summary.md` |
| `leather_unified_eval_test_summary.md` | `experiments/results/leather_unified_eval_test_summary.md` |
| `leather_unified_eval_all_summary.md` | `experiments/results/leather_unified_eval_all_summary.md` |
| `fps_platform_decision.md` | `experiments/fps_platform_decision.md` |
| `next_rerun_decision_after_protocol_lock.md` | `experiments/next_rerun_decision_after_protocol_lock.md` |
| `codex_experiment_verification_report.md` | `experiments/codex_experiment_verification_report.md` |
| `experiment_rerun_checklist.md` | `experiments/experiment_rerun_checklist.md` |
| `experiment_command_plan.md` | `experiments/experiment_command_plan.md` |
| `paper_result_safety_decision.md` | `experiments/paper_result_safety_decision.md` |

另外读取了 `README.md`、`experiments/README.md`、`src/legacy_code_notes.md`、`fixed_existing_results.md`、`complexity_results_a100_final_summary.md`、`class_iou_summary.md`、`small_object_group_summary.md` 和定性可视化索引。

## 四、GitHub 仓库状态

Git 仓库因 `safe.directory` 检查无法直接读取；本轮使用 `/tmp/codex_git_home/.gitconfig` 临时配置只读查询，未修改真实全局 Git 配置。

| 项目 | 当前状态 |
|---|---|
| 当前分支 | `main` |
| remote | `origin https://github.com/m828/Industrial-Surface-Defect.git` |
| 最近提交 | `5f5bacc 实验汇总` |
| 跟踪关系 | `main...origin/main` |
| 是否有未提交修改 | 有 |

`git status --short --branch` 显示：

```text
## main...origin/main
 M experiments/results/class_iou_errors.json
 M experiments/results/class_iou_results.csv
 M experiments/results/class_iou_summary.md
 M experiments/results/complexity_results.csv
 ? third_party/LETNet
 M tools/evaluate_class_iou.py
?? experiments/codex_experiment_verification_report.md
?? experiments/eval_protocol_lock.md
?? experiments/experiment_command_plan.md
?? experiments/experiment_rerun_checklist.md
?? experiments/fps_platform_decision.md
?? experiments/leather_unified_eval_manifest.md
?? experiments/next_rerun_decision_after_protocol_lock.md
?? experiments/paper_result_safety_decision.md
?? experiments/results/a2ms_b_neu_candidate_weights.md
?? experiments/results/a2ms_b_neu_weight_sweep.csv
?? experiments/results/a2ms_b_neu_weight_sweep_summary.md
?? experiments/results/complexity_results_a100_final_summary.md
?? experiments/results/first_round_rerun_decision.md
?? experiments/results/leather_unified_eval_all_summary.md
?? experiments/results/leather_unified_eval_test_summary.md
?? experiments/retrain_a2ms_b_neu/
```

仓库目录存在性：

| 目录 | 是否存在 | 说明 |
|---|---:|---|
| `docs/` | 是 | 含 `paper/`、`literature/`、`experiment_plan/`、`templates/`。 |
| `experiments/` | 是 | 当前实验协议、命令计划、结果文件均在此处。 |
| `figures/` | 是 | 含结构图、模块图、数据样本、定性结果。 |
| `src/` | 是 | 仅保留服务器遗留代码说明，不存放完整训练代码。 |
| `third_party/` | 是 | 含 DMC-Net、LETNet、SeaFormer；`LETNet` 在 git status 中显示嵌套状态异常/未跟踪内容，需后续单独确认。 |

当前已有实验结果文件集中在 `experiments/results/`，包括：

- `fixed_existing_results.md`
- `new_results_pending.md`
- `paper_result_update_summary.md`
- `a2ms_b_neu_candidate_weights.md`
- `a2ms_b_neu_weight_sweep.csv`
- `a2ms_b_neu_weight_sweep_summary.md`
- `leather_unified_eval_test_summary.md`
- `leather_unified_eval_all_summary.md`
- `complexity_results.csv`
- `complexity_results_a100_final_summary.md`
- `class_iou_results.csv`
- `class_iou_summary.md`
- `class_iou_errors.json`
- `small_object_group_results.csv`
- `small_object_group_summary.md`
- `experiment_log.md`
- `first_round_rerun_decision.md`

## 五、服务器代码目录状态

### 5.1 `new/`

定位：最完整历史代码目录，后续排查训练配置、重训 A2MS-B NEU 和统一评估时必须重点使用。

主要模型文件：

- Base 系列：`model_dsmo_rs18.py`、`model_dsmo_rs50.py`
- A2MS 系列：`model_dsmo_rs18_eSE_adapt_detailloss_822.py`、`model_dsmo_rs50_eSE_adapt_detailloss.py`
- A2MS 变体：`model_dsmo_rs50_eSE.py`、`model_dsmo_rs50_eSE_adapt.py`、`model_dsmo_rs50_eSE_adapt_detailloss_lightbag.py`、`model_dsmo822.py`
- 对比方法：`model_ddr.py`、`model_dsmo_ddr_s.py`、`pid.py`、`stdcnet.py`、`fcn.py`、`model_unet.py`、`model_dsmo_rs50_ppliteseg.py`
- 其他历史变体：`model_dsmo_rs50_gateatt.py`、`model_dsmo_rs50_csfcn*.py`、`model_dsmo_512*.py`

主要训练脚本：

- NEU：`train_neu_base_s.py`、`train_neu_a2ms_s.py`、`train_neu_resnet_detailloss.py`、`train_neu_ddr.py`、`train_neu_pid.py`
- Leather/Pige：`train_pige_base_s.py`、`train_pige_base_b.py`、`train_pige_a2ms_s.py`、`train_pige_a2ms_b.py`、`train_pige_ddr.py`、`train_pige_pid.py`、`train_pige_stdc.py`
- 传统 baseline：`train_pige_fcn.py`、`train_pige_unet.py`、`train_pige_psp.py`、`train_pige_deeplabv3.py`、`train_pige_enet.py`、`train_pige_hrnet.py`
- 其他数据集：`train_yachi.py`、`train_yachi_res_dsmonet.py`

主要测试/预测脚本：

- NEU：`test_neu.py`、`test_neu_dsmonet.py`、`test_neu_dsmonet_detailloss.py`、`test_neu_resnet_s.py`
- Leather/Pige：`test_pige.py`、`test_pige_resnet.py`、`test_pige_fcn_unet.py`
- Yachi：`test_yachi.py`
- 预测：`predict.py`、`predict_pige.py`、`predict_pige_other.py`

配置与数据：

- `new/dataset` 是 symlink，指向 `/workspace/Industrial Surface Defect/new (copy)/dataset`
- NEU 训练脚本使用 `./dataset/train_neu.txt`，验证/测试日志使用 `./dataset/test_neu.txt`
- Leather 训练脚本使用 `./dataset/pige/train.txt`，训练时验证使用 `./dataset/pige/val.txt`
- 论文最终测试应由统一评估脚本使用 `./dataset/pige/test.txt`

关键权重：

- A2MS-B NEU 唯一候选：`new/model_savePath/dsmonet_resnet_pascal_0109_neu_eSE_adapt_datailloss.pkl`
- Base-B NEU：`new/model_savePath/dsmonet_resnet_pascal_0110_neu_120000.pkl`
- Base-S NEU：`new/model_savePath/dsmonet_resnet_pascal_neu_base_s.pkl`
- A2MS-S NEU：`new/model_savePath/dsmonet_resnet_pascal_neu_a2ms_s.pkl`
- A2MS-B Leather 旧权重：`new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl`
- A2MS-S Leather：`new/model_savePath/dsmonet_resnet_pascal_pige_a2ms_s.pkl`
- Base-S Leather：`new/model_savePath/dsmonet_resnet_pascal_pige_dsmor18_0228_80000.pkl`
- Base-B Leather 新训：`new/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0220ksh.pkl`
- DDRNet：`new/model_savePath/ddr_pascal_0119_neu_ddr_nolabels_23.pkl`、`new/model_savePath/ddr_pascal_pige_ddr23s.pkl`
- PIDNet Leather：`new/model_savePath/fcn_pascal_pige_pid_s.pkl`
- 大量 `dsmonet_resnet_pascal_augnew_*.pkl` 存在，但已被文档判定为 `model_dsmo822.py` 相关变体，不应直接当作论文 A2MS-B 主权重。

### 5.2 `new (copy)/`

定位：整理版本和旧权重集中目录，是当前 Leather 统一评估的重要来源。

主要模型/子目录：

- 顶层保留 Base/A2MS/DDR/PID/STDC 相关模型文件；
- `model/` 下有 `deeplabv3.py`、`enet.py`、`pspnet.py`、`dsmonet.py` 等第三方/旧 baseline；
- `tests/` 下有旧 NEU PSP/UNet 权重与测试脚本；
- `model_savePath/` 和 `model_savePaths/` 保存关键旧权重。

关键权重：

- Base-B Leather 0126：`new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0126.pkl`
- Base-B Leather 0127：`new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl`
- eSE/adapt 旧变体：`new (copy)/model_savePath/dsmonet_resnet_pascal_0120_pige_eSE_adapt.pkl`
- STDC：`new (copy)/model_savePath/sdtdcnet_pige_stdc2_pige.pkl`，文件名含 `stdc2`，但当前脚本按 STDCNet1446/STDC1-Seg 加载，需人工确认。
- PIDNet NEU：`new (copy)/model_savePaths/pid_neu_0119_neu_pid.pkl`
- Sub-region/UNet NEU：`new (copy)/trainedfile/finalModel_best_newmodel_neu.pkl`、`unet_pascal_augnew_neu.pkl`
- `tests/fcn_pascal_neu_psp0121.pkl`、`tests/fcn_pascal_neu_unet.pkl` 存在，但论文可追溯性报告将其标为疑似 PSP/UNet、需重核。

实际数据划分观测：

| 文件 | `sed -n '$='` 行数 | 首尾范围 |
|---|---:|---|
| `dataset/train_neu.txt` | 3630 | NEU training |
| `dataset/test_neu.txt` | 840 | NEU test |
| `dataset/pige/train.txt` | 1638 | `images/468.jpg` 到 `images/2105.jpg` |
| `dataset/pige/val.txt` | 235 | `images/2106.jpg` 到 `images/2340.jpg` |
| `dataset/pige/test.txt` | 468 | `images/0.jpg` 到 `images/467.jpg` |

注意：这与 `eval_protocol_lock.md` 中 Leather train=1637、val=234、test=467 的记录差 1。当前评估摘要也写 468 张。下一步必须人工确认是协议记录误差、空行/表头问题，还是历史划分版本差异。

### 5.3 `subregion unet/`

定位：早期原型和 Sub-region UNet 对比参考。

主要文件：

- 模型：`model_p.py`、`model_unet.py`、`model_dsmo*.py`、`pidnet.py`、`stdcnet.py`
- NEU：`train_neu.py`、`test_neu.py`
- USB：`model/train_usb.py`、`test_usb.py`
- Kaggle/XG：`model/train_xg.py`、`model/datagenerator_kaggle.py`、`model/datagenerator_kaggle_val.py`
- Yachi/Pige：`train_pige.py`、`test_pige.py`、`train_yachi.py`、`test_yachi.py`

权重：

- `subregion unet/trainedfile/finalModel_best_newmodel_neu.pkl`
- `subregion unet/trainedfile/finalModel_best_newmodel_usb.pkl`

限制：

- 顶层没有 `dataset/` 目录；脚本中引用 `./dataset/train_neu.txt`、`./dataset/test_neu.txt`、`./dataset/usbtrain.txt` 等，需要后续确认实际运行时数据路径。
- 当前没有确认到 Leather/Pige 可直接用于论文主结果的 Sub-region UNet 权重。

## 六、当前统一评估协议

以 `experiments/eval_protocol_lock.md` 为正式锁定协议：

| 项目 | NEU-Seg | Leather/Pige |
|---|---|---|
| 测试集 | `new (copy)/dataset/test_neu.txt` | `new (copy)/dataset/pige/test.txt` |
| 协议记录样本数 | 840 | 467 |
| 文件系统实测行数 | 840 | 468 |
| 类别数 | 4，含 background | 8，含 background |
| 输入尺寸 | 200 x 200 | 768 x 768 |
| 预处理 | `TestRescale + ToTensor` | `TestRescale + ToTensor` |
| Normalize | 不使用 ImageNet Normalize | 不使用 ImageNet Normalize |
| mIoU | 包含 background | 包含 background |
| 标签 remap | 无 | 无 |

FPS 锁定建议：

- 推荐统一使用 NVIDIA A100-PCIE-40GB；
- batch size = 1；
- warmup = 50；
- runs = 200；
- 使用 `torch.cuda.synchronize()`；
- 不含后处理；
- 不再混用 RTX 3090 历史 FPS 和 A100 当前 FPS。

协议差异记录：

1. `eval_protocol_lock.md` 与用户给定口径一致写 Leather test=467。
2. 实际 `pige/test.txt` 为 468 行，`leather_unified_eval_test_summary.md` 和 `leather_unified_eval_all_summary.md` 也写 468。
3. 按用户要求，论文口径暂以 `eval_protocol_lock.md` 为准；但下一步实验前必须先确认是否修订该锁定协议。

## 七、当前核心风险

1. A2MS-DefectNet-B on NEU-Seg 严重异常：当前唯一候选权重 mIoU=0.7131，历史 91.3 权重不在服务器。
2. Leather/Pige 上 A2MS-B 不优于 Base-B：统一 test 结果中 Base-B(0127)=0.8991，而 A2MS-B=0.8607，原“91.0 优于 89.2”的结论不成立。
3. Leather 测试集数量存在协议与实际文件差异：467 vs 468。
4. checkpoint `best_iou` 多数来自训练时 `val.txt`，不能直接当作论文 `test.txt` 结果。
5. `tools/evaluate_class_iou.py` 当前不是通用 `--model_type/--weight_path` 接口，只支持 `--dataset`，且模型配置硬编码；NEU loader 使用 `Image` 但未导入，导致 NameError。
6. DDRNet Leather 统一评估失败：脚本使用了错误/不匹配的加载文件，需修正 loader 后重评。
7. 类别级 IoU 文件混入不同数据集/不同口径，且 `class_iou_summary.md` 当前标题写 NEU 但表中包含 Leather 结果，不能直接用于论文。
8. 小目标/细长缺陷评价存在已知 bug：无缺陷组 mIoU 约 0.125，明显不合理。
9. 可视化样本存在选择偏差：偏向 A2MS 优于 Base 的案例，缺失败案例和预测路径记录。
10. FPS 文件版本不一致：`complexity_results.csv` 仅含 4 个主模型且 `verified=no`，`complexity_results_a100_final_summary.md` 汇总更多模型；绝对 FPS 两次 A100 测量存在 12%-23% 波动。

## 八、当前不能写入论文主结论的结果

以下结果暂不能作为论文主实验或主结论：

- NEU-Seg A2MS-B 历史 91.3：当前不可复现，唯一候选权重只有 0.7131。
- Leather A2MS-B 历史 91.0：统一 test 结果为 0.8607，且低于 Base-B(0127)。
- Leather 历史 91.0/89.7/88.1/89.2 等值：除 Base-B(0127) 接近历史外，其余需以统一 test 重评结果为准。
- 所有 val/best_iou：只能作为训练选 checkpoint 的依据，不能写成 test 结果。
- 类别级 IoU：当前来源混杂，需统一重跑。
- 小目标/细长缺陷分组评价：当前无缺陷组异常，需修 bug 后重跑。
- 对比方法中无代码/无权重的方法：FCN、U-Net、HRNet、PSPNet、DeepLabV3+、ENet、BiSeNetV1/V2、SFNet、STDC2-Seg、PP-LiteSeg-T/B、FDSNet、PGA-Net 等，除非作为“引用原论文”并明确标注。
- AAM/eSE/联合损失中间消融点：无独立权重或训练记录，不能写定量值。
- 混合 RTX 3090 和 A100 的 FPS：不能放入同一主表直接比较。

## 九、下一步总体路线

第一阶段只核验，不训练：

- 修订或确认 Leather test 样本数协议；
- 确认统一评估脚本接口和模型配置；
- 确认 NEU/Leather 数据路径、权重路径、日志路径；
- 修复 `evaluate_class_iou.py` 的 NEU loader `Image` 导入问题和 DDRNet loader 映射；
- 输出“可执行但不启动”的训练/评估命令清单。

第二阶段执行阻塞实验：

- 重训或找回 A2MS-DefectNet-B on NEU-Seg 91.3 权重；
- 对 Leather 上 Base-B、A2MS-B、Base-S、A2MS-S、STDC、DDRNet、PIDNet 做统一 test 评估；
- 统一 A100 FPS 和复杂度统计，明确是否采用 A100 作为论文速度平台。

第三阶段修复细粒度分析：

- 重建类别级 IoU；
- 修复小目标/细长缺陷分组评价；
- 重新生成可视化，加入失败案例、Base 优于 A2MS 的案例和样本 ID。

第四阶段扩展对比实验：

- DMC-Net on NEU-Seg / Leather；
- PIDNet-S on NEU-Seg / Leather 的统一复核；
- 视时间补 LETNet 或 SeaFormer、Boundary F1、更多可视化样本。
