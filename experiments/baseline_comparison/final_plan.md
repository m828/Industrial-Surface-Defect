# Baseline 最终实验设计 (final_plan)

> 日期：2026-09-29
> 状态：**代码就绪，未开训**。等待用户确认后按批次启动。
> 目的：为 CJIG 投稿稿补充公平 baseline 比较。论文主线（DSMONet → A2MS-DSMONet）不变。

## 一、最终 baseline 名单与统一配置

| 模型 | 代码来源 | commit/版本 | 输入尺寸 | batch | optimizer | lr | 训练轮次 | 评价脚本 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| DDRNet23-slim | 内部 `new/model_ddr.py`（实测 20.30M） | 内部链，无 git hash（仓库外代码，sha256 可追溯） | 200×200 | 16 | Adam | 1e-4 恒定 | 240k 固定终点 | `tools/evaluate_class_iou.py` | ✅ 脚本/配置就绪，smoke 通过 |
| STDC-Seg（STDCNet1446，16.07M） | 内部 `new (copy)/stdc.py` | 同上 | 200×200 | 16 | Adam | 1e-4 恒定 | 240k 固定终点 | 同上 | ✅ 新建脚本（`train_neu_stdc_bisenet_240k.py`），smoke 通过 |
| PIDNet-S（7.72M） | 内部 `new/pid.py`（≡官方 XuJiacong/PIDNet 结构） | 官方 commit `4c158cf`（参照） | 200×200 | 16 | Adam | 1e-4 恒定 | 240k 固定终点 | 同上 | ✅ 脚本/配置就绪，smoke 通过 |
| LETNet（0.95M） | third_party/LETNet @`faaa065`（已修复 4 blocker + NEU 适配） | `faaa065` + 本地适配（`results/letnet_adaptation_log.md`） | 208×208（200 pad，表注声明） | 16（协议化时） | Adam | 1e-4 恒定（协议化时） | 待定（协议化 ≈240k iters，见下注） | 同上（需注册 pad/crop wrapper） | ⏸ 备选，待用户决策 |

共同协议（NEU-Seg）：train 3630 / test 840，4 类含背景（background + crazing + inclusion + patches），from scratch，seed 1337，weight decay 2e-6，**训练中不构建任何 val/test loader、不做模型选择**，仅终点保存 checkpoint（含 `model_state`+`iter` 键）。

评价统一：`tools/evaluate_class_iou.py`，strict 加载，test 840 全量，mIoU 与 per-class IoU **含背景**。**禁止**使用各原仓库自带评价脚本产生论文数字。

## 二、训练批次与优先级

| 批次 | 方法 | 理由 |
|---|---|---|
| 第一批 | **DDRNet23-slim → STDC-Seg** | 最快验证协议链（3.6h / 4.5h）；STDC 为新写脚本需首轮观察 |
| 第二批 | **PIDNet-S** | 脚本成熟（5.6h） |
| 第三批 | **LETNet**（决策后） | 需先解决评价注册与协议化决策；预计 ≈18h（208×208） |

## 三、预计训练时间（NEU，实测 s/iter 外推，A100-40GB）

| 方法 | s/iter | 240k 预估 |
|---|---:|---:|
| DDRNet23-slim | 0.055 | ≈3.6 h |
| STDC-Seg | 0.068 | ≈4.5 h |
| PIDNet-S | 0.084 | ≈5.6 h |
| LETNet | 0.27 | ≈18 h |

第一批合计 ≈8h；前三方法合计 ≈14h。

## 四、是否需要重新下载代码

**不需要**。
- 三个内部 baseline 代码已在工作区并经 smoke 验证；
- LETNet 已克隆并完成适配（修复记录已存档）；
- 官方仓库（BiSeNetV2/STDC-Seg/DDRNet）已克隆作参照；PIDNet 官方仓库因网络故障未落地，**但内部实现 `new/pid.py` 结构与官方一致、已注册且 smoke 通过，不阻塞**；网络恢复后可补拉（地址与 commit 已固定于 `results/repo_clone_log.md`）。

## 五、训练启动命令（确认后按批执行）

```bash
PY=/opt/conda/bin/python
BC="/workspace/Industrial Surface Defect/Industrial-Surface-Defect/experiments/baseline_comparison"

# 第一批
$PY $BC/scripts/train_neu_ddrnet_240k.py       --config $BC/configs/neu_ddrnet_240k.yaml
$PY $BC/scripts/train_neu_stdc_bisenet_240k.py --config $BC/configs/neu_stdc_bisenet_240k.yaml

# 第二批
$PY $BC/scripts/train_neu_pidnet_240k.py       --config $BC/configs/neu_pidnet_240k.yaml
```

每个训练完成后统一评价（示例 DDRNet；其余替换 --model-name 与 --weight）：

```bash
$PY "/workspace/Industrial Surface Defect/Industrial-Surface-Defect/tools/evaluate_class_iou.py" \
  --dataset neu \
  --test-list "/workspace/Industrial Surface Defect/new (copy)/dataset/test_neu.txt" \
  --data-root "/workspace/Industrial Surface Defect/new (copy)" \
  --model-name ddr \
  --weight "/workspace/Industrial Surface Defect/repro_runs/baseline_neu_240k/checkpoints/neu_ddrnet_240k_iter240000.pkl" \
  --num-classes 4 --input-size 200 200 \
  --save-log $BC/logs/eval_ddrnet_240k.log
```

日志约定：完整训练日志写 `repro_runs/baseline_neu_240k/logs/`（不入库）；摘要与评价日志入 `experiments/baseline_comparison/logs/`。

## 六、论文表格设计建议（供参考，不改正文）

建议 NEU 主表在现有内部对比基础上扩展为：

| Method | Backbone | Input | mIoU/%（含背景） | per-class IoU（crazing/inclusion/patches） | Params/M | FPS |
|---|---|---|---|---|---|---|
| DDRNet23-slim | —（双分辨率） | 200×200 | 待训 | 待评 | 20.30* | 待测 |
| STDC-Seg | STDCNet1446 | 200×200 | 待训 | 待评 | 16.07 | 待测 |
| PIDNet-S | —（三分支） | 200×200 | 待训 | 待评 | 7.72 | 待测 |
| DSMONet-B | ResNet-50 | 200×200 | 91.24（冻结） | 84.15/93.49/88.71 | 29.37 | 38.16 |
| A2MS-DSMONet-B | ResNet-50 | 200×200 | 91.45（冻结） | 84.47/93.55/89.16 | 29.46 | 37.37 |

注意事项（论文写作纪律）：
1. \* DDRNet 参数量以统一脚本重测为准（实测 20.30M 与旧表 6.71M 冲突，见 `repository_audit.md`）；
2. baseline 与本文方法同协议（240k 固定终点、seed 1337、统一评价）一事需在表注写明；
3. baseline FPS 需与本文方法同环境同方式测量（A100、batch1、FP32），不得引用各论文官方 FPS；
4. LETNet 若纳入，表注声明输入为 200 pad 至 208；
5. baseline 结果先写入 `results/`（pending），经核验后才可进正文；
6. 冻结数字（91.24/91.45/89.91/90.97）在任何情况下不被新结果覆盖。

## 七、开跑前检查单

- [x] 正式配置 checkpoint_dir 已指向独立目录 `repro_runs/baseline_neu_240k/`
- [x] 协议化脚本通过 1-epoch smoke（4/4）
- [ ] 用户确认启动批次
- [ ] （LETNet 决策）统一评价工具注册 pad-208/crop-200 wrapper；协议化训练配置
- [ ] 每个模型训练完成后：统一评价 → 结果 pending → 核验 → `checkpoint_metadata/` 登记
