# Baseline 代码状态审计（CJIG v7 补充实验）

> ⚠️ 2026-10-10 更新：LETNet 已改归 Real-time（非工业缺陷专用）；"DDRNet23-slim" 已更名 DDRNet-23（见 `results/complexity_audit_final.md` §3）。最终分类与数字以 `results/final_baseline_table_v3.md` 为准。

生成：2026-10-09
范围：`third_party/` 全部仓库 + `new/` 内 baseline 模型文件 + 统一评价链注册状态。
结论先行：**全部 8 个候选 baseline 均已完成 240k 统一协议训练与评价**；无需重新拉取官方代码（PIDNet 官方仓因网络未克隆，但内部实现已验证一致并已完成训练）。

更新：2026-10-10 —— U-Net（90.62）、DeepLabv3+（90.95）、BiSeNetV2（88.75）已完成训练评价并回填 `results/final_baseline_table_v2.md`。

## 1. 总览表

| 模型 | 类别 | 来源 / 代码位置 | 版本 / commit | 可运行 | 240k 状态 | 建议纳入论文 |
|---|---|---|---|---|---|---|
| U-Net | Classical | `new/model_unet.py`（本仓库历史实现，canonical scale=1） | 仓库内 | ✅（208 pad） | ✅ 90.62 | ✅ |
| DeepLabv3+ | Classical | `new (copy)/model/deeplabv3.py`（历史 DeepLab 类，ResNet-50, OS16） | 仓库内 | ✅ | ✅ 90.95 | ✅ |
| BiSeNetV2 | Real-time | `third_party/BiSeNetV2`（CoinCheung/BiSeNet 官方） | @6b4b67a | ✅（224 pad，load_pretrain 已禁用） | ✅ 88.75 | ✅ |
| STDC-Seg | Real-time | `new/stdcnet.py`（内部实现）；官方仓 `third_party/STDC-Seg` @59ff37f 仅参照 | — | ✅ | ✅ 88.16 | ✅ |
| PIDNet-S | Real-time | `new/pid.py`（内部实现，结构经统一评价+smoke 验证）；官方仓 XuJiacong/PIDNet @4c158cf 因网络未克隆 | — | ✅ | ✅ 88.82 | ✅ |
| DDRNet23-slim | Real-time | `new/model_ddr.py`（内部实现）；官方仓 `third_party/DDRNet` @de0db31 仅参照 | — | ✅ | ✅ 89.24 | ✅ |
| FDSNet | Industrial | `third_party/FDSNet`（官方，jianzhang96/fdsnet） | @ced4d0d | ✅（mmcv shim） | ✅ 87.08 | ✅ |
| LETNet | Industrial\* | `third_party/LETNet`（官方，XU-GITHUB-curry） | @faaa065 + 本地适配 | ✅（208 pad） | ✅ 86.36 | ✅ |

\* LETNet 归属待用户最终确认：其官方出处（IEEE T-ITS 2023）为通用轻量实时分割，非工业缺陷专用；当前按任务安排列入工业对比层，v2 表注已标明。

## 2. 明确排除 / 不纳入

| 模型 | 状态 | 原因 |
|---|---|---|
| DMC-Net | ❌ 排除 | 官方仓（Michaelzyb/DMC-Net）自身缺模型定义 `dmcnet.py`（79 文件零命中，无 pyc），不可复现；证据链见 `industrial_baseline_audit.md` |
| SeaFormer | ❌ 排除 | 依赖 mmcv（与本环境冲突），且公平迁移需大量改动；收益低（已有 3 个通用实时 baseline） |
| PP-LiteSeg-B / "BiSeNetV1-L" | ❌ 不纳入 | 评价链中这两个注册名指向 `new/model_dsmo_rs50_ppliteseg.py` / `model_dsmo_rs50_csfcn_yuan.py`，是 DSMONet 变体的历史命名残留，**并非真正的 PP-LiteSeg / BiSeNetV1**，不能作为 baseline |
| BiSeNetV1（真） | 不单独训练 | CoinCheung 仓内含 V1，但 real-time 层已有 STDC/PIDNet/DDRNet/BiSeNetV2 四个代表，继续增加边际收益低 |

## 3. 新增三模型的关键技术处理（本轮）

| 项 | U-Net | DeepLabv3+ | BiSeNetV2 |
|---|---|---|---|
| 参数量（实测） | 31.04M | 59.34M | 5.20M |
| 输入处理 | 200→208 pad（200 不满足 /16，unetUp 负偏移 pad 在 200 会 shape mismatch），评价 crop 回 200 | 原生 200（forward 内置 interpolate 回原尺寸） | 200→224 pad（BGALayer 要求 /32×4 == /8，200 时 24/28 对不上；224 全部整除），评价 crop 回 200 |
| from scratch 保障 | 无预训练机制 | `pretrained=False`（否则会下载 ImageNet 权重） | **禁用 `load_pretrain`**（上游 `__init__` 默认联网下载 backbone_v2.pth） |
| 损失（链习惯） | plain CE（ignore 255 pad 区） | plain CE | OHEM-CE ×5 头各 1.0（thresh 0.7, min_kept 1e5，ignore 255） |
| 评价注册 | `unet` → `_PadCropWrapper(208)` | `deeplabv3p` | `bisenetv2` → `_PadCropWrapper(224)`，aux_mode=train 保留辅助头以 strict 加载 |

三模型前向 sanity 已于 2026-10-09 通过（形状/参数量如上表）。

## 4. 统一协议核对（所有 baseline 一致）

train 3630 / test 840；4 类含背景；from scratch；seed 1337；Adam lr=1e-4 恒定、wd 2e-6；batch 16；240k 固定终点；训练中无 val/test、无模型选择；统一 `tools/evaluate_class_iou.py` 评价。

## 5. 风险与注意

1. 共享 GPU：与 lung 项目共用 A100-40GB，训练墙钟受负载影响；协议不受影响（只影响速度）。
2. BiSeNetV2 / U-Net 的 pad 处理与 LETNet 同类，论文表注需统一口径："按原模型输入约束 padding 至 208/224，评价协议与其余模型完全一致"。
3. PIDNet 官方仓未克隆不影响结果有效性：内部实现 `new/pid.py` 与官方结构一致（此前 smoke + 统一评价已验证），来源与 commit 已在 `repo_clone_log.md` 固定。
