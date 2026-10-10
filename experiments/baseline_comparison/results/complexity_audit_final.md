# 复杂度（Params / MACs）最终审计——统一真实推理口径

日期：2026-10-10
统计脚本：`repro_runs/baseline_neu_240k_classical/complexity_final_audit.py`（thop 0.1.1，model.eval()，batch1，FP32，原始数据 `results/complexity_audit_raw.json`）
任务入口：统一 200×200 输入；各模型保留其必要的内部尺寸处理（U-Net/LETNet 内部 208×208，BiSeNetV2 内部 224×224），MACs 按真实推理执行路径统计。

## 1. 口径判定规则

1. 推理时结构上会被执行的分支（包括"执行了但未返回"的辅助分支）→ 计入；
2. 训练专用分支若推理时结构上不执行（`self.training` 分支跳过 / 推理形态下模块不存在）→ 不计入（与本文模型"辅助细节监督仅参与训练"的口径对称）；
3. 参数量以完整实现（named_parameters 去重全量）报告，与冻结值 29.37M/29.46M 的历史口径一致；thop 只统计前向执行模块，遇到多路径复用会重复计数（STDC 即如此），因此 thop 的 params 仅作交叉参考，不作报告口径；
4. 不为了统一计算量而修改任何模型结构。

## 2. 全量结果（200×200 任务入口）

| 模型 | 报告 Params（完整实现） | 报告 MACs（推理路径） | 内部输入尺寸 | 辅助头计入？ | thop 交叉参考 |
|---|---|---|---|---|---|
| U-Net | 31.04M | 36.10G | 208×208 | 无辅助头 | 31.040M / 36.102G ✓一致 |
| DeepLabv3+ | 59.34M | 14.29G | 200×200 | 无辅助头 | 59.340M / 14.290G ✓一致 |
| BiSeNetV2 | 5.20M | **2.36G** | 224×224 | 否（aux_mode='eval' 时辅助头不存在；训练态 5 头全执行 = 3.42G 备查） | eval 3.343M/2.364G；train 5.196M/3.417G |
| STDC-Seg | 16.07M | 6.02G | 200×200 | **是**——forward 无条件计算 conv_out16/32 与全部 sp 分支，结构上无法跳过，按规则 1 计入 | 6.017G ✓（thop params 22.30M 为多路径复用重复计数，不采用） |
| PIDNet-S | 7.72M | 0.97G | 200×200 | 否（eval 模式 `self.training` 分支跳过 seghead_p/d） | eval 7.624M / 0.969G ✓ |
| DDRNet-23（见 §3） | 20.30M | 2.90G | 200×200 | 否（eval 模式跳过 seghead_extra；该头参数 0.148M 含在完整实现内） | eval 20.147M / 2.905G ✓ |
| FDSNet | 2.59M | **0.17G** | 200×200 | 否（aux=False 部署形态下 edge/se 辅助层不存在；as-evaluated aux=True = 0.33G 备查） | auxT 2.592M/0.333G；auxF 0.962M/0.168G |
| LETNet | 0.95M | 1.21G | 208×208 | 无辅助头 | 0.950M / 1.207G ✓一致 |
| DSMONet-B | 29.37M（冻结） | 6.65G（冻结） | 200×200 | 否（eval 跳过训练监督分支） | 29.321M / 6.648G ✓一致（params 差 0.05M 为训练专用模块，见 §4） |
| A2MS-DSMONet-B | 29.46M（冻结） | 6.65G（冻结） | 200×200 | 否（AAM 在推理主路径内；辅助细节监督仅训练） | 29.337M / 6.648G ✓一致（差 0.12M 同上） |

相对此前结果 JSON 的变化（仅 MACs，且均有审计依据）：

- BiSeNetV2：3.42G → **2.36G**（改为 aux_mode='eval' 部署推理路径；与本文"训练专用监督不计入推理"口径对称；as-evaluated 3.42G 保留备查）
- FDSNet：0.33G → **0.17G**（aux=False 为上游构造器默认部署形态；as-evaluated 0.33G 保留备查）
- 其余 8 模型 MACs 与既有报告一致（≤0.01G 舍入差）

## 3. DDRNet 身份审计（重点复核结论）

**结论：内部实现的"DDRNet23-slim"实际是 DDRNet-23，论文应使用名称 "DDRNet-23"。**

证据链：

1. 官方仓（`third_party/DDRNet/segmentation/`，ydhongHIT/DDRNet @de0db31）两个变体的唯一结构差异是宽度配置：
   - `DDRNet_23_slim.py`：`planes=32, head_planes=64`
   - `DDRNet_23.py`：`planes=64, head_planes=128`
2. 内部实现 `new/model_ddr.py:358`：`planes=64, spp_planes=128, head_planes=128` —— 与官方 **DDRNet-23** 完全一致；文件中第 357 行被注释掉的才是 slim 配置（planes=32, head_planes=64）。
3. 参数/算力逐位验证（num_classes=4、augment=False）：
   - 官方 DDRNet-23：20.147M / 2.905G
   - 内部实现 eval 模式：20.147M / 2.905G —— **完全一致**
   - 官方 DDRNet-23-slim：5.695M / 0.743G —— 与内部实现明显不符
4. 历史表中"6.71M"与任何官方变体都对不上（真 slim 为 5.70M@nc=4），判定为历史命名/统计错误，不采信；当前实测 20.30M（含 0.148M 训练专用 aux 头的完整实现）为正确值。

与原方法的区别说明（供论文表注）：训练按本文统一协议（from scratch、Adam 恒定 lr、240k 固定终点），非原论文的 ImageNet 预训练 + Cityscapes 配方；结构本身与官方 DDRNet-23 一致。

## 4. 本文模型复核

- 冻结值复核通过：DSMONet-B 6.648G≈6.65G ✓、A2MS-B 6.648G≈6.65G ✓（thop eval 路径）。
- 参数量两口径说明：冻结 29.37M/29.46M 为 checkpoint state_dict 全量（含训练专用监督模块 0.05M/0.12M）；thop 前向执行口径为 29.32M/29.34M。论文沿用冻结全量口径，本审计仅记录差异，不改动冻结值。
- A2MS 的 AAM（eSE+ACW）位于推理主路径内（其参数/计算已含在 6.65G 内）；辅助细节监督仅训练阶段存在，推理零开销 ✓。

## 5. 可比性说明（论文可用表述）

- 所有模型以 200×200 为任务输入统一统计；U-Net（208）、LETNet（208）、BiSeNetV2（224）因原始结构约束需要内部 padding，MACs 按其实际内部计算尺寸计入——这属于真实推理开销的一部分。
- 训练专用辅助监督分支不计入推理 MACs（适用：DDRNet-23、PIDNet-S、FDSNet、BiSeNetV2、DSMONet-B、A2MS-DSMONet-B）；STDC-Seg 的内部实现无推理精简路径，其前向始终执行全部辅助分支，按实际执行计入。
- Params 为完整实现参数量（含训练专用模块），与论文历史口径一致。
- 统一训练协议下的结果不等于各方法在其原始最佳专用训练方案下的峰值性能。
