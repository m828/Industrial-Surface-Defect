# NEU-Seg Baseline 最终对比表 v3（CJIG v7 实验章节候选，pending 用户确认，未进论文）

生成：2026-10-10。取代 `final_baseline_table_v2.md`（v2 保留为历史版本，勿再引用）。
相对 v2 的变化：① LETNet 改归 Real-time（其官方出处为通用轻量实时分割，IEEE T-ITS 2023）；② "DDRNet23-slim" 更名为 DDRNet-23（身份审计见 `complexity_audit_final.md` §3）；③ 新增 Background IoU 列；④ MACs 统一为真实推理路径口径（BiSeNetV2 3.42→2.36G、FDSNet 0.33→0.17G，依据同上审计）；⑤ FPS 移出本表，待统一复测。

| Model | Category | Backbone | mIoU | Background | Crazing | Inclusion | Patches | Params | MACs |
|---|---|---|---|---|---|---|---|---|---|
| U-Net | Classical | 五级编解码（VGG 式） | 90.62 | 98.53 | 83.06 | 93.15 | 87.73 | 31.04M | 36.10G |
| DeepLabv3+ | Classical | ResNet-50 (OS=16) | 90.95 | 98.55 | 84.01 | 93.08 | 88.15 | 59.34M | 14.29G |
| BiSeNetV2 | Real-time | 双分支（Detail+Semantic） | 88.75 | 98.27 | 80.72 | 93.00 | 83.00 | 5.20M | 2.36G |
| STDC-Seg | Real-time | STDCNet1446 | 88.16 | 98.17 | 80.59 | 92.54 | 81.33 | 16.07M | 6.02G |
| PIDNet-S | Real-time | PIDNet-S | 88.82 | 98.28 | 81.44 | 92.84 | 82.70 | 7.72M | 0.97G |
| DDRNet-23 | Real-time | DualResNet-23 | 89.24 | 98.33 | 82.03 | 93.44 | 83.17 | 20.30M | 2.90G |
| LETNet | Real-time | 轻量 CNN+Transformer | 86.36 | 97.59 | 74.77 | 89.62 | 83.47 | 0.95M | 1.21G |
| FDSNet | Industrial | Fast-SCNN 式轻量骨干 | 87.08 | 97.95 | 78.69 | 91.46 | 80.24 | 2.59M | 0.17G |
| DSMONet-B | Ours | ResNet-50 | **91.24** | 98.62 | 84.15 | 93.49 | 88.71 | 29.37M | 6.65G |
| A2MS-DSMONet-B | Ours | ResNet-50 | **91.45** | 98.64 | 84.47 | 93.55 | 89.16 | 29.46M | 6.65G |

FPS：待统一复测——按 `fps_rebenchmark_plan.md` 在 GPU 空闲时对全部 10 模型同批交错测量后单独成表；本表不引用任何既有/历史 FPS 值。

表注（论文候选表述）：

1. 所有方法均在 NEU-Seg 统一协议下重新训练与评价：训练集 3630 张、测试集 840 张全量；4 类（crazing / inclusion / patches + background，mIoU 含背景）；输入 200×200；从头训练（不使用 ImageNet 预训练权重）；seed 1337；Adam（初始学习率 1e-4 恒定、weight decay 2e-6）；batch 16；240k 迭代固定终点评价；训练全程不使用验证/测试集，无测试集模型选择；统一 `tools/evaluate_class_iou.py` 评价。
2. 输入尺寸差异：U-Net、LETNet 因原模型下采样整除约束内部 padding 至 208×208，BiSeNetV2 因 BGALayer 结构约束内部 padding 至 224×224；评价输出均裁回 200×200，评价协议与其余模型完全一致。
3. 参数量与计算量为本文实际实现的统一实测值（thop，200×200 任务入口；padding 模型按其内部实际计算尺寸计入）；训练专用辅助监督分支不计入推理 MACs（STDC-Seg 实现的前向始终执行全部分支，按实际执行计入）；详见 `complexity_audit_final.md`。
4. DDRNet-23：本文实现与官方 DDRNet-23（planes=64）结构逐位一致（20.147M/2.905G 相同）；早期内部记录误标为 23-slim，真实 23-slim（planes=32，5.70M/0.74G）未纳入本轮训练。
5. 统一训练配置不代表各方法在其原始最佳专用训练方案下的最高性能；本表用于在相同数据、相同训练预算与相同评价口径下的公平对比。
6. 工业缺陷专用方法仅纳入 FDSNet：其余近年方法（DMC-Net / TAG-Net / CDARNet / LGGFormer 等）官方代码未公开或不完整，不具备可核验的公平复现条件，见 `recent_industrial_methods_audit.md`。

## 数据来源

- 8 个 baseline：`results/{unet,deeplabv3p,bisenetv2,stdc,pidnet,ddrnet,fdsnet,letnet}_results.json` + 对应 `_audit.md`
- 本文模型：`experiments/audit/long240k_endpoint_results.json`（冻结值，未改动）
- 复杂度：`complexity_audit_final.md` + `complexity_audit_raw.json`（2026-10-10 统一实测）
