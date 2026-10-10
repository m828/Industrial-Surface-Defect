# NEU-Seg Baseline 最终对比表 v2（历史版本——已被 v3 取代，勿再引用）

> ⚠️ 2026-10-10：本表已由 `final_baseline_table_v3.md` 取代。v3 修正：LETNet 改归 Real-time（非工业专用）；"DDRNet23-slim" 更名 DDRNet-23（身份审计）；MACs 统一真实推理路径口径（BiSeNetV2 2.36G、FDSNet 0.17G）；新增 Background IoU；FPS 移出待统一复测。mIoU 数字两版一致。

更新：2026-10-10（第二版：新增 Classical 层与 BiSeNetV2，共 10 模型，全部完成）
统一协议：train3630 / test840；4 类含 background；from scratch；seed 1337；Adam lr=1e-4 恒定；wd 2e-6；batch 16；240k 固定终点；训练全程无 val/test、无 test-based 模型选择。
评价：`tools/evaluate_class_iou.py`，840 全量，TestRescale+ToTensor，无 ImageNet Normalize。
Params/MACs：统一实测（thop @200×200）。
FPS：不进本表——按 `fps_rebenchmark_plan.md` 在 GPU 空闲时统一复测后另行回填；既有 FPS 值仅留存 audit。

| Method | Category | Backbone | mIoU | Crazing | Inclusion | Patches | Params | MACs |
|---|---|---|---|---|---|---|---|---|
| U-Net | Classical | 五级编解码（VGG 式） | 90.62 | 83.06 | 93.15 | 87.73 | 31.04M | 36.10G§ |
| DeepLabv3+ | Classical | ResNet-50 (OS=16) | 90.95 | 84.01 | 93.08 | 88.15 | 59.34M | 14.29G |
| BiSeNetV2 | Real-time | 双分支（Detail+Semantic） | 88.75 | 80.72 | 93.00 | 83.00 | 5.20M | 3.42G§ |
| STDC-Seg | Real-time | STDCNet1446 | 88.16 | 80.59 | 92.54 | 81.33 | 16.07M | 6.02G |
| PIDNet-S | Real-time | PIDNet-S | 88.82 | 81.44 | 92.84 | 82.70 | 7.72M | 0.97G |
| DDRNet23-slim | Real-time | DualResNet | 89.24 | 82.03 | 93.44 | 83.17 | 20.30M‡ | 2.90G |
| FDSNet | Industrial | Fast-SCNN 式轻量骨干 | 87.08 | 78.69 | 91.46 | 80.24 | 2.59M | 0.33G |
| LETNet | Industrial\* | 轻量 CNN+Transformer | 86.36 | 74.77 | 89.62 | 83.47 | 0.95M | 1.21G |
| DSMONet-B | Ours | ResNet-50 | **91.24** | 84.15 | 93.49 | 88.71 | 29.37M | 6.65G |
| A2MS-DSMONet-B | Ours | ResNet-50 | **91.45** | 84.47 | 93.55 | 89.16 | 29.46M | 6.65G |

\* LETNet 归属待用户确认：官方出处（IEEE T-ITS 2023）为通用轻量实时分割，非工业缺陷专用；当前按任务安排列入工业对比层。

† 输入处理：U-Net 200→208 padding（/16 整除约束）、BiSeNetV2 200→224 padding（BGALayer /32×4==/8 几何约束）、DeepLabv3+ 原生 200；padding 模型的评价输出均裁回 200×200，评价协议与其余模型完全一致（与 LETNet 208 同例）。

§ MACs 口径：U-Net 按 208×208 内部前向实测；BiSeNetV2 按 224×224 内部前向且含 5 个输出头（与 FDSNet aux=True 口径一致），推理仅消费主头。

‡ 内部实现"DDRNet23-slim"实测 20.30M，与历史表 6.71M 不一致，以实测为准。

## 四级比较结构（论文叙事用）

1. **Classical**：U-Net（Ronneberger 2015）、DeepLabv3+（Chen 2018）——经典分割范式参照；
2. **Real-time**：BiSeNetV2、STDC-Seg、PIDNet-S、DDRNet23-slim——实时语义分割代表；
3. **Industrial**：FDSNet、LETNet——工业表面缺陷方向方法；
4. **Ours**：DSMONet-B（基础架构）、A2MS-DSMONet-B（工业缺陷增强）。

不再增加模型（SegFormer 等暂缓，除非人工确认）。

## 数据来源

- 已完成 8 个 baseline：`results/{fdsnet,letnet,stdc,pidnet,ddrnet,unet,deeplabv3p,bisenetv2}_results.json` + 对应 `_audit.md`
- DSMONet-B / A2MS-DSMONet-B：`experiments/audit/long240k_endpoint_results.json`（冻结值，未改动）
- DMC-Net 排除证据链：`industrial_baseline_audit.md`（上游缺 `dmcnet.py`，由 FDSNet 承担工业专用对比）

## 与上一版的差异

- 新增 Classical 层（U-Net 90.62、DeepLabv3+ 90.95）与 Real-time 的 BiSeNetV2（88.75），2026-10-10 全部训练+评价完成；
- 按任务要求改用 Method/Category/Backbone/mIoU/逐类/Params/MACs 九列；Input 差异并入表注，FPS 移至 `fps_rebenchmark_plan.md` 统一复测后处理；
- 原有 5 个 baseline 与本文 2 个模型的全部数字未改动。
