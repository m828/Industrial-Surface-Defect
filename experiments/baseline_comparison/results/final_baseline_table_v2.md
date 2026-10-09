# NEU-Seg Baseline 最终对比表 v2（CJIG v7 实验章节候选，pending 用户确认，未进论文）

生成：2026-10-09
统一协议：train3630 / test840；4 类含 background；from scratch；seed 1337；Adam lr=1e-4 恒定；wd 2e-6；batch 16；240k 固定终点；训练全程无 val/test、无 test-based 模型选择。
评价：`tools/evaluate_class_iou.py`，840 全量，TestRescale+ToTensor，无 ImageNet Normalize。
Params/MACs：统一实测（thop @200×200）。
FPS：**待统一复测**——按 `fps_rebenchmark_plan.md` 在 GPU 空闲时 7 模型同批交错复测后回填；既有 FPS 值（含 2026-10-09 交错参考值、顺序测量值、论文历史值 38.16/37.37）仅留存 audit，不进本表。

| Method | Backbone | Category | Input | mIoU | Crazing | Inclusion | Patches | Params | MACs | FPS |
|---|---|---|---|---|---|---|---|---|---|---|
| FDSNet | Fast-SCNN 式轻量骨干 | 工业缺陷专用 | 200×200 | 87.08 | 78.69 | 91.46 | 80.24 | 2.59M | 0.33G | 待复测§ |
| LETNet | 轻量 CNN+Transformer | 轻量实时\* | 208×208（pad）† | 86.36 | 74.77 | 89.62 | 83.47 | 0.95M | 1.21G | 待复测§ |
| STDC-Seg | STDCNet1446 | 通用实时 | 200×200 | 88.16 | 80.59 | 92.54 | 81.33 | 16.07M | 6.02G | 待复测§ |
| PIDNet-S | PIDNet-S | 通用实时 | 200×200 | 88.82 | 81.44 | 92.84 | 82.70 | 7.72M | 0.97G | 待复测§ |
| DDRNet23-slim | DualResNet | 通用实时 | 200×200 | 89.24 | 82.03 | 93.44 | 83.17 | 20.30M‡ | 2.90G | 待复测§ |
| DSMONet-B | ResNet-50 | 本文 | 200×200 | **91.24** | 84.15 | 93.49 | 88.71 | 29.37M | 6.65G | 待复测§ |
| A2MS-DSMONet-B | ResNet-50 | 本文 | 200×200 | **91.45** | 84.47 | 93.55 | 89.16 | 29.46M | 6.65G | 待复测§ |

\* **类别归属待用户确认**：LETNet 官方出处为通用轻量实时语义分割（IEEE T-ITS 2023，Cityscapes/CamVid），并非工业缺陷专用方法；当前按任务安排与 FDSNet 同列于工业缺陷对比层，若论文按"工业专用 / 通用实时"严格分层，LETNet 建议归入通用实时层（见其 `letnet_audit.md` 第 5 节"轻量实时（非工业专用）"）。

† LETNet 按原模型输入要求采用 208×208 padding（200×200 右/下补边，满足 /16 整除），评价输出 crop 回 200×200，其余评价协议一致。

‡ 内部实现"DDRNet23-slim"实测 20.30M，与历史表 6.71M 不一致，以实测为准。

§ FPS 待统一复测（`fps_rebenchmark_plan.md`）；2026-10-09 交错式参考值：FDSNet 205.57 / LETNet 21.82 / STDC 116.40 / PIDNet 123.67 / DDRNet 168.11 / DSMONet-B 90.43 / A2MS 91.26，仅供 audit。

## 三级比较结构（论文叙事用）

1. **工业缺陷方法**：FDSNet（ICASSP 2022，官方代码）；LETNet（见上 \* 归属说明）；
2. **通用实时语义分割**：STDC-Seg、PIDNet-S、DDRNet23-slim；
3. **本文方法**：DSMONet-B（基础架构）、A2MS-DSMONet-B（工业缺陷增强）。

不再增加模型（DeepLabV3+ / BiSeNetV2 / SeaFormer / SegFormer 均暂缓，除非人工确认）。

## 数据来源

- FDSNet：`results/fdsnet_results.json`（sha256 942c8bb1…），audit `results/fdsnet_audit.md`
- LETNet：`results/letnet_results.json`（sha256 6ab2806a…），audit `results/letnet_audit.md`，适配记录 `results/letnet_adaptation_log.md`
- STDC-Seg / PIDNet-S / DDRNet23-slim：`results/{stdc,pidnet,ddrnet}_results.json` + 对应 audit
- DSMONet-B / A2MS-DSMONet-B：`experiments/audit/long240k_endpoint_results.json`（冻结值，未改动）
- FPS 参考值：`results/fps_idle/interleaved_fps.json`

## DMC-Net 处理说明

DMC-Net 未纳入：官方仓库（Michaelzyb/DMC-Net）自身缺失模型定义文件 `dmcnet.py`，无法复现，证据链见 `industrial_baseline_audit.md`。工业缺陷专用对比由 FDSNet（官方代码完整 @ced4d0d）承担。

## 与 v1（`final_baseline_table.md`）的差异

- 行序调整为 FDSNet → LETNet → STDC → PIDNet → DDRNet → DSMONet → A2MS（工业 → 通用 → 本文）；
- 新增 Input 列（标注 LETNet 208 padding）；
- FPS 列由 v1 的交错参考值改为"待复测"，待 `fps_rebenchmark_plan.md` 执行后回填；
- mIoU / 逐类 IoU / Params / MACs 与 v1 完全一致，未改动。
