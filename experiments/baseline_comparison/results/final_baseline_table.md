# NEU-Seg Baseline 最终对比表（7 模型，pending 用户确认，未进论文）

生成：2026-10-09
统一协议：train3630 / test840；200×200；4 类含 background；from scratch；seed 1337；Adam lr=1e-4 恒定；wd 2e-6；batch 16；240k 固定终点；训练全程无 val/test、无 test-based 模型选择。
评价：`tools/evaluate_class_iou.py`，840 全量，TestRescale+ToTensor，无 ImageNet Normalize。
Params/MACs：统一实测（thop @200×200）。
FPS：交错式 10×100 段测量（7 模型逐段轮转，取段均值中位数），A100、batch1、FP32、eval+no_grad（`results/fps_idle/interleaved_fps.json`）。

| Method | Backbone | Category | mIoU | Crazing | Inclusion | Patches | Params | MACs | FPS |
|---|---|---|---|---|---|---|---|---|---|
| FDSNet | Fast-SCNN式 | 工业缺陷专用 | 87.08 | 78.69 | 91.46 | 80.24 | 2.59M | 0.33G | 205.57 |
| PIDNet-S | PIDNet-S | 通用实时 | 88.82 | 81.44 | 92.84 | 82.70 | 7.72M | 0.97G | 123.67 |
| LETNet | 轻量CNN+Transformer | 轻量实时 | 86.36 | 74.77 | 89.62 | 83.47 | 0.95M | 1.21G | 21.82* |
| STDC-Seg | STDCNet1446 | 通用实时 | 88.16 | 80.59 | 92.54 | 81.33 | 16.07M | 6.02G | 116.40 |
| DDRNet23-slim | DualResNet | 通用实时 | 89.24 | 82.03 | 93.44 | 83.17 | 20.30M† | 2.90G | 168.11 |
| DSMONet-B | ResNet-50 | 本文 | **91.24** | 84.15 | 93.49 | 88.71 | 29.37M | 6.65G | 90.43 |
| A2MS-DSMONet-B | ResNet-50 | 本文 | **91.45** | 84.47 | 93.55 | 89.16 | 29.46M | 6.65G | 91.26 |

\* LETNet 按原模型输入要求采用 208×208 padding，其余评价协议一致；其 FPS 为第三方实现在本环境（PyTorch 2.7.1）实测，TransBlock patch 操作在当前框架下效率偏低，与官方报告值不可直接比较。
† 内部实现"DDRNet23-slim"实测 20.30M，与历史表 6.71M 不一致，以实测为准。

注：DSMONet-B 与 A2MS-DSMONet-B 的 FPS 差（90.43 vs 91.26，约 0.9%）在交错测量的噪声范围内——两者 MACs 相同、参数仅差 0.31%，论文中建议表述为"推理速度基本相当"。冻结 mIoU（91.24/91.45）与 Params/MACs 未改动。

## 数据来源

- FDSNet：`results/fdsnet_results.json`（sha256 942c8bb1…），audit `results/fdsnet_audit.md`
- LETNet：`results/letnet_results.json`（sha256 6ab2806a…），audit `results/letnet_audit.md`
- PIDNet-S / STDC-Seg / DDRNet：`results/{pidnet,stdc,ddrnet}_results.json` + 对应 audit
- DSMONet-B / A2MS：`experiments/audit/long240k_endpoint_results.json`（mIoU/逐类，冻结）
- FPS：`results/fps_idle/interleaved_fps.json`（7 模型同批）

## DMC-Net 处理说明

DMC-Net 未纳入：官方仓库（Michaelzyb/DMC-Net）自身缺失模型定义文件 `dmcnet.py`，无法复现，证据链见 `industrial_baseline_audit.md`。工业缺陷专用对比由 FDSNet（ICASSP 2022，官方代码完整）承担。
