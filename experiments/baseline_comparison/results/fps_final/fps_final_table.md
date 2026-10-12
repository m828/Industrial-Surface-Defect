# NEU-Seg 十模型统一 FPS 最终表（CJIG v7 补充指标）

测量时间：2026-10-12。数据来源：`fps_final_results.json`（attempt 3，经三重复测验证）。
协议：NVIDIA A100-PCIE-40GB、batch=1、FP32、`model.eval()`、`torch.no_grad()`；每模型 warmup 100 次，10 个交错 block × 每模型 100 次逐次计时（共 1000 样本）；FPS = 1000 / 中位延迟。
Params / MACs 为 `final_baseline_table_v3.md` 冻结值，本轮未重新统计。

| Model | Category | Input（入口） | 内部计算尺寸 | Params | MACs | 中位延迟 (ms) | FPS |
|---|---|---|---|---|---|---|---|
| U-Net | Classical | 200×200 | 208×208 | 31.04M | 36.10G | 2.76 | 361.82 |
| DeepLabv3+ | Classical | 200×200 | 200×200 | 59.34M | 14.29G | 11.69 | 85.54 |
| BiSeNetV2 | Real-time | 200×200 | 224×224 | 5.20M | 2.36G | 5.40 | 185.21 |
| STDC-Seg | Real-time | 200×200 | 200×200 | 16.07M | 6.02G | 8.81 | 113.50 |
| PIDNet-S | Real-time | 200×200 | 200×200 | 7.72M | 0.97G | 8.29 | 120.63 |
| DDRNet-23 | Real-time | 200×200 | 200×200 | 20.30M | 2.90G | 6.16 | 162.29 |
| LETNet | Real-time | 200×200 | 208×208 | 0.95M | 1.21G | 44.74 | 22.35 |
| FDSNet | Industrial | 200×200 | 200×200 | 2.59M | 0.17G | 4.42 | 226.00 |
| DSMONet-B | Ours | 200×200 | 200×200 | 29.37M | 6.65G | 10.45 | 95.65 |
| A2MS-DSMONet-B | Ours | 200×200 | 200×200 | 29.46M | 6.65G | 10.48 | 95.41 |

说明：

1. BiSeNetV2 / FDSNet 按推理部署形态测量（BiSeNetV2 `aux_mode='eval'`，FDSNet `aux=False`；辅助头参数从 checkpoint 剥离，主输出已验证与训练形态逐位一致）。表中 Params 列为含辅助头的训练形态冻结值，部署形态参数量分别为 3.34M / 0.96M，详见 `fps_final_audit.md`。
2. STDC-Seg 的实现在 eval 模式下仍无条件执行全部辅助分支，按时际执行路径计时（与其 MACs 口径一致）。
3. DDRNet-23 / PIDNet-S 的辅助分支由 `self.training` 门控，eval 模式不执行；DSMONet-B / A2MS-DSMONet-B 的辅助分割头与细节监督头同样仅在训练时执行。
4. A2MS-DSMONet-B 与 DSMONet-B 的 FPS 差异为 0.25%（<5%），不构成速度差异主张。
5. 历史 FPS（38.16/37.37 等）为早期不同测量链的 audit 数据，与本表不可比，不得混用。
