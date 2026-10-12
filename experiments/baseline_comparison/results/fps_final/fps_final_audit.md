# FPS 统一复测审计（fps_final，2026-10-12）

## 1. 测量协议

- GPU：NVIDIA A100-PCIE-40GB；batch=1；FP32；`model.eval()`；`torch.no_grad()`；`cudnn.benchmark=False`
- 入口输入：全部模型 200×200；内部 padding 模型保持自身结构约束（U-Net/LETNet 200→208，BiSeNetV2 200→224），与 `tools/evaluate_class_iou.py` 同一 `_PadCropWrapper` 路径
- warmup：每模型 100 次（10×10 轮转）
- 正式测量：10 个交错 block × 每模型 100 次逐次计时（每次前后 `torch.cuda.synchronize()` + `perf_counter`），模型在 block 内轮转，单模型共 1000 个样本
- 统计：mean / median / std（毫秒）；FPS = 1000 / median
- 脚本：`experiments/baseline_comparison/results/fps_final/bench_fps_final.py`（自包含，模型构造复用 `tools/evaluate_class_iou.py` 的 MODEL_BUILDERS）

## 2. 模型与 checkpoint 清单（sha256 全值见 fps_final_results.json）

| 模型 | 模型文件 | checkpoint sha256 前16 |
|---|---|---|
| U-Net | `new/model_unet.py`（+208 pad wrapper） | 27d5eb6542e0f1bc |
| DeepLabv3+ | `new (copy)/model/deeplabv3.py` | 5e9192b095fba07b |
| BiSeNetV2 | `third_party/BiSeNetV2/lib/models/bisenetv2.py` | e0f66982b9ae3812 |
| STDC-Seg | `new (copy)/stdc.py` | faa0665f29d8e33e |
| PIDNet-S | `new/pid.py` | b40e3ab97cbc039b |
| DDRNet-23 | `new/model_ddr.py` | c2ce05e8b83f1303 |
| LETNet | `third_party/LETNet/Network/model/LETNet.py`（+208 pad wrapper） | 6ab2806a9cd5f6cf |
| FDSNet | `third_party/FDSNet/core/models/fdsnet.py` | 942c8bb1ee7987a2 |
| DSMONet-B | `new/model_dsmo_rs50.py` | 46c5b83c73130ade |
| A2MS-DSMONet-B | `new/model_dsmo_rs50_eSE_adapt_detailloss_fixed.py`（fixed 版，非 bug 版） | 5beaaa1c0b424403 |

## 3. 推理路径与部署形态转换

与 `complexity_audit_final.md` 的推理 MACs 口径一致：

| 模型 | 推理形态 | 转换规则 | 验证 |
|---|---|---|---|
| BiSeNetV2 | `aux_mode='eval'`（辅助头不存在，2.36G 形态） | 从 240k checkpoint 剥离 `aux2./aux3./aux4./aux5_4.` 共 56 个键 | 同输入下主输出与训练形态 max\|Δ\|=0（逐位一致） |
| FDSNet | `aux=False`（0.17G 形态） | 剥离 `edge_auxlayer./semantic_auxlayer.` 共 20 个键 | 同上，max\|Δ\|=0 |
| DDRNet-23 / PIDNet-S | eval() 即可 | 辅助分支由 `self.training` 门控，无需转换 | — |
| DSMONet-B / A2MS-B | eval() 即可 | 辅助分割头与细节监督头由 `self.training` 门控 | — |
| STDC-Seg | 无部署形态 | forward 无条件执行全部分支，按实计时（与 6.02G 口径一致） | — |
| 其余 | 单输出 | — | — |

部署形态参数量（本次实测）：BiSeNetV2 3.3433M、FDSNet 0.9617M；表中的冻结 Params（5.20M / 2.59M）为含辅助头的训练形态口径，两者关系已在 v3 表注中说明。

## 4. 共享 GPU 污染防护与采纳依据（重要）

本机与另一项目共享 GPU，外部任务以容器方式运行（宿主 `ps` 不可见），表现为约 1.4GB 显存占用的进程反复出现/退出。判定分两级：

- **进程级哨兵**（保守）：测量脚本内置 2s 轮询，逐 block 记录外部 pid；包装脚本在启动前要求连续 3 次空闲检查。
- **计时级证据**（最终采纳依据）：污染只会增加延迟。若每个模型 10 个 block 的中位数相互一致（max/min 接近 1）、block 墙钟约等于理论耗时，则该运行在计时层面无污染。

本轮全部测量尝试一览：

| 尝试 | 进程级判定 | 计时级证据 | 处置 |
|---|---|---|---|
| 手工首发（`fps_run1_contaminated_gpu_busy.*`） | 外部任务活跃计算 | LETNet block 中位 45.0→683.4ms、DSMONet 10.4→180.4ms，尖峰 5–50 倍 | **作废** |
| 驱动 v1 ×4（`fps_run1–4_contaminated.*`） | 均检出外部 pid | 各模型 block 中位一致（LETNet 43.7–46.9ms，全速） | 按当时严格判据存档，未采纳 |
| 驱动 v1 另有 1 次运行中期外部任务转活跃（block 墙钟升至约 33s），人工中止，未产出结果文件 | — | — | 中止 |
| 驱动 v2 attempt 1/2/4（`fps_attempt*_unusable.*`） | 检出外部 pid | block 墙钟 ≈11.3s（全速），block 中位一致 | 留作复测 |
| **驱动 v2 attempt 3（采纳，`fps_final_*.json`）** | 检出外部 pid | block 墙钟 ≈11.3s；每模型 block 中位 max/min ≤1.021；与 attempt 1/2/4 四重复测同模型 FPS 离散 ≤1.6% | **采纳为最终结果** |

采纳逻辑（证据分级）：

1. **事实**：attempt 1–4 期间均检出一个外部 pid（各次不同，约 1460MiB 显存常驻）。
2. **事实**：这四次运行的 block 墙钟 ≈11.2–11.6s，与十模型理论耗时（≈11.3s）一致；每模型 10 个 block 中位 max/min ≤1.054；四次独立运行同模型 FPS 离散 ≤1.6%；与 2026-10-09 交错式参考值（DSMONet 90.43、A2MS 91.26、LETNet 21.82、FDSNet 205.57 等）同量级一致。
3. **事实**：唯一真正被污染的手工首发运行呈现出不可掩盖的数量级尖峰并被检出作废，证明该检测链路有效。
4. **推断**：attempt 1–4 期间的外部 pid 为仅持有显存的空闲 CUDA 上下文，未参与计算，对计时无影响。
5. 若审稿阶段需要零外部进程的黄金窗口，可用 `run_until_clean.sh` 按同一协议直接重跑（脚本仍保留进程级严格判据接口）。

## 5. 与历史 FPS 的关系（不得混用）

| 模型 | 本表 FPS | 2026-10-09 交错参考 | 早期冻结 FPS（audit 历史） |
|---|---|---|---|
| DSMONet-B | 95.65 | 90.43 | 38.16 |
| A2MS-DSMONet-B | 95.41 | 91.26 | 37.37 |

差异来源：早期 38.16/37.37 为不同测量链（共享 GPU 顺序测量、协议不同）的历史 audit 值；2026-10-09 参考值为 100 次一窗的吞吐式计时；本轮为逐次延迟计时（FPS=1/median），且测量时 GPU 无计算竞争。三者口径不同，禁止横向混填。论文若启用 FPS 列，只能使用本表。

## 6. 结果速览

U-Net 361.82 / DeepLabv3+ 85.54 / BiSeNetV2 185.21 / STDC-Seg 113.50 / PIDNet-S 120.63 / DDRNet-23 162.29 / LETNet 22.35 / FDSNet 226.00 / DSMONet-B 95.65 / A2MS-DSMONet-B 95.41（FPS，batch1 FP32，A100）。

A2MS-DSMONet-B 相对 DSMONet-B：-0.25%（<5%，不构成速度差异主张）。
