# CJIG v10.6-P1 — GPU 成本只读调查（不使用 FPS 推算训练成本）

日期：2026-10-26
数据来源：`repro_runs/long240k_seed1337/{base_b,a2_full}` 的 train_summary.json、train_metrics.jsonl、run log、manifest、`INCIDENTS.md`；`experiments/audit/multiseed_*_manifest.json`。均为既有记录，未发起任何新测量。

## 1. 硬件与环境（事实）

- GPU：NVIDIA A100-PCIE-40GB（driver 535.247.01）；torch 2.7.1+cu126，CUDA 12.6，python 3.11.13。
- 启动时共享情况（manifest 记录）：另有一无关 nnUNet 任务，约 7.3GB 显存 / 87% util；两模型在同一 A100 上**顺序**运行（先 Base 后 A2）。
- 10k multiseed：每个 seed 的 base/a1/a2 三运行**并发**共用一张 A100（manifest `concurrency` 字段），其墙钟时间同样受争用影响，不用于校准。
- **95.41 FPS 效率测试未满足干净 GPU 区块准入，本文件不使用任何 forward FPS 推算训练成本。**

## 2. 已完成的 240k 训练：真实墙钟与速度（事实）

| 项 | Base-B 240k | A2MS-B 240k |
|---|---|---|
| 实际优化更新 | 239,999（标签 240000） | 239,999 |
| 训练墙钟（train_summary） | **75,220.84 s = 20.89 h** | **32,418.14 s = 9.00 h** |
| metrics 末条 wall_time | 75,218.1 s | 32,416.2 s |
| Time/Image 均值（×16 = s/iter） | 0.01886 → 0.3017 s/iter | 0.00793 → 0.1269 s/iter |
| Time/Image 中位数 | 0.01908 | 0.00422 |
| 前 100 记录均值 | 0.01713 | 0.01956（≈0.31 s/iter） |
| 后 100 记录均值 | 0.01869 | 0.00417（≈0.067 s/iter） |
| run log 抽点 Time/Image | 全程约 0.019 | 10k 0.0193 / 20k 0.0179 / 30k 0.0198 / 40k 0.0211 / 60k 0.0116 / 120k 起 0.0041 |
| 恢复训练 | 无（resumed_from: None） | 无 |

## 3. 对速度差异的解释（高概率推断，非直接日志事实）

- Base 全程约 0.30 s/iter 且前后期稳定；A2 早期与 Base 同速（≈0.31 s/iter），40k–120k 之间过渡到约 0.067 s/iter 并保持至结束。
- A2 结构比 Base 多 eSE/ACW/detail head，**不可能在同等条件下快 4–5 倍**；最合理的一致解释是：Base 运行期间持续存在计算争用，而争用在 A2 运行中段消失（与启动时 nnUNet 87% util 的记录相容）。
- 服务器没有逐时段 GPU util 日志，故该解释标记为**高概率推断**；无论解释如何，**Base 的 20.89 h 是受争用污染的墙钟，不能当作干净 GPU 成本**；A2 的 9.00 h 是混合段墙钟。

## 4. 可用于预算的校准值与缺口

- **A2 干净段校准（近似）**：后段 Time/Image 0.0041–0.00417 s/img ×16 ≈ **0.066 s/iter**。
  按同一脚本 239,999 更新估算：239,999 × 0.066 ≈ 15,840 s ≈ **4.4 h/运行**（不含 checkpoint 保存、评价与少量循环开销；按 Base 实测墙钟/计时比约 +4% 估计时约 4.6 h）。
  该值仅校准于本 A100、本协议、A2 模型、后段实测；不是承诺值。
- **Base 干净成本：缺失。** 没有任何干净区段的 Base 实测，不能给出校准数字；只允许写公式：
  `T_run = N_updates × t_iter_clean + t_ckpt + t_eval`，N_updates=239,999（同脚本）。
- **评价耗时**：未取得统一 840 评价的结构化耗时记录（缺口）；如需预算，应在未来授权的评价运行中直接记录墙钟，而不是估算。
- **数据加载耗时**：Time/Image 含每个训练 step 的主体时间（含前向/反向，不含 checkpoint 保存与评价）；未单独分解 dataloader 等待时间（缺口）。

## 5. 失败与恢复记录（事实）

- `INCIDENTS.md` INC-001：两次 240k 训练均已完成（Base 2026-09-23 11:36、A2 20:36）后，约 23:57 首次评价时发现 NVML 初始化失败、CUDA 不可用；checkpoint/日志/metrics 已落盘且事后 SHA 校验完整；仅评价延迟，由 watcher 等 GPU 恢复后自动执行预注册 E1 评价；**无重训、无重选 checkpoint、无协议变更**。
- 两 240k 运行均无 resume；10k 历史复现运行曾有一次因保存目录缺失导致的启动即崩溃记录（`neu_10k_reproduction_manifest.json` incidents），与现行正式资产无关。

## 6. 对未来四运行（2 模型 × 2 新 seed）的预算口径

- 只能给出**条件公式 + A2 单侧校准**：A2 每运行约 4.4–4.6 h（干净 A100、实测后段外推）；Base 每运行缺干净校准，暂以 `239,999 × t_iter_clean(Base)` 表示。
- 若在与 Base 原运行相同程度的争用下执行，墙钟可能接近或超过 20 h/运行（以 Base 实测为上限参照，不是预算承诺）。
- **不给出未经校准的总 GPU 小时预算。** 执行前应在授权后的首个运行记录干净区段 Time/Image 再回填预算。
