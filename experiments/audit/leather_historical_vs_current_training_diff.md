# Leather 历史高分链 vs 当前 60k 复现链逐项差异 (leather_historical_vs_current_training_diff)

> 日期：2026-09-28
> 对比对象：
> - 历史链 H（RTX3090 "61"，2024-01~03）：`new (copy)/runs_other/dsmonet_resnet_pige/dsmor50_pige/`（Base 0126→0127）、`runs/dsmonet_resnet_detailloss/pige_4_0228/`（822 链）、A2 r50 130k 链（日志缺失，以 checkpoint 元数据+同族脚本为准）
> - 当前链 C（A100，2026-09）：`repro_runs/leather_60k_seed1337/`（Base 60k / A2 60k，seed 1337）

## 逐项对比

| 项目 | 历史链 H | 当前链 C | 差异影响评估 |
|---|---|---|---|
| 模型文件(Base) | `new (copy)/model_dsmo_rs50.py` 同 `new/model_dsmo_rs50.py` | 同 | 无差异 |
| 模型文件(A2) | `model_dsmo_r50_eSE_adapt_detailloss.py`（head_seg1=**out**） | `model_dsmo_rs50_eSE_adapt_detailloss_fixed.py`（head_seg1=**out**） | **计算图一致**（仅注释/导入差异） |
| 输入 | 768×768 | 768×768 | 无 |
| train/val/test | 1638/235/468（同文件） | 同 | 无 |
| train transform | Transforms_PIL(768)+ToTensor，无 Normalize | 同 | 无 |
| optimizer | Adam lr=1e-4, wd=2e-6 | 同 | 无 |
| **LR scheduler** | **常数 1e-4（无 schedule）** | **cosine T_max=48000, eta_min=1e-6** | **重大**：C 在 48k 后 LR≈0，后 12k 几乎不学；H 全程满 LR |
| **总迭代** | Base: 80k（0126 scratch 0→55k+，0127 resume 55k→80k）；A2: ≥130k | 60k | **重大**：H 多 1.3–2.2 倍有效训练 |
| batch size | 12 | 8 | 中等：等效样本吞吐量 H 更高 |
| **checkpoint 选择** | **val235 best（训练中每 1000 iter 监控，仅按 val 提升保存）** | **无 val/test 监控，固定 60k 终点** | **重大**：H 的 0.9091/0.8930 是"最优截获"，C 是"终点快照"；H 终点瞬时 val 通常比 best 低 1–1.5 pt |
| loss 权重 | Base: [10,1,3]；A2: [10,1,3,1]（detail_aggregate_loss 权重 1） | 同（[10,1,3] / [10,1,3,3]？以 repro config 为准，详见下） | 待核注 |
| seed | 未显式固定（脚本 `cfg.get("seed",1337)`，历史 config 无 seed 键→默认 1337 但未固定 numpy/torch 全链） | 顶层 seed 1337 显式固定 | 小 |
| resume 链 | 0127 resume 0126@55k；pige_4 多段 resume 至 128k+ | from scratch | 中 |
| 训练机 | RTX3090（旧） | A100-PCIE-40GB | 环境差异（非决定性） |
| 评价预处理 | TestRescale+ToTensor，无 Normalize | 同 | 无 |

> loss 权重注：当前 A2 repro 使用 [10,1,3,3]（NEU 协议沿用），历史 detailloss 脚本（`train_pige_resnet_loss.py:143`、`train_pige_resnet_loss904.py:143`）为 **[10,1,3,1]**——detail loss 权重 1 vs 3 是真实差异；`train_pige_4_0228.py:133` 主循环甚至只对前 3 路加权、第 4 路单独 `loss_fn[3](outputs[3], labels)` 追加（权重 1）。历史 detail 监督强度仅为当前 repro 的 1/3。

## 为什么当前 60k（87.21/84.99）低于历史（89.91/90.97）：原因排序

**P1（几乎确定）：训练长度 × scheduler 组合。** 历史链在常数 LR 下训练 78k–130k 并在 val 最优点截获；当前链 60k 且 cosine 在 48k 后把 LR 压到 ~1e-6，相当于有效训练只有 ~48k。历史 Base 链自身的轨迹显示 55k(0.8723)→78k(0.8930) 仍在上涨；当前 Base 60k 终点 0.8721 恰好落在历史 55k 的水平上——与"少训 + LR 提前衰减"完全一致。

**P2（确定）：checkpoint 选择口径。** 历史数字是 val235 best（多次监控取峰），当前是固定终点。历史链终点瞬时 val 比 best 低 1–1.5 pt（如 0127 终点 0.8794 vs best 0.8930）。仅口径差异就值 ~1 pt。

**P3（较可能）：batch 12 vs 8 与 resume 链带来的优化轨迹差异。**

**排除项：** 数据划分（相同文件、无泄漏）、评价协议（相同）、模型结构（A2 fixed = 历史 r50 计算图）、标签映射（历史 checkpoint 在当前评价脚本下 per-class 全部正常——若映射漂移，历史权重会整体崩塌，实测没有）。

## 关于"大论文 91%"的最终解释

> 大论文/旧稿的 Leather A2MS-B ≈91% = **历史 A2 r50 链（常数 LR、~130k、batch12、val235-best）checkpoint 的 validation 最优 mIoU 0.909084（四舍五入 91%）**。
> 该 checkpoint 在当前锁定 test468 上的真实 test mIoU = **0.909713**（正确 forward）。
> 旧审计的 0.860749 是同一权重经 bug forward（head_seg1=high_feats）评价的伪低值。
> 当前 60k 复现的 84.99% 是"短训练 + cosine 早衰 + 无 val 选择 + detail 权重 3"的另一协议产物，与历史不可直接比较。
