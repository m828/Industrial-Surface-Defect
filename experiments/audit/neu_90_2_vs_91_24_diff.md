# 90.2 vs 91.24 差异归因 (neu_90_2_vs_91_24_diff)

> 日期：2026-09-28
> 前提：IDENTITY-A 成立（同一计算图，max_abs_diff=0.0）。因此差异只能来自训练链与评价口径。

## 四件事分开

| 维度 | 大论文 90.2 | 当前 91.24 |
|---|---|---|
| Architecture identity | DSMONet rs50（subregion unet/model_dsmo_rs50.py） | 同一计算图（new/model_dsmo_rs50.py） |
| Training result（权重） | 历史链 ~52k–60k 阶段权重（9028/9034/0110 族） | 当前 240k 固定终点权重（seed 1337，from scratch） |
| Evaluation protocol | 训练中 test-monitor：832/840（batch16, drop_last=True），monitor 峰值 | 训练后统一 840、batch1、固定终点、无 test 选择 |
| Result source | 大论文表格（原文不在服务器） | `long240k_endpoint_results.json` + 本轮逐位复评 |

## 差异原因排序（按证据强度）

**P1（直接证据）：训练预算。** 历史 90.2 对应 ~52k–60k；当前为 240k。两条独立轨迹都显示 60k→240k 约 +1 pt（历史 0.9028→0.913+；当前 0.9007→0.9124）。仅此一项即可解释绝大部分差距。

**P2（直接证据）：评价口径。** 历史为 832/840 张的 monitor 峰值（test-driven best）；当前为 840/840 固定终点。旁证：0110 权重 monitor 值 0.9047 → 统一 840 复评 0.9039，口径差约 0.1 pt。

**P3（次要）：实现/环境。** 2023 旧 PyTorch/CUDA vs 当前 PyTorch 2.7.1/CUDA 12.6；batch16 shuffled monitor 的随机子集波动（每次 832/840 且 shuffle）等。幅度约 ±0.1–0.2 pt，不可单独归因。

**排除项**：模型结构（IDENTITY-A）、数据划分（同一 3630/840 列表）、归一化（两端均无 ImageNet Normalize）、指标定义（4 类含背景、runningScore/confusion 等价）。

## Scenario 判定

同时满足 N1 与 N2 的特征，主判定为 **N1（同一架构，90.2=历史较短训练阶段，91.24=当前 240k 重训终点）**，N2 的口径差异（832-monitor-best vs 840-fixed-endpoint）为叠加因素。
不适用 N3 的"假结果"情形：历史 90.2 是真实的 monitor 记录，只是口径不同；历史 test-monitor 选峰做法在当前论文纪律下不再使用。

## 对论文的含义

- 当前论文报告 91.24 是合法的：它是同一架构在当前干净协议下的真实重训结果，且可逐位复现。
- 表述禁区：不得写"DSMONet-B 从 90.2 提升到 91.24"（模型没变，是实验链变了）；可写"在当前统一训练与评价协议下，基础网络（DSMONet-B）取得 91.24% mIoU"。
- 大论文 90.2 不需要进入当前论文；保留在本审计档案中。
