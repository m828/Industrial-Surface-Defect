# DSMONet 历史结果说明 (dsmonet_history_note)

> 日期：2026-09-29
> 用途：向 baseline 实验参与者说明 DSMONet-B 两个历史数字的关系。**不删除任何历史记录。**

## 两个数字

| 来源 | 数字 | 口径 |
|---|---:|---|
| 硕士大论文 / 历史正式实验 | DSMONet-B = **90.2%** mIoU | 历史 test-monitor 峰值：训练中按 batch16、drop_last 实际评价 832/840 张，约 52k–60k 迭代阶段的监控最优值 |
| 当前论文（CJIG 稿） | DSMONet-B = **91.24%** mIoU | 当前统一协议：from scratch、240k 固定终点、seed 1337、统一 840 张全量终评一次、mIoU 含背景、无测试集模型选择 |

## 两者区别的原因

1. **训练预算不同**：历史值为约 52k–60k 阶段表现，当前为 240k 充分训练；
2. **评价协议不同**：历史为 832 张监控峰值（存在 monitor 选优口径），当前为 840 张固定终点一次评价。

两条独立证据链（历史 detail 链 60k→240k：0.9028→0.913+；当前链 60k→240k：0.9007→0.9124）均显示约 +1 pt 来自训练预算；其余约 0.1 pt 量级来自评价口径。

## 纪律

- **禁止写**："DSMONet 从 90.2 提升到 91.24"——模型架构没有变化，不存在"模型提升"；
- **禁止**把 90.2% 写入当前论文正文；正文唯一正式数字为 91.24%；
- 90.2% 仅保留于 audit 档案，作为历史记录。

## 证据链

- `experiments/audit/neu_90_2_vs_91_24_diff.md`（差异归因，P1 训练预算 / P2 评价口径 / P3 环境）
- `experiments/audit/neu_historical_90_2_evidence.md`（90.2 出处取证：run_2023_08_21 日志 60k monitor=0.90283）
- `experiments/audit/neu_dsmonet_base_identity.md`（当前模型与大论文 DSMONet-B 计算图同一，IDENTITY-A）
- `experiments/audit/neu_dsmonet_training_trajectory.md`（训练轨迹对照）
- `experiments/audit/neu_current_base_9124_manifest.json`（91.24 复核 manifest，含 checkpoint sha256）
