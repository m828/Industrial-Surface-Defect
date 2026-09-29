# Smoke 日志摘要 — DDRNet (DualResNet_imagenet) / NEU

- 日期： 2026-09-29
- 脚本： `experiments/baseline_comparison/scripts/train_neu_ddrnet_240k.py`（以 `new/train_neu_ddr.py` 为底版协议化）
- 配置： `experiments/baseline_comparison/configs/neu_ddrnet_240k_smoke.yaml`（train_iters=226, bs16, n_workers=4, seed=1337, Adam 1e-4 恒定, wd=2e-6）
- 完整控制台日志： `repro_runs/baseline_smoke/logs/smoke_ddrnet_console.log`
- 指标流： `repro_runs/baseline_smoke/results/neu_ddrnet_240k_train_metrics.jsonl`
- 摘要： `repro_runs/baseline_smoke/results/neu_ddrnet_240k_train_summary.json`
- checkpoint： `repro_runs/baseline_smoke/checkpoints/neu_ddrnet_240k_iter000226.pkl`（含 model_state/iter 键）
- eval 接口日志： `repro_runs/baseline_smoke/logs/eval_ddrnet_smoke.log`，CSV `repro_runs/baseline_smoke/results/eval_ddrnet_smoke.csv`

## 关键数字（接口验证，非结果）

- 训练数据： 3630 样本，226 iters（= 1 epoch，drop_last），无 val/test loader
- loss（[ohem_cross_entropy, bce_with_logits_loss] × 权重 [1.0, 0.4]，同旧链）： iter20 3.1567 → iter226 1.1174，有限且下降
  - 分项 iter226： [0.999, 0.118]
- 实测速度： 稳态 ≈0.055 s/iter（n_workers=4，共享 A100）
- eval 接口（840 张 test_neu，strict 加载成功）： mIoU 0.507556（1-epoch smoke，无意义）
- 训练 wall time（不含数据集初始化）： 16.6 s
