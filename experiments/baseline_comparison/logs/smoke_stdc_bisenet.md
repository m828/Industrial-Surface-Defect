# Smoke 日志摘要 — STDC-Seg (BiSeNet + STDCNet1446) / NEU

- 日期： 2026-09-29
- 脚本： `experiments/baseline_comparison/scripts/train_neu_stdc_bisenet_240k.py`
- 配置： `experiments/baseline_comparison/configs/neu_stdc_bisenet_240k_smoke.yaml`（train_iters=226, bs16, n_workers=4, seed=1337, Adam 1e-4 恒定, wd=2e-6）
- 完整控制台日志： `repro_runs/baseline_smoke/logs/smoke_stdc_console.log`
- 训练日志： `repro_runs/baseline_smoke/logs/run_*.log`（get_logger 落盘）
- 指标流： `repro_runs/baseline_smoke/results/neu_stdc_bisenet_240k_train_metrics.jsonl`
- 摘要： `repro_runs/baseline_smoke/results/neu_stdc_bisenet_240k_train_summary.json`
- checkpoint： `repro_runs/baseline_smoke/checkpoints/neu_stdc_bisenet_240k_iter000226.pkl`（158 MB，含 model_state/iter 键）
- eval 接口日志： `repro_runs/baseline_smoke/logs/eval_stdc_smoke.log`，CSV `repro_runs/baseline_smoke/results/eval_stdc_smoke.csv`

## 关键数字（接口验证，非结果）

- 训练数据： 3630 样本，226 iters（= 1 epoch，drop_last），无 val/test loader
- loss（总 = CE×3 + 边界 BCE）： iter20 3.6907 → iter226 1.6567，有限且下降
  - 分项 iter226： [main_ce 0.345, aux16_ce 0.424, aux32_ce 0.538, boundary_bce 0.350]
- 实测速度： 稳态 ≈0.068 s/iter（n_workers=4，共享 A100）
- eval 接口（840 张 test_neu，strict 加载成功）： mIoU 0.388201（1-epoch smoke，无意义）
- 训练 wall time（不含数据集初始化）： 18.6 s
