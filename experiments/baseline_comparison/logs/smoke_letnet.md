# Smoke 日志摘要 — LETNet / NEU（third_party 适配）

- 日期： 2026-09-29
- 仓库： `Industrial-Surface-Defect/third_party/LETNet/`（嵌套 git 仓库，基线 commit `faaa065` "Add files via upload" + 本地适配修改，详见 `results/letnet_adaptation_log.md`）
- 运行命令（cwd = `third_party/LETNet/Network/`）：
  `PYTHONPATH=".:./model" /opt/conda/bin/python train.py --model LETNet --dataset neuseg --max_epochs 1 --batch_size 16 --num_workers 4 --savedir "/workspace/Industrial Surface Defect/repro_runs/baseline_smoke/letnet_checkpoint/"`
- 超参： LETNet 原生 recipe（Adam lr=5e-4 + warmpoly、wd=1e-4、batch16、输入 208×208、random scale/mirror）。本阶段只做接口验证，未按 240k 协议改造其超参。
- 完整控制台日志： `repro_runs/baseline_smoke/logs/smoke_letnet_console.log`
- checkpoint： `repro_runs/baseline_smoke/letnet_checkpoint/neuseg/LETNetbs16gpu1_trainval/model_1.pth`（4.8 MB，dict 键 {epoch, model}）
- 训练 log/曲线： 同目录 `log.txt`、`loss_vs_epochs.png`、`iou_vs_epochs.png`
- inform 缓存（自动生成）： `third_party/LETNet/Network/dataset/inform/neuseg_inform.pkl`（mean [113.59]*3，classWeights [1.46, 8.73, 6.25, 8.45]，histogram range (0,3)）
- 临时 eval 脚本（接口验证）： `repro_runs/baseline_smoke/code/eval_letnet_smoke_20img.py`

## 关键数字（接口验证，非结果）

- 数据读取： train 3630 / val 840 样本，226 iters（= 1 epoch，drop_last）
- forward/loss： iter1 1.631 → iter226 0.744（有限且下降）；epoch 平均 train loss 1.0795
- 参数量： 0.95 M
- checkpoint： model_1.pth 生成 ✓
- 训练内建 val（840 张，LETNet 自带流程）： mIOU 0.2993（无意义）
- 临时 eval 脚本（前 20 张 test_neu，strict 加载 + pad 208 forward + argmax + 裁回 200×200）： mIoU 0.263370（无意义，仅验证接口）
- 实测速度： 稳态 ≈0.27 s/iter（bs16@208×208，共享 A100）
- 全程 wall time： 140 s（含 inform 首次统计生成）
