# 主流方法对比审计 (mainstream_comparison_audit_v5)

> 日期：2026-09-28
> 结论：**Leather 表 ADDED（3 个 Level A 方法）；NEU 表 REMOVED（无 Level A 证据）**。

## 一、候选方法盘点与分级

| Model | Checkpoint | 数据集 | 训练监控集 | 选择方式 | 统一评价 | 级别 |
|---|---|---|---|---|---|---|
| STDC-Seg (STDCNet1446) | `new (copy)/model_savePath/sdtdcnet_pige_stdc2_pige.pkl` | Leather | val235（`train_pige_stdc.py:78`） | val-best | test468 = 0.882401（2026-06-16 初评；2026-09-28 复评一致） | **A** |
| PIDNet-S | `new/model_savePath/fcn_pascal_pige_pid_s.pkl` | Leather | val235（`train_pige_pid.py:69`） | val-best | test468 = 0.862009（两轮一致） | **A** |
| DDRNet23slim | `new/model_savePath/ddr_pascal_pige_ddr23s.pkl` | Leather | val235（`train_pige_ddr.py:69`） | val-best | test468 = 0.842052（两轮一致） | **A** |
| PIDNet-S | `new (copy)/model_savePaths/pid_neu_0119_neu_pid.pkl` | NEU | **test_neu.txt（840 测试集）**（`train_neu_pid.py:79`） | **test-driven best@58500** | 0.8667（训练日志，test 监控值） | **C（测试集泄漏，禁用）** |
| DDRNet23slim | `new/model_savePath/ddr_pascal_0119_neu_ddr_nolabels_23.pkl` | NEU | **test_neu.txt**（`train_neu_ddr.py:77`） | **test-driven best@56000** | 0.8843（同上） | **C（禁用）** |
| STDC / BiSeNet / PP-LiteSeg (NEU) | 无本地 NEU 权重 | NEU | — | — | — | 不可用 |
| 文献报告值（DMC-Net/TAG-Net 等） | — | NEU 系 | 各文献 split 不一 | — | — | B（仅量级参考，本轮不入表） |

## 二、NEU 主表决策：REMOVED（删除"待补充"声明）

理由：
1. 本地仅存的两个 NEU 主流方法权重（PIDNet-S、DDRNet23slim）均为**测试集监控选优**产物，进入论文将造成协议不公平（且与本文 NEU 固定终点纪律冲突）；
2. STDC/BiSeNet/PP-LiteSeg 无 NEU 权重；
3. 本轮禁止为主流表重新训练；
4. 文献值 split/口径异质，混入会误导。

正文处理（v5 2.2 节）：删除"将在核验后补充"，改为"不同文献数据划分与评价口径存在差异，本文主要报告统一协议下的内部对比"；3.2 局限性第（6）条同步说明。

## 三、Leather 对比表：ADDED（表 4）

证据链（全部 Level A）：
- 训练：train1638，val235 监控 + val-best 保存（脚本行号见上表）；
- 评价：`tools/evaluate_class_iou.py` + `leather_valid_test_list.txt`（468，sha256 锁定），TestRescale+ToTensor，无 Normalize，batch1，strict load；
- 复评 CSV/log：`repro_runs/leather_historical_audit/baseline_reverify/`（2026-09-28，与 2026-06-16 审计值逐位一致）；
- 参数/FPS：`repro_runs/leather_historical_audit/baseline_reverify/baseline_efficiency_768.json`（独占 A100，768²，FP32，batch1）。

| 方法 | mIoU(468) | Params/M | FPS(768², 独占A100) |
|---|---:|---:|---:|
| DDRNet23-slim（内部复现） | 84.21 | 20.30 | 150.62 |
| PIDNet-S（内部复现） | 86.20 | 7.72 | 109.83 |
| STDC-Seg（内部复现） | 88.24 | 16.08 | 85.18 |
| Base-B | 89.91 | 29.37 | 64.68 |
| A2MS-DSMONet-B | **90.97** | 29.46 | 62.83 |

诚实注记（已写入正文 2.6）：本文两模型帧率低于三个轻量基线，参数量居中；accuracy 优势与速度劣势同时如实呈现。
DDRNet 参数量实测 20.30M 与 2026-05 复杂度审计记录的 6.71M 不一致（后者可能未计辅助分支），本表统一采用本轮同一构建器实测值。

## 四、结论

- Leather：表 4 进入正文（5 行，含 3 个 Level A 基线）。
- NEU：主流对比 REMOVED，2.2 仅内部对比 + 口径说明。
