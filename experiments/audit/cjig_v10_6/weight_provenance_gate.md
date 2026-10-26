# CJIG v10.6-P1 — 独立权重复用门禁与来源核验

日期：2026-10-26
方式：只读。实测每个 checkpoint 字节 SHA-256 并与 manifest 登记值对照；解析实际配置、训练日志与 train_metrics；不加载模型、不推理。
明细：`independent_weight_inventory.csv`（18 对象）、`candidate_prediction_cache_inventory.csv`（29 项）、`weight_asset_verification_details.json`、`e0_checkpoint_sha_crosscheck.json`。

## 一、门禁判定汇总

| 对象 | 权重字节核验 | 模型源码 | 训练入口/配置 | 数据清单 | 评价件 | 逐图预测缓存 | 判定 |
|---|---|---|---|---|---|---|---|
| E0 10k seed2026 Base/A1/A2 | best 实测 SHA 与 manifest 一致 | 与正式 SHA 一致 | 一致（cosine,10k,bs16,顶层seed） | 一致 | unified_eval_840.json SHA 一致 | **MISSING** | AVAILABLE_BUT_INCOMPLETE |
| E0 10k seed3407 Base/A1/A2 | 同上 | 同上 | 同上 | 一致 | 一致 | **MISSING** | AVAILABLE_BUT_INCOMPLETE |
| E0 10k seed1337 Base 锚点 | 无中央 manifest；本次首次登记实测 SHA `32382567fca78968…` | `new/model_dsmo_rs50.py` = fab316c…；训练脚本 6fb632eb… 与 multiseed origin 一致 | 一致（seed 缺省=1337） | 用 `new/dataset/*`，与 `new (copy)` 清单同 SHA | 统一评价 CSV mIoU 0.854558 | **MISSING** | AVAILABLE_BUT_INCOMPLETE |
| E0 10k seed1337 A2/A1 锚点 | best 实测 SHA 与 manifest 一致 | 与正式 fixed SHA 一致 | 一致 | 一致 | unified_eval_840_best.json SHA 一致 | **MISSING** | AVAILABLE_BUT_INCOMPLETE |
| E0 detail_only（3 seed，补充） | best+last 均与 manifest 一致 | 独立 detail_only 模型（非 Base/A2MS） | 一致 | 一致 | 一致 | **MISSING** | AVAILABLE_BUT_INCOMPLETE |
| **E1 240k seed1337 Base** | iter240000 实测 SHA = manifest = cache_manifest（46c5b83c…） | fab316c… 前后一致 | 一致（常数 LR,bs16,from scratch） | 一致 | unified840_iter240000.json mIoU 0.9124083744680408 | **840 PNG 已核验** | **AVAILABLE_AND_VERIFIED** |
| **E1 240k seed1337 A2MS** | iter240000 实测 SHA = manifest = cache_manifest（5beaaa1c…） | a0e6fd52… 前后一致 | 同上 | 一致 | mIoU 0.9145442177671137 | **840 PNG 已核验** | **AVAILABLE_AND_VERIFIED** |
| E1 240k seed2026/3407 Base/A2MS（4 项） | 不存在 | — | — | — | — | — | **MISSING** |

## 二、seed1337（240k）复用门禁：通过（附一项必须记录的协议事实）

逐项核验结果：

1. **原权重字节**：iter240000 两文件实测 SHA 与 `long240k_*_manifest.json`、`checkpoint_metadata/*`、`mechanism_eval_240k/cache_manifest.json` 三处登记一致。
2. **实际源码身份**：Base `fab316c…`、A2 fixed `a0e6fd52…`，训练前后 SHA 不变（manifest before==after）；A2 `head_seg1 = out` fixed forward。
3. **配置与训练依赖**：config.yml 实测：seed 顶层 1337、train_iters 240000、batch16、Adam lr=1e-4 wd=2e-6、`lr_schedule: null`（ConstantLR）、resume null、loss 权重 Base [10,1,3] / A2 [10,1,3,3]。
4. **完整训练记录**：run log + stdout/stderr + tensorboard + `train_metrics.jsonl`（各 2400 条，iter 100…240000，每 100 一条）+ train_summary.json 均在。
5. **LR 与实际优化更新**：两条 metrics 全程 LR 唯一值 0.0001（常数 LR 实证）。
   **实际更新次数 = 239,999，而非 240,000**（见第三节）。
6. **随机数初始化**：训练脚本第 102–106 行以 `cfg.get("seed",1337)` 同时设置 torch/cuda/cuda_all/numpy/random。
7. **数据清单**：train 3630（SHA 498b038f…）、test 840（SHA 2c45b319…），与现行锁定协议一致。
8. **固定终点与选权规则**：脚本第 128 行显式记录不构建 test/monitor loader；训练中无 test 评价、无 best-by-test、无 early stopping；保存点全部由 iteration 预注册。
9. **权重→预测缓存身份链**：`cache_manifest.json` 记录两 checkpoint 路径+SHA 与 840 样本 ID；v10.4 已从缓存 mask 独立复算 mIoU 与冻结值逐位一致。**链闭合。**
10. **中断/事故**：`INCIDENTS.md` INC-001 为训练全部完成后的 GPU/NVML 故障，仅延迟评价，未影响训练与 checkpoint 完整性；两运行 `resumed_from: None`，无恢复训练。

**结论：seed1337 通过复用门禁，可在新运行严格同脚本同协议的前提下作为第三对描述性统计成员；不得单独当作"多种子稳定性"证据。**

## 三、必须记录的协议事实：保存点标签比实际更新数多 1

两个 240k 训练脚本（Base 第 175–290 行，A2 第 179–290 行同构）：

```
i = start_iter (=0)
while i <= train_iters:
    for batch in trainloader:
        i += 1                      # 本次更新序号
        ...
        optimizer.step()            # 第 i 次优化更新
        cur = i + 1                 # 显示/保存用计数
        if cur in SAVE_ITERS: save(epoch=cur)
        if cur == train_iters: 写 summary 并 break
```

- 循环在 `cur == 240000` 时终止，即 `i = 239999`：**实际优化更新 239,999 次**。
- 所有永久 checkpoint 的文件名/`epoch` 标签为 N，但实际对应 **N−1 次更新**（iter240000→239999；iter010000→9999，依此类推）。
- `train_metrics.jsonl` 的 `iter` 字段同样是 `cur`，因此"iter 240000"日志对应第 239,999 次更新。
- GradAudit 按 `i` 触发，故 `i=240000` 的审计点天然缺失——与 manifest 中 `grad_audit_note` 一致；manifest 将其称为 "logging-only"，但同一 `cur` 还驱动保存与终止，**保存点标签偏移是本任务必须显式记录的事实**，不只是日志措辞。
- 10k 脚本（`train_base_b_10k.py` 第 236/267/287 行等）同样以 `i+1` 触发 monitor/保存，故 10k 端点标签 10000 对应 **9,999 次更新**；单次 endpoint monitor 评价使 best==last（两个文件、同一保存点、不同字节）。
- **可比性处理**：Base 与 A2 两运行共享同一入口与偏移，彼此可比。未来新种子运行必须**沿用同一脚本与同一循环结构**，不得"修正"偏移后再宣称与 1337 严格可比；若另写新入口，必须显式记录更新次数差异。

## 四、E0（10k）能回答什么、不能回答什么

- 已有：3 seed × Base/A1/A2（+detail_only 补充）完整权重、配置、日志、统一 840 聚合评价（含混淆矩阵），best checkpoint 实测 SHA 均与 manifest 一致。
- 缺失：**全部 10k 运行均无 840 逐图预测缓存，也无逐例 TP/FP/FN 或边界指标缓存**。只有 checkpoint 不等于有预测缓存。
- 因此：不重推理即可复用的是**聚合级 10k 多种子描述**（mIoU/类 IoU/混淆矩阵）；RQ1/RQ2 所需的逐图区域—边界分析在 10k 上**必须重新推理**（本轮未授权、未执行）。
- 10k 为 cosine 调度、9,999 次更新，与 240k L1（常数 LR、239,999 次更新）协议不同，**E0 不能充当 240k 多种子重复，也不得与 240k 结果混合统计**。

10k 统一 840 mIoU（实测评价件 SHA 已核）：

| seed | Base | A1 | A2 |
|---|---|---|---|
| 1337 | 0.854558（CSV） | 0.8577030329146066 | 0.8627271601588227 |
| 2026 | 0.8474726575362161 | 0.8608698075765292 | 0.857755345260512 |
| 3407 | 0.8544933640708361 | 0.8522812838679086 | 0.8518702689212362 |

## 五、E1（240k）现存与缺口

- 仅存 `repro_runs/long240k_seed1337/` 一对正式 240k 运行；`baseline_neu_240k*` 为其他模型（DDR/STDC/PIDNet 等），不属于本任务两模型，不得混入。
- seed2026/3407 的 Base/A2MS 240k：**4 项全部 MISSING**（目录与 checkpoint 均不存在，不能凭文件名推断）。
- 因此：**240k 协议下的独立权重重复验证目前无法在不训练的情况下开展**；唯一可立即复用的是 seed1337 的发现集证据（已核验）。

## 六、状态标签

- E0 10K ASSETS：**INCOMPLETE**（权重与聚合评价已核验；逐图预测缓存缺失；协议不同层）。
- E1 240K ASSETS：**INCOMPLETE**（1337 一对 AVAILABLE_AND_VERIFIED；其余 4 项 MISSING）。
- seed1337 复用门禁：**PASS（附 239,999 更新偏移记录）**。
