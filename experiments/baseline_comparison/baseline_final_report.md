# Baseline 补充实验最终报告（CJIG 投稿前）

> ⚠️ 2026-10-10 更新：本报告中的 LETNet 已改归 Real-time 层、"DDRNet23-slim" 已更名 DDRNet-23（身份审计）、MACs 已统一真实推理路径口径。最终口径以 `results/final_baseline_table_v3.md` + `results/complexity_audit_final.md` 为准；本文件保留为过程记录。

生成：2026-10-09；**收尾：2026-10-10（全部完成）**
范围：CJIG v7 实验章节 baseline 对比链的当前完整状态。
纪律遵守：未修改论文正文；未修改冻结数字（NEU 91.24/91.45，Leather 89.91/90.97）；未覆盖任何已有 checkpoint 与结果；本轮未做 git 提交（等待本地论文同步）。

## 1. 模型完成状态

### ✅ 已完成（240k 统一协议，结果已产出并审计，共 8 个 baseline）

| 模型 | Category | mIoU | Crazing | Inclusion | Patches | Params | MACs | checkpoint sha256 前16 |
|---|---|---|---|---|---|---|---|---|
| DeepLabv3+ | Classical | 90.95 | 84.01 | 93.08 | 88.15 | 59.34M | 14.29G | 5e9192b095fba07b |
| U-Net | Classical | 90.62 | 83.06 | 93.15 | 87.73 | 31.04M | 36.10G | 27d5eb6542e0f1bc |
| DDRNet23-slim | Real-time | 89.24 | 82.03 | 93.44 | 83.17 | 20.30M | 2.90G | c2ce05e8b83f1303 |
| BiSeNetV2 | Real-time | 88.75 | 80.72 | 93.00 | 83.00 | 5.20M | 3.42G | e0f66982b9ae3812 |
| PIDNet-S | Real-time | 88.82 | 81.44 | 92.84 | 82.70 | 7.72M | 0.97G | b40e3ab97cbc039b |
| STDC-Seg | Real-time | 88.16 | 80.59 | 92.54 | 81.33 | 16.07M | 6.02G | faa0665f29d8e33e |
| FDSNet | Industrial | 87.08 | 78.69 | 91.46 | 80.24 | 2.59M | 0.33G | 942c8bb1ee7987a2 |
| LETNet | Industrial\* | 86.36 | 74.77 | 89.62 | 83.47 | 0.95M | 1.21G | 6ab2806a9cd5f6cf |
| DSMONet-B | Ours | 91.24 | 84.15 | 93.49 | 88.71 | 29.37M | 6.65G | 46c5b83c…（冻结） |
| A2MS-DSMONet-B | Ours | 91.45 | 84.47 | 93.55 | 89.16 | 29.46M | 6.65G | 5beaaa1c…（冻结） |

注：任务书中"优先训练 PIDNet-S / STDC-Seg / DDRNet23-slim"已于 9-30 完成，本轮**未重复训练**（重复训练既浪费算力也违反不覆盖纪律）。

新增 3 模型（U-Net / DeepLabv3+ / BiSeNetV2）于 2026-10-09 13:47 启动、10-10 凌晨全部完成（wall-time 分别约 14.5 / 12.7 / 13.5h，共享 GPU 并发）：

- 运行目录：`repro_runs/baseline_neu_240k_classical/{unet,deeplabv3p,bisenetv2}/`
- 配置：`experiments/baseline_comparison/configs/neu_{unet,deeplabv3p,bisenetv2}_240k.yaml`
- 脚本：`experiments/baseline_comparison/scripts/train_neu_{unet,deeplabv3p,bisenetv2}_240k.py`
- 协议：与先完成 5 个 baseline 完全一致（train3630 / 4类含背景 / from scratch / seed1337 / Adam lr1e-4 恒定 / wd2e-6 / batch16 / 240k 固定终点 / 训练中无 val/test / 无模型选择）
- 评价：`tools/evaluate_class_iou.py` 统一 840 全量 strict 评价（2026-10-10）

### ❌ 失败 / 排除

| 模型 | 结论 | 原因 |
|---|---|---|
| DMC-Net | 排除 | 官方仓自身缺 `dmcnet.py`（79 文件零命中，无 pyc），不可复现；证据链 `industrial_baseline_audit.md`；工业专用对比由 FDSNet 承担 |
| SeaFormer | 排除 | 依赖 mmcv（环境冲突）+ 迁移改动量大；通用实时层已有 4 个代表，边际收益低 |
| PIDNet 官方仓克隆 | 失败但不影响 | GitHub 网络不稳定（TLS 中断）；训练用内部实现 `new/pid.py`（结构一致性已验证），仓库 URL 与 commit 已固定于 `repo_clone_log.md` |
| PP-LiteSeg-B / "BiSeNetV1-L" 注册名 | 不纳入 | 评价链中这两个名字指向 DSMONet 变体文件（历史命名残留），非真实对应模型，不能当 baseline |

## 2. 推荐论文纳入列表（10 模型，四级结构）

1. **Classical**：U-Net、DeepLabv3+
2. **Real-time**：BiSeNetV2、STDC-Seg、PIDNet-S、DDRNet23-slim
3. **Industrial**：FDSNet、LETNet（\*归属待确认，见下）
4. **Ours**：DSMONet-B、A2MS-DSMONet-B

表格载体：`results/final_baseline_table_v2.md`（10 模型全部回填完毕）。

## 3. 待用户决策事项

1. **LETNet 分层归属**：官方出处为通用轻量实时分割（IEEE T-ITS 2023，Cityscapes/CamVid），非工业缺陷专用；当前按任务安排列入 Industrial 层并在表注标明，最终归属请拍板。
2. **论文 FPS 列口径**：维持冻结值 38.16/37.37，还是统一复测后整列刷新——复测计划已冻结于 `results/fps_rebenchmark_plan.md`，两种方案的表述都已备好。

## 4. 收口状态与剩余事项

~~训练完成后收口~~ 已于 2026-10-10 全部执行完毕：

1. ✅ 三模型 240k 终点 checkpoint → `tools/evaluate_class_iou.py` 统一评价（840 全量 strict）
2. ✅ `results/{unet,deeplabv3p,bisenetv2}_results.json` + 对应 `_audit.md`（sha256、配置、逐类 IoU、Params/MACs 齐备）
3. ✅ `final_baseline_table_v2.md` 已回填为 10 模型完整版

剩余事项：

4. ⏳ GPU 空闲窗口执行 `fps_rebenchmark_plan.md`（10 模型同批交错，含新增 3 模型）
5. ⏳ 用户确认后统一 git 提交（本轮未提交）
6. ⏳ 两个决策点待拍板：LETNet 分层归属；论文 FPS 列口径（冻结 38.16/37.37 vs 统一复测刷新）
