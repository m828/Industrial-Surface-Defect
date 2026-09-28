# Detail-only 模型静态审计（2×2 factorial 第四格）

日期：2026-09-14
作者：Kimi Code 审计链
状态：训练前静态审计（in-training GradAudit 数值在训练完成后补充于文末）

---

## 0. 审计对象

| 项 | 值 |
|---|---|
| 新模型文件 | `new/model_dsmo_rs50_detailloss_only.py` |
| 模型 SHA256 | `473518197a704af468b5dd388f007f0b46e9a73cdcece0f9cd65b642683fda42` |
| 训练脚本 | `train_detail_only_10k.py`（各 run `code/` 内，SHA256 `1294e8921d86a255c8773e4ad5b17f2e64c7dbb2d8acf9b8714136294542b5dd`） |
| 探针脚本 | `repro_runs/multiseed_10k/seed_1337/detail_only/analysis/static_probe.py` |
| 探针原始输出 | 同目录 `static_probe_output.json` |

D-only 定义：**Base-B 架构（SELayer、无 eSE/AdaptiveChannelWeight）+ 与 A2 完全相同的第四路 detail head 与 detail supervision**，loss_weights `[10, 1, 3, 3]`。

---

## 1. Detail-only vs Base-B（`new/model_dsmo_rs50.py`，SHA256 `fab316c5…`）

逐行 diff 结果——**全文件仅 3 行差异**：

```diff
35c35
<         # self.seg_head_detailloss = SegHead(128, 64, 1)
---
>         self.seg_head_detailloss = SegHead(128, 64, 1)
116,117c116,117
<             # x4 = self.seg_head_detailloss(edge_laplacian)
<             # logit_list.append(x4)
---
>             x4 = self.seg_head_detailloss(edge_laplacian)
>             logit_list.append(x4)
```

state_dict key 集合比较（probe 实测）：

- `D-only − Base-B` = 仅 7 个 key，全部属于 `seg_head_detailloss.*`（conv._conv.weight、conv._batch_norm.{weight,bias,running_mean,running_var,num_batches_tracked}、conv_out.weight）；
- `Base-B − D-only` = **空集**。

即除新增第四路 detail head 及其训练态输出外，D-only 与 Base-B 计算图逐项一致：SELayer、DAPPM、SqueezeBodyEdge、UAFM arm1/arm2、edge_fusion、conv_up、SegHead、backbone 全部相同。

参数量：Base-B 29,370,629 → D-only 29,444,549（+73,920，即 SegHead(128,64,1) 的参数量）。

---

## 2. Detail-only vs A2（`model_dsmo_rs50_eSE_adapt_detailloss_fixed.py`，SHA256 `a0e6fd52…`）

state_dict key 集合比较（probe 实测）：

- `D-only − A2` = `se.conv1.{weight,bias}`、`se.conv2.{weight,bias}`（SELayer(128, reduction=128)）；
- `A2 − D-only` = `se.fc.{weight,bias}`、`se.channel_weight.weight`（eSE + AdaptiveChannelWeight）。

**唯一结构差异就是 `self.se` 模块**：D-only 用 Base-B 的标准 SELayer，A2 用 eSE+ACW（论文所称 AAM）。第四路 detail branch 二者完全相同：同一 `SegHead(128,64,1)`、同一输入 `edge_laplacian`（56×56）、同一 `detail_aggregate_loss`（Laplacian 3×3 kernel、阈值 0.1、×1/×2/×4 金字塔 fuse_kernel [0.6,0.3,0.1]、BCE+Dice）、同一权重 3。

参数量：D-only 29,444,549 vs A2 29,460,804（差 16,255 = eSE/ACW 与 SELayer 的参数差）。

---

## 3. 主输出正确性（forward regression 不复现）

`model_dsmo_rs50_detailloss_only.py:98`：

```python
high_feats = high_feat + seg_body + high_feat_se   # 7×7
high_feat = self.conv_up(high_feats)               # 7×7 → 56×56
out = self.arm2(seg_edge, high_feat)               # 56×56 detail-semantic fusion
head_seg1 = out                                    # ✓ 与 Base-B / fixed 一致
```

已逐行确认不存在 `head_seg1 = high_feats`。主输出路径：

```text
x32(7×7) → DAPPM/cm → arm1 → high_feat(7×7)
  → SqueezeBodyEdge → seg_body/seg_edge
  → high_feats = high_feat + seg_body + se(high_feat)   (7×7)
  → conv_up → 56×56
  → arm2(seg_edge[edge_fusion 后], high_feat) → out(56×56)
  → head_seg1 = out → seg_heads → output0 (200×200)
```

---

## 4. Tensor shape（probe 实测，batch=2，输入 200×200，train mode）

| tensor | shape |
|---|---|
| cm (DAPPM) 输出 | 2×128×7×7 |
| arm1 输出 high_feat | 2×128×7×7 |
| squeeze_body_edge 输出 (seg_body, seg_edge) | 2×128×7×7 / 2×128×7×7 |
| bot_fine(edge_shallow) | 2×128×50×50 |
| laplacian(edge_shallow) | 2×128×50×50 |
| conv_up ×3 次调用（seg_edge / se(high_feat) / high_feats） | 均 2×128×56×56 |
| edge_fusion 输出 seg_edge | 2×128×56×56 |
| arm2 输出 out | 2×128×56×56 |
| seg_heads(head_seg1) / (head_seg2) / (head_seg3) | 2×4×56×56 / 2×4×56×56 / 2×4×7×7 |
| seg_head_detailloss(edge_laplacian)（edge_laplacian 已插值到 56×56） | 2×1×56×56 |
| **train outputs ×4** | 2×4×200×200 ×3 + **2×1×200×200**（detail） |
| **eval output** | 2×4×200×200（仅 output0） |

与 Base-B / fixed 模型的既有审计 shape 完全一致（除新增的第 4 路 detail 输出）。

---

## 5. Gradient 验证（probe 反向传播实测，seed=1337 初始化）

**Test A — loss = outputs[0].sum()**（验证主输出路径）：

| 模块 | grad norm | grad=None 的张量数 |
|---|---:|---:|
| arm2 | 86515.00 | 6 |
| se (SELayer) | 2.41 | 0 |
| edge_fusion | 170877.60 | 0 |
| seg_heads | 494192.48 | 0 |
| seg_head_detailloss | 0.0 | **4（全部）** |
| arm1 | 2336.99 | 0 |
| conv_up | 531.36 | 0 |

→ arm2 真实参与 output0 的反向传播（非 dead branch）；detail head 不在 output0 路径上（符合预期）。

**Test B — loss = Σ w_j·outputs[j].sum()，w=[10,1,3,3]**（模拟训练 loss 路由）：

| 模块 | grad norm | grad=None 的张量数 |
|---|---:|---:|
| arm2 | 933171.61 | 6 |
| se (SELayer) | 21.74 | 0 |
| edge_fusion | 1900503.61 | 0 |
| seg_heads | 6995047.18 | 0 |
| **seg_head_detailloss** | **747019.41** | **0** |
| arm1 | 27741147.87 | 0 |
| conv_up | 5935.87 | 0 |

→ 第四路 detail loss 开启后 detail head 获得真实梯度。

**关于 arm2 的 6 个 grad=None 张量**：UAFM(128,128,tokens=False) 的 forward 走 `fuse_onlys`，只使用 `conv_x`/`conv_xy_atten2`/`conv_out`；`conv_xy_atten1`（2 conv 权重 + 2 BN weight + 2 BN bias = 6 个张量）在 tokens=False 下不参与计算。该行为与 Base-B、fixed 模型完全相同（同一 UAFM 实现），不是 D-only 引入的问题。

---

## 6. 训练 loss 路由（与 A2 完全一致）

训练脚本 `train_detail_only_10k.py` 与 A2 的 `train_fixed_a2ms_b_10k.py` 仅 2 处差异：

1. import 改为 `model_dsmo_rs50_detailloss_only`；
2. GradAudit 模块组改为 `arm2 / SELayer / edge_fusion / seg_heads / detail_head`（SELayer 无 `channel_weight` 属性，原 eSE/ACW 条目会 AttributeError）。

loss 公式不变（zip 路由，j<3 双项、j=3 仅 detail 项）：

```text
L_total = Σ_{j=0..2} w_j · [ L_j(2ch_fg_bg(output_j), label1) + L_j(output_j, labels) ]
        + w_3 · detail_aggregate_loss(output_3, labels)
w = [10, 1, 3, 3]
L_0 = L_2 = ohem_cross_entropy, L_1 = bce_with_logits_loss
```

detail target 在 loss 内由 labels 即时生成（Laplacian 3×3、clamp(min=0)、阈值 0.1、stride 1/2/4 金字塔、fuse_kernel [0.6,0.3,0.1]、BCE+Dice），与 A2 使用的 `tools/loss/loss.py::detail_aggregate_loss` 为同一函数。

---

## 7. 协议锁死确认

| 项 | 值 |
|---|---|
| train/test | 3630 / 840（`new (copy)/dataset/train_neu.txt`、`test_neu.txt`） |
| 输入 | 200×200，4 类含 background |
| transform | train: Compose([Transforms_PIL(200,200), ToTensor])；test: TestRescale+ToTensor，无 Normalize |
| optimizer | Adam lr=1e-4 wd=2e-6 |
| scheduler | cosine_annealing T_max=48000 eta_min=1e-6 |
| iters / batch | 10000 / 16 |
| seed 读取 | 顶层 `seed` 键（`cfg.get("seed", 1337)`），三配置均已写顶层 `seed: 1337/2026/3407` |
| val_interval | 10000（仅终点 monitor；正式结果为统一 840 终点评价，best==last by design） |
| loss_weights | [10, 1, 3, 3]（detail 权重=3，不调） |

---

## 8. In-training GradAudit（训练完成后补记）

三个 matched seed 训练过程中 iter 1 / 100 / 1000 的模块梯度范数（取自各 run `logs/detail_only_s{seed}_10k_codex1/run_*.log`）：

| seed | iter | arm2 | SELayer | edge_fusion | seg_heads | detail_head |
|---|---:|---:|---:|---:|---:|---:|
| 1337 | 1 | 31.029 | 0.000068 | 28.315 | 56.078 | 4.401 |
| 1337 | 100 | 21.469 | 0.008012 | 151.105 | 25.609 | 2.079 |
| 1337 | 1000 | 17.224 | 0.076499 | 68.249 | 30.632 | 4.143 |
| 2026 | 1 | 33.401 | 0.000902 | 157.109 | 46.787 | 3.842 |
| 2026 | 100 | 11.753 | 0.002448 | 23.201 | 19.667 | 1.958 |
| 2026 | 1000 | 19.431 | 0.070025 | 38.590 | 23.098 | 2.521 |
| 3407 | 1 | 47.395 | 0.000316 | 152.708 | 61.328 | 5.497 |
| 3407 | 100 | 44.667 | 0.014245 | 93.699 | 35.230 | 2.513 |
| 3407 | 1000 | 27.289 | 0.295526 | 39.068 | 35.922 | 11.132 |

结论：

1. arm2 在全部三个 seed 的全部采样点获得真实非零梯度——主输出融合分支非 dead branch；
2. detail_head 在 detail supervision 开启下梯度量级（约 2–11）与其余模块同数量级，无 gradient explosion；
3. SELayer 梯度正常；
4. 三个 seed 的 iter-1 梯度指纹互不相同（如 arm2：31.03 / 33.40 / 47.39；seg_heads：56.08 / 46.79 / 61.33），佐证顶层 `seed` 键实际生效（effective seed = requested seed）。
