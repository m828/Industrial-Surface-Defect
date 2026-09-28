# Leather 历史 91.x 证据台账 (leather_historical_91x_evidence)

> 日期：2026-09-28
> 范围：Leather/Pige 数据集（2341 张，768×768，8 类含背景）历史 89.x–91.x 结果的来源取证。
> 原则：FACT = 有直接文件/日志/元数据支持；STRONG INFERENCE = 多证据一致但缺关键环节；UNCONFIRMED = 仅旧文本记录。
> 本轮未训练任何模型；所有统一评价基于 `experiments/protocol/leather_valid_test_list.txt`（468 张，768×768，8 类含背景，TestRescale+ToTensor，无 Normalize，batch=1，pred=outputs[0]，严格加载）。

---

## 一、证据总表

| # | 结果值 | 性质 | 来源 | 证据级别 |
|---|---|---|---|---|
| E1 | val235 best mIoU = **0.909084** | 历史训练监控值 | `new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl` 内嵌 `best_iou` 字段 | **FACT**（checkpoint 元数据） |
| E2 | test468 mIoU = **0.909713** | 本轮统一复评 | 同一 checkpoint + `new/model_dsmo_rs50_eSE_adapt_detailloss_fixed.py`（正确 forward），strict load 成功，manifest `hist_a2_160k_fixedfwd.json` | **FACT**（本轮实测） |
| E3 | test468 mIoU = 0.860749 | 旧复评（2026-06-16） | 同一 checkpoint + `new/model_dsmo_rs50_eSE_adapt_detailloss.py`（**含 `head_seg1 = high_feats` bug**） | **FACT**（旧审计，但走了错误 forward） |
| E4 | val235 best = 0.892961 | 历史训练监控值 | `dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl` 内嵌 `best_iou`，与 `new (copy)/runs_other/dsmonet_resnet_pige/dsmor50_pige/run_2024_01_27_14_05_37.log` 中 val 轨迹最大值 0.8929607（iter 78000）**逐位一致** | **FACT** |
| E5 | test468 mIoU = 0.899120 | 本轮复评复核 | 0127_80000 checkpoint + `new/model_dsmo_rs50.py`，与前序审计 0.899120 完全一致 | **FACT** |
| E6 | val235 best = 0.901785 | 历史训练监控值 | `dsmonet_resnet_pascal_pige_4_0229_160000.pkl` 内嵌 `best_iou`，与 `runs/dsmonet_resnet_detailloss/pige_4_0228/run_2024_02_29_21_55_15.log` 轨迹 max 一致 | **FACT** |
| E7 | test468 mIoU = **0.917625** | 本轮统一复评 | pige_4_0229_160000 + 822 架构（修复磁盘文件 debug `return [edge_laplacian]` 为 `return logit_list` 后），strict load 成功 | **FACT**（注意：模型为 822/Light_Bag 变体，非论文 A2MS-B 结构） |
| E8 | val235 best = 0.889354 → test468 = 0.890033 | 0120 eSE_adapt（无 detail head） | checkpoint 元数据 + 本轮复评 | **FACT** |
| E9 | val235 best = 0.872281 → test468 = 0.880614 | 0126 Base-B @55k | 同上 | **FACT** |
| E10 | val235 best = 0.827827 → test468 = 0.858620 | 0220ksh Base-B | checkpoint 元数据与 `new/runs_other/train_all/base_b_pige.log`（2026-05 A100，60k，cosine）max val 0.827827 逐位一致 | **FACT**（该 ckpt 实为 2026-05 A100 重训产物，文件名"0220"有误导性） |
| E11 | 大论文/旧稿 "A2MS-B Leather ≈ 91.0%" | 旧文本 claim | 服务器上**未找到**硕士大论文原稿（已搜索 /workspace 三层深度及 /root，无 docx/pdf 原稿） | **UNCONFIRMED**（文本层面）；但与 E1/E2 数值吻合（0.909 四舍五入为 91.0） |

---

## 二、历史 91% 的完整证据链（E1/E2 为主）

### 2.1 它是哪个 checkpoint

`new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl`
- SHA256：`336c5a4c2e5c1def67e34dc5061c5ed384829c8fdf318ce67b64f04e1a54f01f`
- `new/` 与 `new (copy)/` 根下两份拷贝 SHA256 完全相同。
- 内嵌元数据：`epoch=130000`（实际保存迭代为 130k，文件名 160000 是目标/系列名）、`best_iou=0.9090842792341525`、`scheduler_state._last_lr=[0.0001]`（**常数 LR**）。
- state_dict：512 keys，含 `seg_head_detailloss`(7)、`se.fc`(2)、`channel_weight`(1，即 AdaptiveChannelWeight)、`arm2`、`squeeze_body_edge` → **完整 A2 结构**（eSE + ACW + detail head）。

### 2.2 它是哪个模型 forward

- 该 checkpoint 结构匹配 `new (copy)/model_dsmo_r50_eSE_adapt_detailloss.py`，该文件 forward 为 **`head_seg1 = out`（正确）**。
- 当前 `new/model_dsmo_rs50_eSE_adapt_detailloss_fixed.py` 与该历史 r50 文件的差异仅为：注释、死代码行、`from stdcnet import STDCNet1446` 导入——**计算图完全一致**。
- 当前 `new/model_dsmo_rs50_eSE_adapt_detailloss.py`（bug 版，`head_seg1 = high_feats`）在 `new/` 与 `new (copy)/` 中字节相同；bug 版与历史 r50 版在同目录共存，bug 为后期文件整理时引入的变体。
- 结论：**历史 91% 对应的是正确 forward**；用 bug 版评价同一 checkpoint 得到 0.860749 是评价侧 bug 导致的低估。

### 2.3 它的 0.9091 是 validation 还是 test

**FACT：validation（val235）best。**
- 历史训练脚本族（`new (copy)/train_pige_dsmor50.py`、`train_pige_4_0228.py`、`train_pige_resnet_loss904.py` 等）只加载 `./dataset/pige/val.txt`（235 张）做监控，训练循环内无任何 test 加载。
- checkpoint 保存逻辑：`if score["Mean IoU"] >= best_iou: torch.save(...)` —— 只按 **val235 最优**保存。
- 0127 链证据闭合：log 中 val 轨迹 max=0.8929607@iter78000 与 checkpoint 内嵌 best_iou 逐位一致，checkpoint epoch 字段=78000。
- 训练时 val 的预处理（TestRescale+ToTensor，无 Normalize）与当前 test468 协议一致，因此 val→test 的数字可直接比较。

### 2.4 它的训练配方（FACT + STRONG INFERENCE）

- FACT（checkpoint/同族 log/config）：Adam lr=1e-4、wd=2e-6、**常数 LR（无 scheduler，`_last_lr=0.0001` 全程不变）**、val_interval=1000、输入 768×768、val-best checkpoint。
- FACT（同族 config `runs/dsmonet_resnet_detailloss/pige_4_0228/dsmonet_resnet_detailloss.yml` 等）：batch_size=12、loss 族含 `ohem_cross_entropy + bce_with_logits + ohem_cross_entropy + detail_aggregate_loss`。
- STRONG INFERENCE：训练脚本为 `train_pige_*_loss` 族（loss weights `[10,1,3,1]`，见 `train_pige_resnet_loss.py:143`）；**该 130k 链的完整训练日志未在本服务器找到**（可能在原 RTX3090 机器"61"上，仅 checkpoint 于 2025-05-18 拷贝至本机）。
- 迭代数：checkpoint epoch=130000；文件名 160000 表明计划目标为 160k，实际保存于 130k（val 最优点）。

### 2.5 数据划分

- 历史与当前使用同一套文件：`train.txt=1638 / val.txt=235 / test.txt=468`（`new (copy)/dataset/pige/`）。
- sha256：train `5fa37b63…`、val `142c66d1…`、test `449b0eb2…`。
- 跨 split 路径/内容重复：**0**（全部 2341 张图像内容哈希无重复）。
- 旧稿"val=468/test=235"为论文文本对调错误，数据文件本身从未对调。

---

## 三、各历史链 val→test468 对照（本轮实测）

| 链 | checkpoint | 结构 | 保存 iter | val235 best | test468 mIoU |
|---|---|---|---:|---:|---:|
| pige_4 (822/Light_Bag+eSE+detail) | pige_4_0229_160000 | 491 keys | 128000 | 0.901785 | **0.917625** |
| **A2MS-B（论文结构=r50/fixed forward）** | eSE_adapt_detailloss_160000 | 512 keys | 130000 | 0.909084 | **0.909713** |
| pige_4 (822) 中段 | pige_4_0228 | 491 keys | 74000 | 0.891943 | 0.900675 |
| Base-B | dsmor50_0127_80000 | 506 keys | 78000 | 0.892961 | 0.899120 |
| eSE_adapt（无 detail head） | 0120_pige_eSE_adapt | 505 keys | 43000 | 0.889354 | 0.890033 |
| Base-B 中段 | dsmor50_0126 | 506 keys | 55000 | 0.872281 | 0.880614 |
| Base-B（2026-05 A100 60k cosine 重训） | dsmor50_0220ksh | 506 keys | 56500 | 0.827827 | 0.858620 |

统一趋势：test468 普遍比 val235 best 高 0.3–1.6 pt（test 分布略易），两个集合口径一致、无泄漏。

---

## 四、明确排除的干扰项

1. `new/runs_other/train_all/a2ms_b_pige.log`（2026-05-20）：其脚本 `new/train_pige_a2ms_b.py` **第 32 行 `from model_dsmo822 import DSMONet`** —— 该"60k A2MS-B"重训实际训练的是 model_dsmo822（augnew 系列，val best 0.8560），**不是论文 A2MS-B**。文件名误导，不得引用。
2. `dsmonet_resnet_pascal_augnew_51500.pkl`：model_dsmo822 架构，非 A2MS-B。
3. `pige_4_0229_160000`（0.917625）：虽然当前测试最高，但为 822/Light_Bag 变体结构（`model_dsmo_rs50_eSE_adapt_detailloss_822.py`），与论文 A2MS-B 结构不同；且磁盘上该文件 forward 留有 debug `return [edge_laplacian]`（本轮仅在副本上修复后评价）。**不得作为论文 A2MS-B 的结果**。
4. 当前 60k 复现（Base 0.872137 / A2 0.849932）：固定终点、无 val 选择、cosine 调度——协议与历史不同，不能直接对比（见 diff 文档）。
