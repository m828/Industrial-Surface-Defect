# Leather 正式结果溯源 (leather_result_provenance_v4)

> 日期：2026-09-28
> 目的：清楚区分"历史 val 分数 / 历史 checkpoint / 当前统一 test468 复评 / 论文报告值"，防止混用。

## 一、Base-B

| 层级 | 值 | 性质 |
|---|---|---|
| 历史 val235 best（训练监控） | 0.8929607 | 监控集分数，checkpoint 内嵌 `best_iou`；与 `run_2024_01_27_14_05_37.log` val 轨迹 max 逐位一致 |
| 历史 checkpoint | `new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl` | sha256 `a96c78ea…`；文件名 80000，实际保存 iter=**78000**（val-best 截获点） |
| 当前统一 test468 复评 | **0.899120** | 本轮审计实测（2026-06-15 初测、2026-09-28 复核一致） |
| **论文报告值** | **89.91%** | = 当前统一复评值，非历史 val 数字 |

## 二、A2MS-DSMONet-B

| 层级 | 值 | 性质 |
|---|---|---|
| 历史 val235 best（训练监控） | 0.909084 | checkpoint 内嵌 `best_iou`；旧稿"约91%"即此数的四舍五入 |
| 历史 checkpoint | `new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl` | sha256 `336c5a4c…`；文件名 160000，实际保存 iter=**130000** |
| 模型结构确认 | `new (copy)/model_dsmo_r50_eSE_adapt_detailloss.py`（`head_seg1 = out`） | 与 `new/model_dsmo_rs50_eSE_adapt_detailloss_fixed.py` 计算图一致；512 keys strict load |
| 旧审计伪低值 | 0.860749 | 同一权重经 bug forward（`head_seg1 = high_feats`）评价，弃用 |
| 当前统一 test468 复评 | **0.909713** | 本轮审计实测（fixed forward），strict load |
| **论文报告值** | **90.97%** | = 当前统一复评值 |

## 三、相对差

Δ(A2 − Base) = 0.909713 − 0.899120 = **+0.010593（+1.06 个百分点）**。

## 四、口径声明（正文必须体现）

- NEU-Seg：固定 240k 迭代训练终点，test840 终点一次性评价，训练中不使用 val/test 做模型选择。
- Leather：沿用历史训练协议，train1638 训练、val235 选择 checkpoint、test468 独立最终评价一次。
- 两个数据集均**不使用测试集进行模型选择**；Leather 的 checkpoint 选择集是 validation，不是 test。

## 五、明确不进入论文的数字

| 数字 | 原因 |
|---|---|
| 0.909084（A2 val235 best） | validation 监控值，非测试结果 |
| 0.917625（pige_4_0229_160000） | 822/Light_Bag 变体，非论文 A2MS-DSMONet 结构 |
| 0.860749 | bug forward 评价伪低值 |
| 87.21% / 84.99%（60k 复现） | 固定终点短训协议，仅存档于 audit |
| −2.22 pt / rotten −12.46 pt | 同上，60k 协议产物，不代表历史链能力 |
