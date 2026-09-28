# 大论文 DSMONet-B 90.2% 来源取证 (neu_historical_90_2_evidence)

> 日期：2026-09-28
> 置信标记：FACT / STRONG INFERENCE / UNCONFIRMED

## 一、大论文原文

**UNCONFIRMED（文本层）**：服务器上未找到硕士大论文原稿（PDF/Word/LaTeX 均不存在，两轮搜索确认）。"DSMONet-B / ResNet-50 / 90.2%"来自用户对大论文表格的转述，表号、指标定义、上下文无法直接核对。

## 二、90.2 的最可能实验来源

**FACT（日志层）**：历史 NEU 训练链的 test-monitor mIoU 在 52k–60k 迭代区间反复出现 0.902x：

| 记录 | 值 | 位置 |
|---|---|---|
| 0.90283（detail 链 @60k，随后 checkpoint 重命名 `_detailloss_9028.pkl`） | 0.9028278… | `subregion unet/runs/dsmonet_resnet_detailloss/all_img/run_2023_08_21_10_54_21.log:1753` |
| 0.90291 / 0.90294 / 0.90273 / 0.90229…（非 detail 链 52k 起多个 monitor 点） | 0.9022–0.9029 | `subregion unet/runs/dsmonet_resnet/all_img/run_2023_08_11_14_49_33.log` 多处 |
| 0.9034（非 detail 链 checkpoint 重命名 `_9034.pkl`，52k） | 0.9034 | 同上日志 resume 记录 |

**STRONG INFERENCE**：大论文 90.2% = 上述历史链在约 52k–60k 迭代阶段的 test-monitor mIoU 的截断/约写（0.902x → 90.2）。依据：①数值区间精确重合；②checkpoint 命名证实当时以 monitor 值标记模型；③同一架构的更短训练轨迹（当前复现 60k = 0.9007）也落在 90.0–90.3。

**FACT（checkpoint 层）**：现存历史 Base-B 结构权重 `new/model_savePath/dsmonet_resnet_pascal_0110_neu_120000.pkl`（506 keys，无 eSE/ACW/detail head）内嵌 best_iou=0.90472（保存于 iter 50500；文件名 120000 为计划目标）。本轮在当前统一 840 测试协议下复评该权重：**mIoU = 0.903864**（strict 加载成功；manifest：本轮 neu_current_base_9124_manifest.json 同法生成，CSV 见 `repro_runs/leather_historical_audit/neu_0110_base_unified840.csv`）。

## 三、90.2 的口径

| 维度 | 判定 | 依据 |
|---|---|---|
| 指标 | mIoU，4 类含背景 | 历史 runningScore(4)（FACT） |
| 评价集 | NEU-Seg 测试集（840 张列表） | 历史 val_loader 直接读 `test_neu.txt`（FACT） |
| 每次实际评价张数 | **832/840**（batch16 + drop_last=True，训练中 monitor） | 训练脚本（FACT） |
| best 还是 final | **monitor 最优**（test-driven best）——checkpoint 按 monitor 峰值重命名 | 命名 + 日志（FACT） |
| 训练阶段 | 约 52k–60k 迭代 | 日志迭代号（FACT） |
| 对应 checkpoint | 9028/9034/0110 系列；9028/9034 原文件已丢失，0110 在盘并可复评 | FACT |

## 四、负面证据

- 未发现任何 90.2 对应的独立 test 终评脚本输出；90.2 不属于 fixed-endpoint 评价。
- 历史 9130/9145（91.3–91.6）是同一架构在 165k–240k 阶段的 monitor 峰值，与 90.2 不属同一训练阶段，不得混用。
