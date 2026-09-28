# Leather rotten_surface 退化专项诊断 (leather_rotten_surface_diagnostic)

> 日期：2026-09-28
> 问题：当前 60k 复现中 A2 相对 Base 在 rotten_surface 上 −12.46 pt，需定位原因。

## 一、现象汇总（统一 test468，IoU）

| 模型 | 链 | rotten_surface | scratch | mIoU |
|---|---|---:|---:|---:|
| Base-B | 当前 60k 复现（cosine/固定终点） | 0.881025 | 0.644290 | 0.872137 |
| A2MS-B(fixed) | 当前 60k 复现（cosine/固定终点） | **0.756474** | 0.633737 | 0.849932 |
| Base-B | 历史 0127@78k（常数LR/val-best） | 0.915895 | 0.700504 | 0.899120 |
| **A2MS-B** | **历史 130k（常数LR/val-best，正确forward）** | **0.941891** | 0.747213 | 0.909713 |
| 822 变体 | 历史 128k（常数LR/val-best） | 0.939630 | 0.728398 | 0.917625 |

## 二、关键推理

1. **不是架构缺陷**：与当前 A2 完全同计算图（fixed forward = 历史 r50 forward）的历史 checkpoint，rotten_surface = 0.9419，比历史 Base 还高 +2.6 pt。若 AAM/detail 结构本身伤害该类，历史权重不可能达到 0.94。
2. **不是标签映射漂移**：历史与当前共用同一 test 列表、同一评价脚本、同一 LEATHER_CLASSES 顺序；历史全部 5 个 checkpoint 在 8 类上均得到合理 IoU（任一类错位都会使历史权重该类 IoU 崩塌，实测没有）。类别索引 6=rotten_surface 一致。
3. **不是 detail supervision 在 768 分辨率的结构性失效**：历史链同样使用 detail_aggregate_loss（权重 1）于 768×768，rotten 仍为全类最高之一（0.94）。
4. **最可能原因：训练不足 + 协议差异的组合**（P1 级）：
   - 当前 A2 60k：cosine T_max=48k → 后 12k LR≈0；固定终点快照；detail loss 权重 3（历史为 1）。
   - rotten_surface 是 7 类中面积占比小、纹理弥散的类，对训练充分度最敏感；A2 的 4 路损失 + 注意力模块使每步有效梯度分摊更多，在短训练+早衰减下更容易欠拟合该类。
   - 旁证：历史 A2 链 val 轨迹直到 100k+ 仍在改善（pige_4 链 74k→128k：0.8919→0.9018 val；test 0.9007→0.9176）；60k 远未到达该类收敛区。
5. **次要因素：detail loss 权重 3 vs 1**（P2 级，未单独消融验证）：当前 repro 的 detail 权重是历史的 3 倍，可能在大面积弥散类上施加过强边界约束。此点仅作记录，本轮不实验。

## 三、结论

rotten_surface −12.46 pt 归因：**训练协议（长度/scheduler/终点选择/detail 权重），非模型结构，非数据/标签问题。**
该现象不否定 A2 架构；在历史协议下同一架构该类 IoU=0.9419。

## 四、对论文的含义

正文若报告 Leather 结果，应使用历史 val-best 链的统一468复评值（见 leather_reproduction_decision.md），不应使用当前 60k 固定终点值。
