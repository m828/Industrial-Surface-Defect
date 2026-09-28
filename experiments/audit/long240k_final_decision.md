# long240k Final Decision (seed 1337, Protocol L1)

## Scenario verdict: **B — 两者都达到91%**

| Model   | endpoint mIoU (unified-840, fixed 240k) | ≥ 0.910000? |
| ------- | --------------------------------------: | :---------: |
| Base-B  |                                0.912408 |     YES     |
| A2 Full |                                0.914544 |     YES     |

ΔLong = **+0.002136** (+0.21 pt), A2 numerically ahead for this seed.

Per the frozen protocol, Scenario B reads: 历史 Base-like 长训练能力得到复现（在当前更严格的统一840固定终点协议下 Base-B 也越过 0.91），fixed A2 Full 在完全相同条件下同样达到 91%，且数值略高。单 seed，不得宣称 A2 稳定优于 Base-B。

## 历史91.x 与当前结果的口径差异（必须分开陈述）

| 维度 | 历史 0.913–0.916 | 本轮 |
| ---- | --------------- | ---- |
| 评价集 | test split 反复 monitor（832/840） | 统一 840，全程仅在终点正式评价 |
| 模型选择 | test-driven best checkpoint | 固定终点 iter240000（best=last by design） |
| 环境 | 旧 Python/PyTorch | py3.11.13 / torch 2.7.1+cu126 / A100 |
| 结论词 | historical reference | current unified fixed endpoint |

正确表述：历史 91.x 是 Base-like DSMONet 长训练能力的参考证据；本轮在**无 test 反馈、固定终点**的更干净协议下，Base-B (0.912408) 与 A2 (0.914544) 双双越过 0.91，A2 终点落在历史 0.913–0.916 区间内。不得写作"完全复现历史 0.9159"。

## 10k → 240k 证据链收敛

- 10k 2×2（3 seeds）：AAM 与 detail 均"暂不支持稳定增益"，Full 弱正向（+0.53 pt，2/3）；crazing/patches 方向一致小增益。
- 240k（1 seed）：Full − Base +0.21 pt；crazing 6/6 轨迹点为正；patches 终点为正但轨迹混合。
- 结论：长训练没有放大也没有逆转 10k 信号。AAM+detail 的优势维持在小幅正向区间，主要集中在 crazing。

## 唯一主建议：**B — 重点重新审视新增模块的论文价值**

依据：

1. Base-B 自身已达 0.912408。240k 下 Full 的 +0.21 pt 数值优势处于两模型轨迹间噪声带（|Δ| 全程 ≤ 0.8 pt）之内，单 seed 无法与随机性区分。
2. 10k 阶段已显示 AAM（2/3 正）与 detail（有 AAM 时 1/3 正）单独效应均不稳定；240k 未改变这一判断。
3. 若论文继续以 A2MS 组合为核心贡献，需要长训多 seed（≥3）才能支撑任何"consistently"级别表述——成本约为本轮的 3 倍（≈6 GPU·日），预期收益是验证一个 ~+0.2 pt 量级的差异。
4. crazing 的 6/6 轨迹一致性是全轮最强单点证据，可作为论文中 detail-sensitive 类别的机制性叙述素材，但不能替代 mIoU 层面的统计证据。

明确的下一步选项（不在本轮执行）：(a) 长训 3-seed 验证 A2 vs Base；(b) 将论文贡献重心从 mIoU 增益转向 crazing 类/边界质量/效率等其他可验证轴；(c) 接受 Base-B 即达 91% 的事实并简化方法叙述。三者择一需用户决策。

## 纪律确认

- 训练期间 0 次 test 评价；无 best-by-test；无 early stopping；预算 240k 未因任何结果调整
- 终点结果先于任何中间轨迹评价冻结（E1→E2→E3 顺序执行）
- 未修改论文、`fixed_existing_results.md`、模型文件、超参、seed；未启动任何额外训练
- INC-001（训练后 GPU/NVML 故障）仅延迟评价，已记录
