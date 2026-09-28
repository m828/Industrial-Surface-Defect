# 长训练准备度评估与协议设计（仅设计，本轮不执行任何长训练）

日期：2026-09-22
输入证据：2×2 factorial（`a2ms_2x2_factorial_analysis.md`）、类别级分析（`a2ms_2x2_classwise_analysis.md`）、历史取证（`historical_neu_91x_evidence.md`）

---

## 1. 要回答的问题

> **fixed A2MS-DSMONet-B（A2 Full）能否在统一 NEU-Seg 协议下达到 91% mIoU？**

已知约束（历史取证）：

- 历史 0.913–0.916 出现在 **165k–240k iters**（常数 LR、Adam 1e-4、batch16、Base-like 模型）；10k ≈ 0.857、20k ≈ 0.874、30k ≈ 0.882——**91.x 是长训练阶段产物，10k 的 85–86% 不代表长训终点**；
- 历史协议的 monitor 直接用 840 张 test 并按 test mIoU 保存 best——**新正式实验不得沿用此做法**。

## 2. 模型选择（一个主模型 + 一个必要对照）

### 主模型：A2 Full（`new/model_dsmo_rs50_eSE_adapt_detailloss_fixed.py`）

依据（2×2 证据）：

1. 它是论文当前声称的完整方法（AAM + detail supervision）——91% 问题指向的就是它；
2. 10k 均值四格最高（0.857451），ΔFull = +0.53 pt（2/3 正）；
3. 类别级：crazing 3/3 正（+0.87 pt）、patches 3/3 正（+1.10 pt），且**任何 seed 下缺陷类无退化**；
4. 无崩塌风险（最差 seed 仅 −0.26 pt，且由 inclusion 随机波动造成）。

不选 D-only 作主模型的理由：D-only 虽方向最稳（3/3），但它不含论文声称的 AAM；它已完成了了归因使命。不选 A1：缺少 detail，不能代表完整方法。

### 对照模型：Base-B（`new/model_dsmo_rs50.py`）

唯一必要对照：

- 历史 91.x 属于 Base-B 族模型——它是"91% 能否复现"的锚点；
- 它回答"新增模块在长训练下到底有无价值"：A2@240k vs Base@240k 是论文最终绕不开的比较；
- 不加第三、第四个长训 run（成本翻倍且 10k 已排除"某格明显更优"的可能）。

## 3. Scheduler 分析（为什么不能照搬 T_max=48000）

当前 10k 控制使用 cosine_annealing T_max=48000。若 240k 直接沿用：

- LR 在 48k 处降到 eta_min=1e-6，随后 **192k iters（80% 训练）贴在 LR 地板**——实际变成"先 cosine 衰减、后近似零学习率"的混合 schedule，既不等于历史常数 1e-4，也不是任何有依据的设计；
- 历史 91.x 的 LR 轨迹是**常数 1e-4 全程 240k**（历史 config 无 scheduler 衰减，见 `historical_training_chain_diff.md`）。

因此长训必须在下面两个候选中显式选择。

## 4. 候选协议（仅设计）

公共部分（锁死）：train 3630 / test 840；200×200；4 类含 background；train transform = Compose([Transforms_PIL(200,200), ToTensor])；test = TestRescale+ToTensor，无 Normalize；batch16；Adam lr 1e-4 wd 2e-6；from scratch；两模型同 seed（matched）；seed 写入顶层 `seed` 键；非覆盖 run 目录 + manifest（同本轮规范）。

### Protocol L1：Historical-like（推荐，见第 6 节）

- scheduler：**常数 LR = 1e-4，全程 240k**（复刻产出 91.x 的真实历史 recipe）；
- train_iters = 240000；
- 用途：直接回答"在接近历史条件下 fixed 模型能否达到 91%"；与历史曲线（10k 0.857 / 20k 0.874 / 30k 0.882 / 240k 0.913+）逐点可比。

### Protocol L2：Modern controlled

- scheduler：cosine_annealing，**T_max = 240000**（schedule 匹配总预算），eta_min = 1e-6；
- train_iters = 240000；
- 用途：若论文主结果需要更现代的 schedule；作为 L1 的敏感性对照（资源允许时的第二批）。

### 明确排除

- T_max=48000 延用到 240k（第 3 节）；
- 任何 20k/30k/60k"先试试"式中间实验（历史已给出这些点的参考值，单独跑它们回答不了 91% 问题）。

## 5. 长训评价规则（比历史更干净，预注册）

1. **固定预算 240k，不用 test 做模型选择**：不根据 test mIoU 保存 best、不 early stop、不因 test 曲线改任何超参；
2. checkpoint 只按**预注册迭代点**保存：iter 10000 / 30000 / 60000 / 120000 / 240000（schedule 驱动，非 test 驱动）；
3. **正式 840 test 评价只对 iter240000 终点 checkpoint 做一次**（统一脚本、batch1、TestRescale+ToTensor、无 Normalize、pred=outputs[0]、含 background）；
4. 预注册中间 checkpoint 可做**描述性**轨迹评价（10k/30k/60k/120k），必须在报告中声明为 descriptive trajectory，不得用于选 checkpoint、不得与历史"best"比较时混用；
5. 训练过程监控只用：train loss、LR、梯度范数（GradAudit 采样）；840 test 不得在训练中途出现；
6. seed：资源约束下**单 seed**（两模型 matched），在报告中显式声明此局限；若预算允许扩到 2–3 seeds，seed 列表须在启动前冻结；
7. 若训练发散（loss NaN / mIoU 崩塌），按预注册规则终止并记录，不得换 seed 重跑凑数。

## 6. 建议（对应任务书第 23 节第 9 问）

| 项 | 建议 | 依据 |
|---|---|---|
| 主长训模型 | **A2 Full** | 论文完整方法；10k 四格均值最高；缺陷类 3/3 无退化 |
| 对照模型 | **Base-B**（仅此一个） | 91.x 所属族；回答模块长训价值 |
| 推荐总 iteration | **240000** | 91.x 出现于 165k–240k；预算必须覆盖该区间才能回答原问题 |
| 推荐 scheduler | **Protocol L1：常数 LR 1e-4** | 91% 问题本质是历史可比性问题；常数 LR 是产出 91.x 的真实 recipe；L2 列为可选敏感性协议 |
| 是否值得投入长训 | **值得** | 见下 |

"值得"的依据：

1. 10k 2×2 已排除结构性风险（forward bug 修复；四格无崩塌）；
2. Full 呈弱正向趋势且缺陷类信号方向一致（AAM→crazing、Detail→patches）——长训是唯一可能放大这些信号的实验；
3. 10k 差异在噪声量级是**预期内**的：历史 Base-like 在 10k 也只有 0.857，91.x 全部来自 10k 之后的 230k iters——10k 结果对 240k 终点几乎没有否证能力；
4. 不跑长训，"能否达到 91%"永远无法回答，论文实验表也只能停在 10k 诊断协议。

同时如实声明风险：

- 10k 证据显示 AAM 与 detail 的增益**不稳定且组合不可加**——即使长训，A2 相对 Base-B 的终点优势也可能不显著；论文必须接受"A2≈Base-B 长训相当"的可能结局；
- 单 seed 240k 的结论强度有限；
- 历史 91.x 的复现本身存在环境差异风险（py3.9→3.11、torch 版本、无 checkpoint 留存）。

## 7. 资源估算

- 10k iters ≈ 71 min（本轮实测，含 GPU 共享）→ 240k ≈ 28–29 h/模型；A2 + Base-B 两 run 并行 ≈ 29–32 h（独占时更快）；
- checkpoint：354 MB × 5 个预注册点 × 2 模型 ≈ 3.6 GB；
- 正式评价：每模型终点 1 次 ×~25 s；描述性轨迹 4 次 ×~25 s。

## 8. 启动前检查单（下一轮执行时逐项勾选）

- [ ] 用户明确批准长训（本轮红线：不跑 20k/30k/60k/240k）
- [ ] seed 冻结并写入两模型顶层 config
- [ ] run 目录非覆盖、manifest 预生成
- [ ] config 中 val_interval 关闭或 = 240000（不启用 test monitor）
- [ ] 预注册 checkpoint 点写入训练脚本（schedule 驱动）
- [ ] 训练脚本 GradAudit 采样保留（只读）
- [ ] 确认不会写入 `new/model_savePath` 或任何历史目录
