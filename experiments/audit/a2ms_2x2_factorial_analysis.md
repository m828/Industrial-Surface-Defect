# A2MS-B 2×2 factorial 分析（matched seeds 1337 / 2026 / 3407）

日期：2026-09-22
数据：统一 840 终点评价（10k、batch16、Adam 1e-4、cosine T_max=48000、含 background；best==last by design，无 test 选择）
四格定义：B = Base-B（无 AAM / 无 detail）；A = A1（AAM / 无 detail）；D = Detail-only（无 AAM / detail）；AD = A2 Full（AAM / detail）

---

## 1. 完整 2×2 结果表（mIoU）

| Seed | Base-B | A1 (AAM-only) | D-only (Detail-only) | A2 (Full) |
|---|---:|---:|---:|---:|
| 1337 | 0.854558 | 0.857703 | 0.855020 | 0.862727 |
| 2026 | 0.847473 | 0.860870 | 0.852310 | 0.857755 |
| 3407 | 0.854493 | 0.852281 | 0.855968 | 0.851870 |
| **mean ± SD** | **0.852175 ± 0.004072** | **0.856951 ± 0.004343** | **0.854433 ± 0.001898** | **0.857451 ± 0.005435** |

D-only 是四格中方差最小的（SD 0.19 pt，约为 Base-B 的一半）。

## 2. 五类效应（paired，逐 seed）

| 效应 | 1337 | 2026 | 3407 | mean ± SD | 正向 |
|---|---:|---:|---:|---:|---:|
| **1. ΔA_noD = A − B**（AAM 主效应，无 detail 背景） | +0.003145 | +0.013397 | −0.002212 | **+0.004777 ± 0.007932** | 2/3 |
| **2. ΔD_noA = D − B**（Detail 主效应，无 AAM 背景） | +0.000462 | +0.004838 | +0.001475 | **+0.002258 ± 0.002291** | **3/3** |
| **3. ΔD_withA = AD − A**（Detail 效应，有 AAM 背景） | +0.005024 | −0.003114 | −0.000411 | **+0.000500 ± 0.004145** | 1/3 |
| **4. ΔA_withD = AD − D**（AAM 效应，有 detail 背景） | +0.007707 | +0.005445 | −0.004098 | **+0.003018 ± 0.006265** | 2/3 |
| **5. Interaction = AD − A − D + B** | +0.004562 | −0.007952 | −0.001886 | **−0.001759 ± 0.006258** | 1/3 |
| 参考：ΔFull = AD − B | +0.008169 | +0.010283 | −0.002623 | +0.005276 ± 0.006922 | 2/3 |

## 3. 判读

### 3.1 Detail 主效应（ΔD_noA）：方向稳定、幅度小

- 3/3 seeds 为正（+0.05 / +0.48 / +0.15 pt），是五个效应中**唯一方向完全一致**的；
- 但均值 +0.23 pt < Base-B seed 间波动（SD 0.41 pt），3 seeds 下 sign test p = 0.25，统计功效不足；
- 结论：**detail supervision 在无 AAM 时呈一致的小幅正向信号，但幅度低于 run 噪声量级，不能宣称“显著提升”**。

### 3.2 AAM 主效应（ΔA_noD）：均值最大、方向不稳定

- mean +0.48 pt 为各单模块效应中最大，但 seed3407 为负（−0.22 pt），2/3；
- SD 0.79 pt > 均值，方向一致性不足；
- 结论：**AAM 暂不支持稳定增益**（维持第三轮结论），也无明显负向证据。

### 3.3 Detail 在 AAM 存在时（ΔD_withA）：增益消失

- mean +0.05 pt，1/3 为正；
- 与 ΔD_noA（3/3 正）形成鲜明对照：**detail 的小幅增益在加入 AAM 后不再出现**。

### 3.4 AAM 在 detail 存在时（ΔA_withD）

- mean +0.30 pt，2/3——与 ΔA_noD 一样不稳定；AAM 的效应不因 detail 存在而变稳定。

### 3.5 Interaction：均值弱负、方向不一致

- mean −0.18 pt，1/3 为正；方向不一致，**不能宣称稳定的负交互**；
- 但结构上值得注意：AD − A ≈ 0 且 D − B > 0（3/3），意味着两个模块的增益**不可加**——组合后并没有把 D-only 的一致小增益带进 Full；
- 模式符合“增益互相吸收/抵消”的弱迹象，但证据强度不足以定论。

## 4. 冻结决策规则对照（任务书第十六节）

| 情况 | 条件 | 本轮数据 |
|---|---|---|
| A（Detail 强于 AAM） | D>B 多数/全部 且 A≈B | 部分吻合：D>B 3/3 ✓；A−B 2/3、mean +0.48pt（"≈B"勉强） |
| B（Detail 无价值） | A>B 稳定 且 D≈B、AD≈A | 不吻合：A>B 不稳定 |
| C（Full 最强支持） | A>B、D>B、AD>A,D 多数一致 | 不吻合：AD−A 仅 1/3 |
| D（整体无价值） | A≈B、D≈B、AD≈B | 不吻合：D>B 3/3、AD>B 2/3 |
| **E（Detail 有效但与 AAM 负交互）** | A≈B、D>B、AD≈A 且 interaction<0 | **最接近**：A≈B（均值正但不稳定）✓-；D>B（3/3）✓；AD≈A（mean +0.05pt）✓；interaction 均值 −0.18pt（方向不一致）✓- |

**本轮数据最接近情况 E 的弱版本**：detail supervision 自身有一致的小幅增益；AAM 增益方向不稳定；二者组合后 detail 的增益消失，interaction 均值略负但不稳定。

## 5. 结论（只对证据负责）

1. **Detail supervision**：单独使用时方向一致为正（3/3），但效应量（+0.23 pt）低于 Base-B 的 seed 噪声（0.41 pt）；在 AAM 存在时增益消失（1/3）。判定：**弱正向信号，暂不足以宣称稳定有效**。
2. **AAM（eSE + AdaptiveChannelWeight）**：均值正向最大但不稳定（2/3，SD > mean）。判定：**暂不支持稳定增益**。
3. **Full A2MS-B vs Base-B**：mean +0.53 pt，2/3，最差 seed −0.26 pt——整体与 Base-B 基本相当、呈弱正向趋势，无崩塌风险。
4. **组合不可加**：两个模块各自的（不稳定/小）增益在组合中不叠加，interaction 均值略负。
5. 10k 协议下没有任何变体显著优于 Base-B；所有差异都在 run-to-run 噪声的 1–2 倍以内。10k 阶段的作用是**筛查结构问题**（已完成：forward bug 已排除），而不是证明模块价值——模块价值需要在更长训练预算下重新评估（见 `long_training_readiness.md`）。

## 6. 数据来源

- B/A/AD 三格：`a2ms_multiseed_10k_results.md` 及其 manifests（1337 复用、2026/3407 新建，协议一致性已核验）；
- D 格：`detail_only_multiseed_results.md` 及 `detail_only_seed{1337,2026,3407}_manifest.json`；
- 计算脚本：`repro_runs/multiseed_10k/analyze_2x2.py`（确定性重算，输出 `/tmp/2x2_analysis.json`）。
