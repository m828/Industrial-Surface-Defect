> 阶段说明：以下为数据冻结前的历史准备记录。当前数据已交付KSDD2_DATA_FROZEN_V1；请以KSDD2_SOURCE_INTEGRATION_REPORT.md和KSDD2_SERVER_RUN.md为准。不要再按本文旧候选640×256输入或旧待冻结流程执行。

# KSDD2代码适配验收

日期：2026-11-09。基底commit `8abd026bb6b0fce051196379604f9c0ccabd6aac`，已读取远端研究分支确认一致。本地准备位于同一Git仓库的独立 `.ksdd2-prep-worktree`；本轮不提交或推送。E0/E1-A/E1-B/E1-C工作树与实验资产不修改。

## 直接回答

1. **无需修改模型源码。** 工厂build_model(variant,num_classes=2,seed=...)已经支持四种条件；三次共享SegHead调用同步输出2类，DetailHead维持1通道，transport输入/公式/305参数不变。
2. **二类结构兼容；历史损失有数值和语义条件。** 历史mask-aware等同重复二类分割三路，严格空hard集合仍NaN；不能直接把“能调用”当成已批准新数据集训练协议。新增显式候选任务损失，不修改历史损失。
3. **可做组级开发划分。** 已实现纯元数据分组/一次划分函数和合成测试；没有真实KSDD2文件扫描、真实清单或阈值。
4. **12次矩阵合理，费用尚不可确认。** 输入原生padding的成本和batch可行性需后续有限GPU成本测量，不编造A100小时数。
5. **当前未达到端到端KSDD2_SOURCE_READY。** 已通过的是模型/任务适配基础组件；原NEU训练和导出仍有任务绑定。下一节列出确切接入缺口，不通过放宽NEU门禁伪装为可运行。

## 已实施的最小修改

| 文件 | 实际改动/状态 |
|---|---|
| scripts/initialization_audit.py | audit新增可选num_classes=4，构建时使用该参数；默认及所有旧调用保持4类。KSDD2 wrapper传2类及四模型。 |
| scripts/ksdd2_support.py（新增） | 显式0/1或0/255解码；不缩放原图padding及valid；候选masked OHEM/BCE/detail；二值全局/病例/空GT指标；确定性已知同源组划分；训练阳性面积P33。没有训练main、数据扫描或网络副作用。 |
| configs/ksdd2_protocol_candidate_v1.json（新增） | 四模型三seed候选配置，input/mask_encoding待真实数据确认；训练/预算/冻结均false，DRAFT_NOT_RUNNABLE。 |
| tests/test_ksdd2.py（新增） | 四模型二类训练/eval输出、共享头、三seed初始化、历史重复mask-aware、空hard行为、候选loss有限值和梯度、padding外零梯度、指标分母与组隔离等合成检查。 |

models/factory.py、transport.py、losses.py、metrics.py、vendor以及所有旧配置未修改。未新增模型、未复制整个工程。四份本地已有外置vendor按原SHA提供给独立工作树进行CPU测试，仍未跟踪、不纳入发布清单。

## 现有入口为什么不能直接运行

| 现有位置 | 真实限制 | 下一次最小接入方案 |
|---|---|---|
| train_e0_ready.main | 读取固定neu_train.txt=3630，核验quarantine与{'1','2','3'}阈值，并使用NEU协议检查 | 增加显式dataset/task选择，保留NEU分支原样；KSDD2分支调用独立任务契约和数据manifest，不删原检查。 |
| SegmentationData | 无条件resize到固定H/W，只返回x,y | 为同一Dataset增加KSDD2适配器，原尺寸读入→训练期同步翻转→pad_native，返回valid和native_hw；不影响旧NEU返回格式。 |
| train_e0_ready loss路由 | 只允许现有两个NEU loss ID，不传valid | 在任务分支显式调用binary_joint_loss，记录新ID和fallback事件；不把新loss加入NEU白名单。 |
| validate | 强制.cuda()且没有valid crop；前景类别缺失会拒绝 | 复用模型forward，KSDD2按native_hw去pad后调用binary_metrics；正常病例不可因此拒绝，数据级类别缺失另报；逐例与分组均保存。 |
| server_preflight | NEU官方列表、类别、D4/quarantine及执行协议硬绑定 | 增加KSDD2独立任务预检分支，继续使用git_identity、verify_loaded_sources、config/split/threshold SHA、报告绑定；不接受把NEU counts改成任意值的绕过。 |
| export_e0_artifacts / e0_contract | MATRIX和全局字段为NEU，完整测试要求旧各类阈值；候选支撑函数未接入 | 扩展显式KSDD2轻量schema，继续白名单/大小/SHA限制，保存2×2混淆矩阵、binary指标、逐例、正常图错误和按阶段响应；旧导出不改。 |
| summarize_e1a/e1b/e1c | 固定NEU来源/seed组合/4类/726案例，不能用于KSDD2 | 在现有目录增加薄的KSDD2配对汇总入口，复用统计与校验工具，但验证二类和四条件×3seed新来源；不冒用NEU结果。 |

这些是有限、明确的任务适配代码工作，并非缺少新网络思想。实际文件结构、mask编码、输入和loss方案未冻结时，当前不生成虚假的可执行训练配置，也不写一个会自动训练的临时入口。服务器元数据返回后按上述表进行一次接入，实现/测试新的显式任务契约，再进行源码发布和真实preflight。

## CPU检查和已知问题

首次针对性运行发现两项测试问题：测试logits误用了整数dtype；64×96整网输入小于现有SqueezeBodyEdge两次无padding下采样的空间要求。已分别改为浮点合成logits和224×256矩形整网输入，并让padding适配器拒绝各轴不足224的32倍数目标。没有修改网络来迁就测试。

针对性11项通过。随后运行全部106项CPU合成回归，耗时38.425秒；实际输出、耗时和环境保存于`evidence/ksdd2_cpu_tests.txt`。CPU测试只证明接口/张量/数值规则，不提供分割性能、GPU显存或速度证据。原vendor的upsample/grid_sample弃用警告保留。

可重跑：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PYTHON" - <<'PY'
from pathlib import Path
import unittest
modules=['research.industrial_defect_english.tests.'+p.stem
         for p in sorted(Path('research/industrial_defect_english/tests').glob('test_*.py'))]
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
raise SystemExit(0 if result.wasSuccessful() else 1)
PY
```

## 进入正式接入前需冻结

- 官方资产实际身份、pairing/空GT计数和允许的mask编码。
- 开发组证据、parent未知范围、一次清单及SHA、训练派生面积阈值。
- 输入真实H/W、共同pad目标和valid/细节边带规则。
- 候选loss ID、是否取消重复二类mask-aware、路由权重、空hard显式fallback规则。
- batch、更新数、验证间隔、统一采样/增强、初始化、固定端点；候选并非已批准配置。
- CPU源码任务接入、服务器成本测量和12次正式预算分别批准，不能混为一次通过。

工程基础组件：PASS。真实数据/分组/阈值：PENDING_SERVER。完整训练/预检/导出接入：PENDING_IMPLEMENTATION。科学参数与GPU预算：PENDING_AUTHOR_FREEZE。因此总体为**KSDD2_COMPONENTS_READY / KSDD2_SOURCE_READY尚未达到**。
