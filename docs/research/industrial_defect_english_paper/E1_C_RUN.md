# E1-C — Shared vs Recomputed Final Transport

## 科学问题与边界

只检验跨阶段共享第一阶段T，与最终阶段使用同一个predictor对当前D、M重新计算T_final的差异。新增 `recomputed_final_transport × {1337,2027,3407}` 三次训练；三个同seed coupled端点直接复用。E0、E1-A、E1-B不重跑、不改写。开发集、阈值、监督、初始化和评价协议不变；本次源码发布不构成GPU预算或训练批准。

源码从E1-B `faed7033e78e5a7a9e6dc5a2f798545aa31e52e7` 开始，独立工作树准备。研究发布分支为 `research/e0-coupled-transport`。后续轻量结果建议使用 `results/e1c-neu-shared-vs-recomputed`，本地本轮不创建空结果分支。

## 实际计算路径核对

实现位于原 `models/transport.py` 与 `models/factory.py`，本轮两文件及损失源码均不修改。

| 路径 | coupled_transport | recomputed_final_transport |
|---|---|---|
| 第一阶段 | T=transport(L,R) | 完全相同 |
| 细节形成 | D=edge_fusion(E,L+T,R−T) | 完全相同 |
| 原语义回流 | M=conv_up(s+body+AAM(s)) | 完全相同 |
| 最终语义输入 | M−T | T_final=transport(D,M)，输入M−T_final |
| 训练期细节头 | DetailHead(L+T) | 完全相同，不使用T_final |
| 主预测 | SegHead(arm2(D,最终语义输入)) | 同一路径 |
| predictor参数对象 | 一份 | 同一份，调用两次 |
| 每次forward的调用数 | 1 | 2，训练与评估均如此 |
| 新增参数 | — | 相对coupled为0；两者transport各305参数 |

共享SegHead、AAM、四路训练输出和推理单主输出保持原行为。增加的是一次相同predictor的执行和相应描述量计算，不能称MACs/时延完全相同。CPU测试将最后一层bias设为非零来核对路径；正式模型仍保留原零初始化，不修改机制或正式初始化。

## 冻结协议与配置派生

原训练2903、验证726、quarantine1，split seed1337；训练seed为1337/2027/3407。200×200、batch16、Adam、constant LR1e-4、WD2e-6、10000真实optimizer更新、loss权重[10,1,3,3]、`neu_a2ms_mask_aware_candidate_v1`、`isolated_v1`、从零初始化和固定最终端点评价保持。官方840测试集不用于选模；不重新冻结或调整面积/复杂度阈值。

五项冻结资产SHA完整保存在 `configs/e1c_protocol_v1.json`，与作者指定值一致。`prepare_e1c`必须读三组真实 `run.json` 和 `executed_config.json`；不接受口述数值、模板或仅有export manifest代替run.json。要求真实完成10000更新、source/执行配置SHA正确、冻结文件实际字节SHA一致、原划分计数正确，以及三参考运行的科学协议一致。若实际值与描述不同，失败并报告，不重新生成数据或改写值。

参考source：seed1337为 `135d7ccfba6f112b60598dad38f5da19ac1bbc99`；2027/3407为 `e946787b58f4584268fe66478b2f4d2ccad8d344`。已有E1-A轻量结果commit `0d95303507017568109be95afeaf9c32a7d588c3`。

公共初始化必须等于同seed原run记录。transport初始化通过相同字节factory/transport、原seed与原transport内部seed173重建并比较两个条件；若旧run已登记transport初始SHA，还须相等。旧run没有该字段时，配置明确记录 `RECONSTRUCTED_FROM_BYTE_IDENTICAL_FACTORY_AND_SEED_NOT_HISTORICAL_TENSOR_RECORD`，不声称取得旧初始tensor。预检和训练前再次核验。原权重不加载。

生成三份新配置及其SHA，不改旧配置；三份GPU/budget批准默认false。配置SHA与共享协议文件SHA分别保留。新batch_id前缀e1c，每seed独立，不覆盖旧输出。

## 服务器接收与准备命令

使用服务器既有同一仓库、Python和四份同SHA外置vendor；不重新检查通用服务器环境或扫描数据。路径变量使用服务器真实已有路径。`E1C_SOURCE_SHA`由本地本次发布提供完整40字符值；下列prepare/preflight命令不启动训练。

```bash
git fetch origin research/e0-coupled-transport
git cat-file -e "${E1C_SOURCE_SHA}^{commit}"
git worktree add --detach "$NEW_E1C_SOURCE_WORKTREE" "$E1C_SOURCE_SHA"
cd "$NEW_E1C_SOURCE_WORKTREE"
test "$(git rev-parse HEAD)" = "$E1C_SOURCE_SHA"
sha256sum -c docs/research/industrial_defect_english_paper/e1c_source_sha256.txt
# 按原方式装载已有四份外置vendor，仍受provenance SHA限制；不提交vendor源码。
"$PYTHON" -m research.industrial_defect_english.scripts.prepare_e1c \
  --references "$E0_COUPLED_1337" "$E1A_COUPLED_2027" "$E1A_COUPLED_3407" \
  --output "$NEW_EXTERNAL_E1C_CONFIG_DIR"
```

每个seed只运行一个模式。读取对应配置的batch_id，使用工作树外全新的preflight报告路径：

```bash
CFG="$NEW_EXTERNAL_E1C_CONFIG_DIR/e1c_recomputed_final_transport_s${SEED}.json"
VARIANT=recomputed_final_transport
BATCH_ID=$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["batch_id"])' "$CFG")
RUN_ID="$BATCH_ID:$VARIANT:s$SEED"
"$PYTHON" -m research.industrial_defect_english.scripts.server_preflight \
  --expected-commit "$E1C_SOURCE_SHA" --config "$CFG" \
  --variant "$VARIANT" --run-id "$RUN_ID" --output "$NEW_PREFLIGHT_JSON"
```

预检识别E1-C唯一条件，核对当前source及真实import路径、配置SHA、五项冻结SHA、同seed参考运行文件、公共和transport初始化。保持原真实数据与门禁检查；GO_READY仍仅表示可以提交训练申请。源commit、run/config/split/threshold SHA缺失或不一致时BLOCKED。所有输出拒绝覆盖。

## 预算、批准与正式训练模板

先回报三配置的继承预算、既有同seed coupled实际训练耗时与峰值显存，以及可用的原运行记录。只新增3次、串行执行是建议；不假定额外一次小predictor调用等于零成本，也不凭CPU测试估算A100速度。原run预算只是提案，不自动成为E1-C批准。若作者要求修改资源上限，另存版本并重新计算SHA/preflight，训练科学参数保持不变。

另行获批后，在三份外置配置中记录真实授权，兼容现runner执行标记 `scientific_gate=AUTHOR_APPROVED_E0`，保持experiment_stage=E1-C；GPU及budget批准置true。任何配置字节变更后重新preflight。下面只为未来执行模板，本次不运行：

```bash
PREFLIGHT_SHA=$(sha256sum "$NEW_PREFLIGHT_JSON" | cut -d ' ' -f1)
"$PYTHON" -m research.industrial_defect_english.scripts.train_e0_ready \
  --config "$CFG" --variant recomputed_final_transport \
  --source-commit "$E1C_SOURCE_SHA" --run-id "$RUN_ID" \
  --preflight-report "$NEW_PREFLIGHT_JSON" --preflight-sha256 "$PREFLIGHT_SHA" --authorize-gpu
```

不为新batch训练formal/capacity/coupled基线。性能、beta或T不触发早停；数值、协议、资源异常按原规则停止留证。未完成10000更新一律INCOMPLETE，保留负向seed，不追加或筛选seed。

## 轻量导出与配对汇总

```bash
"$PYTHON" -m research.industrial_defect_english.scripts.export_e0_artifacts \
  --run-dir "$COMPLETED_E1C_RUN" --preflight-report "$MATCHED_PREFLIGHT" --output "$NEW_LIGHT_PACKAGE"
"$PYTHON" -m research.industrial_defect_english.scripts.export_e0_artifacts \
  --verify --output "$NEW_LIGHT_PACKAGE"
"$PYTHON" -m research.industrial_defect_english.scripts.summarize_e1c \
  --recomputed-packages "$E1C_PACKAGE_1337" "$E1C_PACKAGE_2027" "$E1C_PACKAGE_3407" \
  --reference-packages "$COUPLED_PACKAGE_1337" "$COUPLED_PACKAGE_2027" "$COUPLED_PACKAGE_3407" \
  --source-commit "$E1C_SOURCE_SHA" --output "$NEW_SUMMARY_JSON"
```

汇总方向固定 **coupled_transport − recomputed_final_transport**，保留三个seed各自方向及均值/样本标准差(ddof=1)。输出含背景/前景mIoU、四类IoU、4×4混淆矩阵、二值前景FP/FN、前景跨类错误和逐类FP/FN、面积/复杂度分组、参数量、训练耗时和峰值显存。FP定义为GT背景→预测前景；二值FN为GT前景→预测背景，跨类错误另列；逐类FN包含错为其他前景类，不混用。

验证逐例CSV必须为同一726图像×3前景类别；各GT像素总数与混淆矩阵一致。小面积和高复杂度沿用原阈值（complexity_group=large为高组）。拒绝不完整、错seed、重复、错参考、SHA或协议不匹配的包。

原响应文件保留有序调用列表；E1-C `mechanism_response.json`另分stage1/final，每阶段726条验证响应和各1条最终训练batch响应。beta_mean、transport_rms及rho_difference=T_RMS/difference_RMS均按阶段报告。完整coupled只调用一次，最终复用同T；不能把新模型两阶段混合均值与旧模型单次均值直接解释为机制效果，两个阶段difference分母也不同。训练batch诊断与验证图像诊断分开。特征响应是描述性证据；三个seed不构成显著性证明。实际训练时长包含日志/诊断开销，不作为独占推理时延。

白名单沿用轻量JSON/CSV/ID-hash视图及校验记录，上限4MB/文件、15MB/包。禁止图像、GT、checkpoint和大型训练日志进入Git。本地本轮只交付代码、文档、小体积CPU测试日志及源码SHA。

## 本地验收

本轮实际运行14个CPU测试模块，95项全部通过，其中E1-C新增12项；耗时30.971秒。覆盖非零T中间路径、D/M再次调用、同一predictor与梯度、细节头及主输出、参数量、三个seed初始化、三个配置SHA/路径/防覆盖、预检拒绝、轻量包往返、同seed汇总及INCOMPLETE拒绝，并回归E0/E1-A/E1-B。测试均为合成数据。测试日志：`evidence/e1c_cpu_tests.txt`；逐文件SHA：`e1c_source_sha256.txt`。原vendor弃用警告保留，不修改历史实现消除警告。

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
