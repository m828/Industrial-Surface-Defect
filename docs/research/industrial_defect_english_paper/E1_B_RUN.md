# E1-B：三种机制消融，三个配对训练seed

本次只发布代码和协议，不授权服务器训练。新增 stage1_only、head_shift_only、no_head_shift × seed1337/2027/3407，共9次训练。复用E1-A归档的完整 coupled_transport 三个终点，不重跑、不修改E0/E1-A。源码分支仍为 `research/e0-coupled-transport`；未来轻量结果分支为 `results/e1b-neu-mechanism`，本次不创建空结果分支。

## 模式定义与实现

模型及损失源码沿用E1-A原字节，本次扩展运行入口、预检、配置生成和结果导出。`T=beta*(R-L)`；E为语义边缘，L为对齐细节，R为增强语义，M为语义回流。三种模式都是已有factory/transport分支，没有新增网络算子。

| 模式 | edge_fusion输入 | arm2语义输入 | 训练期DetailHead输入 |
|---|---|---|---|
| 完整coupled（已有） | E, L+T, R-T | M-T | L+T |
| stage1_only | E, L+T, R-T | M | L+T |
| head_shift_only | E, L, R | M | L+T |
| no_head_shift | E, L+T, R-T | M-T | L |

`head_shift_only`是辅助细节监督路径消融，主预测不执行最终语义修正；推理时跳过transport和DetailHead，验证期T响应为空是预期行为，不记作响应为零。新增的训练forward hook仅在既有评价时点记录最后一个训练batch的T/beta；不增加模型forward、不改变损失、不消耗随机数。`mechanism_response.json`与原验证响应文件区分训练/推理范围，均不构成因果机制证明。

## 不变的实验协议

原开发划分 train2903/validation726/quarantine1，split seed1337、原面积/复杂度阈值不变。200×200、batch16、Adam、constant LR1e-4、WD2e-6、mask-aware loss `neu_a2ms_mask_aware_candidate_v1`、isolated_v1、从零初始化、10000次真实更新，固定最终端点评价。原840测试集不用于选模型；验证性能和响应不触发早停；异常/预算停止仍为INCOMPLETE。每seed核验公共初始化与该seed既有完整模型记录一致，并核验完整模型与三消融的transport初始参数一致。

完整模型参考身份：seed1337源码 `135d7ccfba6f112b60598dad38f5da19ac1bbc99`；seed2027/3407源码 `e946787b58f4584268fe66478b2f4d2ccad8d344`。归档结果分支 `results/e1a-neu-three-seed`，commit `0d95303507017568109be95afeaf9c32a7d588c3`。固定的五项数据文件SHA登记于 `research/industrial_defect_english/configs/e1b_protocol_v1.json`，预检逐项比对。实际执行配置另有各自字节SHA，不冒用E0/E1-A配置SHA。

## 服务器Codex可直接接收的任务

继续使用同一仓库及现有运行Python、NEU开发数据、冻结manifest和四份同SHA外置vendor；不要重新做服务器/数据审计，不重新划分，不改原结果，不训练。先完成以下准备后提交9组preflight和预算供作者批准。

执行前把 `E1B_SOURCE_SHA` 设置为本次本地交付的完整40字符commit；其余变量使用服务器已有的真实路径，含空格路径全程加引号。新目录必须不存在；不要清理旧目录。

```bash
git fetch origin research/e0-coupled-transport
git cat-file -e "${E1B_SOURCE_SHA}^{commit}"
git worktree add --detach "$NEW_E1B_SOURCE_WORKTREE" "$E1B_SOURCE_SHA"
cd "$NEW_E1B_SOURCE_WORKTREE"
test "$(git rev-parse HEAD)" = "$E1B_SOURCE_SHA"
sha256sum -c docs/research/industrial_defect_english_paper/e1b_source_sha256.txt
```

按既有外置vendor装载方法提供四份已持有文件，字节受provenance锁定；不提交这些文件。用三个真实coupled运行目录派生配置；每个目录必须有原 `executed_config.json` 和 `run.json`（已导出包则为 `run_manifest.json`），并且原配置指向的冻结文件仍可读取。不要用指标摘要或模板代替原执行证据。

```bash
"$PYTHON" -m research.industrial_defect_english.scripts.prepare_e1b \
  --references "$E0_COUPLED_1337" "$E1A_COUPLED_2027" "$E1A_COUPLED_3407" \
  --output "$NEW_EXTERNAL_E1B_CONFIG_DIR"
```

该命令只读核对3个完成端点及源commit、配置SHA、五项冻结文件SHA、2903/726/1计数、原模型/损失字节和跨seed共同协议；生成9份独立配置及SHA。每seed三配置同batch_id，variant被锁定；不同seed不同batch，输出目录拒绝覆盖。配置包含同seed完整模型的源码、执行配置及初始化身份。生成文件保留GPU/budget批准false；继承的预算仅是待审提案，不是本次9次训练的资源授权。

对每个配置读取seed、variant、batch_id，运行独立preflight：

```bash
# CFG为9份生成配置中的一份；NEW_PREFLIGHT_JSON为工作树外全新文件。
read -r SEED VARIANT BATCH_ID < <("$PYTHON" -c \
 'import json,sys;c=json.load(open(sys.argv[1]));print(c["seed"],c["variant"],c["batch_id"])' "$CFG")
RUN_ID="$BATCH_ID:$VARIANT:s$SEED"
"$PYTHON" -m research.industrial_defect_english.scripts.server_preflight \
  --expected-commit "$E1B_SOURCE_SHA" --config "$CFG" \
  --variant "$VARIANT" --run-id "$RUN_ID" --output "$NEW_PREFLIGHT_JSON"
```

只报告实际GO_READY/BLOCKED、配置/预检SHA及资源提案，等待作者确认preflight和9次训练预算。不得自动开始训练。预检不执行optimizer更新。E1-B不需要在新batch重训formal_a2ms；初始化参照取同seed已完成coupled记录。

仅在另行获批后：在工作树外9配置中记录批准（兼容既有执行标记 `scientific_gate=AUTHOR_APPROVED_E0`，保持experiment_stage=E1-B；gpu_training_approved及budget.author_approved置true）。如资源上限获批修改，9组采用相同批准规则。重新计算配置SHA并重新preflight，不复用修改前报告。保持原冻结数据、阈值、模型和训练协议不变。正式命令模板如下，本次不执行：

```bash
PREFLIGHT_SHA=$(sha256sum "$NEW_PREFLIGHT_JSON" | cut -d ' ' -f1)
"$PYTHON" -m research.industrial_defect_english.scripts.train_e0_ready \
  --config "$CFG" --variant "$VARIANT" --source-commit "$E1B_SOURCE_SHA" \
  --run-id "$RUN_ID" --preflight-report "$NEW_PREFLIGHT_JSON" \
  --preflight-sha256 "$PREFLIGHT_SHA" --authorize-gpu
```

## 轻量导出与同seed汇总

每组保留实际更新、状态、配置/预检身份、指标和运行开销。使用既有工具，输出新目录：

```bash
"$PYTHON" -m research.industrial_defect_english.scripts.export_e0_artifacts \
  --run-dir "$COMPLETED_RUN" --preflight-report "$MATCHED_PREFLIGHT" --output "$NEW_LIGHT_PACKAGE"
"$PYTHON" -m research.industrial_defect_english.scripts.export_e0_artifacts \
  --verify --output "$NEW_LIGHT_PACKAGE"
```

白名单只含轻量JSON/CSV、逐例/分层统计、ID/hash视图与SHA；不包含mask、图像、权重或大日志。每个包沿用4MB/文件及15MB/包上限。原九个E1-A结果包不改写。待9个消融全部完成，汇总命令：

```bash
"$PYTHON" -m research.industrial_defect_english.scripts.summarize_e1b \
  --ablation-packages "${E1B_PACKAGES[@]}" \
  --reference-packages "$E0_COUPLED_PACKAGE_1337" "$E1A_COUPLED_PACKAGE_2027" "$E1A_COUPLED_PACKAGE_3407" \
  --source-commit "$E1B_SOURCE_SHA" --output "$NEW_SUMMARY_JSON"
```

`E1B_PACKAGES`必须为9个真实轻量包路径的bash数组；不可漏组或替换seed。汇总拒绝INCOMPLETE、SHA不符、重复/缺失组合、不同训练/划分/阈值和错误配对；检查同seed公共初始SHA、三消融transport初始SHA、逐例ID/GT面积/分层一致。输出四条件mIoU/前景mIoU/各类IoU均值及样本标准差(ddof=1)、逐seed“消融−完整”差、FP/FN/跨类错误、面积/复杂度差值、T/beta和开销。结果分支另行提交白名单轻量文件，不能改源码；三seed不作统计显著性或SOTA结论。

## 本地工程验收

CPU合成测试覆盖非零T的三模式路径、主输出和共享SegHead、辅助头训练/推理分离、三个seed公共/transport初始化、九配置SHA及路径带空格、旧E0/E1-A配置、输出防覆盖、INCOMPLETE拒绝及同seed配对汇总。它们验证工程行为，不提供真实模型性能证据。

本地实际运行13个test模块，共83项全部通过（含新增6项E1-B测试），耗时20.410秒；环境未升级。旧vendor的upsample/grid_sample警告保持原行为，未修改模型以消除警告。可重跑命令（仅合成CPU测试）：

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
