# E1-A：两个新训练seed，六次独立重复

新增seed2027、3407，每个seed均为formal_a2ms、capacity_control、coupled_transport；保留原E0 seed1337结果，禁止重跑后替换。split seed固定1337，原冻结2903/726/1和阈值不变；从零初始化、10k真实更新、200×200、batch16、Adam/constant1e-4/WD2e-6、完整mask-aware及isolated_v1保持。模型和loss源码与135d7cc逐字节一致；初始化核验改为使用当前训练seed。每seed一个共同配置，通过variant区分三组，输出batch目录含训练seed，禁止覆盖。

## 配置和启动

获取本次研究分支完整commit，在同仓库新source工作树使用该commit；原E0目录不动。四份外置vendor沿用原SHA。下面都使用服务器已有路径，不扫描数据或重新生成split：

```bash
git fetch origin research/e0-coupled-transport
git worktree add --detach "$NEW_E1_SOURCE_WORKTREE" "$APPROVED_E1_SOURCE_COMMIT"
cd "$NEW_E1_SOURCE_WORKTREE"
"$PYTHON" -m research.industrial_defect_english.scripts.prepare_e1a \
 --e0-config "$E0_FORMAL_RUN/executed_config.json" \
 --e0-run-manifest "$E0_FORMAL_RUN/run.json" --output "$NEW_EXTERNAL_E1_CONFIG_DIR"
```

该命令仅核对原E0完成记录、原配置字节及冻结manifest/阈值SHA和2903/726/1计数，生成两份新配置和SHA，不训练或重新划分。若实际文件缺失/不符，停止，不用本地模板或作者口述数字填入真实证据。仓库seed模板供审阅，不是已绑定真实数据的可运行配置。原E0文件不改，配置保存原E0 source/config/manifest身份。新配置的预算和GPU授权仍为false；既有E0预算只作为待审上限继承，不代表已经批准六次训练或估算出实际耗时。

作者确认新配置、GPU预算后，在两份配置中分别记录真实授权：沿用现runner的运行标记 `scientific_gate=AUTHOR_APPROVED_E0`（这是通用执行门禁名，不把experiment_stage改回E0），`budget.author_approved=true`、`gpu_training_approved=true`；重新计算配置字节SHA；保持各seed内部三组同一配置。先formal_a2ms，之后其余两组；两个seed分不同batch目录。

```bash
# SEED取2027或3407，VARIANT先formal_a2ms，后capacity_control和coupled_transport
CFG="$NEW_EXTERNAL_E1_CONFIG_DIR/e1a_seed${SEED}.json"
"$PYTHON" -m research.industrial_defect_english.scripts.server_preflight \
 --expected-commit "$APPROVED_E1_SOURCE_COMMIT" --config "$CFG" \
 --variant "$VARIANT" --run-id "$BATCH_ID:$VARIANT:s$SEED" --output "$NEW_PREFLIGHT_JSON"
# 仅GO_READY、实际配置和预算获批后：
"$PYTHON" -m research.industrial_defect_english.scripts.train_e0_ready \
 --config "$CFG" --variant "$VARIANT" --source-commit "$APPROVED_E1_SOURCE_COMMIT" \
 --run-id "$BATCH_ID:$VARIANT:s$SEED" --preflight-report "$NEW_PREFLIGHT_JSON" \
 --preflight-sha256 "$NEW_PREFLIGHT_SHA256" --authorize-gpu
```

BATCH_ID读取当前CFG，不能用模板ID替代prepare生成的ID。预检校验当前训练seed的共同初始化；split audit仍必须seed1337。验证不早停，不修改T、loss或数据；失败及预算停止保留INCOMPLETE。不能称三个seed为显著性证明。

## E0轻量结果回传（原导出文件尚未交到本地）

本地及GitHub目前未发现E0_TRAINING_RESULTS.md或结果分支；作者口述指标不当作已导出的文件。服务器Codex使用实际已导出的三组目录，先逐组运行：

```bash
"$PYTHON" -m research.industrial_defect_english.scripts.export_e0_artifacts --verify --output "$EXISTING_EXPORTED_PACKAGE"
```

只在包SHA/白名单通过后，将三个原导出包和E0_TRAINING_RESULTS.md放入同仓库独立结果工作树的全新 `experiments/results/industrial_defect_english/e0_seed1337/`，建议分支 `results/e0-seed1337`（存在则核对复用，不能覆盖）。逐路径git add、commit、push该结果分支，核验远端SHA后将branch/commit告知本地。保留原source SHA135d7cc；结果提交不是训练源码commit。只收原导出的JSON/CSV/ID与SHA，禁止图像、GT、checkpoint、大日志，禁止修改源码及原运行。分支名是本次建议，不声称已经创建或已回传。

## 汇总及机制后续

每个完整run继续用原export_e0_artifacts生成轻量包。九个包收齐后：

```bash
"$PYTHON" -m research.industrial_defect_english.scripts.summarize_e1a \
 --packages "$E0_FORMAL_PACKAGE" "$E0_CONTROL_PACKAGE" "$E0_COUPLED_PACKAGE" \
 "$E1_S2027_FORMAL_PACKAGE" "$E1_S2027_CONTROL_PACKAGE" "$E1_S2027_COUPLED_PACKAGE" \
 "$E1_S3407_FORMAL_PACKAGE" "$E1_S3407_CONTROL_PACKAGE" "$E1_S3407_COUPLED_PACKAGE" \
 --e1-source-commit "$APPROVED_E1_SOURCE_COMMIT" --output "$NEW_E1_SUMMARY_JSON"
```

核验包SHA、每seed三组共同config/init SHA、各seed同split/阈值/隔离/超参数，拒绝不完整或缺少条件。输出含背景/前景mIoU与类别IoU的均值和样本标准差(ddof1)、各seed coupled对两个基线差值/方向、FP/FN/前景跨类错误、面积/复杂度条件、逐seed T/beta及耗时/显存；逐例CSV保留在原轻量包。E0与E1训练源码SHA不同是执行适配，prepare已检查模型和loss不变，不把新SHA冒用旧E0身份。三seed只报告方向和不确定性，不自动宣称显著或SOTA。

stage1_only、head_shift_only、no_head_shift已由factory/transport现有实现支持，本轮不改、不训练。若后续批准推理T=0诊断，可在独立评价副本用临时forward hook将transport的T置零、保留原checkpoint与权重字节；另标干预评价，不替换正常预测。本轮未执行该推理。
