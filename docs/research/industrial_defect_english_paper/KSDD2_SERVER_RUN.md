# KSDD2端到端源码运行说明

本文件取代冻结前的KSDD2_SERVER_HANDOFF.md。只使用现有工程；数据已声明冻结为KSDD2_DATA_FROZEN_V1，不重新划分、不更改P33，不访问官方test做模型选择。当前仅完成本地源码和CPU验收，**没有批准GPU训练、12组预算或候选损失**。

## 1. 锁定本次源码

本地交付回执提供完整40字符`KSDD2_SOURCE_SHA`，不能用浮动分支HEAD代替。按既有同仓库worktree方式取得版本，不覆盖NEU运行目录：

```bash
git fetch origin research/e0-coupled-transport
git worktree add --detach "$NEW_KSDD2_SOURCE_WORKTREE" "$KSDD2_SOURCE_SHA"
cd "$NEW_KSDD2_SOURCE_WORKTREE"
test "$(git rev-parse HEAD)" = "$KSDD2_SOURCE_SHA"
sha256sum -c docs/research/industrial_defect_english_paper/ksdd2_source_integration_sha256.txt
```

沿用已有四份外置vendor的provenance SHA，不提交vendor；任何源码/配置更改均需重新发布和预检，不在服务器偷偷修源码后沿用旧SHA。使用既有真实训练Python，所有含空格路径加引号。

## 2. 绑定已冻结资产，不生成新划分

外部冻结目录线索为`/workspace/ksdd2_development_frozen_v1/`，通过CLI传入，不是模型硬编码。必需：dev_train.jsonl、dev_validation.jsonl、preprocessing_protocol.json、area_threshold_p33.json、dataset_fingerprint.json、split_freeze_report.json、SHA256SUMS.txt，以及实际数据根目录。

源代码固定train SHA `dd7d50550c23887da15234f425378a0ce4e07974d0ebe533a13254db457c29c3`、validation SHA `482a8c16e4c096f8a3a3035783fb519c2570555b768dfb04da537105c9a58ee6`；两份不符立即失败。其余元数据按真实文件字节计算SHA，核对原SHA256SUMS并写入执行配置，checksum清单自身也被配置SHA锁定。支持清单中的basename或与配置资产完全一致的绝对路径；不会跟随checksum条目读任意文件。

**实际冻结JSON字段格式未传给本地，不能假设等于示例。** 把`configs/ksdd2_schema_map_example.json`复制到工作树外，逐项映射真实字段；不要修改冻结文件去迎合代码。使用RFC6901 JSON pointer，独立height/width字段可映射为两个pointer的列表，标量JSON文档可用空字符串pointer。manifest必须映射sample_id、相对图像/GT路径、两个文件SHA、native_hw、gt_positive和original_split；group_id若原清单没有则在映射中删除/设null，不伪造父图信息。

area.p33为必需的显式映射，数值必须精确等于0.009815324081003446。其他元数据仅对实际存在且显式映射的字段解码核对；不存在的示例字段应删去映射。所有七个文件仍进行字节SHA绑定，预检报告明确列出解码过的JSON字段，不把未映射字段宣称为语义独立核验。冻结预处理由运行契约强制执行，并由作者在批准配置前对照真实元数据确认。

```bash
"$PYTHON" -m research.industrial_defect_english.scripts.prepare_ksdd2 \
  --frozen-root "$REAL_FROZEN_DIRECTORY" \
  --data-root "$REAL_KSDD2_DATA_ROOT" --runs-root "$EXISTING_EXTERNAL_RUN_ROOT" \
  --schema-map "$EXTERNAL_VERIFIED_SCHEMA_MAP" --output "$NEW_EXTERNAL_CONFIG_DIRECTORY"
```

prepare只读冻结清单和元数据，核对来源、计数及SHA，生成3份配置及SHA，**不重划数据，不模型forward，不训练**。1865/466、197/49阳性计数固定；真实preflight再核验原文件SHA/shape/0或255编码与训练small66/other131/normal1668。

每seed一份`ksdd2_seed<seed>.json`，四模型共享同一配置字节。初始化为2类从零公共张量一致，三个transport条件初始transport相同，isolated_v1不变。执行配置保存在源码工作树外，避免污染源码身份。

## 3. 待作者确认的训练候选

候选为四模型×seed1337/2027/3407；batch16、Adam lr1e-4、WD2e-6、constant、10000真实更新、每1000验证、固定最终端点。冻结数据规则不能因预算而变：输入H672×W256，无缩放/裁剪/ImageNet Normalize，RGB/255、右下复制padding、独立valid；0/255映射为0/1、二logits argmax、无阈值调优/后处理。

候选loss ID为ksdd2_binary_masked_ohem_detail_candidate_v1，路由权重[10,1,3,3]、OHEM(.7,100000)、空hard集合valid CE fallback计数、细节有效区内缩4px。它没有重复NEU的二类mask-aware路由。候选ID、权重、OHEM、fallback和有效区规则均须作者批准，不因源码完成而成为正式科学冻结。

新配置scientific_protocol_approved、gpu_training_approved、budget.author_approved、stopping.author_approved均为false；预算数值未捏造。预检会因此BLOCKED，清楚返回缺失项。实际成本/显存不支持时，必须在正式比较前统一提出四模型新候选并进行相应协议/源码版本审阅，不暗改某模型配置。

max_batch_gpu_hours按该seed的四模型batch记账；整12次的上限为三份seed批预算之和，必须分别核对并整体批准，不能把一个seed的cap误称为12次总cap。输出根目录必须可写、位于源码工作树之外，并满足批准的可用磁盘上限。

## 4. 独立preflight

```bash
SEED=1337  # 另两值2027、3407
VARIANT=formal_a2ms  # 其余capacity_control、coupled_transport、recomputed_final_transport
CFG="$NEW_EXTERNAL_CONFIG_DIRECTORY/ksdd2_seed${SEED}.json"
BATCH_ID=$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["batch_id"])' "$CFG")
RUN_ID="$BATCH_ID:$VARIANT:s$SEED"
"$PYTHON" -m research.industrial_defect_english.scripts.server_preflight \
  --expected-commit "$KSDD2_SOURCE_SHA" --config "$CFG" --variant "$VARIANT" \
  --run-id "$RUN_ID" --output "$NEW_EXTERNAL_PREFLIGHT_REPORT"
```

核验完整commit、实际import路径和提交字节、source snapshot、vendor锁、独立任务/数据版本、七资产SHA、原始配对与valid、四模型初始化、输出防覆盖、当前GPU/磁盘及批准状态。任何不符BLOCKED。CPU诊断不能得到真实服务器GO_READY，preflight不step/backward。源码/配置/资产/预算改变后必须重做预检；不能复用旧报告。

本轮未执行真实服务器preflight。服务器先回报实际检查及资源提案，等待作者批准。只有批准科学候选、GPU和具体费用上限后，才能把四个批准字段记录为true、填入预算数值、重新计算配置SHA并重新预检。报告training_authorized始终false，表示预检本身不是训练。

## 5. 将来另行获批后的训练模板（本轮禁止执行）

```bash
PREFLIGHT_SHA=$(sha256sum "$NEW_EXTERNAL_PREFLIGHT_REPORT" | cut -d ' ' -f1)
"$PYTHON" -m research.industrial_defect_english.scripts.train_e0_ready \
  --config "$CFG" --variant "$VARIANT" --source-commit "$KSDD2_SOURCE_SHA" \
  --run-id "$RUN_ID" --preflight-report "$NEW_EXTERNAL_PREFLIGHT_REPORT" \
  --preflight-sha256 "$PREFLIGHT_SHA" --authorize-gpu
```

每seed先formal_a2ms完成10000更新，再其余三条件；共12新训练，不用NEU checkpoint补齐。不按验证表现或T/beta早停/挑seed。数值、资源、协议异常保留INCOMPLETE；所有损失fallback按路由计数。每验证点保存滚动recovery及RNG，终点保存endpoint及SHA；本版不提供自动resume/自动重试，不能把恢复文件冒用为已完成终点。

## 6. 轻量导出与12组汇总

```bash
"$PYTHON" -m research.industrial_defect_english.scripts.export_e0_artifacts \
  --run-dir "$COMPLETED_KSDD2_RUN" --preflight-report "$MATCHED_PREFLIGHT" --output "$NEW_LIGHT_PACKAGE"
"$PYTHON" -m research.industrial_defect_english.scripts.export_e0_artifacts --verify --output "$NEW_LIGHT_PACKAGE"
"$PYTHON" -m research.industrial_defect_english.scripts.summarize_ksdd2 \
  --packages "${TWELVE_VERIFIED_PACKAGES[@]}" --source-commit "$KSDD2_SOURCE_SHA" --output "$NEW_SUMMARY_JSON"
```

TWELVE_VERIFIED_PACKAGES为真实12包的bash数组，不得用模板或缺失组合替代。KSDD2使用独立ksdd2_light_v1 schema，2×2混淆矩阵、全局前景IoU/Dice、固定GT阳性病例均值、正常图误报和small/other、逐例原图坐标、按阶段T/beta、fallback及资源统计；无图像/GT/权重/大日志。沿用4MB/文件、15MB/包限制。完整包要求真实10000更新、所有SHA/预检绑定正确；不完整包可保全，但汇总拒绝。

三组比较均为coupled−对照，保留每seed方向、三seed均值及样本标准差；NA保留缺失计数。FP/FN差值为正代表coupled错误更多。正常图双方都空的病例前景分数为NA而非1，仍保留TN与误报分母。T/beta是包含padding影响的特征网格诊断，重计算条件按两阶段分开，不作因果/显著性证明。

轻量结果未来另用结果分支回传，源码分支不写结果；不自动提交或上传原始资产。官方1004测试集评价须后续另行授权，本任务没有该入口的放行配置。
