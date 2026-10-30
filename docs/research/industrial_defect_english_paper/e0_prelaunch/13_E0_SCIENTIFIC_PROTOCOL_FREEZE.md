# E0 作者科学协议冻结记录

唯一记录 ID：**e0_scientific_protocol_v1**。决策日期：2026-10-29。依据：作者在本次会话中明确给出的《E0作者科学协议裁决》。本记录接续 P2.7，改变其“科学协议待批准”状态；P2.7 及更早文件保留为历史，不再代表当前科学批准状态。

**科学协议已批准；真实 P3 冻结、源码发布、GPU训练和最终资源分配均未批准。** 本记录与同 ID 的 JSON 是同一次裁决的可读和机器可读载体，不构成两套协议。未来更改科学内容必须另有作者裁决和新版本；不能通过修改同 ID 文件或换损失后继续原运行来规避。

## 1. 已批准内容和保留边界

| 项目 | 裁决 |
|---|---|
| 三个条件 | formal_a2ms、coupled_transport、capacity_control；不改变模型机制 |
| loss_protocol_id | neu_a2ms_mask_aware_candidate_v1，三组完整 mask-aware；保留历史空 hard-pixel 非有限行为，停止并留证，不回退或自动修改公式 |
| rng_policy | isolated_v1；核验公共初始张量，同一采样与增强策略，各自独立随机流；不作跨GPU位级确定性声明 |
| 数据治理方案 | neu_exact_d4_quarantine_candidate_v1；批准方案供 P3 准备，未批准真实执行，不能在 parent 证据不足时强行冻结 |
| 终止策略 | fixed_endpoint_safety_only_candidate_v1；固定最终10000次真实optimizer更新；禁止验证性能及beta/T响应早停 |
| 初始化 | 三组从零；不加载旧checkpoint或ImageNet权重 |
| 候选固定超参数 | seed1337、batch16、200×200、Adam、常数LR1e-4、weight_decay2e-6、损失权重[10,1,3,3] |
| 验证及选权 | 仅冻结开发验证集；每1000更新及最终更新评价，固定末步权重；不使用历史840测试作新机制选择 |
| 允许停止 | 非有限损失/梯度/参数、协议/运行异常、批准的资源上限；未完成10000真实更新一律INCOMPLETE，失败或异常端点不得伪装完成 |
| 资源预定上限 | 单GPU串行；每组3600秒、3,000,000,000字节；整批3分配GPU小时；启动至少10,000,000,000字节可用磁盘。待服务器环境及成本验收后重新申请运行，非实测耗时或最终运行批准 |

`e0_fixed_endpoint_candidate_v2` 保留为现有代码认可的执行规则版本名，若名称含candidate不表示本次裁决仍未发生；本次批准状态由唯一科学协议记录和配置标记确定。E0不称为历史240k训练的完全同协议复现；训练范围、预算、初始化/随机流和部分执行细节有区别，历史成绩不能成为本次同协议端点。

## 2. 损失计算核对

前三个多类输出共享SegHead，依次接arm2联合特征、edge_fusion特征和分解前high_feat；第四路为独立细节头，原A2MS接对齐浅层细节，新模型使用其修正后特征。第三路不是独立seg_body输出。

对 `j=0,1,2`，`y∈{0,1,2,3}`：

```python
b = (y != 0).long()
q_j = torch.stack((z_j[:, 0], z_j[:, 1:].sum(dim=1)), dim=1)
```

二类前景项是**前景通道 logits 求和**，不替换为概率求和、softmax后求和或logsumexp。令 `ell=(OHEM,BCE,OHEM)`，`w=(10,1,3,3)`：

`L = Σ(j=0..2) w_j ell_j(q_j,b) + Σ(j=0..2) w_j ell_j(z_j,y) + w_3 [BCE(z_3,t)+Dice(sigmoid(z_3),t)]`。

细节目标沿用整数标签拉普拉斯响应、full/x2/x4、0.6/0.3/0.1融合及严格阈值。候选OHEM在有效标注域内按历史阈值及严格小于规则取hard-pixel均值，空集可为NaN。当前公式、原件及数值局限见 [P2.7损失附件](evidence/p2_7_loss_rights_decisions.md)；原历史源码未修改。CPU定点检查直接截取实际函数路由，核对二类logits和非零目标，而非只核对函数名称。总损失或组件非有限时保留failure_event；其他非有限及异常保留run状态/实际更新数；不得更换公式后自动继续。

## 3. 配置和SHA的唯一入口

共享科学文件：[e0_shared_scientific_protocol_v1.json](../../../../research/industrial_defect_english/configs/e0_shared_scientific_protocol_v1.json)。

`shared_protocol_sha256 = e3e567b9ce75377906a475d0a53e8e94d700f846c28b64765a4f5b4ba8c90fac`

共同运行模板：[e0_science_locked_runtime_template_v1.json](../../../../research/industrial_defect_english/configs/e0_science_locked_runtime_template_v1.json)。

`config_sha256 = f170ddaf52b1c43a397754cafd42911da723be708da797d8c644d090d0d09f0d`

两者均为 **SHA-256(实际文件完整字节)**，包括空格及末尾换行；不对重新序列化的JSON对象求另一个散列，不在文件中自引用其自身SHA。共享文件只锁定科学规则；共同运行配置另包含批准边界、源快照、路径及未来数据/运行信息。因此这两个SHA不同，不能相互替代。

现有运行器要求三组的完整配置字节SHA相同，通过CLI变更variant和run_id。本次不制造三份不同可执行配置；各arm的config_sha256均指上述同一文件，故相同是预期行为：

| arm | --config | config_sha256 | 独立标识 |
|---|---|---|---|
| formal_a2ms | 同一共同运行模板 | f170ddaf52b1c43a397754cafd42911da723be708da797d8c644d090d0d09f0d | variant + batch_id:formal_a2ms:s1337 |
| coupled_transport | 同一共同运行模板 | f170ddaf52b1c43a397754cafd42911da723be708da797d8c644d090d0d09f0d | variant + batch_id:coupled_transport:s1337 |
| capacity_control | 同一共同运行模板 | f170ddaf52b1c43a397754cafd42911da723be708da797d8c644d090d0d09f0d | variant + batch_id:capacity_control:s1337 |

未来服务器填写真实路径、split/threshold/quarantine等SHA、实际环境和获批运行信息后，必须保存新的共同配置字节、重新计算config_sha256并重新预检，三组仍共同使用；不得把本模板SHA冒称为已执行配置SHA。shared_protocol科学内容不随路径填充变化。独立arm启动描述文件若将来需要，其SHA只能称启动描述SHA，不等于运行器完整配置SHA；本轮未创建这类平行配置。

新配置记录实际批准的loss/RNG/governance ID，`stopping.author_approved=true`表示安全终止科学规则已批准；同时：

- `scientific_gate=AUTHOR_APPROVED_SCIENTIFIC_PROTOCOL_ONLY`，不是旧运行器要求的`AUTHOR_APPROVED_E0`。
- `p3_data_freeze_approved=false`、`source_publication_approved=false`、`gpu_training_approved=false`、`budget.author_approved=false`。
- 所有真实数据路径、冻结manifest/阈值、quarantine、runs_root均未填，不创建虚假路径或SHA。

因此当前配置不可启动训练。旧P2.7模板完整保留为历史候选，不再是本次批准后的配置入口。

**门禁实现边界：** 当前代码自动绑定完整运行配置SHA、source commit/实际导入路径、preflight及split/threshold/quarantine证据；新增独立shared协议文件的SHA由本轮本地字节核验，并要求服务器接收时执行同样检查。代码本轮未新增独立shared文件自动校验接口，不能声称该接口已经实现。原始/执行配置会记录新字段；调用现有preflight前仍须核对shared文件真实字节及其关键值与共同配置一致。正式运行申请必须附此核对及新的运行配置SHA。

## 4. 服务器P3数据验收任务说明（尚未授权执行）

以下可交给服务器Codex作为待授权任务；**本次作者裁决没有授权实际P3冻结、发布或GPU训练，当前不得执行真实扫描/冻结。** 只有另获明确P3执行批准后，按以下流程只读检查原件、写全新的外部证据目录。

1. 确认实际项目、运行容器及Python解释器。核对获批完整40字符源码commit、干净源工作树及实际模块路径；本地历史HEAD不包含新E0代码，不得冒充将来的发布SHA。若源码取得/使用权限或版本未确定，BLOCKED，不自行发布vendor/GT、连接替代仓库、安装升级依赖。
2. 对接收的共享协议和共同模板分别执行字节SHA核验；核对上节两种SHA和科学条目，不混用。使用当前声明的loss/RNG/治理/终止规则；任何变更返给作者，不在服务器改源码后继续记旧SHA。
3. 根据实际文件确认数据根路径及原始3630训练/840测试列表，核对清单SHA、真实配对、可读性、shape、0/1/2/3类别值；必须完整覆盖两组，不抽样冒充全量。只为身份和隔离读取测试GT，不运行模型/训练/测试指标。
4. 用现有 `scan`、`d4_evidence` 核对精确D4匹配和GT差异，记录原始字节及解码身份。核对parent记录的真实来源、覆盖与冲突；不能因CSV齐全就宣称来源可信，不能编造parent_id。若证据不足，先输出BLOCKED/UNKNOWN清单，**不继续生成正式开发划分**。历史004097/000717只作为待复核线索，不能假装当次已扫描。
5. 仅在P3操作授权和可靠parent证据均满足时，使用原脚本显式选择quarantine策略，输出目录必须全新、在数据及source工作树外。示意命令仅供届时填入真实值：

```bash
"$VERIFIED_PYTHON" -m research.industrial_defect_english.scripts.freeze_neu_development \
  --train-list "$VERIFIED_TRAIN_LIST" --test-list "$VERIFIED_TEST_LIST" \
  --data-root "$VERIFIED_DATA_ROOT" --family-map "$VERIFIED_PARENT_MAP" \
  --output "$NEW_EXTERNAL_FREEZE_DIR" --seed 1337 \
  --quarantine-policy-id neu_exact_d4_quarantine_candidate_v1
```

6. 全训练D4+可信parent连通组中，凡与测试确认同源的训练种子所在整组均quarantine。原件和官方列表原位不变。保存D4证据、GT差异、隔离原因/样本ID、manifest、各集合ID指纹及SHA。核对唯一训练ID的互斥并集：`3630 = eligible_train_count + quarantined_train_count`；`eligible = development_train ∪ development_validation`且不交；组不跨子集。不可强求2904/726。
7. 验证开发各类阳性例数/像素分母、缺类/小样本、原test排除；面积与复杂度阈值只用eligible中分出的开发训练manifest调用现有build_conditions.py构建、保存SHA，不能把隔离或验证GT用于阈值拟合。未知parent或其他未解决同源继续BLOCKED。D4不能排除裁剪、重采样或颜色变化，不能宣布全面独立性。
8. 汇报真实原始/eligible/quarantine/开发样本数、数据与清单指纹、已获/未知parent证据、类别及阈值、异常清单和SOURCE SHA。GPU/磁盘/软件成本资料仅按届时授权核验，不能擅自训练或推理测速。重新提交三组E0运行申请与实际预检配置；任何GO_READY只是申请条件，不替代训练批准。

P3交付应包括现有脚本生成的d4_matches.json、quarantine.jsonl、dev_train/validation.jsonl、split_audit.json、训练阈值文件及SHA；父系不足时改交阻塞报告与证据，不伪造FROZEN。原始图像、GT、checkpoint与大日志留服务器；是否可回传元数据仍按来源/隐私边界审核，不能自动git add/commit/push。

## 5. 本轮执行和证据

实际新增仅两份配置、本冻结记录及evidence中的输入SHA基线、CPU日志、定点核对JSON和最终SHA清单；现有模型、源码、测试文件和P2.7模板均未修改。当前本地main/HEAD为`0426b978ef76cd082a060874b3eff2a16184ce6d`；原`.gitignore`修改及未跟踪研究文件保留，暂存区为空。未fetch、提交、推送、访问服务器、执行真实P3或训练。

- 原有 **68项CPU合成回归全部通过**，14.015秒：[science_freeze_cpu_regression.txt](evidence/science_freeze_cpu_regression.txt)，SHA `13ba8f7035053be2ecab5b0dc6d94a8bd0f490e6e37dbb3f93ae3b64e96c877b`。
- 另4项定点核对通过：[science_freeze_validation.json](evidence/science_freeze_validation.json)：共享/共同配置字节与关键值一致；科学与操作批准分离；实际调用二类logits/非零目标正确；科学仅批准配置在preflight和optimizer前拒绝训练。合成检查不构成服务器GO_READY或性能结果。
- 模型源码快照SHA仍为`dafcbc117e20e9c5100fb5731f7eca858679e3260711e22d4b727e6428ac638f`。该算法覆盖现有工程Python及vendor provenance，**不包含配置JSON**；配置及共享文件分别按上节字节SHA跟踪，不混称为源码commit。尚无获准发布的新E0 commit。
- 历史GT与vendor权利继续按[P2.7权利决策附件](evidence/p2_7_loss_rights_decisions.md)保留阻塞；科学协议批准不提供第三方再分发权，也未改变公开历史。

最终状态：**SCIENTIFIC_PROTOCOL_FROZEN；DATA_FREEZE_NOT_AUTHORIZED；SOURCE_PUBLICATION_NOT_AUTHORIZED／RIGHTS_BLOCKED；GPU_TRAINING_NOT_AUTHORIZED；RESOURCE_CAPS_PROVISIONAL。** 本轮在此停止。
