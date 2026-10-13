# CJIG v9 Final Submission Audit

日期：2026-10-13

## Scientific status

READY FOR SCIENTIFIC REVIEW

本轮最终文本审查未发现阻断性科学措辞问题。此状态表示可继续人工科学评阅，不等于全部实验限制已消除或最终投稿元数据、版式已经完成。

审查仅使用现有稿件与库内冻结证据，不训练、不运行模型推理、不添加 baseline、不改实验代码或 JSON。参考文献本轮未改写，也未重新进行外部逐条存在性核验。

## File status

| 文件 | 本轮操作 |
|---|---|
| [paper_draft_cjig_v9_submission_candidate.md](paper_draft_cjig_v9_submission_candidate.md) | 修改摘要第 23、35、37 行三段；引言以后全文不变 |
| [cjig_revision_log_v9.md](cjig_revision_log_v9.md) | 追加最终摘要修订记录，保留旧哈希并登记新哈希 |
| [leather_extension_integration_audit_v9.md](leather_extension_integration_audit_v9.md) | 更新分阶段哈希与摘要状态，提交清单与最新指令一致 |
| cjig_final_submission_audit_v9.md | 新增本报告 |

已按用户指定路径备份编辑前稿件：`/tmp/paper_draft_cjig_v9_before_final_edit.md`。备份与编辑前稿件哈希一致，未覆盖 v8。

| 对象 | SHA-256 |
|---|---|
| v9 编辑前 / 指定备份 | `882b44eaeb08f5650324b906162574e059319613811cbbdcf2cacc278e96e765` |
| v9 摘要修订后 | `0308dd3a94307bbda75aec7707b87e2c991400535ed44ebf7cad0e11fb5e65fd` |
| v8 未覆盖输入稿 | `979552a9c007c38df523d165241cb00a25114e7d10ccef74bcc3c23ee1f7327d` |

摘要调整：中文 Leather 句按要求改为另一工业材质的补充应用评价；FPS 保留 A100、95.41 FPS、batch=1、FP32，并明确为单图等效帧率、仅反映前向计算效率。英文同步精简重复的训练链限制与前向速度限制句，不删除正文中的相关说明。英文 Method 仍保留皮革验证监控选权与 468 张测试图像信息。

## Frozen results

NEU: 91.24 / 91.45

Leather: 89.91 / 90.97

单位均为 mIoU/%，顺序均为 DSMONet-B / A2MS-DSMONet-B。

### NEU baseline 状态：表 2

以下摘要直接提取自最终稿表 2；表内全部列与编辑前 v9 逐字一致，十模型逐列与 [v3 冻结源表](../../experiments/baseline_comparison/results/final_baseline_table_v3.md) 一致：

| Model | mIoU/% |
|---|---:|
| U-Net | 90.62 |
| DeepLabv3+ | 90.95 |
| BiSeNetV2 | 88.75 |
| STDC-Seg | 88.16 |
| PIDNet-S | 88.82 |
| DDRNet-23 | 89.24 |
| LETNet | 86.36 |
| FDSNet | 87.08 |
| DSMONet-B | 91.24 |
| A2MS-DSMONet-B | 91.45 |

方法分类保持经典 2 种、通用实时 5 种、工业缺陷参照 1 种、本文方法 2 种。LETNet 属于通用实时方法；DDRNet-23 命名未改变。NEU-Seg 仍是主要基准，十模型采用共同 240k 固定终点协议，不把文献报告值并入统一复现排名。

### Leather 主结果：表 5

| Method | mIoU/% |
|---|---:|
| DDRNet-23（内部复现） | 84.21 |
| PIDNet-S（内部复现） | 86.20 |
| STDC-Seg（STDCNet1446，内部复现） | 88.24 |
| DSMONet-B | 89.91 |
| A2MS-DSMONet-B | 90.97 |

主表保留既有 DDRNet-23，不删除任何历史对比或实验。表 5 与新增表 7 不合并排名，不以测试集相同推断训练协议完全一致。

### 其余冻结内容

全部 9 张 Markdown 表的单元、顺序与编辑前备份逐字一致。10k 三种子消融、240k 两模型比较、类别/边界/尺度/形态分析、皮革逐类别分析、十模型前向计时均未改变。图 1—4 图注不变，图件未编辑。

四项核心精度的全文出现次数编辑前后保持：91.24 为 6 次，91.45 为 5 次，89.91 与 90.97 均为 6 次。

## Added evidence

Leather supplementary:

- U-Net 82.42
- DeepLabv3+ 87.60
- FDSNet 78.19

最终稿表 7：

| Method | Category | mIoU/% |
|---|---|---:|
| U-Net | Classical | 82.42 |
| DeepLabv3+ | Classical | 87.60 |
| FDSNet | Industrial defect | 78.19 |

三项值与 [Leather 扩展结果 JSON](../../experiments/baseline_comparison/leather_extension/leather_baseline_extension_results.json) 一致，均为本文补充复现，不是文献报告值。三模型采用独立的 60k 固定终点、无验证/测试监控选权方案；与主表本文两模型的验证监控选权、批量和学习率设置不匹配。因此只补充考察不同分割方法在皮革场景中的表现差异，不计算跨表提升、不作全方法排名。

FDSNet 仅作为具有公开可复现实现的工业缺陷参照，不代表全部工业方法。Leather 定位为真实工业场景中另一材质的应用补充；模型在相应材质训练集上训练后评价，不是钢材训练权重直接测试皮革。

## Claim 风险检查

执行用户指定英文词的 grep 筛查，得到 4 个词级命中，并逐项阅读上下文：

| 行号 | 命中 | 判断 |
|---|---|---|
| 35 | best | “without representing each method's best original recipe”；限定统一方案不等于原方法最佳训练方案，不是本文最佳 claim |
| 37 | higher | “higher than eight non-proposed comparison implementations”；明确限定 NEU-Seg 统一固定终点的 8 个非本文提出的复现模型，允许保留 |
| 39 | higher | “do not imply uniformly higher measured speed”；否定从低 MACs 推导普遍速度优势 |
| 370 | superior | PP-LiteSeg 原论文参考文献题名；不是本文结果 claim，不擅改原题名 |

其余目标词 outperform、state-of-the-art、real-time deployment、generalization、domain generalization、faster 为 0 命中。没有主动的 Leather 优越性、跨材质提升、工业领域最佳或速度最优结论；中文讨论中的相关否定句是限制说明，不作为主动 claim 统计。

中文摘要的 95.41 FPS 明确是指定 A100、batch=1、FP32 的单图等效帧率；英文还保留共享环境、200×200 输入及中位延迟口径。所有速度解释仍是 forward inference benchmark / single-image equivalent FPS，不是端到端部署或生产线吞吐。

两路独立只读复核（数值/协议与措辞/论证）均未发现本轮修改的阻断项。

## Limitations retained

- single seed endpoint
- Leather protocol mismatch
- shared GPU timing
- incomplete reproduction of recent industrial methods

上述限制继续在实验设置、相关表注与 Discussion 保留。皮革训练链未完全匹配见 2.1、2.5、表 5 注、3.3、3.4；新增表 7 与主表不能跨表排名见 2.6、3.3、3.4。单种子 +0.21 个百分点不解释为稳定或显著的训练增益，皮革 +1.06 个百分点不解释为严格同预算因果消融。

## 禁止项检查

按用户指定命令执行：

```bash
grep -R "87.21\|84.99" docs/paper/paper_draft_cjig_v9_submission_candidate.md
```

结果：0 命中，grep 退出码 1 表示无匹配；另以转义小数点的正则复核，仍为 0 命中。

废弃模型名称、旧组件名、禁止的历史精度字符串和用户指定的主动过强比较字符串均为 0 命中；新增皮革 FPS 未进入正文。

行为与文件检查：

- 无重新训练、无模型推理执行、无新增 baseline、无已有实验删除。
- 无实验代码、JSON、checkpoint、数据、权重或图件修改。
- 三份 Leather 证据文件哈希与初次整合登记一致。
- v8 哈希不变，编辑前 v9 备份完整。
- 未执行广域 Git 暂存，也未执行任何其他 Git 暂存、commit 或 push。
- 暂存区为空，已跟踪文件无修改；本轮四份文档仍为未跟踪 Markdown。

投稿前尚需人工核定稿件已有的作者、单位、中图分类号、基金等占位信息；本审计不替代最终期刊版式和元数据核对。

## Git 提交清单

仅允许人工确认后提交以下四份文件：

1. `docs/paper/paper_draft_cjig_v9_submission_candidate.md`
2. `docs/paper/cjig_revision_log_v9.md`
3. `docs/paper/leather_extension_integration_audit_v9.md`
4. `docs/paper/cjig_final_submission_audit_v9.md`

不提交 checkpoint、repro_runs、logs、third_party、dataset、weights 或其他实验资产；其余 10 份历史 v7/v8 未跟踪文档不纳入此次清单，也不删除。本地 v8 与旧审计继续保留，不声称其已在 GitHub 提交；如需补入历史版本链，须另行确认。

建议 commit message：

```text
docs(paper): finalize CJIG v9 wording and submission audit
```

当前阶段只准备清单，等待人工确认；未自动执行提交命令。

## Git status

```text
On branch main
Your branch is up to date with 'origin/main'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	docs/paper/baseline_integration_audit_v7.md
	docs/paper/baseline_section_audit_v8.md
	docs/paper/cjig_claim_audit_v7.md
	docs/paper/cjig_claim_final_audit_v8.md
	docs/paper/cjig_final_submission_audit_v9.md
	docs/paper/cjig_revision_log_v7.md
	docs/paper/cjig_revision_log_v8.md
	docs/paper/cjig_revision_log_v9.md
	docs/paper/cjig_table_list_v7.md
	docs/paper/leather_extension_integration_audit_v9.md
	docs/paper/paper_draft_cjig_v7_baseline_integrated.md
	docs/paper/paper_draft_cjig_v8_submission_candidate.md
	docs/paper/paper_draft_cjig_v9_submission_candidate.md
	docs/paper/submission_remaining_items_v7.md

nothing added to commit but untracked files present (use "git add" to track)
```
