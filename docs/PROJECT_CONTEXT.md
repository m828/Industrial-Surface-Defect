# PROJECT_CONTEXT.md — AI 辅助论文开发项目上下文

> 用途：为后续 Codex / 本地 AI 辅助论文修改提供完整、权威的项目上下文。
> 建立日期：2026-09-28
> 效力：**本文件是项目事实源之一**。任何 AI 辅助修改必须遵守本文件的命名、数字与写作纪律。
> 纪律：不修改论文正文数字；不修改实验结果；不重新实验（除非用户明确指令）。

---

## 1. Project Overview

- **项目名称**：Industrial Surface Defect Semantic Segmentation
- **当前目标**：投稿中文期刊《中国图象图形学报》（CJIG 方向）
- **论文当前版本**：`docs/paper/paper_draft_cjig_v6_1_submission_candidate.md`（CJIG v6.1 submission candidate）
- **论文标题**：细节—语义互优化与自适应增强的工业表面缺陷实时语义分割网络

**一句话主线**：

> 构建细节—语义互优化基础架构 DSMONet，并针对工业缺陷低占比、多尺度和形态差异特点，引入自适应语义增强与训练阶段细节监督形成 A2MS-DSMONet，通过公开 NEU-Seg 和真实皮革数据验证。

**方法演进链（论文叙事结构，不可退回）**：

```
通用实时语义分割矛盾（高层语义 vs 局部细节）
        ↓
DSMONet：细节—语义互优化基础架构
        ↓
工业缺陷任务特性（低占比 / 弱纹理 / 多尺度 / 形态差异）
        ↓
A2MS-DSMONet：AAM + 辅助细节监督（仅训练）
        ↓
NEU-Seg + Leather 双数据集验证
        ↓
类别 / 边界 / 尺度 / 形态复杂度细粒度分析
```

---

## 2. Model Hierarchy

### DSMONet-B

- 细节—语义互优化基础架构，ResNet-50 backbone；
- 当前论文的基础模型（不是"baseline 配角"，是第一层方法）；
- 参数量：29,370,629；
- 代码：`new/model_dsmo_rs50.py`（sha256 `fab316c5…`）；
- 与 2023 年历史版 `subregion unet/model_dsmo_rs50.py` 计算图同一（IDENTITY-A：state_dict keys/shapes 一致、同权重前向 max_abs_diff=0.0），证据见 `experiments/audit/neu_dsmonet_base_identity.md`。

### A2MS-DSMONet-B

= DSMONet-B + 两项面向工业缺陷的增强：

1. **AAM（自适应语义增强模块）**
   - eSE 高效通道注意力（GAP + 1×1 Conv + HSigmoid）；
   - ACW 自适应通道权重（可学习逐通道权重）；
   - 嵌入互优化路径的通道重标定位置，输出同时参与细节融合与语义回流；
2. **辅助细节监督（detail supervision）**
   - 仅训练阶段（training only），推理时移除，不增加推理开销。

- 参数量：29,460,804（较 DSMONet-B +0.31%）；
- 代码：`new/model_dsmo_rs50_eSE_adapt_detailloss_fixed.py`（sha256 `a0e6fd52…`）；
- 关键 forward 事实：主输出 `head_seg1 = out`（arm2 细节—语义融合特征）。历史整理期曾出现 `head_seg1 = high_feats` 退化，已修复并实证（`experiments/audit/a2ms_b_fixed_forward_audit.md`）。

### 命名禁令

- **禁止使用 Base-B**：该名称只是 2026-05 实验整理期人工简称，仅存于历史 audit 档案，正文/新文档一律用 DSMONet-B；
- 同样废止：A2MS-DefectNet、Base-S、baseline-B。

---

## 3. Historical Relation

- DSMONet 与团队此前工作同源（旧英文稿曾发表后撤稿，原因与期刊同行评审流程有关，非学术不端；披露草案见 `docs/paper/prior_publication_disclosure_draft_v6.md`，仅用于 Cover Letter，不进正文）。
- 当前论文将方法重新组织为：**DSMONet 基础架构 + 工业缺陷适配增强**。

### 历史 90.2% 的正确理解

硕士大论文中 DSMONet-B = 90.2% 与当前 91.24% **不是旧模型和新模型的差异**，而是同一架构在不同实验链下的两个结果：

- 训练预算不同（历史 test-monitor 峰值约 52k–60k 阶段 vs 当前 240k 固定终点）；
- 评价协议不同（历史 832/840 monitor 口径 vs 当前统一 840 全量终评）。

证据链：`experiments/audit/neu_90_2_vs_91_24_diff.md`、`neu_historical_90_2_evidence.md`、`neu_dsmonet_training_trajectory.md`。

**禁止表述**："DSMONet 从 90.2 提升到 91.24"。90.2% 不进入正文；正文唯一正式数字为 91.24%。

---

## 4. Frozen Experimental Results（冻结，不得改动）

### NEU-Seg（train 3630 / test 840，200×200，4 类含背景，240k 固定终点，seed 1337，from scratch，无测试集模型选择）

| 模型 | mIoU/% | 背景 | 裂纹 | 夹杂物 | 斑块 | FPS | Params/M |
|---|---:|---:|---:|---:|---:|---:|---:|
| DSMONet-B | 91.24 | 98.62 | 84.15 | 93.49 | 88.71 | 38.16 | 29.37 |
| A2MS-DSMONet-B | **91.45** | 98.64 | 84.47 | 93.55 | **89.16** | 37.37 | 29.46 |

- Δ = **+0.21 pt**；参数量 **+0.31%**；MACs 持平（6.648G）。
- 10k 消融（3 种子 mean±SD）：DSMONet-B 85.22±0.41 / +AAM 85.70±0.43 / +细节监督 85.44±0.19 / 联合 85.75±0.54。
- 效率环境：A100，batch1，FP32；NEU 为共享 GPU 交错测量（相对值），Leather 为独占测量。

### 评价协议要点

- TestRescale + ToTensor，**不使用 ImageNet Normalize**；mIoU 含背景；pred=outputs[0]；batch=1。

---

## 5. Leather Dataset（真实工业皮革数据）

- 划分：train 1638 / val 235 / test 468；768×768；7 缺陷类 + 背景（8 评价类）。
- 协议：历史训练链以 val235 选择最优 checkpoint；最终仅在 test468 评价一次。**与 NEU 的固定终点协议不同**，正文已分别说明。

| 模型 | mIoU/%（test468 统一复评） |
|---|---:|
| DSMONet-B | 89.91 |
| A2MS-DSMONet-B | **90.97** |

- Δ = **+1.06 pt**；7 类缺陷中 6 类正向，**刺猴（wart）−3.75 pt** 回退（如实报告）。
- 同协议复现对比：STDC-Seg 88.24 / PIDNet-S 86.20 / DDRNet23-slim 84.21。
- 定位：**跨材质适用性考察**。**禁止**写成"泛化能力证明"。
- 已废弃的旧 60k 结果（87.21/84.99）仅存于 audit，永远不得回到正文。

---

## 6. Mechanism Findings（机制结论，必须保持双向性）

**不是**所有类别统一提升：

- **斑块（patches）**：区域与边界质量同步改善（BIoU@1/2/3px +0.55/+0.54/+0.55 pt，bootstrap CI 不含零）；
- **裂纹（crazing）**：区域完整性改善（空洞填补，聚合 IoU +0.32 pt），但边界定位存在 trade-off（典型样本 BIoU/BF1 略降；形态复杂度越高回退越明显，高复杂度组 ΔIoU −1.74 pt）；
- **夹杂物（inclusion）**：基本稳定。

**总结论**：enhancement is morphology-dependent（增强效果具有缺陷形态依赖性）。

---

## 7. Writing Rules（写作纪律）

**禁止**（无依据强 claim）：

- 显著提升、大幅、全面优于、普遍提升、所有类别提升、证明泛化能力、首次提出/全新提出。

**允许**：

- 改善、增强、数值提高 X 个百分点、体现应用潜力、形态相关差异、在一定条件下取得较好结果。

**其他纪律**：

- 不使用"基础模型/基础网络"指代 DSMONet-B（用"基础架构"或直接指名）；
- 不做时间优先声明（"首先构建"已清除，统一为"本文构建"）；
- 比较范围必须写清（"优于本文复现的 STDC、PIDNet 和 DDRNet 方法"）；
- 内部审计词（取证/审计/manifest/锁定/冻结/复评）不进正文；
- mask-aware loss 与 detail supervision 思想均有已有来源，不得声称原创损失函数。

---

## 8. Important Files

| 路径 | 用途 |
|---|---|
| `docs/paper/` | 论文稿（当前候选 v6.1）、修订日志、claim 审查、命名审计、主线说明、投稿清单；`figures/` 子目录含 Fig1–4 终版（SVG/PDF/600dpi PNG）及图 2/3/4 选例 manifest |
| `experiments/audit/` | 全部审计证据：NEU 历史结果溯源（90.2 vs 91.24）、Leather checkpoint 取证、统一复评 manifest、边界/尺度/形态/消融结果 JSON |
| `experiments/configs/` | 4 个正式训练配置（NEU/Leather × DSMONet-B/A2MS-DSMONet-B），论文结果对应 |
| `experiments/protocol/` | 评价协议锁定记录；`splits/` 含双数据集划分清单与类别映射 |
| `checkpoint_metadata/` | 论文 checkpoint 元信息（sha256/iter/seed/mIoU）；**权重本体不上传** |
| `experiments/results/` | 类别 IoU、复杂度、小目标分组等结果表 |
| `tools/` | 统一评价（`evaluate_class_iou.py`）、复杂度测量等脚本 |
| `docs/repository_sync_report.md`、`docs/missing_files_report.md` | 仓库同步记录与差异扫描 |
| `docs/paper/prior_publication_overlap_audit_v6.md`、`prior_publication_disclosure_draft_v6.md` | 旧稿重叠审计与撤稿披露草案（Cover Letter 用） |

---

## 9. Future AI Instructions（后续 AI 修改论文时必须遵守）

1. **实验数字冻结**：第 4、5 节所有数字不得改动；任何新数字必须来自新的、经用户批准的实验与审计；
2. **不改变模型命名**：DSMONet / DSMONet-B / A2MS-DSMONet / A2MS-DSMONet-B 锁定；
3. **不恢复 ELMM**：该模块已从论文模型中移除，任何版本不得回写；
4. **不使用旧 FPS**（127.9 / 115.6 / 71.6 等旧链数字禁用）；效率数字以第 4 节为准；
5. **不使用 90.2 作为正文结果**（仅 audit 档案）；
6. **所有 claim 必须有证据**：引用 `experiments/audit/` 或 `docs/paper/cjig_claim_audit_v6.md` 中的具体证据文件；
7. 不改动数据划分、评价协议、图件选例规则；
8. 修改后必须重跑禁用词检查。

### 废弃术语终检（针对论文正文文件）

论文正文（`paper_draft_cjig_v6_1_submission_candidate.md`）中不得出现：

- `Base-B`
- `A2MS-DefectNet`
- `ELMM`

当前状态（2026-09-28 核验）：三项均为 **0 命中**。

---

*本文件由 2026-09-28 项目状态整理生成；如后续实验或论文版本更新，应同步更新本文件。*
