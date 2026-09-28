# 历史英文稿重叠关系审计 v6 (prior_publication_overlap_audit_v6)

> 日期：2026-09-28
> 对象：历史英文稿《Real-time Semantic Segmentation via Mutual Optimization of Spatial Details and Semantic Information》（曾发表后撤稿）与当前中文稿 v6。
> 重要限制：**该英文稿全文未在服务器上找到**（已搜索 /workspace 及项目目录，无 PDF/TeX/Word）。本审计基于代码世系、checkpoint 世系、实验记录推断；文字级重叠为 UNCONFIRMED，需用户提供旧稿全文后复核。
> 相对 v5 审计的变化：v6 将 DSMONet 作为论文第一层方法完整写入正文（1.2 节），方法维度重叠如实上调。

## 一、方法层重叠（依据代码世系）

| 内容 | 世系证据 | 与旧稿关系判断 |
|---|---|---|
| 细节—语义互优化主干（DAPPM / UAFM×2 / SqueezeBodyEdge / Laplacian 细节路径 / edge_fusion / 语义回流 / arm2 / 共享 SegHead） | `subregion unet/model_dsmo_rs50.py`（2023 年文件）即历史 DSMONet；`new/model_dsmo_rs50.py` 与其计算图同一（IDENTITY-A，max_abs_diff=0.0） | **历史已有**（旧英文稿核心机制），v6 完整描述 |
| DSMONet-B（ResNet-50 主干） | 同上 | **历史已有** |
| AAM（eSE + ACW） | 历史 `model_dsmo_r50_eSE_adapt_detailloss.py` 已含 eSE/ACW 结构 | **大部分历史已有**；本次贡献在于修复 forward（head_seg1=out 实证）、嵌入互优化路径的定位表述、统一协议评价与消融 |
| 辅助细节监督分支 | 历史 detailloss 链已存在（2023 年记录） | **历史已有** |
| 当前论文真正新增 | 见下节 | — |

## 二、当前稿真实新增内容（实证支持）

1. 发现并修复历史代码的 forward 退化（head_seg1 路径），明确论文模型计算图；
2. NEU-Seg 固定终点、无测试集选择的 240k 协议与多 seed 10k 消融（2×2 factorial）；
3. 边界质量（BIoU/BF-score 多容差）、尺度分层、形态复杂度等细粒度机制分析；
4. Leather 历史 checkpoint 的统一 test468 复评与完整溯源链（sha256 级），以及同协议 STDC/PIDNet/DDRNet 对比；
5. 统一的复杂度/实时性测量协议；
6. 类型依赖性结论（patches 正向 / crazing 权衡 / wart 回退）——旧稿无此分析层级；
7. 两层方法叙事（通用互优化架构 → 工业缺陷适配增强）的组织方式。

## 三、实验数据重叠

| 数据 | 旧稿（推断） | 当前稿 | 重叠判断 |
|---|---|---|---|
| NEU-Seg 数字 | 历史 91.3–91.6 系 test 监控选优产物（审计已证） | 固定终点 91.45/91.24 | **数字不沿用旧稿** |
| Leather 数字 | 约 91.0（val235 best） | test468 复评 90.97/89.91，checkpoint 相同 | **权重相同、评价口径不同且文中如实说明协议** |
| 图件 | 旧稿图未知 | Fig1 按 fixed 架构重绘（v6 增加两层标注）；Fig2/3/4 全部本轮新生成 | 概念同源、无复制 |

## 四、重叠风险分级（v6 更新）

| 维度 | v5 等级 | v6 等级 | 说明 |
|---|---|---|---|
| method | MODERATE | **HIGH** | v6 将 DSMONet 完整架构作为第一层方法写入正文，与旧稿核心机制同源；这是论文定位决定的，必须经披露管理 |
| text | LOW–MODERATE | LOW–MODERATE | 中文重写、两层叙事重组；无旧稿全文无法逐字比对（UNCONFIRMED） |
| figures | LOW | LOW–MODERATE | Fig1 与旧稿架构图描绘同一基础结构（概念同源），但全部重绘、新增层次标注；其余图件新生成 |
| experiments | MODERATE | LOW–MODERATE | NEU 全新实验；Leather 复用历史权重但评价口径更新且文中如实说明 |
| overall | MODERATE | **MODERATE** | 属"同一研究线的延续、修复与规范化"；method 维度 HIGH 通过投稿披露管理 |

## 五、建议

1. 投稿时在 Cover Letter 中主动披露旧英文稿的存在、撤稿状态与两稿技术关系（见 disclosure 草案 v6）；
2. 请用户提供旧英文稿 PDF 以完成文字级重叠复核（当前 text 维度为 UNCONFIRMED）；
3. 不在正文引用该撤稿稿；若编辑要求，按其要求以规范形式标注 retracted 状态。
