# 历史英文稿重叠关系审计 (prior_publication_overlap_audit_v5)

> 日期：2026-09-28
> 对象：历史英文稿《Real-time Semantic Segmentation via Mutual Optimization of Spatial Details and Semantic Information》（已撤稿）与当前中文稿 v5。
> 重要限制：**该英文稿全文未在服务器上找到**（已搜索 /workspace 及项目目录，无 PDF/TeX/Word）。本审计基于代码世系、checkpoint 世系、实验记录推断；文字级重叠为 UNCONFIRMED，需用户提供旧稿全文后复核。

## 一、方法层重叠（依据代码世系）

| 内容 | 世系证据 | 与旧稿关系判断 |
|---|---|---|
| 细节—语义互优化主干（SqueezeBodyEdge / UAFM / DAPPM / arm2） | `subregion unet/model_dsmo_rs50.py`（2023 年文件）即历史 DSMONet | **历史已有**（旧英文稿核心机制） |
| Base-B = ResNet-50 + 互优化 | `new/model_dsmo_rs50.py` 与其一致 | **历史已有** |
| AAM（eSE + ACW） | 历史 `model_dsmo_r50_eSE_adapt_detailloss.py` 已含 eSE/ACW 结构 | **大部分历史已有**；本次贡献在于修复 forward（head_seg1=out 实证）、统一协议评价与机制分析 |
| 细节监督分支 | 历史 detailloss 链已存在（2023-09 subregion pige_loss） | **历史已有** |
| 当前论文真正新增 | 见下节 | — |

## 二、当前稿真实新增内容（实证支持）

1. 发现并修复历史代码的 forward 退化（head_seg1 路径），明确论文模型计算图；
2. NEU-Seg 固定终点、无测试集选择的 240k 干净协议与多 seed 10k 消融（2×2 factorial）；
3. 边界质量（BIoU/BF-score 多容差）、尺度分层、形态复杂度等细粒度机制分析；
4. Leather 历史 checkpoint 的统一 test468 复评与完整溯源链（sha256 级）；
5. 统一的复杂度/实时性测量协议；
6. 类型依赖性结论（patches 正向 / crazing 权衡 / wart 回退）——旧稿无此分析层级。

## 三、实验数据重叠

| 数据 | 旧稿（推断） | 当前稿 | 重叠判断 |
|---|---|---|---|
| NEU-Seg 数字 | 历史 91.3–91.6 系 test 监控选优产物（审计已证） | 固定终点 91.45/91.24 | **数字不沿用旧稿** |
| Leather 数字 | 约 91.0（val235 best 0.909084） | test468 复评 90.97/89.91，checkpoint 相同 | **权重相同、评价口径不同且已披露** |
| 图件 | 旧稿图未知 | Fig1 按 fixed 架构重绘；Fig2/3/4 全部本轮新生成 | 低重叠 |

## 四、重叠风险分级

| 维度 | 等级 | 说明 |
|---|---|---|
| method | **MODERATE** | 核心机制同源（这是论文定位允许的），AAM/细节监督为既有结构的规范化呈现 |
| text | LOW–MODERATE | 中文重写、结构与叙事重组；无旧稿全文无法逐字比对（UNCONFIRMED） |
| figures | **LOW** | 全部重绘/新生成，无复制 |
| experiments | **MODERATE** | NEU 全新实验；Leather 复用历史权重但评价口径更新且文中如实说明协议 |
| overall | **MODERATE** | 属"同一研究线的延续与规范化"，非独立无关工作 |

## 五、建议

1. 投稿时主动向编辑披露旧英文稿的存在与撤稿状态（见 disclosure 草案）；
2. 请用户提供旧英文稿 PDF 以完成文字级重叠复核；
3. 若期刊要求，在正文引言或脚注中加一句相关工作说明。
