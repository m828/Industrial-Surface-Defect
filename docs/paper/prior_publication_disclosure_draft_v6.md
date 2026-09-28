# 历史撤稿披露草案 v6 (prior_publication_disclosure_draft_v6)

> 日期：2026-09-28
> 用途：投稿系统 / Cover Letter 需要时使用；**不进入正文**。
> 状态：草案，事实部分需用户最终确认后提交。
> v6 更新：按用户说明更正撤稿表述（旧稿**曾发表后撤稿**，撤稿原因与期刊同行评审流程问题有关，不涉及作者数据造假或学术不端）；按两层主线重写技术关系。

## 事实陈述草案（中文，供 Cover Letter 改写）

本稿与作者团队此前英文稿件《Real-time Semantic Segmentation via Mutual Optimization of Spatial Details and Semantic Information》属于同一研究方向的延续工作。该稿件曾发表后撤稿（retracted），撤稿原因与期刊同行评审流程问题有关，不涉及作者数据造假或学术不端。旧稿提出了细节—语义互优化的基础网络结构，即本稿第一层方法 DSMONet 的技术来源。

相对旧稿，本稿的主要差异与新增内容包括：

1. 对网络输出路径进行了核查与修正，明确了正式模型计算图（主输出取自细节—语义融合特征）；
2. 将自适应语义增强模块（AAM）与辅助细节监督作为面向工业表面缺陷的任务适配层，给出了系统消融（多随机种子 2×2 组合）与长训练终点验证；
3. 采用不使用测试集进行模型选择的评价协议重新开展 NEU-Seg 实验（固定 240k 训练终点、统一 840 张测试图像）；
4. 新增缺陷类别、边界质量、尺度分层与形态复杂度等细粒度分析，结论呈现缺陷形态依赖性；
5. 皮革数据集结果为对历史训练所得模型权重在统一测试协议（468 张独立测试图像）下的评价，评价集与协议均在文中说明，并新增同协议下 STDC/PIDNet/DDRNet 实时分割方法的对比。

## 英文要点（供 Cover Letter 使用）

- The previous article was **retracted**; the retraction concerned the journal's peer-review process and involved no data fabrication or academic misconduct by the authors.
- The present manuscript extends the earlier detail-semantic mutual optimization architecture (DSMONet) with defect-oriented enhancements (AAM and auxiliary detail supervision), and reports new experiments under protocols that do not use the test set for model selection.

## 待用户确认的事实点

- [ ] 旧稿发表与撤稿的具体时间、期刊与 DOI（本文未掌握，需作者确认）；
- [ ] 旧稿是否在任何平台以预印本形式公开（如 arXiv/ResearchSquare）；
- [ ] 撤稿通知的正式措辞（如有截图/邮件，建议存档）；
- [ ] 目标期刊投稿系统是否有"相关工作披露"字段（CJIG 官网未明示，建议投稿信中主动说明）。
