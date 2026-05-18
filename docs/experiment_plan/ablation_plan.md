# 消融实验整理计划

## 已有消融结果

当前固定结果中已有 ELMM、AAM 和联合损失消融实验。后续论文更新时必须以 `experiments/results/fixed_existing_results.md` 为准，不从口头描述或临时日志直接写入论文。

## 待整理事项

1. 核对 ELMM 消融中各模块名称与论文方法章节一致；
2. 核对 AAM 插入数量、插入位置和表格字段；
3. 核对联合损失中 `l0 + l1 + l2`、`ledge`、`lmask` 的公式、权重和实现来源；
4. 在论文中明确边缘细节损失借鉴 STDCNet，掩码感知损失借鉴 Sub-region UNet，不将两个损失本身写成原创。

## 记录位置

- 固定结果：`experiments/results/fixed_existing_results.md`；
- 新结果：`experiments/results/new_results_pending.md`；
- 论文更新协议：`docs/paper/paper_update_protocol.md`。
