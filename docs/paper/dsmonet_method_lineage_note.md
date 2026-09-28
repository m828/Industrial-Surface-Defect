# DSMONet 方法谱系说明 (dsmonet_method_lineage_note)

> 日期：2026-09-28
> 用途：明确当前论文方法与技术历史的关系，供引言/相关工作/披露使用。

## 谱系

```
历史 DSMONet（2023–2024，subregion unet/model_dsmo_rs50.py）
  = 大论文中的 DSMONet-B（ResNet-50 主干）
  = 旧英文撤稿稿的核心结构
        │  共享组件：ResNet-50 主干、DAPPM、SqueezeBodyEdge、UAFM×2、
        │           Laplacian 浅层细节、edge_fusion、共享 SegHead、3 路训练输出、
        │           mask-aware(bce) 监督、OHEM-CE
        ▼
当前论文 Base-B（new/model_dsmo_rs50.py）
  = 与历史版同一计算图（key/shape/输出逐位一致，max_abs_diff=0.0）
  = 审计期命名（2026-05 experiment_entry_mapping.md 首次引入 "Base-B"）
        │  新增组件：AAM（eSE 通道注意力 + ACW 自适应通道权重）、
        │           细节监督分支（seg_head_detailloss + detail_aggregate_loss，仅训练态）
        │  工程修正：forward 主输出固定为 head_seg1 = out（修复历史整理期引入的退化）
        ▼
当前论文 A2MS-DSMONet-B（new/model_dsmo_rs50_eSE_adapt_detailloss_fixed.py）
```

## 训练与评价变化（相对历史链）

| 维度 | 历史链 | 当前论文 |
|---|---|---|
| 训练预算 | 分阶段 resume 至 165k–240k | 单次 from scratch 240k |
| LR | 常数 1e-4 | 常数 1e-4（同） |
| 模型选择 | test-monitor 峰值（832/840） | 固定终点，无 test 选择 |
| 最终评价 | 840（test 脚本）/ 832（monitor） | 统一 840、batch1 |
| Leather | val235 选择 + test468 终评 | 同（沿用历史链权重统一复评） |

## 论文写作纪律

- 不把 Base-B 写成新设计；Base-B 即历史 DSMONet-B 的当前协议实例。
- A2MS-DSMONet 的新增性仅限：AAM、细节监督分支、以及整套干净协议与细粒度分析。
- 与撤稿旧稿的关系披露见 `prior_publication_disclosure_draft_v5.md`。
