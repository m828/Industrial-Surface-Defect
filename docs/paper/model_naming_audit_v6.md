# 模型命名审计 v6 (model_naming_audit_v6)

> 日期：2026-09-28
> 结论：**v6 起正式正文统一使用 DSMONet / DSMONet-B / A2MS-DSMONet / A2MS-DSMONet-B；Base-B 完全退出正文。**

## 一、命名决策依据

1. **IDENTITY-A 实证**（`experiments/audit/neu_dsmonet_base_identity.md`）：
   - `new/model_dsmo_rs50.py` ≡ `subregion unet/model_dsmo_rs50.py`（≡ 大论文 DSMONet-B）；
   - state_dict keys/shapes 完全一致；参数量均 29,370,629；
   - 同一 240k 权重 strict 加载、相同输入 forward：`max_abs_diff = 0.0`。
2. "Base-B" 为 2026-05-18 `experiment_entry_mapping.md` 整理期引入的人工简称，非独立模型。
3. 既然架构同一，继续用 Base-B 会制造"看似新 baseline"的误导；恢复 DSMONet-B 本名同时保证学术连续性。

## 二、正式命名体系（锁定）

| 名称 | 含义 | 代码文件 | 论文角色 |
|---|---|---|---|
| DSMONet | 细节—语义相互优化网络（通用基础架构） | — | 第一层方法 |
| DSMONet-B | DSMONet，ResNet-50 主干 | `new/model_dsmo_rs50.py` | 基础模型，NEU 91.24 / Leather 89.91 |
| A2MS-DSMONet | DSMONet + AAM + 辅助细节监督 | — | 第二层方法 |
| A2MS-DSMONet-B | A2MS-DSMONet，ResNet-50 主干 | `new/model_dsmo_rs50_eSE_adapt_detailloss_fixed.py` | 完整模型，NEU 91.45 / Leather 90.97 |

废弃：Base-B、baseline-B、基础B模型、A2MS-DefectNet（v4 前旧名）。

## 三、v6 执行记录

- 正文：`paper_draft_cjig_v6_dsmonet_a2ms_integrated.md`，`DSMONet-B` 30 处，覆盖摘要（中/英）、贡献、方法、NEU 表1/表2、Leather 表4/表5、消融表6/表7、效率表8、图1/3/4 图注、讨论、结论。
- 图件：Fig1/3/4 全部重渲（脚本 `fig1_architecture_v6.py` / `visualize_cases_v6.py` / `leather_viz_final_v6.py` / `assemble_fig34_v6.py`），图内文字无 Base-B。
- 自动核验：`grep -c "Base-B" v6正文` = 0；对 figures 目录新生成 PNG 目检确认列标题为 DSMONet-B。

## 四、不受影响范围

- audit 档案（`experiments/audit/*`、`repro_runs/*` 旧脚本、v3–v5 文档）保留历史命名，用于溯源；
- checkpoint 文件名（如 `dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl`）不改；
- `docs/paper/neu_baseline_naming_recommendation.md` 中的"方案 B"建议已被本轮用户决策取代（用户选择全面恢复 DSMONet-B 命名，即原方案 A 的加强版）。

## 五、首次定义句（1.1 节，全文锚点）

> 采用 ResNet-50 主干的模型记为 DSMONet-B 与 A2MS-DSMONet-B。

2.1 训练设置段补充模型角色定义：

> DSMONet-B 为采用细节—语义互优化机制的基础网络（ResNet-50 主干）；A2MS-DSMONet-B 为在 DSMONet-B 上引入自适应语义增强模块与辅助细节监督分支的完整模型。
