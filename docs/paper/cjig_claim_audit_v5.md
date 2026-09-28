# CJIG v5 claim 审计 (cjig_claim_audit_v5)

> 对象：paper_draft_cjig_v5_submission_candidate.md
> 日期：2026-09-28

## 一、强表述扫描（全文 grep 实测）

| 关键词 | 命中 | 判定 |
|---|---|---|
| 显著提升/大幅提高/全面优于/普遍提升/稳定优于/所有类别均/跨域 | 0 | ✓ |
| significantly/substantially/consistently outperform/superior generalization | 0 | ✓ |
| robust | 1（文献 [10] 标题内） | 合法，非本文 claim |
| "泛化能力不足"（引言，指传统方法） | 1 | 合法背景表述 |
| "显著的类别间形态差异"（讨论，指数据属性） | 1 | 合法，非统计 claim |

## 二、关键数字全文一致性（grep 计数）

| 数字 | 出现次数 | 含义核验 |
|---|---:|---|
| 91.45 / 91.24 / +0.21 | 多处 | NEU 主结果，与 `long240k_endpoint_results.json` 一致 ✓ |
| 90.97 / 89.91 / +1.06 | 7 / 4 / 5 | Leather 主结果，与 `leather_paper_final_results.json` 一致 ✓ |
| 37.37 / 38.16 | 多处 | NEU 200² FPS（共享环境） ✓ |
| 64.68 / 62.83 | 2.6 节 | Leather 768² FPS（独占环境，已标注） ✓ |
| 85.18 / 109.83 / 150.62 | 2.6 节 | 基线 768² FPS 同环境实测 ✓ |
| 84.21 / 86.20 / 88.24 | 表 4 | 基线 test468 两轮复评一致 ✓ |
| 29.37 / 29.46 / 0.31% | 多处 | 参数量 ✓ |
| 1638 / 235 / 468 / 3630 / 840 | 多处 | 划分口径一致 ✓ |

## 三、禁止数字残留（Gate 6）

127.9 / 115.6 / 71.6 / +1.1(旧) / 91.3(旧 NEU) / 87.21 / 84.99 / −2.22 / 12.46 / ELMM：**0 命中**。

## 四、训练协议表述合规

- 无"相同训练轮次/same budget/完全相同训练条件"表述；
- 无 80000/160000/78k/130k 等迭代细节讨论；
- Leather 仅表述为"验证集选择 + 468 张独立测试图像评价"。

## 五、新增 claim 的证据映射（相对 v4 新增项）

| 新 claim | 证据 | 判定 |
|---|---|---|
| "高于相同协议下复现的 STDC、PIDNet、DDRNet"（摘要/结论） | mainstream_comparison_audit_v5.md，Level A | ✓（注意限定词"相同协议下复现的"已保留） |
| 表 4 基线 mIoU/Params | baseline_reverify + baseline_efficiency_768.json | ✓ |
| "本文模型帧率低于轻量方法但仍实时" | 64.68/62.83 vs 85–150 FPS | ✓ 如实双向呈现 |
| 图 2/3/4 选例规则 | Fig2 manifest / case_selection.json / final_viz_manifest.json | ✓ |

## 六、结论

v5 无超出证据的 claim；Leather 叙事为"总体正向 + 类别依赖"，NEU 叙事不变。claim 审计 **PASS**。
