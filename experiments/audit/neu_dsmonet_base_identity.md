# Base-B 与历史 DSMONet-B 模型身份审计 (neu_dsmonet_base_identity)

> 日期：2026-09-28
> 结论：**IDENTITY-A —— 当前 Base-B 与历史 DSMONet-B 为同一计算图，仅命名不同。**

## 一、实证结果（本轮实测，inference only）

| 检查项 | 结果 |
|---|---|
| `new/model_dsmo_rs50.py` SHA256 | `fab316c5…`（md5 `112b00fa…`） |
| `new (copy)/model_dsmo_rs50.py` | 与 new/ **逐字节相同** |
| `subregion unet/model_dsmo_rs50.py` | 文本 diff 仅 3 处：2 处注释/空行；1 处 subregion 保留 3 行"先赋值 logit_list 再在分支内重写"的**死代码**（dead store），forward 行为无差异 |
| state_dict key 集合 | **完全相同**（逐 key、逐 shape，0 处不匹配） |
| 参数量 | 两文件实例化均 **29,370,629** |
| 同一组权重（当前 Base-B 240k checkpoint）分别加载后同输入前向 | **max_abs_diff = 0.0**（严格加载 strict=True 均成功） |
| 2023 年编译 pyc 与现行源文件函数级指纹 | 此前审计已证完全一致（`historical_neu_91x_evidence.md` §2） |

## 二、结构要素核对（三文件共同）

ResNet-50 主干（自定义 `model_resnet.py`，pretrained=False）→ DAPPM → SqueezeBodyEdge（语义主体/边缘分解）→ Laplacian 浅层细节 → edge_fusion → UAFM×2（arm1 通道+空间 / arm2 空间）→ 共享 SegHead；训练态返回 3 路输出（head_seg1=out 主输出 + 2 路辅助），推理态仅主输出；无 eSE、无 ACW、无 seg_head_detailloss。

## 三、命名变更来源

- 大论文/历史代码：`DSMONet`（ResNet-50 版本在大论文中称 **DSMONet-B**）。
- 2026-05-18 `experiment_entry_mapping.md` 首次出现 "**Base-B**（细节-语义互优化，大模型）= DSMONet-ResNet50-SE → `new (copy)/model_dsmo_rs50.py`"，与 Base-S（ResNet-18）对应。
- 此后审计与论文草稿沿用 Base-B。**"Base-B"是 2026 年整理期引入的审计/消融命名，不是新模型。**

## 四、结论

当前中文稿的 Base-B = 大论文 DSMONet-B 的同一架构。两数字（90.2 / 91.24）的差异不来自模型结构，只能来自训练链与评价口径（见 `neu_90_2_vs_91_24_diff.md`）。
