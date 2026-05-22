# 固定实验结果记录

> 本文件由当前项目 `fixed_existing_results.md` 整理而来，实验数值未新增；仅按第四版论文命名规则统一为 A2MS-DefectNet、Base-S、Base-B。新结果写入论文前必须先进入 `new_results_pending.md` 并完成核验。
>
> **状态更新（2026-05-22）**：11 个新训练结果已写入 `new_results_pending.md`，均标注 verified=false。待人工核验后同步到本文件。详见 `paper_result_update_summary.md`。

# 固定已有实验结果

本文档是本轮中文期刊论文初稿的数值基准。`paper_draft.md` 中出现的实验数值只允许来自本文档；未在本文档中列出的数值统一写为 `TODO`。

## 1. 来源与使用边界

- 主要来源：`source/面向多尺度目标的工业表面缺陷检测算法及轻量化研究.docx` 的已抽取表格与正文。
- 辅助来源：`source/撤稿论文-JIFS235929-DSMONet.docx` 与 `source/撤稿论文-JIFS235929.pdf` 仅作为 DSMONet 基础框架来源和待核验材料；当前 Word 抽取无可读正文和表格，不直接复用其文字。
- PDF 说明材料当前未能完成自动文本解析，涉及撤稿说明或 PDF 内独有信息时，在正文中保留 `TODO` 或 `待确认`。
- 公式文本未能完整可靠抽取，所有公式写作均需标记 `TODO_FORMULA_CHECK`。

## 2. 数据集与实验设置

| 项目 | 已有信息 | 来源 | 初稿处理 |
|---|---:|---|---|
| 主实验数据集 | NEU-Seg、自建皮革缺陷数据集 | 硕士论文第 3、4 章 | 可直接写入 |
| NEU-Seg 使用类别 | 夹杂物、补丁、划痕三类 | 硕士论文第 4 章实验设置 | 可直接写入 |
| NEU-Seg 样本数量 | TODO | 未在当前抽取片段中确认 | 不补数 |
| 皮革数据集总量 | 2341 张 | 硕士论文第 3 章 | 可直接写入 |
| 皮革数据集划分 | 训练 1638 张、验证 468 张、测试 235 张 | 硕士论文第 3 章 | 可直接写入 |
| 皮革图像尺寸 | 768 x 768 | 硕士论文第 3 章 | 可直接写入 |
| 皮革缺陷类别 | 开创伤、刺刮伤、烙印、破洞、皮肤藓、烂面、刺猴 | 硕士论文第 3 章 | 可直接写入 |
| CPU | Intel Core i9-10900X | 硕士论文第 3 章表 3-1 | 可直接写入 |
| GPU | NVIDIA GeForce RTX 3090 | 硕士论文第 3 章表 3-1 | 可直接写入 |
| 内存 | 24GB | 硕士论文第 3 章表 3-1 | 可直接写入 |
| 操作系统 | Ubuntu 18.04 | 硕士论文第 3 章表 3-1 | 可直接写入 |
| 框架 | PyTorch 1.10 | 硕士论文第 3 章表 3-1 | 可直接写入 |
| CUDA/cuDNN | CUDA 11.2、cuDNN 8.2 | 硕士论文第 3 章表 3-1 | 可直接写入 |
| Python | 3.9.7 | 硕士论文第 3 章表 3-1 | 可直接写入 |
| batch size | 16 | 硕士论文第 3、4 章 | 可直接写入 |
| 最大迭代次数 | 80000 | 硕士论文第 3 章 | 可直接写入 |
| 优化器 | Adam | 硕士论文第 3、4 章 | 可直接写入 |
| 初始学习率 | 0.0001 | 硕士论文第 3、4 章 | 可直接写入 |
| 权重衰减 | 2e-6 | 硕士论文第 3 章 | 可直接写入 |
| 评估指标 | mIoU、FPS、Params | 硕士论文第 3 章 | 可直接写入 |

## 3. DSMONet 基础框架已有结果

该组结果用于交代 DSMONet 作为基础框架的证据，不作为中文稿主创新主线。

### 3.1 不同方法在 NEU-Seg 和皮革数据集上的结果

| 模型类别 | Method | Backbone | Params | NEU-Seg mIoU | NEU-Seg FPS | Leather mIoU | Leather FPS |
|---|---|---|---:|---:|---:|---:|---:|
| Large Model | FCN | - | 134.2 | 87.4 | 94.8 | 84.3 | 25.2 |
| Large Model | U-Net | - | 31.1 | 89.2 | 164.1 | 87.7 | 24.9 |
| Large Model | HRNet | - | 65.8 | 88.3 | 32.9 | 84.7 | 27.4 |
| Large Model | PSPNet | - | 49.1 | 86.3 | 70.9 | 85.1 | 21.5 |
| Large Model | Deeplabv3+ | ResNet50 | 59.3 | 89.8 | 61.2 | 89.7 | 29.9 |
| Large Model | Base-B | ResNet50 | 29.3 | 90.2 | 115.6 | 89.2 | 67.3 |
| Small Model | ENet | - | 0.36 | 83.1 | 90.7 | 60.2 | 86.9 |
| Small Model | PIDNet-S | - | 7.62 | 85.4 | 138.2 | 85.9 | 117.2 |
| Small Model | BiSeNetV1 | ResNet18 | 13.4 | 87.5 | 218.7 | 86.9 | 180.8 |
| Small Model | DDRNet23slim | - | 20.3 | 86.9 | 160.4 | 88.1 | 149.5 |
| Small Model | Base-S | ResNet18 | 14.0 | 88.4 | 178.5 | 88.1 | 167.2 |

## 4. A2MS-DefectNet 消融结果

### 4.1 高效轻量映射模块 ELMM

| 方法 | mIoU | FPS |
|---|---:|---:|
| Baseline | 85.1 | 128.6 |
| DM | 90.1 | 117.6 |
| ELMM | 91.3 | 126.7 |

### 4.2 自适应注意力模块 AAM

| 方法 | 数量 | mIoU | FPS |
|---|---:|---:|---:|
| SE | 1 | 90.5 | 122.9 |
| eSE | 1 | 90.7 | 126.3 |
| AAM | 1 | 91.1 | 126.9 |
| AAM | 2 | 91.3 | 126.7 |

### 4.3 联合损失函数

| l0 + l1 + l2 | ledge | lmask | mIoU |
|---|---|---|---:|
| √ |  |  | 90.4 |
| √ | √ |  | 90.7 |
| √ |  | √ | 90.8 |
| √ | √ | √ | 91.3 |

## 5. A2MS-DefectNet 主实验结果

### 5.1 皮革缺陷数据集

| Dataset | Model | Backbone | mIoU | FPS |
|---|---|---|---:|---:|
| Leather | Sub-region UNet | - | 84.6 | 55.3 |
| Leather | BiSeNetV1-L | ResNet-18 | 86.9 | 180.8 |
| Leather | BiSeNetV2-L | - | 87.1 | 213.1 |
| Leather | SFNet | DF2 | 89.9 | 142.6 |
| Leather | SFNet | ResNet-50 | 89.1 | 44.9 |
| Leather | STDC1-Seg | STDC1 | 85.0 | 163.4 |
| Leather | STDC2-Seg | STDC2 | 85.3 | 135.1 |
| Leather | PP-LiteSeg-T | STDC1 | 82.2 | 122.7 |
| Leather | PP-LiteSeg-B | STDC2 | 85.5 | 86.9 |
| Leather | DDRNet23slim | - | 88.1 | 149.5 |
| Leather | Base-S | ResNet-18 | 88.1 | 167.2 |
| Leather | Base-B | ResNet-50 | 89.2 | 67.3 |
| Leather | A2MS-DefectNet-S | ResNet-18 | 89.7 | 188.7 |
| Leather | A2MS-DefectNet-B | ResNet-50 | 91.0 | 71.6 |

### 5.2 NEU-Seg 数据集

| Dataset | Model | Backbone | mIoU | FPS |
|---|---|---|---:|---:|
| NEU-Seg | FDSNet | - | 78.8 | 186.1 |
| NEU-Seg | PGA-Net | - | 82.2 | 48.5 |
| NEU-Seg | Sub-region UNet | - | 88.5 | 285.1 |
| NEU-Seg | BiSeNetV1-L | ResNet-18 | 87.5 | 218.7 |
| NEU-Seg | BiSeNetV2-L | - | 85.1 | 223.3 |
| NEU-Seg | SFNet | DF2 | 87.5 | 151.0 |
| NEU-Seg | SFNet | ResNet-50 | 90.5 | 126.3 |
| NEU-Seg | STDC1-Seg | STDC1 | 87.5 | 255.1 |
| NEU-Seg | STDC2-Seg | STDC2 | 87.7 | 169.7 |
| NEU-Seg | PP-LiteSeg-T | STDC1 | 80.5 | 142.2 |
| NEU-Seg | PP-LiteSeg-B | STDC2 | 81.1 | 105.6 |
| NEU-Seg | DDRNet23slim | - | 86.9 | 160.4 |
| NEU-Seg | Base-S | ResNet-18 | 88.4 | 178.5 |
| NEU-Seg | Base-B | ResNet-50 | 90.2 | 115.6 |
| NEU-Seg | A2MS-DefectNet-S | ResNet-18 | 89.7 | 196.5 |
| NEU-Seg | A2MS-DefectNet-B | ResNet-50 | 91.3 | 127.9 |

## 6. 可直接支持的数值结论

- A2MS-DefectNet-B 在 NEU-Seg 上达到 91.3 mIoU 和 127.9 FPS。
- A2MS-DefectNet-B 在皮革缺陷数据集上达到 91.0 mIoU 和 71.6 FPS。
- 与 Base-B 相比，A2MS-DefectNet-B 在 NEU-Seg 上 mIoU 提升 1.1 个百分点，FPS 提升 12.3。
- 与 Base-B 相比，A2MS-DefectNet-B 在皮革缺陷数据集上 mIoU 提升 1.8 个百分点，FPS 提升 4.3。
- ELMM 相比 DM 在消融表中 mIoU 提升 1.2 个百分点，FPS 提升 9.1。
- AAM 两次插入取得 91.3 mIoU 和 126.7 FPS。
- 在联合损失消融中，`l0 + l1 + l2 + ledge + lmask` 组合取得 91.3 mIoU，优于仅使用 `l0 + l1 + l2` 的 90.4 mIoU。

