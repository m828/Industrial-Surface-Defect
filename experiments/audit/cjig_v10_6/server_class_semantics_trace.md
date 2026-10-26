# CJIG v10.6-P1 — NEU-Seg 原始类别语义溯源（服务器只读取证）

日期：2026-10-26
范围：只读检查原始数据、历史代码、转换链、第三方命名冲突。未训练、未推理、未复制 GT/原图/标注 JSON 到证据包。

## 结论先行

**CLASS SOURCE CHAIN = PARTIAL。**

- **整数编码链：VERIFIED。** 上游标注 → 训练标签 → 评价标签全程恒等映射，无重排、无颜色转换生效、无跨版本混用。
- **上游语义命名（整数 ID ↔ 缺陷名称）：MISSING（未获原始证据）。** 服务器不存在下载记录、原始 LabelMe JSON、类别顺序元数据或官方命名文件；`1=crazing, 2=inclusion, 3=patches` 目前只是**项目内部一致命名**，不构成上游语义认证。按任务要求，预注册与后续统计一律以 `class_id1/2/3` 为操作标签，不得凭可视化外形命名。

## 1. 原始数据与版本

- 数据目录（两份）：
  - `new (copy)/dataset/NEUSeg/`：`annotations/{training,test}` 各 3630/840 个 PNG；`images/{training,test}` 各 3630/840 个 JPG；`README.md`（SHA `7c1b2f3b…f6d`）内容仅一行 `# NEU-Seg`，无类别元数据。
  - `new/dataset/NEUSeg/`：与上一份**全部 8940 个文件逐一 SHA-256 比对，0 差异**（同名同字节），排除两目录跨版本混用。
- 标注格式：PIL mode `L`（8bit 灰度）、200×200、uint8。全量扫描：训练/测试标注取值均在 `{0,1,2,3}`，尺寸全部 200×200。
- 服务器无其他 NEU-Seg 数据副本（`/workspace` 全域目录名检索）；无原始 zip/tar 包；无 LabelMe JSON（全项目含 `crazing` 的 JSON/TXT/MD 仅本项目审计与元数据文档）。
- 获取记录：未找到。`~/.bash_history` 中 NEU 相关仅训练命令（如 `train_neu_pid.py`），无下载/kaggle NEU-Seg 记录。

## 2. 3630 训练 GT 与转换链闭合核验（整数级）

| 检查 | 结果 |
|---|---|
| train_neu.txt 行数 / SHA | 3630；`498b038f32cd68834894ab433f11468de54a8ef80ac3a6eb1f812d4afcfc1d64` |
| test_neu.txt 行数 / SHA | 840；`2c45b319e04d2307eb0977720de90b98a4867863bce8eca6f8b682295f928f06` |
| 清单 ↔ 磁盘文件 | 两 split 均集合相等、0 缺失、0 重复、image/annotation basename 一一对应 |
| 训练加载器 | `datagenerator_neu.py`（240k Base 运行副本与 `new (copy)/datagenerator_neu.py` **同 SHA `18f96bfce77ccb9f56822b8b148964fe676ceef45155282fe5511bed8b8058c1`**）：`read_txt` 对 mask 直接 `mask.append(img_mask)`，无重编码；`binarymask = (mask != 0)` |
| 颜色转换分支 | `len(img_mask.size)==3` 对 PIL `.size`（二元组）恒为 False，`convert('L')` 分支为**死代码**，未发生任何颜色→灰度转换 |
| `get_pascal_labels/encode_segmap` | VOC 遗留辅助函数，训练读取路径未调用；`experiment_entry_mapping.md` 中“NEU 标签颜色 [255,0,0] 等”描述仅对应该遗留 palette，**与实际灰度标注读取无关** |
| 统一评价器 | `tools/evaluate_class_iou.py`（SHA `37a159b2…220eb`）第 182 行 `Image.open(label_path).convert("L")`，整数直读，无重排 |
| 840 测试 GT 与缓存 | v10.4 已核验：缓存 GT 与源标注 840/840 逐像素一致 |

**转换链闭合（整数级）：上游灰度 ID 直接作为训练与评价标签，全程同一编码。**

## 3. 训练/测试重叠与重复内容检查（新增数据完整性证据）

见 `class_data_integrity_checks.json`（仅计数/文件名/哈希，不含像素数据）。

- **文件名重叠**：训练与测试目录有 300 个同名 ID（000201–000300、000501–000700）。逐对 SHA 比对：**300 对图像与 300 对标注全部不同字节**。属上游文件命名冲突，非同图泄漏。
- **跨 split 内容重复**：对全部 4470 图像与 4470 标注做内容哈希交叉比对：**0 个跨 split 内容重复**。
- **split 内重复（上游数据属性，需知悉）**：
  - 训练图像 3630 文件中 3 对字节相同：`001281/001285`、`002151/002155`、`003021/003025`；
  - 测试图像 840 文件中 1 对字节相同：`000301/000305`；
  - 这 4 对图像各自对应标注**不同**（每对均恰有 580 个像素不同，取值均为 {0,2}），即上游存在同图不同标注版本的重复样本。测试集 000301/000305 会被计为两个样本，对 mIoU 有轻微重复加权；非本协议人为造成，不据此修改任何冻结结果。

## 4. 类别名称表与冲突证据

| 来源 | 内容 | 性质 |
|---|---|---|
| `tools/evaluate_class_iou.py:50` | `NEU_CLASSES = ["background","crazing","inclusion","patches"]` | 项目内部命名 |
| `repro_runs/mechanism_eval_240k/compute_mechanism_metrics.py:14` | `CLASSES = {1:"crazing",2:"inclusion",3:"patches"}` | 项目内部命名 |
| `repro_runs/mechanism_eval_240k/visualize_cases_v6.py:16` | `CID = {"crazing":1,"inclusion":2,"patches":3}` | 项目内部命名 |
| `third_party/FDSNet/core/data/dataloader/sd_voc.py:132`（SHA `90c11d55…f090`） | `classes = ('background','inclusion','patches','scratches')` | **第三方冲突命名**，但其 `BASE_DIR='SD-900-voc'`（900 图钢缺陷数据集），并非 NEU-Seg；说明类别名不可跨数据集/跨项目假定 |

- 内部三处命名彼此一致，但均不能证明上游语义。
- FDSNet 的 SD-900-voc 命名与本项目 NEU 命名冲突（1=inclusion vs 1=crazing；3=scratches vs 3=patches），正好说明：**没有上游元数据时，任何 ID→名称映射都是未认证假设**。不得将 class_id1 命名为 scratches，也不得因内部表一致就视为已认证。
- `subregion unet/`、`MultiModel_Code_0526/`、`xd/`、`third_party/DMC-Net`、`third_party/LETNet` 均未发现 NEU 类别语义元数据。

## 5. 对论文与后续统计的操作性建议

1. 统计与预注册使用 `class_id1/class_id2/class_id3`（本任务 RQ1/RQ2 已如此）。
2. 论文中 crazing/inclusion/patches 命名应标注为项目内部命名，直至取得 NEU-Seg 官方发布页/原始标注导出等上游证据；本地若持有上游来源，应由作者人工确认后补档。
3. 现有全部整数级结果（mIoU、A/B/C、RQ1/RQ2）不受命名问题影响：它们只依赖 ID 一致性，而 ID 链已验证。

## 6. 证据链文件

- `class_mapping_source_inventory.csv`：逐项来源、SHA、证明力与状态。
- `class_data_integrity_checks.json`：split 闭合、重叠/重复检查原始结果。
- 相关源码行号与 SHA 见本文件第 1–4 节。
