# 统一评估协议锁定 (Evaluation Protocol Lock)

> 锁定日期：2026-05-22
> 状态：已锁定，所有后续评估必须基于此协议
> 用途：确保所有模型评估结果可比较、可复现

---

## 一、NEU-Seg 评估协议

### 1.1 数据集路径

| 项目 | 路径 |
|---|---|
| 训练集 txt | `/workspace/Industrial Surface Defect/new (copy)/dataset/train_neu.txt` |
| 测试集 txt | `/workspace/Industrial Surface Defect/new (copy)/dataset/test_neu.txt` |
| 训练集图像 | `/workspace/Industrial Surface Defect/new (copy)/dataset/NEUSeg/images/training/` (3630 张) |
| 测试集图像 | `/workspace/Industrial Surface Defect/new (copy)/dataset/NEUSeg/images/test/` (840 张) |
| 训练集标签 | `/workspace/Industrial Surface Defect/new (copy)/dataset/NEUSeg/annotations/training/` |
| 测试集标签 | `/workspace/Industrial Surface Defect/new (copy)/dataset/NEUSeg/annotations/test/` |

> 注意：`/workspace/Industrial Surface Defect/new/dataset` 是 symlink → `new (copy)/dataset`

### 1.2 数据集统计

| 项目 | 值 |
|---|---|
| 训练集图像数量 | 3630 |
| 测试集图像数量 | 840 |
| 图像尺寸 | 200 × 200 |
| 标签格式 | 单通道 uint8 PNG，像素值 ∈ {0, 1, 2, 3} |

### 1.3 类别定义

| Class ID | 类别名称 (EN) | 类别名称 (CN) | 测试集像素数 | 占比 |
|---|---:|---|---:|---:|
| 0 | background | 背景 | 29,418,415 | 87.55% |
| 1 | crazing | 裂纹 | 837,548 | 2.49% |
| 2 | inclusion | 夹杂物 | 2,411,879 | 7.18% |
| 3 | patches | 补丁/斑块 | 932,158 | 2.77% |

**类别总数**：4 (含 background)
**mIoU 是否包含 background**：是（标准做法，class 0 参与 mIoU 计算）

### 1.4 输入预处理

```python
# 评估模式预处理（与 training val 一致）
from dataAug_new import Compose, TestRescale, ToTensor

eval_transforms = Compose([
    TestRescale(input_hw=(200, 200)),   # 等比缩放至 200×200
    ToTensor(),                          # 转 tensor，除以 255
])
```

- 输入尺寸：200 × 200
- 归一化：ToTensor() → /255，范围 [0, 1]
- **不使用** ImageNet Normalize (无 mean/std 归一化)
- 标签不进行任何变换 (保持原始 class ID)

### 1.5 标签说明

DataGenerator 返回 3 元组：
```python
(images, labels, label1)
```
- `images`: shape [C, H, W] = [3, 200, 200], float32, range [0, 1]
- `labels`: shape [H, W] = [200, 200], int64, values ∈ {0, 1, 2, 3} — 多类别标签
- `label1`: shape [H, W] = [200, 200], int64, values ∈ {0, 1} — 二值缺陷/背景标签 (class 0 → 0, class {1,2,3} → 1)

**评估使用 `labels`（多类别）**。`label1` 仅用于训练时的 binary loss。

**无标签 remap 需求**：原始标签 class ID 即最终类别序号，0=背景, 1=裂纹, 2=夹杂物, 3=斑块。

### 1.6 评估脚本

**主评估脚本**：
```
/workspace/Industrial Surface Defect/Industrial-Surface-Defect/tools/evaluate_class_iou.py
```

[需人工确认] 脚本的确切参数列表。如果此脚本不支持 NEU-Seg 评估，fallback 方案为直接从训练日志提取 best checkpoint 的 per-class IoU。

### 1.7 历史结果对照

| 来源 | 声称 mIoU | 基于 |
|---|---|---|
| `fixed_existing_results.md` | Base-B 90.2, A2MS-B 91.3, Base-S 88.4, A2MS-S 89.7 | RTX 3090, 硕士论文 |
| 新训练日志 (2026-05-22) | Base-B 90.47, A2MS-B 71.31, Base-S 88.37, A2MS-S 88.58 | A100, 60000 iters |

**关键差异**：A2MS-B NEU 新训练值 71.31 与历史 91.3 差距 -19.99pp → 标记为"异常，需排查"

---

## 二、Leather / Pige 评估协议

### 2.1 数据集路径

| 项目 | 路径 |
|---|---|
| 训练集 txt | `/workspace/Industrial Surface Defect/new (copy)/dataset/pige/train.txt` |
| 验证集 txt | `/workspace/Industrial Surface Defect/new (copy)/dataset/pige/val.txt` |
| 测试集 txt | `/workspace/Industrial Surface Defect/new (copy)/dataset/pige/test.txt` |
| 图像目录 | `/workspace/Industrial Surface Defect/new (copy)/dataset/pige/images/` (2341 张) |
| 标签目录 | `/workspace/Industrial Surface Defect/new (copy)/dataset/pige/labels/` (2341 张) |

### 2.2 数据集统计

| 文件 | 图像数量 | 用途 |
|---|---|---|
| train.txt | 1638 | 训练 |
| val.txt | 235 | 训练时验证 |
| test.txt | 468 | **【锁定为论文测试集】** |
| **总计** | 2341 | — |

> **锁定决策**：论文统一使用 `test.txt` (468 张) 作为测试集。不使用 val.txt (235 张) 作为最终评估。  
> **2026-06-15 修正说明**：`experiments/protocol/leather_test_split_audit.md` 已确认 `pige/test.txt` 共有 468 个非空且有效的 image/label 样本；`new/dataset` 与 `new (copy)/dataset` 通过 symlink 指向同一份列表。

### 2.3 类别定义

| Class ID | 类别名称 (EN) | 类别名称 (CN) |
|---|---:|---|
| 0 | background | 背景 |
| 1 | open_wound | 开创伤 |
| 2 | scratch | 刺刮伤 |
| 3 | brand_mark | 烙印 |
| 4 | hole | 破洞 |
| 5 | skin_disease | 皮肤藓 |
| 6 | rotten_surface | 烂面 |
| 7 | wart | 刺猴 |

**类别总数**：8 (含 background)
**mIoU 是否包含 background**：是

### 2.4 标签格式

- 格式：单通道 uint16 PNG，像素值 ∈ {0, 1, 2, 3, 4, 5, 6, 7}
- 标签值与上述 Class ID 一一对应，无需 remap

### 2.5 输入预处理

```python
# 评估模式预处理
from dataAug_new import Compose, TestRescale, ToTensor

eval_transforms = Compose([
    TestRescale(input_hw=(768, 768)),   # 等比缩放至 768×768
    ToTensor(),                          # 转 tensor，除以 255
])
```

- 输入尺寸：768 × 768
- 归一化：ToTensor() → /255，范围 [0, 1]
- **不使用** ImageNet Normalize

### 2.6 评估脚本

**主评估脚本**：
```
/workspace/Industrial Surface Defect/Industrial-Surface-Defect/tools/evaluate_class_iou.py
```

[需人工确认] 脚本参数。

### 2.7 历史结果对照

| 来源 | 测试集 | Base-B | Base-S | A2MS-B | A2MS-S |
|---|---|---|---|---|---|
| `fixed_existing_results.md` (论文) | 声称 468 张 val | 89.2 | 88.1 | 91.0 | 89.7 |
| 旧评估 (2026-05-18) | test.txt 468 张 | 88.06 | — | 86.07 | — |
| 新训练日志 (2026-05-22) | val.txt 235 张 | 82.78 | 83.97 | 85.60 | 84.72 |

**关键差异**：
1. 论文声称测试集 468 张；2026-06-15 审计确认实际 test.txt = 468 张，val.txt = 235 张
2. 历史 89.2-91.0 无法从当前服务器任何权重复现
3. 旧评估 (test.txt) 与 新训练 (val.txt) 数值差异 2-6pp

**锁定决策**：以 test.txt (468 张) 上的统一评估为准。历史 89.2-91.0 暂时搁置，标记为"来源待确认"。

---

## 三、FPS 测量协议

### 3.1 可用 GPU

| GPU | 可用性 | 论文历史标注 |
|---|---|---|
| NVIDIA A100-PCIE-40GB | ✅ 当前可用 | 否 |
| NVIDIA RTX 3090 | ❓ 待确认 | 是 (`fixed_existing_results.md`) |

### 3.2 FPS 测量参数（来自 `experiments/scripts/complexity_stats.py`）

| 参数 | 值 |
|---|---|
| batch_size | 1 |
| warmup 次数 | 50 |
| 正式计时次数 | 200 |
| torch.cuda.synchronize() | ✅ 使用 |
| 后处理包含 | 否（仅模型前向传播） |
| 输入尺寸 1 | 200 × 200 (NEU-Seg) |
| 输入尺寸 2 | 768 × 768 (Leather) |

### 3.3 FPS 平台决策

**锁定**：详见 `fps_platform_decision.md`。

---

## 四、评估脚本调用规范

### 4.1 通用调用模板

```bash
cd "/workspace/Industrial Surface Defect/Industrial-Surface-Defect"
source /opt/conda/etc/profile.d/conda.sh && conda activate base

python tools/evaluate_class_iou.py \
    --model_type <MODEL_TYPE> \
    --weight_path <WEIGHT_PATH> \
    --test_txt <TEST_TXT_PATH> \
    --input_size <H> <W> \
    --num_classes <N> \
    --class_names <NAME_0> <NAME_1> ... <NAME_N-1> \
    --output_dir <OUTPUT_DIR>
```

### 4.2 每个模型的 model_type 映射

| 论文模型名 | model_type (脚本参数) | 模型定义文件 |
|---|---|---|
| Base-S | dsmors18 | `new/model_dsmo_rs18.py`, backbone=resnet18() |
| Base-B | dsmors50 | `new/model_dsmo_rs50.py`, backbone=resnet50() |
| A2MS-DefectNet-S | dsmors18_eSE_adapt_detailloss | `new/model_dsmo_rs18_eSE_adapt_detailloss_822.py`, backbone=resnet18() |
| A2MS-DefectNet-B | dsmors50_eSE_adapt_detailloss | `new/model_dsmo_rs50_eSE_adapt_detailloss.py`, backbone=resnet50() |
| DDRNet23slim | ddr | `new/model_ddr.py`, DualResNet_imagenet(num_classes=N) |
| PIDNet-S | pid | `new/pid.py` |
| STDC1-Seg | stdc | `new/stdcnet.py`, STDC1 |

---

## 五、协议锁定声明

以下项目已锁定，所有后续评估必须严格遵循：

1. ✅ NEU-Seg 测试集：test_neu.txt (840 张)，4 类，标签值 {0,1,2,3}
2. ✅ Leather 测试集：pige/test.txt (468 张)，8 类，标签值 {0,...,7}
3. ✅ 输入尺寸：NEU-Seg 200×200, Leather 768×768
4. ✅ 预处理：TestRescale + ToTensor，无 ImageNet Normalize
5. ✅ mIoU 口径：包含 background (class 0)
6. ✅ 标签无需 remap
7. ✅ FPS 测量：batch=1, warmup=50, runs=200, 含 sync

以下项目待决策：

- ❓ FPS 平台：A100 vs RTX 3090（详见 `fps_platform_decision.md`）
- ❓ 评估脚本具体参数（需人工确认 `evaluate_class_iou.py` 的实际接口）
