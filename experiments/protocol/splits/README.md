# 数据划分清单（Data Splits）

> 仅包含文件名/图像 ID 与标注对应关系，**不含任何原始图像**。图像与标注数据不上传。

## 文件清单

| 文件 | 数据集 | 样本数 | 用途 |
|---|---|---:|---|
| `neu_train.txt` | NEU-Seg | 3630 | 训练集 |
| `neu_test.txt` | NEU-Seg | 840 | 测试集（固定终点统一评价） |
| `leather_train.txt` | Leather | 1638 | 训练集 |
| `leather_val.txt` | Leather | 235 | 验证集（仅用于历史训练中 checkpoint 选择） |
| `leather_test.txt` | Leather | 468 | 独立测试集（最终一次评价） |

## 格式

每行两个字段：`标注相对路径 图像相对路径`（NEU-Seg）或 `图像相对路径 标注相对路径`（Leather）。

**注意两个数据集字段顺序相反**，这是历史代码链的既有约定，评价脚本已按各自顺序解析。NEU-Seg 行序为"标注 图像"，Leather 行序为"图像 标注"。

## 类别映射

### NEU-Seg（4 类，mIoU 含背景）

| ID | 类别 | 英文 |
|---:|---|---|
| 0 | 背景 | background |
| 1 | 裂纹 | crazing |
| 2 | 夹杂物 | inclusion |
| 3 | 斑块 | patches |

### Leather（7 缺陷类 + 背景 = 8 评价类，mIoU 含背景）

| ID | 类别 | 英文 |
|---:|---|---|
| 0 | 背景 | background |
| 1 | 开创伤 | open wound |
| 2 | 刺刮伤 | scratch |
| 3 | 烙印 | brand mark |
| 4 | 破洞 | hole |
| 5 | 皮肤藓 | skin disease |
| 6 | 烂面 | rotten surface |
| 7 | 刺猴 | wart |

## 一致性校验（sha256）

| 文件 | sha256 |
|---|---|
| `neu_train.txt` | 498b038f32cd68834894ab433f11468de54a8ef80ac3a6eb1f812d4afcfc1d64 |
| `neu_test.txt` | 2c45b319e04d2307eb0977720de90b98a4867863bce8eca6f8b682295f928f06 |
| `leather_test.txt` | f9ac23b3b330759f096d71e8432fcbc5376a0766cc778be9adcce569218bf95c |

NEU 清单 sha256 与 `experiments/audit/long240k_*_manifest.json` 中记录一致。`leather_test.txt` 与上层目录的 `../leather_valid_test_list.txt` 样本集合完全相同（468 条逐一相同），仅行尾符不同（CRLF vs LF）；论文审计评价实际使用的是后者（LF 版，sha256 `f9ac23b3…`）。

来源：服务器 `new (copy)/dataset/` 下的既有划分文件，2026-09-28 原样复制，未做任何修改。
