# Leather/Pige Test Split Audit

> 生成日期：2026-06-15  
> 审计范围：`new (copy)/dataset/pige/test.txt` 与 `new/dataset/pige/test.txt`  
> 结论：当前有效测试样本数为 **468**。

## 一、审计对象

| 项目 | 路径 |
|---|---|
| 主列表 | `/workspace/Industrial Surface Defect/new (copy)/dataset/pige/test.txt` |
| symlink 访问列表 | `/workspace/Industrial Surface Defect/new/dataset/pige/test.txt` |
| 数据根目录 | `/workspace/Industrial Surface Defect/new (copy)/dataset/pige` |
| 有效列表输出 | `experiments/protocol/leather_valid_test_list.txt` |

`/workspace/Industrial Surface Defect/new/dataset` 是指向 `/workspace/Industrial Surface Defect/new (copy)/dataset` 的 symlink。两条 test list 读取内容是否完全一致：**是**。

## 二、行数统计

| 指标 | 数值 |
|---|---:|
| 总行数 | 468 |
| 非空行数 | 468 |
| 空行数 | 0 |
| 格式错误行数 | 0 |
| 图像存在且标签存在的有效行数 | 468 |
| 缺失图像行数 | 0 |
| 缺失标签行数 | 0 |
| 重复完整行数 | 0 |
| 重复 image 路径数 | 0 |
| 重复 label 路径数 | 0 |

## 三、首尾样本

前三个非空样本：

- line 1: `images/0.jpg labels/0.png`
- line 2: `images/1.jpg labels/1.png`
- line 3: `images/2.jpg labels/2.png`

最后三个非空样本：

- line 466: `images/465.jpg labels/465.png`
- line 467: `images/466.jpg labels/466.png`
- line 468: `images/467.jpg labels/467.png`

## 四、异常检查

### 空行

无

### 格式错误行

无

### 重复完整行

无

### 重复 image 路径

无

### 重复 label 路径

无

### 缺失图像

无

### 缺失标签

无

## 五、最终结论

- `test.txt` 总行数为 468，非空行数为 468。
- 未发现空行、重复行、格式错误、缺失图像或缺失标签。
- 468 行均为有效测试样本，因此 Leather/Pige 后续统一测试样本数应修正为 **468**，不是 467。
- 已生成有效测试列表：`experiments/protocol/leather_valid_test_list.txt`。

## 六、后续影响

- `experiments/eval_protocol_lock.md` 中 Leather/Pige 测试集数量应从 467 修正为 468。
- 后续所有 Leather/Pige test mIoU、per-class IoU、可视化和分组评价均应引用本有效列表或与其完全一致的 `pige/test.txt`。
