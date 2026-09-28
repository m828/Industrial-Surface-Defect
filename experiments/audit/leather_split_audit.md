# Leather/Pige 数据划分审计 (leather_split_audit)

> 日期：2026-09-28

## 一、划分文件（当前与历史共用同一套）

路径：`/workspace/Industrial Surface Defect/new (copy)/dataset/pige/`（`new/dataset` 为指向它的 symlink）

| split | 文件 | 有效行数 | SHA256 |
|---|---|---:|---|
| train | train.txt | 1638 | `5fa37b630787c684f708208c4d063bcfd527723d6c2d3452cead5da0cdac719e` |
| val | val.txt | 235 | `142c66d1b92635c14c2455ed73627477be68f8ee4c33c667ee1ccf79ebf58229` |
| test | test.txt | 468 | `449b0eb2ea15a8891cc53b30d2756aa35a9b7702004929cbe083de8f03bf959d` |

注：`wc -l` 显示 1637/234/467 是因为末行无换行符；有效样本数为 1638/235/468（与 2026-06-15 split 审计一致）。

统一测试列表：`experiments/protocol/leather_valid_test_list.txt`（468 行，与 test.txt 内容一致）。

## 二、历史训练实际使用的监控集

- 历史训练脚本（`new (copy)/train_pige_dsmor50.py:70` 等）硬编码 `val_loader = DataGenerator(txtpath='./dataset/pige/val.txt')`。
- 训练循环只出现 train/val 两个 loader；**test.txt 在训练侧从不出现**。
- 历史 val 监控细节：batch=12、`shuffle=True`、`drop_last=True` → 每次实际评价 228/235 张（丢弃最后 7 张），且顺序随机。因此历史 val best_iou 带有 ±0.2–0.5 pt 量级的采样噪声，但其性质仍是 validation 分数。

## 三、泄漏检查

| 检查项 | 结果 |
|---|---|
| train∩val 路径重叠 | 0 |
| train∩test 路径重叠 | 0 |
| val∩test 路径重叠 | 0 |
| 2341 张图像内容 MD5 重复 | 0 |
| 跨 split 内容重复 | 0 |

结论：**无 split 泄漏**。历史 val235-best 选模型 + 当前 test468 终评的流程是合法的"train → val 选择 → test 一次性评价"。

## 四、旧稿 val/test 对调问题

- 旧论文稿曾写 val=468/test=235，与数据文件相反；数据文件本身从未变化。
- CJIG v3 已更正为 test=468。本审计所有数字均基于 test=468。

## 五、结论

历史 91%（val235 best 0.909084）与当前统一 test468 评价（0.909713）口径关系明确：

> 历史 claim = validation 最优；本轮复评 = 对 val 选出的同一 checkpoint 在锁定 test468 上一次性评价。
> 两者数值一致（90.9%），不存在 split 口径篡改。
