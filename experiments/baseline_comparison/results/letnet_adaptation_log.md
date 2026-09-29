# LETNet NEU 适配修改清单

- 日期： 2026-09-29
- 仓库： `Industrial-Surface-Defect/third_party/LETNet/`（嵌套 git 仓库；基线 commit `faaa065` "Add files via upload"；以下均为本地修改，未提交）
- 验证： 修复后 `python train.py --model LETNet --dataset neuseg --max_epochs 1` 完整跑通（见 `logs/smoke_letnet.md`）

## A. 已知 blocker 修复（4 处，逐条）

| # | 文件:行号 | 原样 | 改后 | 说明 |
|---|---|---|---|---|
| 1 | `Network/builders/dataset_builder.py:2` | `import pickle ，` | `import pickle` | 行尾中文逗号为 SyntaxError，整个 builders 包无法 import |
| 2 | `Network/builders/model_builder.py:12` | `from model.ESPNet_v2.SegmentationModel import EESPNet_Seg` | （删除该行） | `model/ESPNet_v2/` 在上游仓库即缺失，断链 import 使 model_builder 无法 import |
| 3 | `Network/model/module/transformer.py:6` | `from model.patch import extract_image_patches, reduce_mean, reduce_sum, same_padding, reverse_patches` | `from module.patch import extract_image_patches, reduce_mean, reduce_sum, same_padding, reverse_patches` | `patch.py` 位于 `model/module/`；`model.patch` 不存在。与 `model/LETNet.py:6` 的 `from module.transformer import TransBlock` 保持一致（运行时需 `Network/model` 在 PYTHONPATH） |
| 4 | `Network/model/LETNet.py:375` | `self.LC3 = LongConnection(32, 16, 3)` | `self.LC3 = LongConnection(32, 32, 3)` | 通道 bug：`forward` 中 `output4 + self.LC3(output3)`，output4（经 DAB_Block_4）为 32 通道，原 LC3 输出 16 通道，相加必然 shape mismatch，不修无法前向 |

## B. ②的连带修复（1 处）

| # | 文件:行号 | 原样 | 改后 | 说明 |
|---|---|---|---|---|
| 5 | `Network/builders/model_builder.py:43-44` | `elif model_name == 'ESPNet_v2':`<br>`    return EESPNet_Seg(classes=num_classes)` | `elif model_name == 'ESPNet_v2':`<br>`    raise NotImplementedError("ESPNet_v2 is unavailable: broken upstream import removed (model/ESPNet_v2 missing)")` | 删除断链 import 后该分支残留未定义符号引用，改为显式报错；不影响 LETNet 及其他模型分支 |

## C. NEU-Seg 数据适配（新增/接线）

| # | 文件 | 内容 |
|---|---|---|
| 6 | `Network/dataset/neuseg.py`（新增） | 仿 `camvid.py`：`NeuSegDataSet` / `NeuSegValDataSet` / `NeuSegTestDataSet` / `NeuSegTrainInform`。要点：① 我们的列表每行是 "标注 图像"（标注在前），adapter 内 `_parse_list()` 交换两列；② 4 类（含背景），标签值 {0,1,2,3}，ignore_label=255；③ 输入 200×200 在 Dataset 内 pad 到 208×208（右/下补边，图像补 0、标注补 255），满足 LETNet /16 下采样整除约束（200/16=12.5 不可行，208/16=13）；训练集再随机 crop 208×208（pad 后 offset 恒为 0）；④ `NeuSegTrainInform` 直方图 `np.histogram(label_img, 4, [0, 3])`（range (0,3)），列序同样交换 |
| 7 | `Network/dataset/neuseg/`（新增目录） | `neuseg_trainval_list.txt` ← 拷贝自 `new (copy)/dataset/train_neu.txt`（3630 行，原文件未动）；`neuseg_val_list.txt`、`neuseg_test_list.txt` ← 拷贝自 `test_neu.txt`（840 行）；`NEUSeg` → symlink 到 `/workspace/Industrial Surface Defect/new (copy)/dataset/NEUSeg` |
| 8 | `Network/builders/dataset_builder.py:6` | 新增 `from dataset.neuseg import NeuSegDataSet, NeuSegValDataSet, NeuSegTrainInform, NeuSegTestDataSet` |
| 9 | `Network/builders/dataset_builder.py:25-27`（build_dataset_train inform 分支） | 新增 `elif dataset == 'neuseg': dataCollect = NeuSegTrainInform(data_dir, 4, ...)` |
| 10 | `Network/builders/dataset_builder.py:69-81`（build_dataset_train loader 分支） | 新增 `elif dataset == "neuseg":` 构建 trainLoader（NeuSegDataSet, drop_last=True, shuffle）+ valLoader（NeuSegValDataSet, batch1） |
| 11 | `Network/builders/dataset_builder.py:99-101`（build_dataset_test inform 分支） | 同 #9 |
| 12 | `Network/builders/dataset_builder.py:137-142`（build_dataset_test loader 分支） | 新增 `elif dataset == "neuseg":` 构建 testLoader（NeuSegValDataSet） |
| 13 | `Network/train.py:128-129`（loss 分支） | 新增 `elif args.dataset == 'neuseg': criteria = CrossEntropyLoss2d(weight=weight, ignore_label=ignore_label)`（weighted CE，ignore 255） |
| 14 | `Network/train.py:398-401`（`__main__` 数据集分支，原 386-396 区块） | 新增 `elif args.dataset == 'neuseg': args.classes = 4; args.input_size = '208,208'; ignore_label = 255` |
| 15 | `Network/train.py:246-247`（保存分支） | 新增 `elif args.dataset == 'neuseg': torch.save(state, model_file_name)`（不加则 neuseg 不落任何 checkpoint） |

## D. 运行期自动生成（非手写修改）

- `Network/dataset/inform/neuseg_inform.pkl`：首次训练时由 `NeuSegTrainInform` 生成（mean [113.59, 113.59, 113.59]，std [30.90]*3，classWeights [1.4646, 8.7251, 6.2481, 8.4495]，histogram range (0,3)）。
- `__pycache__/` 若干。

## E. 运行方式（smoke）

```bash
cd "/workspace/Industrial Surface Defect/Industrial-Surface-Defect/third_party/LETNet/Network"
PYTHONPATH=".:./model" /opt/conda/bin/python train.py --model LETNet --dataset neuseg \
  --max_epochs 1 --batch_size 16 --num_workers 4 \
  --savedir "/workspace/Industrial Surface Defect/repro_runs/baseline_smoke/letnet_checkpoint/"
```

注：`PYTHONPATH` 需同时含 `Network/`（builders/dataset/utils 包）与 `Network/model/`（`module.*` 包，对应修复 #3 的 import 风格）。LETNet 本阶段使用其原生 recipe（Adam 5e-4、warmpoly、epoch 制、训练中自带 val）；是否将其纳入 240k 统一协议（恒定 1e-4、无训练期 val、仅终点保存）留待正式基线阶段决定。
