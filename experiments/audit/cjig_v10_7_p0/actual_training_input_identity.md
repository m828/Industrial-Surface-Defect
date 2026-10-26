# P0-1 实际训练/评价输入身份闭环

## 结论

**ACTUAL TRAINING INPUT IDENTITY = VERIFIED（限定于当前源文件、实际运行路径和固定上游树之间的身份链）。**

- Base-B 与 A2MS 的正式 240k 运行副本均通过各自 `code/dataset` 符号链接解析到 `/workspace/Industrial Surface Defect/new (copy)/dataset`。
- 八个重点记录全部存在于实际 train/test 清单，图像与 GT basename 一一对应，文件 SHA-256、解码像素 SHA-256 已取得。
- 当前全部 4470 张图像和 4470 张 GT 与 `DHW-Master/NEU_Seg` commit `da8cebedb292beca4d101db8c442d5706fac3bfe` 的对应文件**逐字节一致**；两边解码像素也一致。
- 这里的 VERIFIED 不表示恢复了每次在线增强后的历史训练张量，也不表示取得了原始相机采集 ID。

## 1. 实际运行链

1. `repro_runs/long240k_seed1337/base_b/config/config.yml:12-13` 与 A2 对应配置指定 `./dataset/` 和 `./dataset/train_neu.txt`。
2. 两个运行目录的 `code/dataset` 均为符号链接，实际解析到 `new (copy)/dataset`；`new/dataset` 也指向同一目录。
3. `train_long240k_base_b.py:114-126` 和 A2 同位置代码读取 `train_list`，构造 `DataGenerator`；batch size 16、`shuffle=True`、`drop_last=True`。
4. Base 日志 `repro_runs/long240k_seed1337/base_b/logs/base_b_s1337_long240k_codex1/run_2026_09_22_14_42_16.log:5` 实际记录 `Train list: ./dataset/train_neu.txt (3630 samples)`。
5. 训练前 manifest 登记 train/test 清单分别为 3630/840 行；本轮实测 SHA 与其一致：
   - train：`498b038f32cd68834894ab433f11468de54a8ef80ac3a6eb1f812d4afcfc1d64`
   - test：`2c45b319e04d2307eb0977720de90b98a4867863bce8eca6f8b682295f928f06`
6. `datagenerator_neu.py:14-64` 按每行“GT 图像”顺序打开文件，训练脚本 `:180-201` 将图像、label 和 binary mask 送入训练。评价脚本 `repro_runs/multiseed_10k/unified_eval_840_ms.py:18-38` 使用 `test_neu.txt`、`TestRescale+ToTensor` 和主输出 `out[0]`；本轮没有执行该评价脚本。

关键代码身份实测：Base 训练脚本 `7bd9040362b427dade3e5faeded2f0ec12dcb70eb8db18c02bf76d9820ca8a9b`；DataGenerator `18f96bfce77ccb9f56822b8b148964fe676ceef45155282fe5511bed8b8058c1`；配置 `51a0c9f22d4a00bae979e92267e9879771f6816da7515184424b4973a2bd31e3`。

## 2. 八个重点记录

下表 SHA 均取前 12 位；**完整 64 位文件/像素 SHA、实际绝对路径和上游对应路径见 `actual_priority_training_input_identity.csv`，全量值见 `full_image_identity_manifest.csv`。**

| 记录键 | 清单行 | 图像文件 SHA | GT 文件 SHA | 图像像素 SHA | GT 像素 SHA | 上游字节+像素 |
|---|---:|---|---|---|---|---|
| test:000301 | test 550 | `4d2de82731b8` | `75387dfcd6d9` | `44f7092cd9e2` | `4495963b7442` | 全部一致 |
| test:000305 | test 58 | `4d2de82731b8` | `eb06a829f813` | `44f7092cd9e2` | `fce3a2b00605` | 全部一致 |
| train:001281 | train 1475 | `a20d50a78450` | `8f02c1f64201` | `68f4450990c2` | `cff20430f2ea` | 全部一致 |
| train:001285 | train 1588 | `a20d50a78450` | `a43b22cb477d` | `68f4450990c2` | `2d398cc999fe` | 全部一致 |
| train:002151 | train 1849 | `32b8ebddb9a3` | `0966ee0248cc` | `4ab8f8ee6ed5` | `608318c3b384` | 全部一致 |
| train:002155 | train 2951 | `32b8ebddb9a3` | `5c0629492129` | `4ab8f8ee6ed5` | `c6c1a2a4f3c4` | 全部一致 |
| train:003021 | train 273 | `1333a1847b2e` | `53c936278c78` | `9ea811279010` | `3789f9cdcf16` | 全部一致 |
| train:003025 | train 113 | `1333a1847b2e` | `62a39c782d34` | `9ea811279010` | `750bdbb27fdf` | 全部一致 |

四个同图对的图像原始字节及解码像素均相同，但各自 GT 文件和 GT 像素不同；每对 GT 都恰有 580 个像素不同。正例闸门见 `positive_control_verification.json`。

## 3. 在线/离线增强边界

- 正式训练使用 `Compose([Transforms_PIL(200,200), ToTensor])`，见训练脚本 `:110-112`。
- `dataAug_new.py:17-71` 实现在线旋转、翻转、噪声和模糊；`:114-130` 每次随机选择一种。旋转角度集合为 90°/180°/270°，图像和两类标签同步变换。
- DataGenerator 在变换前直接读取现有文件，没有离线写出增强图像或 GT 的步骤。当前源文件中的重复/旋转关系不能由这个在线步骤解释为本项目新生成的文件。
- 日志没有逐样本、逐 epoch 的实际增强参数序列；因此**实际增强张量级重放 = UNRESOLVED / NOT RUN**。本轮只做源代码行为审计，没有重放 DataLoader 或执行模型。

## 4. 固定上游身份

指定参考：[DHW-Master/NEU_Seg@da8cebe](https://github.com/DHW-Master/NEU_Seg/tree/da8cebedb292beca4d101db8c442d5706fac3bfe)。

两次 Git 克隆因网络中断失败后，使用 GitHub 按 commit 生成的归档；没有把不完整克隆当作证据。归档 SHA-256 为 `aae62b176f14f56f90bce313046802ceb6c1efb418c692b925ce77a1cf67a436`，gzip 与安全解包检查通过，共 8941 个普通文件（8940 数据文件 + 1 README）。`upstream_identity_summary.json` 同时记录归档方式、实时 `ls-remote` 匹配和逐文件匹配结果；由于不是 Git checkout，`actual_checkout_head=null` 是如实记录，不影响对固定归档 commit 的核验。

## 5. 证据限制

- 训练前记录保存了清单 SHA，而没有 8940 文件的逐文件运行时快照。当前源文件 mtime 早于正式训练、路径和清单闭合、运行日志存在、当前文件与固定上游完全一致，共同强力支持训练时输入身份；mtime 本身不被当作密码学证明。
- 历史训练日志没有逐 batch 样本 ID；某记录存在于训练清单证明其具备被采样资格和曝光风险，不等于恢复其实际被采样次数。此缺口不改变当前 split 的独立性判定。
- 本结论不证明上游如何采集、增强或划分，也不证明 840 条测试记录对应 840 个独立采集对象。
