# 近重复检测实现与覆盖率审计

## 冻结规则与范围

本审计在发现异常后回顾性选择阈值，不是盲法预注册。RGB 高相似要求 Pearson≥0.990、MAE≤6、PSNR≥24dB、SSIM≥0.95；图像近似层为 Pearson≥0.950、MAE≤12、SSIM≥0.80。GT 近似要求前景 IoU≥0.95 且像素一致率≥0.995。SSIM 参数固定为 7×7、data_range=255、非高斯窗口、样本协方差、K1=0.01/K2=0.03。

先对 9988215 个非自配对计算八方向 pHash/dHash 最小距离，再执行缩略图闸门。预筛并集 3035263 对，3002118 对被缩略图规则拒绝，33145 对进入正式候选。预筛拒绝不是候选，也不代表来源无关。

## 实现缺陷与纠正

- 原 LSH 版本产生 503436 行但只有 500000 个逻辑 pair；3436 个反向重复来自未排序的 D4 种子索引。原汇总及下游计数标记为非权威诊断稿，精确文件/像素/D4 结果不受影响。
- 两次单纯扩大 LSH 容量不能解决桶碰撞，相关尝试未形成结论。随后改为全对 Hamming 预筛并修正索引顺序。
- 第一遍容量精核 20810 对；第二遍只复核既定 CAP backlog 12335 对。最终 33145 对全部精核，阈值未变，剩余未复核为 0。

## 已复核分类

| scope | 类别 | 对数 |
|---|---|---:|
| test/test | `FILE_BYTES_IDENTICAL` | 1 |
| test/test | `IMAGE_NEAR_ONLY_SOURCE_UNRESOLVED` | 5 |
| test/test | `REVIEWED_BELOW_NEAR_THRESHOLD` | 275 |
| train/test | `D4_GT_EXACT_AND_IMAGE_HIGH_SIMILARITY` | 714 |
| train/test | `D4_IMAGE_PIXELS_IDENTICAL` | 1 |
| train/test | `GT_SIMILAR_ONLY_SOURCE_UNRESOLVED` | 7 |
| train/test | `IMAGE_NEAR_HIGH_CONFIDENCE_SHARED_SOURCE_CANDIDATE` | 445 |
| train/test | `IMAGE_NEAR_ONLY_SOURCE_UNRESOLVED` | 1323 |
| train/test | `REVIEWED_BELOW_NEAR_THRESHOLD` | 4614 |
| train/train | `D4_GT_EXACT_AND_IMAGE_HIGH_SIMILARITY` | 2711 |
| train/train | `FILE_BYTES_IDENTICAL` | 3 |
| train/train | `GT_SIMILAR_ONLY_SOURCE_UNRESOLVED` | 15 |
| train/train | `IMAGE_NEAR_HIGH_CONFIDENCE_SHARED_SOURCE_CANDIDATE` | 610 |
| train/train | `IMAGE_NEAR_ONLY_SOURCE_UNRESOLVED` | 1028 |
| train/train | `REVIEWED_BELOW_NEAR_THRESHOLD` | 21393 |

只有 FULLY_REVIEWED 且达到全部高相似条件的记录对才进入高置信候选图。`IMAGE_NEAR_ONLY` 和 GT-only 相似均保留为 `SOURCE_UNRESOLVED`。原始字节、解码像素、D4 图像精确、GT 精确+高图像相似、仅图像近似、仅 GT 近似分别记录，不以单一 Pearson 判定泄漏。

## 验证

四个已知同图对、六组旋转 GT 正例和合成 JPEG+旋转正例均通过。独立脚本重新计算全部源文件/像素哈希、D4 规范哈希、全部 33145 个精核指标及来源组件，`independent_verification_report.json = PASS`。另一脚本独立重算全对八方向 Hamming、缩略图闸门、候选集合、描述子和方向，`retrieval_screen_verification.json = PASS`。新增复核器的首次最小值比较逻辑失败已保留并纠正，未修改原数据或实验结果。没有训练、模型推理、新 Bootstrap 或新模型指标评价。
