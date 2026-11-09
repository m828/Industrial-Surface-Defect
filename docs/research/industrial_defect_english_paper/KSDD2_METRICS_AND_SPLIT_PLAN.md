> 阶段说明：以下为数据冻结前的历史准备记录。当前数据已交付KSDD2_DATA_FROZEN_V1；请以KSDD2_SOURCE_INTEGRATION_REPORT.md和KSDD2_SERVER_RUN.md为准。不要再按本文旧候选640×256输入或旧待冻结流程执行。

# KSDD2指标与开发划分计划

状态：候选定义，本地只对合成记录执行；没有生成真实KSDD2划分或评价正式测试集。主协议见KSDD2_EXPERIMENT_PROTOCOL.md。

## 官方边界、ID和配对

官方元数据train2331/test1004，预期阳性/正常分别246/2085、110/894；服务器须由真实文件验收。sample_id使用带split的规范ID（例如train/相对图像路径），不把train/test同文件名视为同一图，也不把不同文件名当作独立来源证据。

实际检查图像—mask一一配对、RGB/灰度解码、原始H/W、二维mask值域及mask为空的图数。0/1与0/255必须显式选择编码；255可能是前景值，不能沿用ignore=255就丢失缺陷。颜色mask、多取值、错shape、缺失/额外mask先报告，不猜映射。正常图必须有可核对的空mask记录，不能仅据文件夹名认定正常。实际图像/标注SHA、解码像素指纹和来源版本随manifest保存。

## 分组和一次划分

1. 只用官方训练记录建立开发集。官方测试列表只用于边界排除和身份核对，不用测试GT统计选分层、尺寸或阈值。
2. 优先取得官方或团队可信的生产件/采集源parent信息。没有则parent_id=null、来源状态UNKNOWN；禁止把所有未知行填同一个“UNKNOWN”父ID，禁止从文件名猜parent。
3. 以已知parent、解码像素SHA、已有精确内容/D4证据组成连通组。已有精确D4工具可复用；无精确D4记录不伪造。可得同源证据组整体落在同一开发split，原图的衍生patch也继承它。
4. 划分候选seed1337、目标validation_fraction=.2。按阳性/正常图像数的目标偏差进行确定性组级贪心分配；输入先按ID排序，组顺序用局部Random(seed)打散，一次执行，不根据成绩重试seed。
5. 在全部为独立单例、且真实计数等于官方元数据的理想条件下，目标约开发train1865（197阳性/1668正常）、val466（49阳性/417正常）。**这只是目标，不是已生成清单**。组约束优先，实际数量允许不同。
6. 必须满足train/val ID互斥、group互斥、覆盖全部原train；两split均有阳性和正常。若已知组使覆盖不可能，报告BLOCKED而不拆组或偷偷改seed。发现重要来源冲突或官方跨split同源疑点时，报告事实，保持官方边界，不自动迁移或quarantine。
7. parent缺失时最多声明“按当前已知来源和精确内容分组”，不能宣称采集源级独立。可由作者明确批准该限定范围，不要求通过无限搜parent才能推进。

本地实现 `ksdd2_support.development_split(rows, official_train_ids, official_test_ids, validation_fraction=.2, seed=1337)` **只处理已验收元数据，不读数据、不写清单、不授予冻结状态**。返回状态METADATA_PLAN_ONLY_NOT_REAL_DATA_FREEZE。服务器真实冻结还必须核验官方路径/文件SHA和完整配对，再独占写入新目录。

冻结包至少包括：official_train/test列表SHA、完整来源指纹、dev_train/validation JSONL和ID SHA、group证据及未知项、逐split阳性/阴性及像素分母、preprocessing/mask-encoding协议、训练阈值文件及SHA。所有模型和seed只读取相同字节，禁止另划分。

## 尺度阈值

主分层“小面积缺陷”以**开发训练阳性图像**原生坐标GT面积占比的P33定义，numpy quantile(method=linear, q=1/3)。阳性图像gt_fraction<=P33为small，其他阳性为other；正常图单列，不放进small。记录阈值数值、训练manifest SHA、分母图像大小和样本数。并列值可使small数量不等于严格三分之一，不为均分而改阈值。

服务器未读取GT前，不填具体阈值。`area_threshold`仅接受训练阳性记录。形态复杂度如将来加入，只复用明确的内轮廓像素/sqrt(area)代理，训练范围内定阈值；本轮不强制增加复杂度作为新主要终点。

## 主指标：全局像素汇总

所有图像先去padding，在原始坐标以两类logits argmax得到预测；等值tie遵循argmax背景通道优先。不通过验证或测试调概率阈值，不作连通域滤除、膨胀或填洞。

汇总2×2混淆矩阵，GT作行、预测作列：`[[TN,FP],[FN,TP]]`。

- Foreground IoU = TP/(TP+FP+FN)。
- Foreground Dice = 2TP/(2TP+FP+FN)。
- Background IoU = TN/(TN+FP+FN)。
- Precision=TP/(TP+FP)，Recall=TP/(TP+FN)。分母为0时NA，不能任意记1。
- 含背景mIoU为两类IoU均可定义时的等权均值，不能替代主前景指标。

全部正常图的FP仍进入主前景指标分母。不同图大小时按实际像素数聚合。padding一律排除，不能增加大量“正确背景”抬高结果。

## 病例指标和空样本规则

| GT | 预测 | 前景病例IoU/Dice | 纳入和额外报告 |
|---|---|---|---|
| 非空 | 非空 | 通常定义 | 纳入GT阳性病例均值 |
| 非空 | 空 | 0 / 0 | 纳入GT阳性病例，完整漏检 |
| 空 | 非空 | 0 / 0 | 误报病例；像素FP和正常图误报都报告 |
| 空 | 空 | NA / NA | 不计前景病例均值；仍计正常图和全局TN |

同时提供：

1. **固定GT阳性集合的等权病例IoU与Dice**：用于模型配对比较，完全漏检计0。
2. 各模型“GT或预测非空”有效集合的病例宏平均IoU及有效n：作为描述性补充。集合可能因误报不同而变化，不能与固定GT阳性估计混为一谈，不能套用同一配对区间。
3. 空GT图像数、任意前景预测的误报图像数、正常图误报率、误报像素数与其正常像素分母。不用前景均值NA隐藏误报。
4. small阳性组的n、宏平均IoU/Dice及原始TP/FP/FN；n<20标记低信息量。

主全局IoU与病例均值回答不同问题，不能把方向差异直接解释为同一图像内的机制权衡。`binary_metrics`返回上述分离字段及全部病例行，合成测试确认两种均值可以不同。

## 比较与统计

开发验证用于冻结前的统一开发，官方test全程保留。正式矩阵训练完成后，主要报告三种对照下coupled逐seed差与三seed均值/样本标准差；不挑最佳seed。可另行计划同一图像/来源组的配对bootstrap，但它只描述病例抽样，不替代训练随机性；空样本集合规则必须与点估计一致。

本轮没有GT、预测或模型效果数据。用户提供的NEU E1-C三个差值只作为继续跨数据集验证的动机，不代表KSDD2效果，也未在本轮重新核算。
