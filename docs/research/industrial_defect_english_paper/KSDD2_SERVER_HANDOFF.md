> 阶段说明：以下为数据冻结前的历史准备记录。当前数据已交付KSDD2_DATA_FROZEN_V1；请以KSDD2_SOURCE_INTEGRATION_REPORT.md和KSDD2_SERVER_RUN.md为准。不要再按本文旧候选640×256输入或旧待冻结流程执行。

# KSDD2服务器后续任务说明

本轮本地未连接服务器。此文档是下一步任务提示词，不表示下述操作已完成。既有路径线索 `/workspace/Industrial Surface Defect/datasets/KSDD2/` 需在原已确认容器中核实；不猜宿主机路径，不重新开展通用SSH/VPN/GPU环境审计。

## 可直接交给服务器Codex的提示词

继续现有工业缺陷工程，仅验收KSDD2数据元信息并返回协议定版所需事实；不要训练、不要模型forward、不要评价官方测试集。NEU结果与所有既有配置保持原样。

1. 在既有容器定位真实KSDD2根目录、官方train/test分隔和原下载/解压来源说明，列出图像与GT命名配对规则。只读记录来源和文件指纹，不复制数据到Git。
2. 检查2331/1004预期是否成立；训练及测试各自逐图配对、shape/通道/mask值域、正负数量与mask空判定。测试侧读取GT只用于资产完整性和发布计数核对，不计算模型指标或用它确定任何阈值/开发选择。若希望完全封存测试GT，可先只核对测试文件清单与来源SHA，把测试正负计数标为官方元数据未独立复核。
3. 实际输出Pillow(H,W)分布，以**官方训练范围**决定候选H640×W256是否适合；不把网页230×630直接视为数组顺序。若不适合，按训练图列出需要的共同32倍数pad尺寸，保留原像素，不自行缩放细小缺陷。不依据正式测试shape改变训练尺寸；测试较大图的预定单图向上pad规则见主协议。
4. 寻找数据随附或团队已有的生产件/采集组parent信息，不做无边界历史追溯；没有则明确UNKNOWN/null。可复用原精确内容/D4函数记录已确认同源关系，不能把文件名或任意单例ID包装成采集源证明。不同官方split同名不等于同源；真实已知冲突如实报告，不移动官方数据。
5. 为官方train提供规范metadata行：sample_id（split限定）、original_split=train、图像/GT相对路径与各自文件SHA、解码像素SHA、native_hw、mask_encoding、gt_positive、gt_pixels、gt_fraction；可得parent_id及精确D4 SHA另列证据来源。official_train_ids与official_test_ids单独保存SHA。不要提供伪造parent。
6. 尚未批准真实划分前，只提交上述计数/元数据与候选组级划分报告；不运行训练入口。作者确认分组范围、比例和编码后，一次性从train生成开发train/validation，组不跨split；不因分层数量不漂亮反复筛seed。不修改官方清单。
7. 划分后从开发train阳性GT求原生面积占比P33；validation和test不能参与。保存manifest字节SHA、ID指纹、组证据及未知项、各split正负/像素分母和阈值SHA，进入新运行目录，不覆盖NEU或先前实验。
8. 将这些**小体积**JSON/CSV、ID/哈希清单回传；图像、GT、权重和大日志均不回传Git。源码仍以作者随后发布的实际完整commit为准；本地当前改动未发布，不要假定GitHub已经含KSDD2入口。

交付：KSDD2_DATA_ACCEPTANCE.json、官方列表及SHA、训练metadata、分组证据/未知项、实际H/W和编码、划分候选计数与可行性。每项区分VERIFIED_FILE、OFFICIAL_METADATA_ONLY、UNKNOWN；不生成模型成绩。

## 本地支撑函数接口（当前不能替代完整runner）

源码基底为8abd026bb6b0fce051196379604f9c0ccabd6aac；KSDD2支撑代码本轮保留本地未提交。经作者允许传递并按SHA验收后，纯元数据计划可调用：

```python
from research.industrial_defect_english.scripts.ksdd2_support import development_split, area_threshold
# rows和两份官方ID来自服务器真实验收结果；这里不构造真实路径或例数。
train_rows, validation_rows, report = development_split(
    rows, official_train_ids, official_test_ids,
    validation_fraction=0.2, seed=1337)
# 此函数不写文件，返回METADATA_PLAN_ONLY_NOT_REAL_DATA_FREEZE。
# 仅在真实开发划分获得批准后才独占写manifest并绑定其SHA。
thresholds = area_threshold(train_rows)
```

没有真实数据时不要把合成测试fixture用作正式输入。官方图像/标注的文件命名须由实际目录决定，本地没有编造`*_GT.png`等固定配对规则。

## 数据返回后的最短接入路径

按CODE_READINESS中的位置，在原工程添加显式KSDD2任务契约/数据适配，保留NEU分支和所有门禁。接入原训练循环、二类loss/valid、原图评价、轻量导出及四模型三seed汇总；不是重新实现模型或创建第二套工程。完成相应CPU与source/config/split/threshold绑定测试后，才可标记端到端KSDD2_SOURCE_READY并申请服务器preflight。

当前严禁直接把ksdd2_protocol_candidate_v1.json传给train_e0_ready.py，或把NEU的num_classes、count检查直接删除。它是不可运行候选，原NEU入口应继续拒绝它。

## 未来资源申请内容（本次不执行）

完整四模型×三seed=12次，不能只保留最佳seed。正式训练前提交冻结输入、batch、loss、更新数和新配置SHA。单独申请短GPU成本smoke，测每模型训练step中位数/分位数、完整开发验证耗时、峰值显存、checkpoint大小；不访问正式测试、不根据smoke验证成绩选模型。

以实际计时计算12次训练、验证、保存和结果导出总预算；列明分配GPU小时/实际墙钟/争用口径，报告磁盘需求与安全余量，等待作者批准。若资源不足，在任何正式比较前统一调整四模型设置并重新冻结；不能给coupled专属尺寸或降低对照训练预算。正式测试评价须后续单独授权，所有开发决策先完成。

## 当前门禁

| 项目 | 状态 |
|---|---|
| 模型二类输出/transport复用 | CPU已通过 |
| padding、空GT指标、候选loss和纯metadata分组 | CPU已通过 |
| 真实KSDD2文件/官方边界/分组/编码 | 尚未服务器验收 |
| 真实开发划分、阈值与SHA | 尚未冻结 |
| KSDD2训练/preflight/导出任务接入 | 尚未实现完整连接，详见CODE_READINESS |
| 正式loss/输入/更新数/预算 | 待作者冻结 |
| GPU训练及正式test评价 | 未授权、未执行 |
