# E0 探索性 D4 数据规则补充 v1

本补充针对作者提出的探索性 E0，保留原 `13_E0_SCIENTIFIC_PROTOCOL_FREEZE.md` 原字节。正式采用和实际运行须作者确认本补充、传递方式和资源预算；本文件不批准训练。

- 原始采集 parent 未知；NEU 只用于本轮探索性机制筛选，不宣称采集源级独立，精确D4不排除裁剪、缩放、重采样、颜色变化或其他近同源。
- 完整读取官方3630训练和840测试；图像解码后精确D4变换和逐像素验证，GT分别报告，图像一致不意味着GT一致。
- 将与840测试确认同源的训练侧D4连通组整组隔离；只用eligible组划分开发训练/验证。保存匹配证据、quarantine、ID、SHA和集合闭合；不得删改原件或原官方清单。
- 840测试只用于身份检查和隔离，不用于新模型选择、训练/验证、分组阈值或checkpoint选择。此前查看过该测试集，也不得声称它是未触碰的盲测。
- 仅D4隔离的冻结状态为 `FROZEN_EXPLORATORY_D4_ONLY`；parent及采集源独立性仍明确为 `UNKNOWN`。不可将它等同有权威parent的 `FROZEN`。
- 工具默认不启用：只有获得作者对本补充和实际执行的确认后，冻结命令才可显式传入 `--exploratory-unknown-parent-authorized`，共同运行配置才可设 `exploratory_unknown_parent_authorized=true` 且 `study_scope=EXPLORATORY_D4_ONLY_V1`。两个位置不一致时preflight拒绝；还必须通过其余数据、预检、源码及训练授权门禁。
- D4组与quarantine计数/证据、数据清单身份和三组共同协议仍须重新核验。存在shape、标签、匹配证据、集合闭合、源码或配置错误时必须停止，不能以“探索性”覆盖错误。
