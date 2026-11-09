# KSDD2端到端源码接入报告

状态：**KSDD2_SOURCE_READY_LOCAL**，仅在本轮完整CPU回归通过并完成源码发布核验后成立；不代表服务器真实preflight、科学候选或GPU预算获批。源码父版本为8abd026bb6b0fce051196379604f9c0ccabd6aac。本文件随源码提交，避免把自身尚未确定的commit写回造成自引用；最终完整commit及远端核验见交付回执，取得源码后可用git rev-parse HEAD核对。

## 验收结论

| 问题 | 本轮实际实现及边界 |
|---|---|
| 模型源码是否未变 | 是。factory、transport、历史loss/metrics及vendor字节与父版本一致；二类输出沿用原工厂。 |
| 训练入口是否接入 | 是。train_e0_ready按dataset_id=ksdd2显式分发，NEU路径继续原契约；共同更新循环training_loop被两任务复用，没有复制第二个训练工程。 |
| 二类loss是否支持valid | 是。仅候选ID、[10,1,3,3]，有效像素OHEM/BCE和边界内缩4px；fallback逐步记录并累积；无NEU损失修改或默认回退。 |
| 原图验证是否接入 | 是。672×256输出按native_hw裁回；全局像素指标与病例等权分开，GT阳性漏检为0、双方空为NA，正常图误报独立保留，small/other用固定P33。 |
| 独立preflight门禁 | 是。完整commit/import/源码snapshot/vendor锁、任务和数据版本、七资产SHA、实际配对/编码/尺寸/计数、初始化、输出冲突、GPU/预算/科学批准均检查；preflight不更新模型。 |
| 轻量导出是否完整 | 是。独立ksdd2_light_v1，二类矩阵/逐例/分层/响应/配置和资源身份；复用既有白名单及大小/SHA安全检查。 |
| 四模型三seed汇总 | 是。只接受同source及冻结资产的12个完整端点；同seed配置/公共初始化、带transport初始身份、病例和分层相配；三组coupled−对照逐seed及均值/样本std，无显著性宣称。 |
| NEU兼容性 | 保留原Dataset、损失、数量/类别/隔离门禁和导出分支。共享循环的CPU一步损失及更新后参数与已发布8abd循环数值一致；不声称跨GPU逐位复现。 |
| CPU回归 | 全部126项通过，新增20项集成测试，完整日志见KSDD2_CPU_TEST_RESULTS.txt；不是实测模型分割效果。 |
| Git发布 | 仅本轮白名单源码/配置/测试/文档和小CPU日志进入研究分支；实际完整SHA以最终回执及git ls-remote核验为准，不强推、不改main或历史结果。 |
| 服务器仍需资产 | 七个真实冻结文件、真实数据root和manifest列名/元数据JSON指针映射、原四份同SHA外置vendor、真实Python/GPU、外部run目录；本地未取得这些服务器资产。 |
| 作者仍需冻结 | 候选loss ID/权重/OHEM/fallback/细节有效区；Adam/LR/WD/batch/10000更新/验证间隔及具体资源上限。批准数据冻结不等于批准这些训练候选。 |

## 固定数据契约

KSDD2_DATA_FROZEN_V1由作者交付；本地不重新生成划分。train1865=197阳性+1668正常，validation466=49阳性+417正常。train SHA `dd7d50550c23887da15234f425378a0ce4e07974d0ebe533a13254db457c29c3`，validation SHA `482a8c16e4c096f8a3a3035783fb519c2570555b768dfb04da537105c9a58ee6`。P33固定0.009815324081003446，训练small66/other131/normal1668。

运行契约固定H672×W256、RGB/255、无ImageNet Normalize、无裁剪/缩放、右下复制pad、valid排除pad、0/255→0/1（255非ignore）、两logits argmax、无后处理。任何改变拒绝，不沿用上一轮640×256的旧建议。其余五文件SHA由服务器prepare读取真实字节并核对原checksum清单、再绑定配置；本地没有编造这些SHA。

真实冻结JSON和manifest结构未回传本地，因此采用显式字段映射，不猜列名。所有七文件SHA必查；P33必解码核对；其他元数据只解码实际映射字段，并在预检报告精确列出已检查语义，未映射项不冒称独立验证。运行预处理本身受固定契约强制。详见KSDD2_SERVER_RUN。

## 变更范围

- 保留并纳入上一轮有效ksdd2_support、二类初始化API、11项测试、四份准备文档、候选配置及原CPU/SHA记录。原准备SHA是上一轮快照记录，不是当前release的校验入口。
- 新增ksdd2_contract（固定契约/Dataset）、ksdd2_task（独立预检和运行接入）、prepare_ksdd2、ksdd2_artifacts、summarize_ksdd2、runtime模板与字段映射示例。
- 从原训练入口抽出一个共享training_loop，保留NEU的计算顺序、loss与监督；KSDD2通过task hook传valid、原图验证和独立组件名。恢复保存、更新journal、固定端点、有限值及资源停止复用。
- 原106测试中两项按旧源码文本位置检查step，因循环搬出而首次报错；改为同时检查调用前门禁及共享循环内finite-before-step，并增加已发布NEU循环的一步数值对照。未删测试或弱化门禁来取得PASS。
- 实际coupled CPU一步测试最初触发合成fixture的50MB磁盘上限，证明了资源停止。只为临时CPU fixture调整空间上限以容纳真实模型recovery/endpoint；未批准或提高生产GPU预算。

## 测试范围与未执行事项

126项涵盖此前106项及20项接入检查：真实尺寸合成PNG、非固定native_hw、672×256边界、空GT和255前景、同步增强、padding零监督、真实coupled的一次CPU更新和recovery、四真实模型固定尺寸输出、原图crop、fallback计数、预检错误SHA/未批准GPU/冲突拒绝、显式任务路由、轻量schema、12个合成端点配对及INCOMPLETE拒绝。模型、资产、环境和GO_READY的mock仅在单元测试临时作用域内，绝不作为真实服务器验收。

本轮未连接GPU服务器，未真实KSDD2 forward/训练，未评价官方test，未更改P33、划分或NEU资产，未使用NEU checkpoint填矩阵。启动分配失败也保存0更新和INCOMPLETE；实际初始化字段保持null直到成功核验，不冒用预检期望值。没有自动resume、自动超参调整或验证早停。所有三份运行配置默认科学/GPU/预算批准false；本地源码就绪不能替代作者批准。

## 下一步

服务器按KSDD2_SERVER_RUN取得本次明确commit，提供原冻结文件和实际字段映射，生成3份pending配置，进行独立preflight并回报预算/门禁。只有作者批准候选和费用后，才更新外置配置批准记录、重算SHA/重做预检并另行启动12组。不得重新冻结数据或重做NEU研究。
