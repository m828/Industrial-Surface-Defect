# 官方 Baseline 仓库拉取记录 (repo_clone_log)

> 日期：2026-09-29
> 用途：记录 third_party/ 官方仓库来源与版本。仅作结构参照与来源核对；训练走内部链（见 repository_audit.md 第三节）。

| 方法 | 仓库 URL | Commit | 状态 |
|---|---|---|---|
| BiSeNetV2（含 V1/V2/V3） | https://github.com/CoinCheung/BiSeNet | `6b4b67a` (fix khwc usage #352) | ✅ 已克隆（third_party/BiSeNetV2，depth 1） |
| STDC-Seg | https://github.com/MichaelFan01/STDC-Seg | `59ff37f` (Update README.md) | ✅ 已克隆（third_party/STDC-Seg，depth 1） |
| DDRNet | https://github.com/ydhongHIT/DDRNet | `de0db31` (Update README.md) | ✅ 已克隆（third_party/DDRNet，depth 1） |
| PIDNet | https://github.com/XuJiacong/PIDNet | `4c158cf24ce432f0a8cb43364fae38d93cee0dc3`（经 API 固定） | ⚠️ **克隆失败（网络）**：git clone 多次 TLS 中断 / index-pack 损坏，codeload zip 超时；已清理残目录。仓库地址与 commit 已固定，网络恢复后补拉 |

## PIDNet 仓库迁移说明

- 论文引用中常见的旧地址 `XuyangBai/PIDNet` 已 404（作者迁移/删除）；
- 当前官方地址为第一作者维护的 `XuJiacong/PIDNet`（853 stars，非 fork）；
- 不影响实验：PIDNet-S 训练使用内部实现 `new/pid.py`（结构与官方一致，已通过统一评价工具注册与 smoke 验证），官方仓库仅作参照。

## 网络备注

2026-09-29 当日到 github.com 的 HTTPS 连接不稳定（api.github.com 正常）；三次成功克隆均带重试。失败项不涉及代码决策。
