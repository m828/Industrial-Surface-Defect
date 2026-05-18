# 仓库整理总结

## 本轮新增和修改文件清单

- `README.md`：项目说明、方法主线、数据集、当前状态和大文件管理要求；
- `.gitignore`：排除缓存、权重、数据集、日志、压缩包和运行目录；
- `TODO.md`：下一阶段实验优先级；
- `docs/paper/`：论文稿、CJIG 体例说明、投稿体例要点、修订记录、论文更新协议；
- `docs/literature/`：近年文献调研、实验候选清单、相关工作改写、参考文献核验表；
- `docs/experiment_plan/`：实验补充计划、消融计划、复杂度统计计划、类别 IoU 计划、小目标可视化计划；
- `docs/templates/`：CJIG 体例说明模板；
- `experiments/README.md`：实验目的、数据集、主要模型、结果记录规范；
- `experiments/scripts/README.md`：服务器脚本用途梳理；
- `experiments/results/`：固定结果、待核验新结果表、复杂度/类别 IoU/小目标分组 CSV 模板；
- `figures/README.md` 与图件子目录占位；
- `src/README.md`、`src/legacy_code_notes.md`：服务器代码整理边界说明；
- `next_experiment_checklist.md`：下一步服务器实验清单；
- `paper_after_experiment_plan.md`：实验完成后的论文更新计划。

## 目录结构

```text
Industrial-Surface-Defect/
├── README.md
├── TODO.md
├── docs/
│   ├── paper/
│   ├── literature/
│   ├── experiment_plan/
│   └── templates/
├── experiments/
│   ├── configs/
│   ├── scripts/
│   ├── logs/
│   └── results/
├── figures/
│   ├── architecture/
│   ├── modules/
│   ├── dataset_samples/
│   └── qualitative_results/
├── src/
├── next_experiment_checklist.md
├── paper_after_experiment_plan.md
└── repo_update_summary.md
```

## 主要内容说明

本轮整理保留当前论文主线：工业表面缺陷实时语义分割，以细节—语义互优化机制为核心，面向工业缺陷特点设计 ELMM、AAM 和联合监督约束。模型命名统一为 A2MS-DefectNet、A2MS-DefectNet-S/B、Base-S/Base-B。

固定结果文件中的实验数值未新增，仅按第四版论文命名规则统一名称。新实验结果必须先进入 `experiments/results/new_results_pending.md`，核验后再同步到固定结果和论文。

## Git 状态

- 本地目标仓库：`/home/mmy/DeepScientist/quests/002/github/Industrial-Surface-Defect`
- 目标远程仓库：`https://github.com/m828/Industrial-Surface-Defect.git`
- 分支：`paper-experiment-prep`
- 提交状态：已本地提交
- GitHub 推送状态：未成功推送。原因：`git clone` 曾在 120 秒超时，后续非交互式 `git push -u origin paper-experiment-prep` 在 90 秒内无输出并被终止；疑似当前环境无法稳定访问 GitHub HTTPS 或缺少可用认证。已按要求停止重试。

## 需要用户手动执行的 Git 命令

本地提交已经存在，通常只需在具备 GitHub 权限和网络的环境中执行：

```bash
cd /home/mmy/DeepScientist/quests/002/github/Industrial-Surface-Defect
git remote -v
git status --short --branch
git log --oneline -1
git push -u origin paper-experiment-prep
```

若需要在另一台机器重新整理并提交，请先复制本仓库文件，再执行 `git add`、`git commit` 和 `git push`。
