# Codex 下一步行动计划

> 生成日期：2026-06-15  
> 原则：先核验，再训练；先统一协议，再写论文；所有结果必须绑定权重、脚本、命令、日志和输出文件。

## 第一阶段：只核验，不训练

目标：在不启动训练、不覆盖权重、不改论文正文的前提下，把后续实验能否安全执行确认清楚。

### 1. 协议与数据路径核验

- 确认 `eval_protocol_lock.md` 是否需要修订 Leather/Pige 样本数：协议写 test=467，但当前 `pige/test.txt` 实测为 468 行。
- 确认 NEU-Seg 使用 `new (copy)/dataset/test_neu.txt`，840 张，4 类，输入 200 x 200。
- 确认 Leather/Pige 使用 `new (copy)/dataset/pige/test.txt`，实际 468 行，8 类，输入 768 x 768。
- 确认所有评估均为 `TestRescale + ToTensor`，不使用 ImageNet Normalize。
- 确认 mIoU 包含 background。

### 2. 训练脚本和配置核验

- NEU Base-S：`new/train_neu_base_s.py` + `new/config_neu_base_s.yml`
- NEU Base-B / A2MS-B：`new/train_neu_resnet_detailloss.py` + `new/dsmonet_resnet_detailloss.yml` / `config_neu_base_b.yml` / `config_neu_a2ms_b*.yml`
- NEU A2MS-S：`new/train_neu_a2ms_s.py` + `new/config_neu_a2ms_s.yml`
- Leather Base-S/Base-B/A2MS-S/A2MS-B：`new/train_pige_*.py` + `new/config_pige_*.yml`
- DDRNet：`new/train_neu_ddr.py`、`new/train_pige_ddr.py`
- PIDNet：`new/train_neu_pid.py`、`new/train_pige_pid.py`
- STDC：`new/train_pige_stdc.py`

核验重点：

- batch size、学习率、迭代次数、优化器、权重衰减；
- `resume` 字段是否指向历史无关权重；
- A2MS-B NEU 的 detailloss 权重和输出分支是否与历史一致；
- Leather 训练时 `val.txt` 与论文最终 `test.txt` 是否明确区分。

### 3. 权重保存路径核验

- 确认不会覆盖现有 `model_savePath/*.pkl`。
- 新训练输出必须写入单独目录，例如 `experiments/retrain_a2ms_b_neu/` 或新命名的 `model_savePath/codex_*` 路径。
- A2MS-B NEU 重训前，记录当前唯一候选权重：
  - `new/model_savePath/dsmonet_resnet_pascal_0109_neu_eSE_adapt_datailloss.pkl`
- Leather 统一评估前，固定每个模型的候选权重：
  - Base-B：优先 `new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl`
  - A2MS-B：`new/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl`
  - Base-S：`new/model_savePath/dsmonet_resnet_pascal_pige_dsmor18_0228_80000.pkl`
  - A2MS-S：`new/model_savePath/dsmonet_resnet_pascal_pige_a2ms_s.pkl`
  - STDC1：`new (copy)/model_savePath/sdtdcnet_pige_stdc2_pige.pkl`
  - DDRNet：`new/model_savePath/ddr_pascal_pige_ddr23s.pkl`
  - PIDNet：`new/model_savePath/fcn_pascal_pige_pid_s.pkl`

### 4. 日志输出路径核验

- 重训前确认训练日志写入单独文件，不覆盖 `new/runs_other/train_all/*.log`。
- 评估结果统一写入 `experiments/results/` 下的新日期目录或新文件，避免覆盖当前 `class_iou_results.csv`。
- 每次评估至少保存：
  - 命令；
  - 权重路径；
  - 数据集 txt；
  - 输入尺寸；
  - GPU；
  - 输出 CSV/JSON/Markdown；
  - stdout/stderr log。

### 5. 评估脚本核验与修复

- `tools/evaluate_class_iou.py` 当前只支持 `--dataset`，不支持命令计划中的 `--model_type/--weight_path`。
- NEU loader 使用 `Image` 但未导入，需要修复后才能跑 NEU 统一评估。
- DDRNet Leather loader 当前使用不匹配模型文件，需改为正确的 `model_ddr.py` 加载方式。
- 建议先用 Base-B Leather 单模型做 smoke test，再批量评估。

## 第二阶段：阻塞实验执行

目标：解决论文主实验最关键的数值可信度问题。

### 1. 重训 A2MS-B NEU

执行前必须完成：

- 找回历史 91.3 权重的最后一次搜索；
- 对比 Base-B NEU 正常日志与 A2MS-B NEU 异常日志；
- 检查 `dsmonet_resnet_detailloss.yml` 中 `resume`、学习率、损失权重、训练轮次；
- 确认训练输出不会覆盖现有权重。

完成后必须输出：

- best checkpoint 路径；
- 完整训练日志；
- test_neu.txt 统一评估结果；
- per-class IoU；
- 与 Base-B NEU 的差异。

### 2. Leather 统一全模型评估

需要统一评估：

- Base-B；
- A2MS-B；
- Base-S；
- A2MS-S；
- STDC1-Seg；
- DDRNet23slim；
- PIDNet-S。

关键要求：

- 使用同一 `pige/test.txt`；
- 使用同一预处理；
- 使用同一类别顺序；
- 不把 `val.txt` best_iou 写成 test；
- 单独标注 Base-B(0126) 和 Base-B(0127) 的权重版本。

### 3. A100 FPS 统一测试

建议采用：

- GPU：NVIDIA A100-PCIE-40GB；
- batch size = 1；
- warmup = 50；
- runs = 200；
- `torch.cuda.synchronize()`；
- 不含后处理。

输出：

- Params；
- FLOPs@200；
- 如脚本支持，补 FLOPs@768；
- FPS@200；
- FPS@768；
- 模型大小；
- 测量时 GPU 空闲状态记录。

## 第三阶段：细粒度分析重建

目标：修复当前不能写入论文的分析项。

### 1. 类别级 IoU

- 修复 `evaluate_class_iou.py`；
- 按 NEU 和 Leather 分别生成独立结果文件；
- 确保 mIoU 与主实验表一致；
- 确保类别顺序为：
  - NEU：background, crazing, inclusion, patches；
  - Leather：background, open_wound, scratch, brand_mark, hole, skin_disease, rotten_surface, wart。

### 2. 小目标/细长缺陷评价

- 修复无缺陷组 mIoU=0.125 的评估逻辑；
- 明确分组规则是否按语义连通域近似；
- 输出每组样本 ID；
- 扩展到 Base-S/Base-B/A2MS-S/A2MS-B/STDC/DDR/PID；
- 如果 A2MS-B 仍弱于 Base-B，不强行解释为模型优势，先从数据、权重、评估口径、类别顺序排查。

### 3. 可视化重新生成

- 每类场景至少 3-5 个样本；
- 增加 Base 优于 A2MS 的案例；
- 增加两者均失败的案例；
- 保存原图、GT、各模型预测路径；
- 统一英文标题或修复 CJK 字体；
- 记录样本选择规则，避免选择偏差。

## 第四阶段：新增对比实验

目标：增强论文外部对比，但不阻塞核心结果修复。

### 1. DMC-Net

- 先确认官方代码来源和任务设置；
- 适配多类别输出头；
- 适配 NEU-Seg；
- 适配 Leather/Pige；
- 分别训练并统一评估。

### 2. PIDNet-S

- 当前已有 `new/pid.py`、`train_neu_pid.py`、`train_pige_pid.py` 和部分权重；
- 优先统一复核 NEU-Seg 与 Leather test；
- 输出同协议 mIoU、Params、FLOPs、FPS。

### 3. 可选方法

- LETNet 或 SeaFormer：第三优先级，只有在核心结果稳定后再适配。
- Boundary F1：作为边界质量补充指标。
- 更多可视化样本：作为投稿图件增强。

## 当前推荐的立即下一步

先做第一阶段，不训练：

1. 人工确认 Leather/Pige test 样本数到底采用 467 还是实际 468。
2. 修复并冻结统一评估脚本接口。
3. 用 Base-B Leather(0127) 做一次可复现 smoke test。
4. 修复 DDRNet loader 和 NEU `Image` 导入问题。
5. 生成一份不会覆盖旧文件的批量评估脚本。
