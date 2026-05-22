# LETNet / SeaFormer 可复现性评估

> 评估日期：2026-05-19
> 评估目标：判断 LETNet 和 SeaFormer 能否作为 NEU-Seg / Leather 数据集上的对比方法进入主实验表

---

## 评估结论

| 方法 | 可复现性 | 是否进入主实验表 | 理由 |
|---|---|---|---|
| **LETNet** | ✓ 可复现（需修复代码 Bug） | **待定**（可实施但 FPS 过低） | 0.95M 参数，23.5 FPS@208×208（A100），远低于其他实时方法 |
| **SeaFormer** | ❌ 不可复现（环境冲突） | **否** | mmcv-full 1.3-1.5 不兼容 PyTorch 2.x，移植成本过高 |

---

## LETNet 详细评估

### 优势

1. **极轻量**：0.95M 参数，5.04MB 模型大小，远小于 PIDNet-S (7.62M)
2. **代码可用**：模型定义完整，可在当前 PyTorch 2.7 环境下运行
3. **改动量小**：约 90 行代码改动即可适配 NEU-Seg
4. **有学术价值**：IEEE TITS 2023，CNN+Transformer 混合架构的代表

### 劣势

1. **FPS 极低**：23.5 FPS@208×208（A100 GPU），仅为 PIDNet-S (125.1 FPS) 的 18.8%
   - "Lightweight" 指参数量小，不代表推理速度快
   - Transformer 的 patch extraction + attention 在小分辨率下效率低
2. **代码质量差**：
   - 解码器通道维度 Bug（LC3 输出 16ch 与 DAB_Block_4 输出 32ch 不匹配）
   - 中文逗号语法错误
   - 重复定义的 block（copy-paste 错误）
   - 无预训练权重
3. **输入限制**：必须为 16 的倍数，NEU-Seg 200×200 需填充至 208×208
4. **数据集支持差**：硬编码仅支持 Cityscapes 和 CamVid

### 决策分析

**进入主实验表的理由**：
- 作为 CNN+Transformer 混合架构的代表，有对比价值
- 参数量极小，可展示"超轻量"方向的 trade-off
- IEEE TITS 2023 发表，有学术公信力

**不进入主实验表的理由**：
- FPS 23.5 远低于实时要求（通常 >30 FPS），与论文"real-time"定位矛盾
- 代码 Bug 需要修改模型定义，违反"不改模型结构"约束
- 无预训练权重，训练效果不确定
- 仓库 star 数仅 42，影响力有限

### 建议行动

**方案 A（推荐）：仅在相关工作中引用**
- 在论文中提及 LETNet 作为近年 CNN+Transformer 轻量分割方法
- 引用其 Cityscapes/CamVid 结果（来自论文）
- 不进入主实验对比表
- 理由：FPS 过低，不适合作为实时分割对比基线

**方案 B（可选）：修复并训练**
- 修复代码 Bug（~90 行改动）
- 在 NEU-Seg 上训练（预估 ~1 小时）
- 评估 mIoU、FPS、Params
- 风险：FPS 低可能导致审稿人质疑对比公平性

---

## SeaFormer 详细评估

### 阻塞问题

SeaFormer 的分割分支依赖 **mmcv-full 1.3.1-1.4.0**，这是 OpenMMLab 的旧版计算机视觉库，包含编译的 C++/CUDA 扩展（如 `mmcv.ops`）。

**当前环境**：
- PyTorch 2.7.1+cu126
- Python 3.11
- CUDA 12.6

**兼容性**：
- mmcv-full 1.3-1.5 仅支持 PyTorch 1.5-1.13
- PyTorch 2.x 需要 mmcv 2.x + mmengine，API 完全不同
- 尝试安装 mmcv-full 1.4.0 失败（编译超时/不兼容）

### 迁移成本分析

将 SeaFormer 从 mmcv-full 1.x 迁移到 mmcv 2.x + mmengine：

| 工作项 | 估计改动 |
|---|---|
| 替换 mmcv 导入为 mmcv + mmengine | ~30 处 |
| 更新 registry 机制 | ~10 处 |
| 更新 Config/Runner API | ~20 处 |
| 测试和调试 | 不确定 |
| **总计** | 100+ 行，风险高 |

### 建议行动

**放弃复现，在相关工作中引用**
- SeaFormer 是 ICLR 2023 代表性移动端分割方法
- 引用其 ADE20K/Cityscapes 结果（来自论文）
- 在论文中说明：环境依赖（mmcv-full 1.x）与当前实验环境不兼容
- 这是合理的学术做法，许多对比论文也引用未复现的方法

---

## 总结与建议

| 行动 | 优先级 | 预估时间 |
|---|---|---|
| 在相关工作中引用 LETNet 和 SeaFormer | 高 | 30 分钟 |
| （可选）修复 LETNet 并在 NEU-Seg 上训练 | 低 | 2-3 小时 |
| （不推荐）移植 SeaFormer 到 mmcv2 | - | 1-2 天 |

**最终建议**：两者均不进入主实验对比表。LETNet FPS 过低（23.5 vs PIDNet-S 125.1），对比不公平；SeaFormer 环境冲突无法解决。在相关工作中引用即可。
