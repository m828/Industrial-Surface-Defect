# A2MS-DefectNet-B NEU-Seg 权重扫评估摘要

> 评估日期：2026-05-22
> 评估协议：eval_protocol_lock.md (NEU-Seg: test_neu.txt, 840张, 200×200, 4类)

## 扫评估结果

| 候选权重 | 可加载 | mIoU | 接近91.3? |
|---|---:|---:|---|
| dsmonet_resnet_pascal_0109_neu_eSE_adapt_datailloss.pkl | ✅ | 0.7131 | ❌ 差距 -19.99pp |

## 判定

```
❌ 必须重训 A2MS-DefectNet-B NEU-Seg
```

**原因**：
1. 服务器上仅存在 1 个 A2MS-B NEU-Seg 候选权重
2. 该权重 mIoU=71.31，crazing IoU=0.5391，patches IoU=0.5324
3. 历史声称值 91.3 无法从当前任何权重复现
4. 历史权重（推测在 RTX 3090 上训练，文件已不在当前服务器）

## 重训建议

1. 排查当前异常训练 (`train_neu_resnet_detailloss.py`) 的配置问题
2. 确认损失权重、detailloss 配置、学习率调度是否与历史一致
3. 如果配置完全相同但结果仍然异常 → 可能是随机种子或初始化问题
4. 重新训练，监控 crazing 和 patches 类的 IoU 是否正常收敛

## 权重加载验证记录

所有候选权重均已通过 `torch.load()` + `model.load_state_dict()` 加载验证：
- 加载成功：model_dsmo_rs50_eSE_adapt_detailloss.py (A2MS-B, 29.46M params)
- checkpoint['best_iou'] = 0.713059068189787 ≈ 训练日志 0.7131 ✅ 一致
