# Base-B Leather Smoke Test Summary

> 生成日期：2026-06-15  
> 任务范围：单模型 smoke test；未训练、未修改论文、未写入 fixed_existing_results.md、未覆盖权重。

## 一、评估配置

| 项目 | 值 |
|---|---|
| 数据集 | Leather/Pige |
| 测试列表 | `experiments/protocol/leather_valid_test_list.txt` |
| 有效测试样本数 | 468 |
| 模型 | Base-B |
| 模型文件 | `/workspace/Industrial Surface Defect/new/model_dsmo_rs50.py` |
| 模型入口 | `DSMONet(num_classes=8, backbone=resnet50())` |
| 权重 | `/workspace/Industrial Surface Defect/new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl` |
| 输入尺寸 | 768 x 768 |
| 类别数 | 8，包含 background |
| 预处理 | TestRescale 等价 resize + ToTensor(/255)，不使用 ImageNet Normalize |
| 设备 | NVIDIA A100-PCIE-40GB |
| 输出 CSV | `experiments/results/smoke_base_b_leather_class_iou.csv` |
| 日志 | `experiments/logs/smoke_base_b_leather.log` |

## 二、评估命令

```bash
cd "/workspace/Industrial Surface Defect/Industrial-Surface-Defect"
python tools/evaluate_class_iou.py --dataset leather --test-list experiments/protocol/leather_valid_test_list.txt --model-name Base-B --model-file '/workspace/Industrial Surface Defect/new/model_dsmo_rs50.py' --weight '/workspace/Industrial Surface Defect/new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl' --num-classes 8 --input-size 768 768 --output-csv experiments/results/smoke_base_b_leather_class_iou.csv --save-log experiments/logs/smoke_base_b_leather.log
```

## 三、结果

| 指标 | 值 |
|---|---:|
| mIoU (含 background) | 0.899120 |
| 测试样本数 | 468 |
| 权重加载 | 成功 |
| smoke test | 通过 |

## 四、类别级指标

| Class ID | Class | IoU | Dice | Precision | Recall |
|---:|---|---:|---:|---:|---:|
| 0 | background | 0.993250 | 0.996614 | 0.997542 | 0.995687 |
| 1 | open_wound | 0.764801 | 0.866728 | 0.837533 | 0.898032 |
| 2 | scratch | 0.700504 | 0.823878 | 0.839938 | 0.808420 |
| 3 | brand_mark | 0.882528 | 0.937599 | 0.921548 | 0.954219 |
| 4 | hole | 0.990708 | 0.995333 | 0.993048 | 0.997628 |
| 5 | skin_disease | 0.971181 | 0.985380 | 0.980465 | 0.990343 |
| 6 | rotten_surface | 0.915895 | 0.956101 | 0.948373 | 0.963957 |
| 7 | wart | 0.974096 | 0.986878 | 0.980898 | 0.992931 |

## 五、与上一轮结果一致性

- 与上一轮 `leather_unified_eval_all_summary.md` 中 Base-B(0127) mIoU=0.8991 一致。
- 与上一轮 Base-B(0126) mIoU=0.8806 不相同是预期现象，因为本轮 smoke test 使用的是 0127 权重。
- 因此 Base-B Leather(0127) 当前 test mIoU 可追溯到：测试列表、模型文件、权重路径、评估脚本、CSV 和日志。

## 六、注意事项

- 该结果是 smoke test 结果，不写入 `fixed_existing_results.md`。
- 后续进入论文主表前，还需要确认是否固定采用 0127 权重作为 Base-B Leather 主结果权重。
