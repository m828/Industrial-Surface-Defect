# 需要人工确认的问题

> 生成日期：2026-06-15  
> 用途：进入下一轮核验、重评或重训前，需要用户/论文负责人明确的决策点。

## 协议与论文口径

1. 是否最终统一采用 NVIDIA A100-PCIE-40GB 作为论文速度测试平台？
2. 是否完全废弃历史 RTX 3090 FPS，还是在文中保留为“历史/引用平台”并单独标注？
3. Leather/Pige 是否最终使用 `pige/test.txt` 作为唯一 test set？
4. `eval_protocol_lock.md` 写 Leather test=467，但当前 `pige/test.txt` 实际为 468 行，论文和后续实验应采用 467 还是 468？
5. 是否允许将训练时 `val.txt` 的 checkpoint `best_iou` 仅作为选权重依据，而绝不写入论文 test 结果？
6. 类别级 IoU、小目标/细长缺陷评价是否只在统一 test set 上报告，不再混用训练日志 val 结果？

## 权重与结果

7. 是否能找回历史 A2MS-DefectNet-B NEU-Seg 91.3 权重？
8. 如果找不回历史 91.3 权重，是否同意以重训结果作为唯一论文结果，并删除/改写历史 91.3 结论？
9. Leather Base-B 是否采用 `0127_80000` 权重作为主结果权重？该权重当前统一 test mIoU=0.8991。
10. Leather A2MS-B 是否固定使用 `dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl`？该权重 checkpoint best_iou=0.9091，但统一 test mIoU=0.8607。
11. 是否接受如果 Leather 上 A2MS-B 不优于 Base-B，则将 Leather 改写为泛化验证/困难样本分析，而不是主性能优势结论？
12. STDC 权重 `sdtdcnet_pige_stdc2_pige.pkl` 文件名含 `stdc2`，但当前脚本按 STDC1-Seg/STDCNet1446 加载，是否确认它就是 STDC1-Seg？
13. DDRNet Leather 是否优先修复 loader 后统一评估，而不是沿用训练日志 val/best_iou？
14. PIDNet-S NEU 和 Leather 是否需要重新统一评估后再进入主表？

## 训练与评估执行

15. A2MS-B NEU 重训输出是否允许写入新的 `experiments/retrain_a2ms_b_neu/` 目录，并保留 `new/model_savePath` 原权重不动？
16. 新评估结果是否统一写入新日期目录，避免覆盖当前 `class_iou_results.csv` 和 `complexity_results.csv`？
17. 是否需要把 `tools/evaluate_class_iou.py` 改造成通用 CLI：`--model_type`、`--weight_path`、`--test_txt`、`--input_size`、`--num_classes`？
18. 修复 `evaluate_class_iou.py` 的 NEU `Image` 导入和 DDRNet loader 后，是否允许提交代码修改？
19. 是否要求所有训练/评估命令先写入计划文档并由人工确认后再运行？

## 论文写作策略

20. 是否继续优先冲《中国图象图形学报》？
21. 如果核心结果重训后不支持“B 版优于 Base-B”的结论，是否接受调整论文主线为轻量实时性、困难场景分析和局限性讨论？
22. Leather 上 Base-B 当前强于 A2MS-B，是否需要 DeepScientist 同步准备论文表述替代方案？
23. 当前消融中 eSE x1、AAM x1、联合损失中间点缺独立权重，是否要重训，还是在论文中删除这些定量消融点？
24. FDSNet、PGA-Net、SFNet 等不可复现结果是否只作为“引用原论文”对比，不再声称本项目复现？
25. 可视化是否必须加入失败案例和 Base 优于 A2MS 的案例，以降低选择偏差风险？

## 数据与资产管理

26. 是否需要将大权重迁移到 Git LFS、Release、网盘或服务器固定路径，并在仓库只保留路径 manifest？
27. 是否需要为每个论文表格建立一份 `result_manifest.csv`，记录模型、数据集、权重、脚本、命令、日志、输出和 verified 状态？
28. 是否允许将 `third_party/LETNet` 当前嵌套仓库/未跟踪状态整理为明确的 submodule 或普通目录？
29. 是否保留 `new/` 与 `new (copy)/` 两套历史目录并仅记录路径，还是后续整理出一个干净的 `src/` 实验代码副本？
