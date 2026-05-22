# 复杂度统计汇总

- 日期: 2026-05-18
- 设备: NVIDIA A100-PCIE-40GB
- PyTorch: 2.7.1+cu126
- 统计脚本: `tools/measure_complexity.py`
- 权重: 随机初始化（未加载训练权重）
- FPS 测量: warmup=50, runs=200, batch_size=1, cuda.synchronize

## 结果表格

| 模型 | Backbone | Params(M) | FLOPs(G) 200x200 | Size(MB) | FPS@200 | FPS@768 |
|---|---|---|---|---|---|---|
| Base-S | ResNet-18 | 14.00 | 2.51 | 53.49 | 149.7 | 150.5 |
| Base-B | ResNet-50 | 29.37 | 6.65 | 112.36 | 99.9 | 102.6 |
| A2MS-DefectNet-S | ResNet-18 | 14.03 | 2.34 | 53.61 | 170.8 | 169.1 |
| A2MS-DefectNet-B | ResNet-50 | 29.46 | 6.42 | 112.70 | 100.3 | 99.5 |
| DDRNet23slim | DDRNet-Internal | 6.71 | 2.07 | 25.67 | 167.7 | 166.6 |
| STDC1-Seg | STDCNet1446 | 16.07 | 6.02 | 61.40 | 121.5 | 123.7 |
| STDC2-Seg | STDCNet813 | 12.04 | 3.83 | 46.01 | 180.9 | 178.6 |
| PP-LiteSeg-B | ResNet-50 | 30.84 | 6.63 | 117.86 | 110.2 | 110.4 |
| BiSeNetV1-L | ResNet-50 | 29.38 | 6.52 | 112.28 | 127.8 | 99.2 |
| PIDNet-S | PIDNet-Internal | 7.72 | 0.97 | 29.53 | 127.7 | 127.9 |

## 错误/跳过记录

- **PP-LiteSeg-T**: 实例化失败: PP-LiteSeg-T: model_dsmo_rs50_ppliteseg.py 的 backbone_out_chs 硬编码为 [512,1024,2048]，仅支持 ResNet-50。PP-LiteSeg-T (ResNet-18) 需要单独的模型文件或修改 backbone_out_chs，但约束不允许修改模型结构。需人工确认 PP-LiteSeg-T 的正确代码入口。
- **BiSeNetV2-L**: 服务器上未找到 BiSeNetV2 代码文件，需从外部引入
- **Sub-region UNet**: Forward 失败 (200x200): Sizes of tensors must match except in dimension 1. Expected size 192 but got size 200 for tensor number 1 in the list.

## 说明

- FLOPs 仅统计 200x200 输入（NEU-Seg），768x768 输入的 FLOPs 未计算
- FPS 为推理速度（eval 模式，不含后处理）
- 模型大小包含参数和缓冲区
- 所有模型使用随机初始化权重
