# 复杂度统计实验计划

## 目标

统计 A2MS-DefectNet-S/B、Base-S/Base-B 及所有 baseline 的 Params、FLOPs、模型大小和 FPS。

## 已完成

- [x] A2MS-DefectNet-S: Params=14.03M, FLOPs=2.34G (200x200), Size=53.61MB, FPS=170.8@200/169.1@768 (A100, batch=1)
- [x] A2MS-DefectNet-B: Params=29.46M, FLOPs=6.42G (200x200), Size=112.70MB, FPS=100.3@200/99.5@768 (A100, batch=1)
- [x] Base-S: Params=14.00M, FLOPs=2.51G (200x200), Size=53.49MB, FPS=149.7@200/150.5@768 (A100, batch=1)
- [x] Base-B: Params=29.37M, FLOPs=6.65G (200x200), Size=112.36MB, FPS=99.9@200/102.6@768 (A100, batch=1)
- [x] DDRNet23slim: Params=6.71M, FLOPs=2.07G (200x200), Size=25.67MB, FPS=167.7@200/166.6@768 (A100, batch=1)
- [x] STDC1-Seg: Params=16.07M, FLOPs=6.02G (200x200), Size=61.40MB, FPS=121.5@200/123.7@768 (A100, batch=1)
- [x] STDC2-Seg: Params=12.04M, FLOPs=3.83G (200x200), Size=46.01MB, FPS=180.9@200/178.6@768 (A100, batch=1)
  - 注意：权重文件 `sdtdcnet_pige_stdc2_pige.pkl` 实际为 STDCNet1446（STDC1）架构，非 STDCNet813（STDC2）。类别级 IoU 评估中已更正为 STDC1-Seg。
- [x] PP-LiteSeg-B: Params=30.84M, FLOPs=6.63G (200x200), Size=117.86MB, FPS=110.2@200/110.4@768 (A100, batch=1)
- [x] BiSeNetV1-L: Params=29.38M, FLOPs=6.52G (200x200), Size=112.28MB, FPS=127.8@200/99.2@768 (A100, batch=1)
- [x] PIDNet-S: Params=7.72M, FLOPs=0.97G (200x200), Size=29.53MB, FPS=127.7@200/127.9@768 (A100, batch=1)

## 待完成

### 需人工确认后执行

- [ ] 确认 A2MS-DefectNet-S/B 和 Base-S/Base-B 对应的最终代码文件
- [ ] 使用训练后的权重重新统计 FPS（当前为随机初始化权重）
- [ ] 在 RTX 3090 上复测 FPS（与历史结果对齐）
- [ ] PP-LiteSeg-T: backbone_out_chs 硬编码为 [512,1024,2048] 仅支持 ResNet-50，需确认 T 版代码入口
- [ ] BiSeNetV2-L: 服务器上未找到代码文件，需从外部引入
- [ ] Sub-region UNet: pixelshuffle_invert 与 200x200 输入不兼容，需确认正确输入格式

### 未统计的 Baseline（需引入代码）

- [ ] U-Net: `new/model_unet.py`
- [ ] FCN-8s: `new/fcn.py`
- [ ] DeepLabV3+: `new (copy)/model/deeplabv3.py`
- [ ] PSPNet: `new (copy)/model/pspnet.py`
- [ ] ENet: `new (copy)/model/enet.py`
- [ ] HRNet: 需确认模型定义位置

## 脚本

- 统计脚本：`tools/measure_complexity.py`（独立脚本，覆盖 13 个模型）
- 输出文件：`experiments/results/complexity_results.csv`, `experiments/results/complexity_summary.md`

## 运行方法

```bash
cd /workspace/Industrial\ Surface\ Defect/new
/opt/conda/bin/python /workspace/Industrial\ Surface\ Defect/Industrial-Surface-Defect/experiments/scripts/complexity_stats.py
```

## 注意事项

- FLOPs 使用 thop 库计算，输入尺寸需与数据集匹配
- FPS 测量包含 warmup 50 次 + 正式 200 次
- 模型大小包含参数和缓冲区
- 当前结果使用随机初始化权重，FPS 为推理速度参考值
