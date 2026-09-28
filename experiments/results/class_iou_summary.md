# 类别级 IoU 评估汇总

- 日期: 2026-05-26
- 设备: NVIDIA A100-PCIE-40GB
- 评估脚本: `tools/evaluate_class_iou.py`
- 数据集: NEU-Seg (840 张)

## 类别说明

### NEU-Seg (4 类含背景)

| 类别 ID | 英文名 | 中文名 |
|---|---|---|
| 0 | background | 背景 |
| 1 | crazing | 龟裂 |
| 2 | inclusion | 夹杂物 |
| 3 | patches | 补丁 |

## 模型对比 (mIoU)

| 模型 | Backbone | 权重 | mIoU |
|---|---|---|---|
| Base-B | ResNet-50 | 缺少 | N/A |
| A2MS-DefectNet-B | ResNet-50 | 缺少 | 0.8607 |
| Base-S | ResNet-18 | 缺少 | 0.8586 |
| A2MS-DefectNet-S | ResNet-18 | 缺少 | 0.8690 |
| STDC1-Seg | STDCNet1446 | 缺少 | 0.8824 |
| PP-LiteSeg-B | ResNet-50 | 缺少 | N/A |
| DDRNet23slim | DDRNet-Internal | 缺少 | N/A |
| BiSeNetV2-L | BiSeNetV2 | 缺少 | N/A |
| Sub-region UNet | Sub-region-UNet | 有 | N/A |
| PIDNet-S | PIDNet-Internal | 缺少 | 0.8620 |

## 各类别 IoU 详细对比

| 类别 | Sub-region UNet |
|---|---|
| 背景 | N/A |
| 龟裂 | N/A |
| 夹杂物 | N/A |
| 补丁 | N/A |

## 各类别 Dice 详细对比

| 类别 | Sub-region UNet |
|---|---|
| 背景 | N/A |
| 龟裂 | N/A |
| 夹杂物 | N/A |
| 补丁 | N/A |

## 错误/跳过记录

- **PP-LiteSeg-B** (Leather): 缺少权重，未评估
- **DDRNet23slim** (Leather): 模型加载失败: Error(s) in loading state_dict for DSMONet:
	Missing key(s) in state_dict: "cm.scale1.1.weight", "cm.scale1.1.bias", "cm.scale1.1.running_mean", "cm.scale1.1.running_var", "cm.scale1.3.weight", "cm.scale2.1.weight", "cm.scale2.1.bias", "cm.scale2.1.running_mean", "cm.scale2.1.running_var", "cm.scale2.3.weight", "cm.scale3.1.weight", "cm.scale3.1.bias", "cm.scale3.1.running_mean", "cm.scale3.1.running_var", "cm.scale3.3.weight", "cm.scale4.1.weight", "cm.scale4.1.bias", "cm.scale4.1.running_mean", "cm.scale4.1.running_var", "cm.scale4.3.weight", "cm.scale0.0.weight", "cm.scale0.0.bias", "cm.scale0.0.running_mean", "cm.scale0.0.running_var", "cm.scale0.2.weight", "cm.process1.0.weight", "cm.process1.0.bias", "cm.process1.0.running_mean", "cm.process1.0.running_var", "cm.process1.2.weight", "cm.process2.0.weight", "cm.process2.0.bias", "cm.process2.0.running_mean", "cm.process2.0.running_var", "cm.process2.2.weight", "cm.process3.0.weight", "cm.process3.0.bias", "cm.process3.0.running_mean", "cm.process3.0.running_var", "cm.process3.2.weight", "cm.process4.0.weight", "cm.process4.0.bias", "cm.process4.0.running_mean", "cm.process4.0.running_var", "cm.process4.2.weight", "cm.compression.0.weight", "cm.compression.0.bias", "cm.compression.0.running_mean", "cm.compression.0.running_var", "cm.compression.2.weight", "cm.shortcut.0.weight", "cm.shortcut.0.bias", "cm.shortcut.0.running_mean", "cm.shortcut.0.running_var", "cm.shortcut.2.weight", "seg_heads.conv._conv.weight", "seg_heads.conv._batch_norm.weight", "seg_heads.conv._batch_norm.bias", "seg_heads.conv._batch_norm.running_mean", "seg_heads.conv._batch_norm.running_var", "seg_heads.conv_out.weight", "se.fc.weight", "se.fc.bias", "se.channel_weight.weight", "arm1.conv_x.0.weight", "arm1.conv_x.1.weight", "arm1.conv_x.1.bias", "arm1.conv_x.1.running_mean", "arm1.conv_x.1.running_var", "arm1.conv_out.0.weight", "arm1.conv_out.1.weight", "arm1.conv_out.1.bias", "arm1.conv_out.1.running_mean", "arm1.conv_out.1.running_var", "arm1.conv_xy_atten1.0.weight", "arm1.conv_xy_atten1.1.weight", "arm1.conv_xy_atten1.1.bias", "arm1.conv_xy_atten1.1.running_mean", "arm1.conv_xy_atten1.1.running_var", "arm1.conv_xy_atten1.3.weight", "arm1.conv_xy_atten1.4.weight", "arm1.conv_xy_atten1.4.bias", "arm1.conv_xy_atten1.4.running_mean", "arm1.conv_xy_atten1.4.running_var", "arm1.conv_xy_atten2.0.weight", "arm1.conv_xy_atten2.1.weight", "arm1.conv_xy_atten2.1.bias", "arm1.conv_xy_atten2.1.running_mean", "arm1.conv_xy_atten2.1.running_var", "arm1.conv_xy_atten2.3.weight", "arm1.conv_xy_atten2.4.weight", "arm1.conv_xy_atten2.4.bias", "arm1.conv_xy_atten2.4.running_mean", "arm1.conv_xy_atten2.4.running_var", "arm2.conv_x.0.weight", "arm2.conv_x.1.weight", "arm2.conv_x.1.bias", "arm2.conv_x.1.running_mean", "arm2.conv_x.1.running_var", "arm2.conv_out.0.weight", "arm2.conv_out.1.weight", "arm2.conv_out.1.bias", "arm2.conv_out.1.running_mean", "arm2.conv_out.1.running_var", "arm2.conv_xy_atten1.0.weight", "arm2.conv_xy_atten1.1.weight", "arm2.conv_xy_atten1.1.bias", "arm2.conv_xy_atten1.1.running_mean", "arm2.conv_xy_atten1.1.running_var", "arm2.conv_xy_atten1.3.weight", "arm2.conv_xy_atten1.4.weight", "arm2.conv_xy_atten1.4.bias", "arm2.conv_xy_atten1.4.running_mean", "arm2.conv_xy_atten1.4.running_var", "arm2.conv_xy_atten2.0.weight", "arm2.conv_xy_atten2.1.weight", "arm2.conv_xy_atten2.1.bias", "arm2.conv_xy_atten2.1.running_mean", "arm2.conv_xy_atten2.1.running_var", "arm2.conv_xy_atten2.3.weight", "arm2.conv_xy_atten2.4.weight", "arm2.conv_xy_atten2.4.bias", "arm2.conv_xy_atten2.4.running_mean", "arm2.conv_xy_atten2.4.running_var", "edge_fusion.0.weight", "edge_fusion.1.weight", "edge_fusion.1.bias", "edge_fusion.1.running_mean", "edge_fusion.1.running_var", "bot_fine.weight", "laplacian.conv_op.weight", "conv_up.0.weight", "conv_up.2.weight", "body_edge.conv_p.0.weight", "body_edge.conv_p.1.weight", "body_edge.conv_p.1.bias", "body_edge.conv_p.1.running_mean", "body_edge.conv_p.1.running_var". 
	Unexpected key(s) in state_dict: "compression3.0.weight", "compression3.1.weight", "compression3.1.bias", "compression3.1.running_mean", "compression3.1.running_var", "compression3.1.num_batches_tracked", "compression4.0.weight", "compression4.1.weight", "compression4.1.bias", "compression4.1.running_mean", "compression4.1.running_var", "compression4.1.num_batches_tracked", "down3.0.weight", "down3.1.weight", "down3.1.bias", "down3.1.running_mean", "down3.1.running_var", "down3.1.num_batches_tracked", "down4.0.weight", "down4.1.weight", "down4.1.bias", "down4.1.running_mean", "down4.1.running_var", "down4.1.num_batches_tracked", "down4.3.weight", "down4.4.weight", "down4.4.bias", "down4.4.running_mean", "down4.4.running_var", "down4.4.num_batches_tracked", "layer3_.0.conv1.weight", "layer3_.0.bn1.weight", "layer3_.0.bn1.bias", "layer3_.0.bn1.running_mean", "layer3_.0.bn1.running_var", "layer3_.0.bn1.num_batches_tracked", "layer3_.0.conv2.weight", "layer3_.0.bn2.weight", "layer3_.0.bn2.bias", "layer3_.0.bn2.running_mean", "layer3_.0.bn2.running_var", "layer3_.0.bn2.num_batches_tracked", "layer3_.1.conv1.weight", "layer3_.1.bn1.weight", "layer3_.1.bn1.bias", "layer3_.1.bn1.running_mean", "layer3_.1.bn1.running_var", "layer3_.1.bn1.num_batches_tracked", "layer3_.1.conv2.weight", "layer3_.1.bn2.weight", "layer3_.1.bn2.bias", "layer3_.1.bn2.running_mean", "layer3_.1.bn2.running_var", "layer3_.1.bn2.num_batches_tracked", "layer4_.0.conv1.weight", "layer4_.0.bn1.weight", "layer4_.0.bn1.bias", "layer4_.0.bn1.running_mean", "layer4_.0.bn1.running_var", "layer4_.0.bn1.num_batches_tracked", "layer4_.0.conv2.weight", "layer4_.0.bn2.weight", "layer4_.0.bn2.bias", "layer4_.0.bn2.running_mean", "layer4_.0.bn2.running_var", "layer4_.0.bn2.num_batches_tracked", "layer4_.1.conv1.weight", "layer4_.1.bn1.weight", "layer4_.1.bn1.bias", "layer4_.1.bn1.running_mean", "layer4_.1.bn1.running_var", "layer4_.1.bn1.num_batches_tracked", "layer4_.1.conv2.weight", "layer4_.1.bn2.weight", "layer4_.1.bn2.bias", "layer4_.1.bn2.running_mean", "layer4_.1.bn2.running_var", "layer4_.1.bn2.num_batches_tracked", "layer5_.0.conv1.weight", "layer5_.0.bn1.weight", "layer5_.0.bn1.bias", "layer5_.0.bn1.running_mean", "layer5_.0.bn1.running_var", "layer5_.0.bn1.num_batches_tracked", "layer5_.0.conv2.weight", "layer5_.0.bn2.weight", "layer5_.0.bn2.bias", "layer5_.0.bn2.running_mean", "layer5_.0.bn2.running_var", "layer5_.0.bn2.num_batches_tracked", "layer5_.0.conv3.weight", "layer5_.0.bn3.weight", "layer5_.0.bn3.bias", "layer5_.0.bn3.running_mean", "layer5_.0.bn3.running_var", "layer5_.0.bn3.num_batches_tracked", "layer5_.0.downsample.0.weight", "layer5_.0.downsample.1.weight", "layer5_.0.downsample.1.bias", "layer5_.0.downsample.1.running_mean", "layer5_.0.downsample.1.running_var", "layer5_.0.downsample.1.num_batches_tracked", "spp.scale1.1.weight", "spp.scale1.1.bias", "spp.scale1.1.running_mean", "spp.scale1.1.running_var", "spp.scale1.1.num_batches_tracked", "spp.scale1.3.weight", "spp.scale2.1.weight", "spp.scale2.1.bias", "spp.scale2.1.running_mean", "spp.scale2.1.running_var", "spp.scale2.1.num_batches_tracked", "spp.scale2.3.weight", "spp.scale3.1.weight", "spp.scale3.1.bias", "spp.scale3.1.running_mean", "spp.scale3.1.running_var", "spp.scale3.1.num_batches_tracked", "spp.scale3.3.weight", "spp.scale4.1.weight", "spp.scale4.1.bias", "spp.scale4.1.running_mean", "spp.scale4.1.running_var", "spp.scale4.1.num_batches_tracked", "spp.scale4.3.weight", "spp.scale0.0.weight", "spp.scale0.0.bias", "spp.scale0.0.running_mean", "spp.scale0.0.running_var", "spp.scale0.0.num_batches_tracked", "spp.scale0.2.weight", "spp.process1.0.weight", "spp.process1.0.bias", "spp.process1.0.running_mean", "spp.process1.0.running_var", "spp.process1.0.num_batches_tracked", "spp.process1.2.weight", "spp.process2.0.weight", "spp.process2.0.bias", "spp.process2.0.running_mean", "spp.process2.0.running_var", "spp.process2.0.num_batches_tracked", "spp.process2.2.weight", "spp.process3.0.weight", "spp.process3.0.bias", "spp.process3.0.running_mean", "spp.process3.0.running_var", "spp.process3.0.num_batches_tracked", "spp.process3.2.weight", "spp.process4.0.weight", "spp.process4.0.bias", "spp.process4.0.running_mean", "spp.process4.0.running_var", "spp.process4.0.num_batches_tracked", "spp.process4.2.weight", "spp.compression.0.weight", "spp.compression.0.bias", "spp.compression.0.running_mean", "spp.compression.0.running_var", "spp.compression.0.num_batches_tracked", "spp.compression.2.weight", "spp.shortcut.0.weight", "spp.shortcut.0.bias", "spp.shortcut.0.running_mean", "spp.shortcut.0.running_var", "spp.shortcut.0.num_batches_tracked", "spp.shortcut.2.weight", "seghead_extra.bn1.weight", "seghead_extra.bn1.bias", "seghead_extra.bn1.running_mean", "seghead_extra.bn1.running_var", "seghead_extra.bn1.num_batches_tracked", "seghead_extra.conv1.weight", "seghead_extra.bn2.weight", "seghead_extra.bn2.bias", "seghead_extra.bn2.running_mean", "seghead_extra.bn2.running_var", "seghead_extra.bn2.num_batches_tracked", "seghead_extra.conv2.weight", "seghead_extra.conv2.bias", "final_layer.bn1.weight", "final_layer.bn1.bias", "final_layer.bn1.running_mean", "final_layer.bn1.running_var", "final_layer.bn1.num_batches_tracked", "final_layer.conv1.weight", "final_layer.bn2.weight", "final_layer.bn2.bias", "final_layer.bn2.running_mean", "final_layer.bn2.running_var", "final_layer.bn2.num_batches_tracked", "final_layer.conv2.weight", "final_layer.conv2.bias". 
	size mismatch for conv1.0.weight: copying a param with shape torch.Size([64, 3, 3, 3]) from checkpoint, the shape in current model is torch.Size([32, 3, 3, 3]).
	size mismatch for conv1.0.bias: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for conv1.1.weight: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for conv1.1.bias: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for conv1.1.running_mean: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for conv1.1.running_var: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for conv1.3.weight: copying a param with shape torch.Size([64, 64, 3, 3]) from checkpoint, the shape in current model is torch.Size([32, 32, 1, 1]).
	size mismatch for conv1.3.bias: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for conv1.4.weight: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for conv1.4.bias: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for conv1.4.running_mean: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for conv1.4.running_var: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for layer1.0.conv1.weight: copying a param with shape torch.Size([64, 64, 3, 3]) from checkpoint, the shape in current model is torch.Size([32, 32, 3, 3]).
	size mismatch for layer1.0.bn1.weight: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for layer1.0.bn1.bias: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for layer1.0.bn1.running_mean: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for layer1.0.bn1.running_var: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for layer1.0.conv2.weight: copying a param with shape torch.Size([64, 64, 3, 3]) from checkpoint, the shape in current model is torch.Size([32, 32, 3, 3]).
	size mismatch for layer1.0.bn2.weight: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for layer1.0.bn2.bias: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for layer1.0.bn2.running_mean: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for layer1.0.bn2.running_var: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for layer1.1.conv1.weight: copying a param with shape torch.Size([64, 64, 3, 3]) from checkpoint, the shape in current model is torch.Size([32, 32, 3, 3]).
	size mismatch for layer1.1.bn1.weight: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for layer1.1.bn1.bias: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for layer1.1.bn1.running_mean: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for layer1.1.bn1.running_var: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for layer1.1.conv2.weight: copying a param with shape torch.Size([64, 64, 3, 3]) from checkpoint, the shape in current model is torch.Size([32, 32, 3, 3]).
	size mismatch for layer1.1.bn2.weight: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for layer1.1.bn2.bias: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for layer1.1.bn2.running_mean: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for layer1.1.bn2.running_var: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for layer2.0.conv1.weight: copying a param with shape torch.Size([128, 64, 3, 3]) from checkpoint, the shape in current model is torch.Size([64, 32, 3, 3]).
	size mismatch for layer2.0.bn1.weight: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for layer2.0.bn1.bias: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for layer2.0.bn1.running_mean: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for layer2.0.bn1.running_var: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for layer2.0.conv2.weight: copying a param with shape torch.Size([128, 128, 3, 3]) from checkpoint, the shape in current model is torch.Size([64, 64, 3, 3]).
	size mismatch for layer2.0.bn2.weight: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for layer2.0.bn2.bias: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for layer2.0.bn2.running_mean: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for layer2.0.bn2.running_var: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for layer2.0.downsample.0.weight: copying a param with shape torch.Size([128, 64, 1, 1]) from checkpoint, the shape in current model is torch.Size([64, 32, 1, 1]).
	size mismatch for layer2.0.downsample.1.weight: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for layer2.0.downsample.1.bias: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for layer2.0.downsample.1.running_mean: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for layer2.0.downsample.1.running_var: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for layer2.1.conv1.weight: copying a param with shape torch.Size([128, 128, 3, 3]) from checkpoint, the shape in current model is torch.Size([64, 64, 3, 3]).
	size mismatch for layer2.1.bn1.weight: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for layer2.1.bn1.bias: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for layer2.1.bn1.running_mean: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for layer2.1.bn1.running_var: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for layer2.1.conv2.weight: copying a param with shape torch.Size([128, 128, 3, 3]) from checkpoint, the shape in current model is torch.Size([64, 64, 3, 3]).
	size mismatch for layer2.1.bn2.weight: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for layer2.1.bn2.bias: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for layer2.1.bn2.running_mean: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for layer2.1.bn2.running_var: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for layer3.0.conv1.weight: copying a param with shape torch.Size([256, 128, 3, 3]) from checkpoint, the shape in current model is torch.Size([128, 64, 3, 3]).
	size mismatch for layer3.0.bn1.weight: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for layer3.0.bn1.bias: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for layer3.0.bn1.running_mean: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for layer3.0.bn1.running_var: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for layer3.0.conv2.weight: copying a param with shape torch.Size([256, 256, 3, 3]) from checkpoint, the shape in current model is torch.Size([128, 128, 3, 3]).
	size mismatch for layer3.0.bn2.weight: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for layer3.0.bn2.bias: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for layer3.0.bn2.running_mean: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for layer3.0.bn2.running_var: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for layer3.0.downsample.0.weight: copying a param with shape torch.Size([256, 128, 1, 1]) from checkpoint, the shape in current model is torch.Size([128, 64, 1, 1]).
	size mismatch for layer3.0.downsample.1.weight: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for layer3.0.downsample.1.bias: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for layer3.0.downsample.1.running_mean: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for layer3.0.downsample.1.running_var: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for layer3.1.conv1.weight: copying a param with shape torch.Size([256, 256, 3, 3]) from checkpoint, the shape in current model is torch.Size([128, 128, 3, 3]).
	size mismatch for layer3.1.bn1.weight: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for layer3.1.bn1.bias: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for layer3.1.bn1.running_mean: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for layer3.1.bn1.running_var: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for layer3.1.conv2.weight: copying a param with shape torch.Size([256, 256, 3, 3]) from checkpoint, the shape in current model is torch.Size([128, 128, 3, 3]).
	size mismatch for layer3.1.bn2.weight: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for layer3.1.bn2.bias: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for layer3.1.bn2.running_mean: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for layer3.1.bn2.running_var: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for layer4.0.conv1.weight: copying a param with shape torch.Size([512, 256, 3, 3]) from checkpoint, the shape in current model is torch.Size([256, 128, 3, 3]).
	size mismatch for layer4.0.bn1.weight: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer4.0.bn1.bias: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer4.0.bn1.running_mean: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer4.0.bn1.running_var: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer4.0.conv2.weight: copying a param with shape torch.Size([512, 512, 3, 3]) from checkpoint, the shape in current model is torch.Size([256, 256, 3, 3]).
	size mismatch for layer4.0.bn2.weight: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer4.0.bn2.bias: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer4.0.bn2.running_mean: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer4.0.bn2.running_var: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer4.0.downsample.0.weight: copying a param with shape torch.Size([512, 256, 1, 1]) from checkpoint, the shape in current model is torch.Size([256, 128, 1, 1]).
	size mismatch for layer4.0.downsample.1.weight: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer4.0.downsample.1.bias: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer4.0.downsample.1.running_mean: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer4.0.downsample.1.running_var: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer4.1.conv1.weight: copying a param with shape torch.Size([512, 512, 3, 3]) from checkpoint, the shape in current model is torch.Size([256, 256, 3, 3]).
	size mismatch for layer4.1.bn1.weight: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer4.1.bn1.bias: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer4.1.bn1.running_mean: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer4.1.bn1.running_var: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer4.1.conv2.weight: copying a param with shape torch.Size([512, 512, 3, 3]) from checkpoint, the shape in current model is torch.Size([256, 256, 3, 3]).
	size mismatch for layer4.1.bn2.weight: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer4.1.bn2.bias: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer4.1.bn2.running_mean: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer4.1.bn2.running_var: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer5.0.conv1.weight: copying a param with shape torch.Size([512, 512, 1, 1]) from checkpoint, the shape in current model is torch.Size([256, 256, 1, 1]).
	size mismatch for layer5.0.bn1.weight: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer5.0.bn1.bias: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer5.0.bn1.running_mean: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer5.0.bn1.running_var: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer5.0.conv2.weight: copying a param with shape torch.Size([512, 512, 3, 3]) from checkpoint, the shape in current model is torch.Size([256, 256, 3, 3]).
	size mismatch for layer5.0.bn2.weight: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer5.0.bn2.bias: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer5.0.bn2.running_mean: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer5.0.bn2.running_var: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([256]).
	size mismatch for layer5.0.conv3.weight: copying a param with shape torch.Size([1024, 512, 1, 1]) from checkpoint, the shape in current model is torch.Size([512, 256, 1, 1]).
	size mismatch for layer5.0.bn3.weight: copying a param with shape torch.Size([1024]) from checkpoint, the shape in current model is torch.Size([512]).
	size mismatch for layer5.0.bn3.bias: copying a param with shape torch.Size([1024]) from checkpoint, the shape in current model is torch.Size([512]).
	size mismatch for layer5.0.bn3.running_mean: copying a param with shape torch.Size([1024]) from checkpoint, the shape in current model is torch.Size([512]).
	size mismatch for layer5.0.bn3.running_var: copying a param with shape torch.Size([1024]) from checkpoint, the shape in current model is torch.Size([512]).
	size mismatch for layer5.0.downsample.0.weight: copying a param with shape torch.Size([1024, 512, 1, 1]) from checkpoint, the shape in current model is torch.Size([512, 256, 1, 1]).
	size mismatch for layer5.0.downsample.1.weight: copying a param with shape torch.Size([1024]) from checkpoint, the shape in current model is torch.Size([512]).
	size mismatch for layer5.0.downsample.1.bias: copying a param with shape torch.Size([1024]) from checkpoint, the shape in current model is torch.Size([512]).
	size mismatch for layer5.0.downsample.1.running_mean: copying a param with shape torch.Size([1024]) from checkpoint, the shape in current model is torch.Size([512]).
	size mismatch for layer5.0.downsample.1.running_var: copying a param with shape torch.Size([1024]) from checkpoint, the shape in current model is torch.Size([512]).
- **BiSeNetV2-L** (Leather): 缺少权重，未评估
- **Sub-region UNet** (Leather): 缺少权重，未评估
- **Base-B** (NEU-Seg): 缺少权重，未评估
- **A2MS-DefectNet-B** (NEU-Seg): 缺少权重，未评估
- **Base-S** (NEU-Seg): 缺少权重，未评估
- **A2MS-DefectNet-S** (NEU-Seg): 缺少权重，未评估
- **STDC1-Seg** (NEU-Seg): 缺少权重，未评估
- **PP-LiteSeg-B** (NEU-Seg): 缺少权重，未评估
- **DDRNet23slim** (NEU-Seg): 缺少权重，未评估
- **BiSeNetV2-L** (NEU-Seg): 缺少权重，未评估
- **Sub-region UNet** (NEU-Seg): 评估失败: Caught NameError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/opt/conda/lib/python3.11/site-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/workspace/Industrial Surface Defect/Industrial-Surface-Defect/tools/evaluate_class_iou.py", line 522, in __getitem__
    mask = Image.open(ann_path).convert('L')
           ^^^^^
NameError: name 'Image' is not defined

- **PIDNet-S** (NEU-Seg): 缺少权重，未评估
