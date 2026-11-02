# E0 Training Results

## 范围与结论边界

- NEU 探索性 D4-only 开发实验；原始采集 parent 仍为 UNKNOWN。
- 使用冻结的 2903 张 development train 和 726 张 development validation；未使用官方 840 测试集评价或选模。
- seed=1337，200×200，batch=16，Adam（lr=1e-4，weight_decay=2e-6），固定 10000 次真实 optimizer 更新。
- 三组公共初始化 SHA 一致，配置 SHA 一致，未按验证 mIoU 或 beta/T 提前停止。
- 以下为单种子、固定开发集上的描述性结果，不代表统计显著性或最终 SCI 论文结论。

## 1. 固定终点与统一性能对比

三组均为 `COMPLETE`，完成 10000/10000 次更新。指标来自各自第 10000 次更新后的同一组 726 张开发验证图像。

| 模型 | 更新数 | 最终训练 loss | 含背景 mIoU | 前景 mIoU | IoU 0 | IoU 1 | IoU 2 | IoU 3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| formal_a2ms | 10000 | 11.916481 | 0.778215 | 0.717848 | 0.959315 | 0.590826 | 0.828884 | 0.733833 |
| capacity_control | 10000 | 11.971568 | 0.774127 | 0.712399 | 0.959310 | 0.572793 | 0.834158 | 0.730246 |
| coupled_transport | 10000 | 11.205827 | 0.790079 | 0.732616 | 0.962470 | 0.620013 | 0.838301 | 0.739534 |

完整损失分量保留在各运行的 `metrics.jsonl`，本表 loss 为第 10000 个训练 batch 的总损失。

## 2. 相同样本上的配对差异

差值均为前行模型减后行模型；总体指标基于相同验证集合，类别条件指标使用相同 `(sample_id, class_id)` 配对。

| 比较 | Δ含背景 mIoU | Δ前景 mIoU | ΔIoU 0 | ΔIoU 1 | ΔIoU 2 | ΔIoU 3 |
|---|---:|---:|---:|---:|---:|---:|
| capacity_control − formal_a2ms | -0.004088 | -0.005449 | -0.000006 | -0.018033 | +0.005274 | -0.003587 |
| coupled_transport − formal_a2ms | +0.011865 | +0.014768 | +0.003154 | +0.029187 | +0.009417 | +0.005701 |
| coupled_transport − capacity_control | +0.015953 | +0.020217 | +0.003160 | +0.047219 | +0.004143 | +0.009289 |

在本轮单种子终点上，coupled_transport 相对两个对照的四类总体 IoU 均未下降；相对 formal_a2ms 的含背景 mIoU 增加约 1.186 个百分点，相对 capacity_control 增加约 1.595 个百分点。

## 3. 小面积与高复杂边界表现

数值为 GT 阳性病例的等权 case-macro IoU。两个 Δ 列分别为 coupled_transport 减 formal_a2ms、coupled_transport 减 capacity_control 的同病例平均配对差。

### 小面积组（area_small）

| 类别 | n | formal_a2ms | capacity_control | coupled_transport | Δ formal | Δ capacity |
|---|---:|---:|---:|---:|---:|---:|
| class 1 | 144 | 0.482236 | 0.464520 | 0.497282 | +0.015046 | +0.032762 |
| class 2 | 234 | 0.804062 | 0.807812 | 0.817865 | +0.013803 | +0.010053 |
| class 3 | 156 | 0.609977 | 0.600618 | 0.625320 | +0.015344 | +0.024702 |

### 高复杂边界组（complexity_large）

| 类别 | n | formal_a2ms | capacity_control | coupled_transport | Δ formal | Δ capacity |
|---|---:|---:|---:|---:|---:|---:|
| class 1 | 233 | 0.570006 | 0.541327 | 0.587460 | +0.017454 | +0.046134 |
| class 2 | 60 | 0.801286 | 0.808450 | 0.814336 | +0.013050 | +0.005886 |
| class 3 | 160 | 0.642581 | 0.635800 | 0.644394 | +0.001813 | +0.008594 |

并非所有子组都改善：class 1 的 complexity_small 组相对 formal_a2ms 从 0.523297 降至 0.517083，配对均差为 -0.006214（n=66）。因此不能把本轮结果表述为所有复杂度条件一致获益。

## 4. 像素混淆矩阵

行是 GT 类别，列是预测类别，单元格为像素数。

### formal_a2ms

```text
GT\Pred        0        1        2        3
0       24641602   208904   308730   186858
1         173737   575943     1932     6878
2          83782     5512  1968622     1754
3          83040     1904     4694   786108
```

### capacity_control

```text
GT\Pred        0        1        2        3
0       24699434   196556   271648   178456
1         201135   548414     3117     5824
2         105683      928  1951149     1910
3          94179     1464     4629   775474
```

### coupled_transport

```text
GT\Pred        0        1        2        3
0       24759453   176033   246056   164552
1         167444   581415     3647     5984
2         116145     2193  1939784     1548
3          95235     1031     4574   774906
```

## 5. 前景误检、漏检与跨类别误分

- 前景误检：GT background 被预测为任一前景类。
- 前景漏检：GT foreground 被预测为 background。
- 跨类别误分：GT foreground 被预测为另一前景类。

| 模型 | 前景误检像素（率） | 前景漏检像素（率） | 跨类别误分像素（率） |
|---|---:|---:|---:|
| formal_a2ms | 704492（2.7795%） | 340559（9.2195%） | 22674（0.6138%） |
| capacity_control | 646660（2.5513%） | 400997（10.8556%） | 17872（0.4838%） |
| coupled_transport | 586641（2.3145%） | 378824（10.2554%） | 18977（0.5137%） |

相对 formal_a2ms，coupled_transport 的前景误检减少 117851 像素、跨类别误分减少 3697 像素，但前景漏检增加 38265 像素；其中 class 2 和 class 3 的总 FN 分别增加 28838 和 11202 像素。相对 capacity_control，coupled_transport 的前景漏检减少 22173 像素，但跨类别误分增加 1105 像素。总体 IoU 改善不能替代对这些误差方向的复核。

## 6. beta/T 响应

最终验证点的响应如下；formal_a2ms 没有 transport 分支。

| 模型 | beta mean | beta spatial std | abs(beta) q95 | T RMS | T/difference RMS | 最后权重范数 |
|---|---:|---:|---:|---:|---:|---:|
| capacity_control | 0.137450462 | 0 | 0.137450465 | 2.376628643 | 0.137450464 | 0.172088876 |
| coupled_transport | -0.000010194 | 0.000325933 | 0.000702097 | 0.004516571 | 0.000256559 | 0.023152165 |

coupled_transport 的响应轨迹：

| 更新数 | beta mean | beta spatial std | T RMS | T/difference RMS | 权重范数 |
|---:|---:|---:|---:|---:|---:|
| 1000 | 0.000537935 | 0.000331559 | 0.011487441 | 0.000458729 | 0.021664435 |
| 5000 | 0.000059387 | 0.000241407 | 0.004156485 | 0.000201979 | 0.011543185 |
| 10000 | -0.000010194 | 0.000325933 | 0.004516571 | 0.000256559 | 0.023152165 |

T 和 beta 并非恒为零，且 coupled_transport 保留了非零空间变化，说明该分支在 forward 中有实际数值作用。但最终 T/difference RMS 仅约 0.000257（0.0257%），平均 beta 接近零；因此不能仅凭性能增益断言已经证明预期的 transport 因果机制。capacity_control 的全局 beta 更大但总体 mIoU 更低，也说明响应幅度本身不是性能解释。

## 7. 参数量与资源消耗

三组在一张 A100-PCIE-40GB 上串行运行。显存为 `torch.cuda.max_memory_allocated()`，时间为训练及验证的总墙钟时间。

| 模型 | 参数量 | 时间（分钟） | 峰值显存（MiB） | 输出大小（MiB） | 状态 |
|---|---:|---:|---:|---:|---|
| formal_a2ms | 29460804 | 49.318 | 2168.5 | 688.2 | COMPLETE |
| capacity_control | 29461109 | 52.414 | 2222.7 | 690.4 | COMPLETE |
| coupled_transport | 29461109 | 52.452 | 2271.7 | 690.6 | COMPLETE |

总墙钟时间为 9251.064 秒，即 2.5697 GPU 小时；每组均低于 5 小时，总计低于 15 GPU 小时，每组输出低于 3GB。

## 8. 是否进入 E1

**建议有条件进入 E1。** 理由是 coupled_transport 在同一固定终点、相同开发验证样本上相对 formal_a2ms 和等容量 capacity_control 均取得正向总体差异，小面积和高复杂边界组也呈现正向描述性结果。

E1 应优先检验这些方向能否在独立重复中保持，并同时复核 class 1 低复杂度子组下降、class 2/3 漏检变化及较小的 transport 响应。当前结果不应被写成统计显著、跨种子稳定或最终 SCI 结论。

## 9. 身份与 SHA-256

```text
source commit       135d7ccfba6f112b60598dad38f5da19ac1bbc99
source snapshot     adbfee9a28ca674aff64d6899bace22108c4a13724dcb6c109600034a511d219
runtime config      ff8281a9d9c6ccaa327ad23c4a4d4d62ca67b6f4095fadc362ceb2b27d226536
dev train           d02f651e2808579d59a179e43c32e37b40ba7a72db6535864a504900dc4746dd
dev validation      436cd120bf524a985f5546aa3f5db5d208de06a31e4d2e418493bcbda1889298
quarantine          3e68bd4a7f20f38111bdf2efd7b4c1444852db59f28dd81b9685b0a79003b4cd
split audit         6733f512a682295a39c8157708e9b713d79b49642d491f8d6df540b10f93ef69
conditions          df89e01e904e4fe028b0bf775d9a846cb2ce78ef5b6c300aceeddb355d2726ca
common init tensors 2324d732909968e340e62ec7529a492671ad30c258d98f0d3b200b8aa748ed7a
updates journal     e142c87898dc30b4579d724d5cd35fb4babef0e99db25d03bc7f6b1ef529d4ec
```

```text
formal_a2ms checkpoint   4b1571a2f5930a0bd5f66d6f3fb58118cf6c8c51d0d553be11fd378b7dae6563
formal_a2ms metrics      ac3aa8d46ad41bda046d258e36ca011462f70b336a7dc797e0ceea486508ef51
capacity_control checkpoint 72a672313b10905570e192df91c40dd320a178399e66880c82cae81f09221e46
capacity_control metrics    73ced3b5261106dc37455d2aa2f1ab442d9904b524b254c9025e7d655d446de7
coupled_transport checkpoint 1cfcb53e57298c8b9852ee3f795840c202ef1625755fc2f3dfba63c91e0abc2d
coupled_transport metrics    56f56790438069a0fcf126509f798cba12adf91727f7ac305e47cbc787413610
```

配对比较位于 `comparisons/`，三组轻量导出包位于 `exports/`。导出包已由现有工具校验，包含小型指标、逐例结果、响应、运行摘要和 SHA；不包含原始图像、GT 或 checkpoint。当前仅作为 GitHub 回传候选，尚未提交或上传。
