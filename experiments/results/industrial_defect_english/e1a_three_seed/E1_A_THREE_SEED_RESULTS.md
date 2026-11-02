# E1-A Three-Seed Results

## 范围与证据边界

- E1-A 新增 seed 2027、3407，各运行 formal_a2ms、capacity_control、coupled_transport；与原 E0 seed 1337 合并为三个训练 seed。
- 统一使用冻结的 2903 张 development train、726 张 development validation 和 1 张 D4 quarantine；split seed=1337，条件阈值不变。
- 输入 200×200、batch 16、Adam（lr=1e-4、weight_decay=2e-6）、mask-aware loss、isolated_v1 RNG，固定 10000 次真实 optimizer 更新后评价。
- 未使用官方 840 测试集评价或选模；未按验证 mIoU 或 beta/T 提前停止；未修改模型、损失、数据或实验超参数。
- 标准差为三个训练 seed 的样本标准差（ddof=1）。结果仅是探索性开发集描述，不构成统计显著性或最终 SCI 论文结论。

## 1. 六组完成状态与九个固定终点

E1-A 六组均为 `COMPLETE`，均完成 10000/10000 次更新；原 E0 三组也均为完整固定终点。四类 IoU 顺序为 class 0/1/2/3，时间为训练及验证总墙钟时间。

| seed | 模型 | 状态/更新数 | 含背景 mIoU | 前景 mIoU | IoU 0 / 1 / 2 / 3 | 时间（分钟） | 峰值显存（MiB） |
|---|---|---|---:|---:|---|---:|---:|
| 1337 | formal_a2ms | COMPLETE / 10000 | 0.778214574 | 0.717847645 | 0.959315419 / 0.590825915 / 0.828884423 / 0.733832717 | 49.318 | 2168.452 |
| 1337 | capacity_control | COMPLETE / 10000 | 0.774126589 | 0.712398946 | 0.959309638 / 0.572793245 / 0.834158003 / 0.730245531 | 52.414 | 2222.681 |
| 1337 | coupled_transport | COMPLETE / 10000 | 0.790079355 | 0.732615888 | 0.962469637 / 0.620012641 / 0.838300943 / 0.739534080 | 52.452 | 2271.681 |
| 2027 | formal_a2ms | COMPLETE / 10000 | 0.786069155 | 0.727359712 | 0.962197542 / 0.609698951 / 0.837158024 / 0.735222042 | 52.123 | 2168.452 |
| 2027 | capacity_control | COMPLETE / 10000 | 0.789185286 | 0.731336772 | 0.962730885 / 0.613631070 / 0.841941357 / 0.738437891 | 52.103 | 2222.681 |
| 2027 | coupled_transport | COMPLETE / 10000 | 0.785510600 | 0.726469100 | 0.962635219 / 0.607009530 / 0.839771807 / 0.732625902 | 52.162 | 2271.681 |
| 3407 | formal_a2ms | COMPLETE / 10000 | 0.779954076 | 0.719338715 | 0.961800158 / 0.580153823 / 0.836997807 / 0.740864456 | 51.914 | 2168.452 |
| 3407 | capacity_control | COMPLETE / 10000 | 0.779079795 | 0.718437016 | 0.961007953 / 0.588307977 / 0.833249629 / 0.733753562 | 52.186 | 2222.681 |
| 3407 | coupled_transport | COMPLETE / 10000 | 0.787991822 | 0.729938269 | 0.962152421 / 0.614636183 / 0.840717256 / 0.734461308 | 52.508 | 2271.681 |

## 2. 三 seed 统一性能

数值为均值 ± 样本标准差。

| 模型 | 含背景 mIoU | 前景 mIoU |
|---|---:|---:|
| formal_a2ms | 0.781412601 ± 0.004125419 | 0.721515357 ± 0.005115973 |
| capacity_control | 0.780797223 ± 0.007674846 | 0.720724245 ± 0.009673876 |
| coupled_transport | 0.787860592 ± 0.002287203 | 0.729674419 ± 0.003081877 |

| 模型 | IoU 0 | IoU 1 | IoU 2 | IoU 3 |
|---|---:|---:|---:|---:|
| formal_a2ms | 0.961104373 ± 0.001561969 | 0.593559563 ± 0.014961059 | 0.834346751 ± 0.004731194 | 0.736639738 ± 0.003724075 |
| capacity_control | 0.961016158 ± 0.001710638 | 0.591577431 ± 0.020614290 | 0.836449663 ± 0.004777584 | 0.734145661 ± 0.004110231 |
| coupled_transport | 0.962419093 ± 0.000245335 | 0.613886118 ± 0.006533925 | 0.839596669 ± 0.001217640 | 0.735540430 ± 0.003578283 |

## 3. 相同验证样本上的配对差异

差值均为 coupled_transport 减对照模型。总体指标使用相同的 726 张验证图像；这不是统计显著性检验。

| 对照 | seed | Δ含背景 mIoU | Δ前景 mIoU | ΔIoU 0 / 1 / 2 / 3 |
|---|---:|---:|---:|---|
| formal_a2ms | 1337 | +0.011864781 | +0.014768243 | +0.003154218 / +0.029186726 / +0.009416521 / +0.005701363 |
| formal_a2ms | 2027 | -0.000558555 | -0.000890613 | +0.000437677 / -0.002689421 / +0.002613783 / -0.002596140 |
| formal_a2ms | 3407 | +0.008037746 | +0.010599554 | +0.000352263 / +0.034482360 / +0.003719449 / -0.006403148 |
| capacity_control | 1337 | +0.015952766 | +0.020216942 | +0.003160000 / +0.047219396 / +0.004142940 / +0.009288549 |
| capacity_control | 2027 | -0.003674686 | -0.004867673 | -0.000095665 / -0.006621540 / -0.002169549 / -0.005811989 |
| capacity_control | 3407 | +0.008912027 | +0.011501253 | +0.001144469 / +0.026328206 / +0.007467628 / +0.000707746 |

| 对照 | 平均 Δ含背景 mIoU | 平均 Δ前景 mIoU | 平均 ΔIoU 0 / 1 / 2 / 3 | mIoU 正/负 seed 数 |
|---|---:|---:|---|---:|
| formal_a2ms | +0.006447991 ± 0.006362414 | +0.008159061 ± 0.008109682 | +0.001314720 / +0.020326555 / +0.005249918 / -0.001099308 | 2 / 1 |
| capacity_control | +0.007063369 ± 0.009943459 | +0.008950174 ± 0.012735402 | +0.001402934 / +0.022308687 / +0.003147006 / +0.001394769 | 2 / 1 |

平均增益为正，但 seed 2027 对两个对照均为负，不能称为跨 seed 一致获益。class 1 的平均改善最大；class 3 相对 formal_a2ms 平均下降，且在 seed 2027、3407 均为负，属于需要保留的重要类别退化信号。

## 4. 前景误检、漏检与跨类别误分

定义为：FP=GT background 被预测为前景；FN=GT foreground 被预测为 background；跨类误分=GT foreground 被预测为另一前景类。表中为三 seed 像素数均值 ± 样本标准差。

| 模型/配对差 | 前景 FP 像素 | 前景 FN 像素 | 前景跨类误分像素 |
|---|---:|---:|---:|
| formal_a2ms | 600209.333 ± 91302.198 | 401195.333 ± 55935.769 | 19538.000 ± 2716.136 |
| capacity_control | 590773.000 ± 51445.978 | 413423.333 ± 11843.074 | 16876.000 ± 1211.698 |
| coupled_transport | 558208.000 ± 50230.930 | 409712.333 ± 47315.861 | 16450.000 ± 2196.043 |
| coupled − formal 平均差 | -42001.333 | +8517.000 | -3088.000 |
| coupled − capacity 平均差 | -32565.000 | -3711.000 | -426.000 |

相对 formal，coupled 的跨类误分在三个 seed 均减少，但 FP 在 seed 3407 增加 53119 像素，FN 在 seed 1337、2027 分别增加 38265、51939 像素。相对 capacity，FP 在 seed 3407 增加 7503 像素，FN 在 seed 2027 增加 49493 像素。总体 mIoU 改善不能替代这些错误方向。

## 5. 小面积与高复杂边界指标

数值为 GT 阳性病例等权 case-macro IoU 的三 seed 均值；分组阈值完全来自冻结开发训练集。Δ 列分别为 coupled 减 formal、coupled 减 capacity。

| 分组 | 每 seed n | formal | capacity | coupled | Δ formal | Δ capacity |
|---|---:|---:|---:|---:|---:|---:|
| class1_area_small | 144 | 0.493383849 | 0.488452941 | 0.505437583 | +0.012053734 | +0.016984642 |
| class2_area_small | 234 | 0.812281607 | 0.815077025 | 0.818812528 | +0.006530922 | +0.003735503 |
| class3_area_small | 156 | 0.615780633 | 0.613334980 | 0.623015969 | +0.007235336 | +0.009680989 |
| class1_complexity_large | 233 | 0.566320741 | 0.554650755 | 0.576444263 | +0.010123522 | +0.021793508 |
| class2_complexity_large | 60 | 0.807599945 | 0.811202667 | 0.814280553 | +0.006680608 | +0.003077886 |
| class3_complexity_large | 160 | 0.641691008 | 0.639687611 | 0.640904971 | -0.000786037 | +0.001217359 |

小面积组的平均方向均为正。高复杂边界组中 class 1、2 的平均方向为正，但 class 3 相对 formal 略降；这些仍是小样本、探索性分组结果，不能表述为所有复杂度条件一致改善。

## 6. beta/T 跨 seed 响应

响应为固定终点全部验证 batch 的均值。`rho_difference=T RMS/(semantic-detail RMS)`；formal_a2ms 没有 transport 分支，不适用。

| seed | 模型 | beta mean | beta spatial std | T RMS | rho_difference |
|---|---|---:|---:|---:|---:|
| 1337 | capacity_control | 0.137450462 | 0.000000000 | 2.376628643 | 0.137450464 |
| 1337 | coupled_transport | -0.000010194 | 0.000325933 | 0.004516571 | 0.000256559 |
| 2027 | capacity_control | 0.133563049 | 0.000000000 | 2.571497503 | 0.133563052 |
| 2027 | coupled_transport | -0.001510068 | 0.000473881 | 0.029577842 | 0.001566214 |
| 3407 | capacity_control | 0.123891175 | 0.000000000 | 2.412918501 | 0.123891177 |
| 3407 | coupled_transport | -0.000971282 | 0.000404294 | 0.016345601 | 0.000849606 |

coupled_transport 的 T、beta 及空间变化在三个 seed 均非零，说明分支具有实际数值响应；但 rho_difference 仅约 0.000257、0.001566、0.000850，响应幅度随 seed 变化。capacity_control 的全局 beta 更大但没有空间变化。上述结果不能单独证明 transport 的预期因果机制。

## 7. 参数量与计算成本

- formal_a2ms 参数量 29,460,804；capacity_control 和 coupled_transport 均为 29,461,109。
- E1-A 六组在一张 A100-PCIE-40GB 串行运行，总墙钟时间 18779.832 秒，即 5.2166 GPU 小时；低于批准的 8 GPU 小时。
- 每组最长 52.508 分钟，低于 90 分钟；每组输出目录约 689–691 MiB，低于 3 GB。
- E1 两 seed 的 formal/capacity/coupled 分别耗时 104.038、104.290、104.670 分钟。
- 显存为 `torch.cuda.max_memory_allocated()`，不含全部 CUDA 上下文开销；成本来自实际训练日志，不是由推理 FPS 推算。

## 8. 是否建议进入 E1-B

**建议有条件进入 E1-B 机制消融，需另行授权。** 依据是三 seed 平均 mIoU、前景 mIoU 以及多数小面积/高复杂边界分组方向为正，且 coupled 相对等容量对照的平均差也为正。

同时必须把 seed 2027 的反向结果、class 3 相对 formal 的下降、FN/FP 的混合方向以及较小且可变的 transport 响应列为核心问题。E1-B 应用于检验机制与退化来源，不应被表述为确认稳定增益或统计显著性；不建议仅增加 seed 来挑选有利结论。

## 9. 目录与 SHA-256

原始运行保留在：

- `/workspace/e0_neu_development_seed1337_GGzIEeTB/runs/e1a_s2027_05f7bc853e/`
- `/workspace/e0_neu_development_seed1337_GGzIEeTB/runs/e1a_s3407_05f7bc853e/`

轻量结果与报告位于 `/workspace/e1_a_execution/`。checkpoint、原图、GT 和大型训练日志不进入 GitHub 结果包。

共同身份：

```text
E1 source commit       e946787b58f4584268fe66478b2f4d2ccad8d344
E1 source snapshot     b192806e1d999f2a5c888ad064e85c943df325c30def808524f09f2251c1ee14
E0 source commit       135d7ccfba6f112b60598dad38f5da19ac1bbc99
seed2027 config        8d82dcc11128db588e3687f4cac106a03f2aa0b99994e3e2da3cdf76b8e78599
seed3407 config        8c8a0553281f0821fe63ae5c7547eef60ad8f4e1dfb873842ec13c29c9cc4e6e
seed2027 executed cfg  926f7a1267b53dfe02bdf10a1ba812cdfd367cef41d22fa7d9e7731aaf56683a
seed3407 executed cfg  bb00298339f1b7490b49c127e093f18558000ffc9d2ceaecfc381a9535331ade
seed2027 common init   325410dc3ee69064fa736f1c0baaf20bd85d788e677da240d631a99c476c3cae
seed3407 common init   7138f1ff5d7fac0bb1d8e774da944ab38ae81020107639efcee6858e0a989eb9
dev train              d02f651e2808579d59a179e43c32e37b40ba7a72db6535864a504900dc4746dd
dev validation         436cd120bf524a985f5546aa3f5db5d208de06a31e4d2e418493bcbda1889298
quarantine             3e68bd4a7f20f38111bdf2efd7b4c1444852db59f28dd81b9685b0a79003b4cd
split audit            6733f512a682295a39c8157708e9b713d79b49642d491f8d6df540b10f93ef69
conditions             df89e01e904e4fe028b0bf775d9a846cb2ce78ef5b6c300aceeddb355d2726ca
summary JSON           bdea2d7cc03e51f5600a6eb3de020394414b1495cf03bdf9325d6fc6c40a2cb4
```

模型与损失文件 SHA：

```text
models/factory.py     30b4e8e1e984acc0442327f050bf7f385ebad27bbaffbc942007229096b123f1
models/transport.py   69b5789edd27db8f44693282ec8446767a34131ca37cd772fc7641da4f57a0d4
models/losses.py      9858dd9dfa7cfad05f8688473f557f131c88148f499baff1c8e3b5170bfb27ee
models/metrics.py     784b648bde9bd1d7d08d78692e28aacd09289dfe456f9e4384f912f2d5d54502
vendor/model_resnet.py 9a0b255a84894423a9849e65add26e30081e7c020b6f136a3a2e0572ae87690a
vendor/model_dsmo_rs50.py fab316c57518e8273ba676199ccff16500ad60ae89a77c2843585f3ce0c184cf
vendor/model_dsmo_rs50_eSE_adapt_detailloss_fixed.py a0e6fd52b00162ebd16805c473a59983cbd34c2308da115cd170ba1710bb376a
vendor/loss.py        64786a456b502d93e44f451879dfd70bf2a1695868c50cee68a29be22beadf69
```

E1 六组结果身份（`package` 为轻量包内 `artifact_checksums.sha256` 的 SHA；所有组的 updates journal SHA 均为 `e142c87898dc30b4579d724d5cd35fb4babef0e99db25d03bc7f6b1ef529d4ec`）：

```text
2027 formal_a2ms
checkpoint 3fd07e808928e93c018ab9dd4f251cf913db97dca08a33130cdc08240e7f13bd
metrics    0818fd3f1d76e2fe893ace568ef8ad8c2fade7bcba93fb7f735effd49d82ae01
package    b353c6a17aa065d00a6dbbd22e5b6b34aa013cc1d9a4374fea25d0b8396af0ee
2027 capacity_control
checkpoint ee22b6ce4bab0af35a96812ebefaf2ef79f16b27998cffab56e631942d69a578
metrics    88098205bb33cbc3c3cd2e18f7755ef3bc71718a9035ee2647ef77f93f977342
package    10a4a4bf8115561764063a4905807662d4c1e9c9e9acd81290ff29b12df44e66
2027 coupled_transport
checkpoint d1f533e4980c8df7c8662903bacd109cd87ea79e5b4220d4706d15fe4e9a16cb
metrics    fee978f897a13fabc1c65de51d334b916a34346c1d234765aaf38f24576d813c
package    9d5343a767f5b26efe24e95d52f52a0a4ef6364cb42438d2aed04b8132581f72
3407 formal_a2ms
checkpoint 37a8bf28a0d7fc0925c8cb64e95edf2b899a0ff2b52101b437ac1dfe19dd70f6
metrics    0d02ac0fabe3d006559aeb66c7559d2270b6ef9143fff049bf18a493ef7a6335
package    8970a3be47d70221956a37c7defefe7f1728b5ca16c9ee34aafac3f6d50a548f
3407 capacity_control
checkpoint d5ff93dcbd9bff16b0ba669831074d112b5984854b95d99de22dc0b502a464c2
metrics    821bd77bdf5c53d349dc1e9dcee548eeffc45a5cc3c02b610b806a9c49ec480a
package    5e26d3d4d8e04b1cf103e0eaea9dd376e4da4d8a2f2ac23cd9a9dd69150993cc
3407 coupled_transport
checkpoint 1846fa344bde599f6b4ae0f31efc03188a93db6081b47f031338fd5a9d04e7db
metrics    31f1f1b38a8b4967a69fbbc5ea6d3a0c6c4a04438f061078b9253da9f6082e7c
package    a55d85f78461d2677e19b0ac8df5e6e2433f2a30a1638945537742b9a6dc3850
```
