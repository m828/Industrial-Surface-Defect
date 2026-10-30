# E0 服务器接收及运行说明（待作者批准）

状态：本地交付包仅为待接收快照；未发送、未访问服务器、未运行P3/GPU。当前包**不含四个权属未闭合的 vendor Python**，须从服务器已合法持有的版本装配并核对字节哈希；不能擅自公开上传源码或数据。D4-only探索补充需作者确认，不能宣称采集源独立。

## 1. 从GitHub接收

在现有项目仓库中获取 `research/e0-coupled-transport`，由作者提供并确认完整40位 commit；使用干净的同仓库工作树并核对HEAD，不覆盖未提交修改。本分支不含4份vendor源文件。根据仓库 `research/industrial_defect_english/vendor/provenance.json` 的4个SHA，从服务器已合法持有的版本复制到该目录，逐项用sha256校验；缺少/不一致即停止。不要提交或公开vendor原件、数据或权重。

源码门禁仍要求E0源文件、provenance和全部其他文件属于所确认的干净Git版本；只允许该4个未追踪vendor文件按版本化清单做SHA校验，不允许符号链接、哈希不匹配或额外未追踪文件。源代码快照SHA、Git commit、共同配置SHA和共享协议SHA是不同标识。

在实际Python环境运行 `python -c 'from research.industrial_defect_english.models import build_model; print("IMPORT_OK")'`；核查torch/numpy/Pillow/CUDA，不自动安装或升级。已有 `requirements-cpu.txt` 是历史本机参考，不是服务器安装指令。

## 2. 完整数据扫描和D4隔离

作者确认本补充和实际P3操作后，提供真实绝对路径，不猜测或改变历史划分。例：

```bash
"$PYTHON" -m research.industrial_defect_english.scripts.freeze_neu_development \
 --train-list "$OFFICIAL_TRAIN_LIST" --test-list "$OFFICIAL_TEST_LIST" \
 --data-root "$NEU_ROOT" --output "$NEW_EXTERNAL_SPLIT_DIR" --seed 1337 \
 --quarantine-policy-id neu_exact_d4_quarantine_candidate_v1 \
 --exploratory-unknown-parent-authorized
```

必须先确认3630/840真实文件齐全；此命令会读取两组图像和GT，输出新目录，禁止覆盖。查 `split_audit.json`、`d4_matches.json`、`quarantine.jsonl`、`dev_train.jsonl`、`dev_validation.jsonl`：确认 `FROZEN_EXPLORATORY_D4_ONLY`、`UNKNOWN`、总数闭合、D4组隔离、两组ID互斥及各类覆盖；保留各文件SHA。不得重复更换seed/挑划分。原840只用于身份与隔离，不用于模型选择。

随后仅从开发训练manifest产生冻结面积/复杂度阈值：

```bash
"$PYTHON" -m research.industrial_defect_english.scripts.build_conditions \
 --train-manifest "$NEW_EXTERNAL_SPLIT_DIR/dev_train.jsonl" \
 --manifest-sha "$DEV_TRAIN_SHA256" --output "$NEW_EXTERNAL_CONDITIONS_JSON"
```

核对输出目标不存在、阈值文件及开发三manifest、split audit、shared协议的真实文件SHA。修改一份**新的共同运行配置**，将这些路径/SHA填入并记录字节SHA；正式三组使用同一配置字节，variant与run_id由CLI区分。实际source snapshot、包SHA、获批Git commit、配置SHA、split SHA与shared_protocol SHA分别记录，不互相替代。

## 3. GPU smoke：单次合成数据更新

仅作者批准该GPU smoke及资源后，在验证过的source环境运行；独立于正式运行目录，1次更新只验证设备和梯度，不产生正式E0性能或可复用checkpoint：

```bash
"$PYTHON" - <<'PY'
import time, torch
from research.industrial_defect_english.models import build_model
from research.industrial_defect_english.models.losses import joint_loss_with_protocol
assert torch.cuda.is_available()
model=build_model('coupled_transport',4,1337).cuda().train()
optim=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=2e-6)
torch.manual_seed(1337)
x=torch.randn(2,3,200,200,device='cuda'); y=torch.randint(0,4,(2,200,200),device='cuda')
torch.cuda.synchronize(); started=time.perf_counter()
optim.zero_grad(set_to_none=True)
total,_=joint_loss_with_protocol(model(x),y,[10,1,3,3],'neu_a2ms_mask_aware_candidate_v1')
if not torch.isfinite(total): raise SystemExit('SMOKE_INCOMPLETE_NONFINITE_LOSS')
total.backward()
if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()): raise SystemExit('SMOKE_INCOMPLETE_NONFINITE_GRADIENT')
optim.step(); torch.cuda.synchronize()
print({'status':'SMOKE_ONLY','optimizer_updates':1,'seconds':time.perf_counter()-started,'gpu':torch.cuda.get_device_name()})
PY
```

保留环境、输出、耗时和实际资源占用；1步结果不足以准确估算10000步开销，须结合实际环境向作者重新申请预算。不要通过缩短正式训练、使用测试集或改变超参数补偿超限。

## 4. 正式三组：单独授权后执行

作者确认包、D4-only补充、实际数据身份和预算并单独授权正式训练后，依次运行formal_a2ms、coupled_transport、capacity_control。配置中的 `p3_data_freeze_approved`、`exploratory_unknown_parent_authorized`、`scientific_gate`、预算批准字段须反映真实授权；此前始终保持阻塞。给每个variant生成自己的preflight报告和真实SHA：

```bash
"$PYTHON" -m research.industrial_defect_english.scripts.server_preflight \
 --expected-commit "$APPROVED_SOURCE_COMMIT" --config "$COMMON_RUNTIME_CONFIG" \
 --variant "$VARIANT" --run-id "$BATCH_ID:$VARIANT:s1337" --output "$NEW_PREFLIGHT_REPORT"
"$PYTHON" -m research.industrial_defect_english.scripts.train_e0_ready \
 --config "$COMMON_RUNTIME_CONFIG" --variant "$VARIANT" --source-commit "$APPROVED_SOURCE_COMMIT" \
 --run-id "$BATCH_ID:$VARIANT:s1337" --preflight-report "$NEW_PREFLIGHT_REPORT" \
 --preflight-sha256 "$PREFLIGHT_SHA256" --authorize-gpu
```

只在preflight `GO_READY` 且作者已批准GPU后运行第二条；不能因示意命令存在即当作授权。每组10000真实更新、seed1337、batch16、Adam 1e-4/2e-6、从零、完整mask-aware和隔离RNG；不使用验证性能/beta早停。数值或预算异常停且保留INCOMPLETE，不重选权。

各run保存源commit、代码快照、共享科学协议SHA、执行配置SHA、分组SHA、实际更新数、预检和run状态、环境、耗时及日志；使用 `export_e0_artifacts.py --help` 与 `compare_e0_endpoints.py --help` 对照当前实际CLI，导出含/不含背景mIoU、各类IoU、像素混淆矩阵、FP/FN/跨类错误、逐例及条件结果、T/beta（基础模型标注不适用）；未完成或不匹配协议不得作为端点。无需将原始数据、权重、预测或日志提交GitHub。
