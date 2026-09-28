# Historical NEU-Seg mIoU 0.913–0.916 Evidence Forensics

> Date: 2026-09-08
> Scope: read-only forensics over `/workspace/Industrial Surface Defect/` (`new/`, `new (copy)/`, `subregion unet/`, `Industrial-Surface-Defect/`). No file was modified, moved, or deleted.
> Confidence tags: **[FACT]** = directly read from file/log; **[INFER]** = high-probability inference with stated basis; **[UNKNOWN]** = cannot confirm.

---

## 1. All 0.91x records found (NEU-Seg)

### 1.1 Detail-loss chain ("9130" family) — `subregion unet/runs/dsmonet_resnet_detailloss/all_img/`

| Log file | Resume checkpoint (as logged) | Start iter | # evals | mIoU range observed | Time |
|---|---|---:|---:|---|---|
| `run_2023_08_21_10_50_22.log` | `dsmonet_resnet50_neu_128.pkl` (not found → cold start, aborted) | 0 | 0 | — | 2023-08-21 10:50 |
| `run_2023_08_21_10_54_21.log` | none (from scratch) | 0 | 60 | 0.7100 @1k → ~0.90 @60k | 2023-08-21 10:54 |
| `run_2023_08_21_13_58_36.log` | `dsmonet_resnet50_neu_128_detailloss_9028.pkl` | 60000 | 60 | ~0.90x | 2023-08-21 13:58 |
| `run_2023_08_21_16_03_56.log` | `dsmonet_resnet50_neu_128_detailloss_9118.pkl` | 112000 | 68 | ~0.91x | 2023-08-21 16:03 |
| `run_2023_08_21_19_08_17.log` | `dsmonet_resnet50_neu_128_detailloss_9130.pkl` | 177000 | 63 | 0.9090–0.9143 (e.g. line 43 `0.909020885694825` @178k; line 217 `0.9139095249660958`; line 1783 `0.9143070191792786`) | 2023-08-21 19:08 |

**[FACT]** The from-scratch trajectory in `run_2023_08_21_10_54_21.log`:

| iter | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 | 9000 | **10000** | 15000 | 20000 | 30000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mIoU | 0.7100 | 0.7741 | 0.7951 | 0.8202 | 0.8283 | 0.8383 | 0.8412 | 0.8482 | 0.8499 | **0.8572** | 0.8675 | 0.8737 | 0.8822 |

So the historical detail-loss model reached only **≈0.857 at 10k**; 0.913+ required 177k–240k iterations.

### 1.2 Non-detail-loss chain ("9145" family) — `subregion unet/runs/dsmonet_resnet/all_img/`

| Log file | Resume checkpoint | Start iter | # evals | mIoU highlights | Time |
|---|---|---:|---:|---|---|
| `run_2023_08_11_14_49_33.log` | `dsmonet_resnet50_neu_128_9034.pkl` | 52000 | 68 | ~0.90x | 2023-08-11 14:49 |
| `run_2023_08_22_11_05_46.log` | `dsmonet_resnet50_neu_128_180000_9145.pkl` | 164000 | 76 | 0.9023 @165k; `0.9134767925471272` (line 71); `0.9157583951443463` (line 796); **`0.9159367187818162`** (line 2130, max found) | 2023-08-22 11:05 |
| `run_2023_08_22_10_58_29.log`, `11_00_48.log`, `11_04_59.log` | same 9145 ckpt | — | 0 | immediate restarts | 2023-08-22 |

An even earlier link exists: `runs/dsmonet_resnet/all_imgs/dsmonet_resnet.yml` snapshot has `train_iters: 60000, resume: model_savePath/dsmonet__0.8832.pkl` → the non-detail chain was also chained from a 0.8832 checkpoint. **[FACT]**

### 1.3 Checkpoint files

**[FACT]** None of the high-score checkpoints exist on this server (searched all of `/workspace/Industrial Surface Defect/`):

- `dsmonet_resnet50_neu_128_detailloss_9028.pkl` / `_9118.pkl` / `_9130.pkl`
- `dsmonet_resnet50_neu_128_9034.pkl` / `dsmonet_resnet50_neu_128_180000_9145.pkl`
- `dsmonet_resnet822_9128_180000.pkl` (referenced by `subregion unet/test_neu_resnet_dsmo822.py`)

Checkpoint names encode the best monitor mIoU at rename time (`9028`→0.9028, `9118`→0.9118, `9130`→0.9130, `9145`→0.9145, `9128`→0.9128). **[INFER]** — basis: each resume log line is immediately followed by evals in that range.

---

## 2. Model entry / training script / config used by the 0.91x runs

**[FACT]** from `subregion unet/train_neu_resnet_detailloss.py:31`:

```python
from model_dsmo_rs50 import DSMONet     # subregion unet/model_dsmo_rs50.py
...
model = DSMONet(num_classes=4, backbone=resnet50())   # model_resnet.py (custom), pretrained=False
```

**[FACT]** Config snapshot saved by the 9130 run itself (`runs/dsmonet_resnet_detailloss/all_img/dsmonet_resnet_detailloss.yml`):

- `train_iters: 240000`, `batch_size: 16`, `val_interval: 1000`, `n_workers: 16`
- optimizer: Adam, lr 1e-4, weight_decay 2e-6
- `lr_schedule:` **empty** → log line "Using No LR Scheduling" (ConstantLR) **[FACT]**
- loss list: `[ohem_cross_entropy, bce_with_logits_loss, ohem_cross_entropy, detail_aggregate_loss]`
- The 9145 chain used the same config minus `detail_aggregate_loss` (3 losses) **[FACT]**

**[FACT]** Historical model == current source: `subregion unet/__pycache__/model_dsmo_rs50.cpython-39.pyc` (2023-era compiled bytecode) was unmarshalled with python3.10 and compared function-by-function (co_names / co_consts / co_varnames) against the current `subregion unet/model_dsmo_rs50.py`: **all function-level fingerprints identical** (only module-level constant-pool ordering differs). The historical model therefore has exactly: DAPPM, SqueezeBodyEdge, SELayer(128, reduction=128), Laplacian, UAFM×2 (arm1 ch+sp / arm2 sp-only), 3× ConvTranspose2d ×2, one shared SegHead, **3 training outputs, no `seg_head_detailloss`**.

**[FACT]** Same pyc-vs-source audit proves the following zeroed-out files in `subregion unet/` were functionally identical to their `new/` counterparts, which are therefore safe stand-ins for reproduction:

| zeroed file in `subregion unet/` | stand-in used | verification |
|---|---|---|
| `tools/computemIou.py` (0 B) | `new/tools/computemIou.py` | pyc fingerprint IDENTICAL |
| `tools/utils.py` (0 B) | `new/tools/utils.py` | IDENTICAL |
| `tools/loss/loss.py` (0 B) | `new/tools/loss/loss.py` | identical except: `bce_with_logits_loss` has extra dead-code line `weight.view(1,-1,1,1)` (never triggered, weight=None at call sites); `ohem_cross_entropy` param renamed `weight`→`class_weights` (positional calls unaffected) |
| `model_resnet.py` (0 B) | `new/model_resnet.py` | all function fingerprints identical |
| `model_unet.py` (0 B) | `new/model_unet.py` | IDENTICAL |
| `stdcnet.py` (0 B) | `new/stdcnet.py` | IDENTICAL |
| `utils.py` (missing) | `new/utils.py` | IDENTICAL |
| `tools/schedulers/{__init__,schedulers}.py`, `tools/loader/__init__.py`, 3 loader modules (0 B) | `new/` counterparts | needed only for imports; loader classes never instantiated (DataGenerator is used) |

**[FACT]** The 9130/9145 runs' mIoU values are **monitor-set** numbers computed inside the training loop: val loader = `DataGenerator('./dataset/test_neu.txt')`, `TestRescale(200,200)+ToTensor`, no Normalize, `runningScore(4)` (includes background), pred = `outputs[0].max(1)`.

**[FACT]** Eval coverage per monitor pass: valloader `batch_size=16, drop_last=True, shuffle=True` → 52 batches × 16 = **832 of 840 images** per eval (last 8 dropped).

**[FACT]** The final-test script `subregion unet/test_neu_dsmonet2.py` evaluates the same `dataset/test_neu.txt` with the same transforms, `batch_size=16` **without** `drop_last` → all **840** images, loading `dsmonet_resnet50_neu_128_180000_9145.pkl`. So historical "test" and "monitor" are the same 840-image NEU-Seg test split.

**[FACT]** Current data lists: `new (copy)/dataset/train_neu.txt` = 3630 lines, `test_neu.txt` = 840 lines (sha256 recorded in the reproduction manifest). Annotation dirs contain exactly 3630 / 840 files.

**[INFER]** The historical 0.91x evals used the same 840-image test list: basis — the only NEU lists present anywhere in the tree are these; the txts reference `NEUSeg/annotations/{training,test}`; counts 3630/840 match the current locked protocol; the test script reads the same relative path. Strictly, the 2023 copy of the txt cannot be hash-verified (the `subregion unet/dataset/` directory no longer exists), so list *identity* is **[INFER]**, not **[FACT]**.

## 3. Answering the mandated checklist

1. Each 0.913–0.916 record: see tables §1.1/§1.2 (file paths, line numbers, values).
2. Full paths: `subregion unet/runs/dsmonet_resnet_detailloss/all_img/run_2023_08_21_19_08_17.log` (max 0.91431) and `subregion unet/runs/dsmonet_resnet/all_img/run_2023_08_22_11_05_46.log` (max 0.91594).
3. Times: 2023-08-21 19:08→21:07 and 2023-08-22 11:05→13:21.
4. Checkpoints: `..._detailloss_9130.pkl` (resumed @177k), `..._180000_9145.pkl` (resumed @164k) — both **missing**.
5. Model entry: `subregion unet/model_dsmo_rs50.py::DSMONet` (both chains).
6. Training script: `subregion unet/train_neu_resnet_detailloss.py` (9130 chain); the 9145 chain used the 3-loss sibling script (same file family; exact script file **[INFER]** — `train_neu_resnet_detailloss.py` with the 3-loss `dsmonet_resnet.yml`, or a zeroed `train_neu_resnet.py`; both import the same model).
7. Config: §2 (240k iters, batch 16, Adam 1e-4, wd 2e-6, no LR schedule, val every 1000).
8. Iterations: 0.913–0.916 observed at **165k–240k** iterations, after chained resumes (0→60k→112k→177k→240k for detail-loss; …→52k→…→164k→240k for non-detail).
9. Evaluated set: monitor = NEU-Seg **test split (840, drop_last→832 per pass)** during training; final test script covers all 840. NOT a separate validation split; NOT the train set.
10. Background included: yes — 4-class runningScore; per-class lines `0: ~0.985, 1: ~0.84, 2: ~0.93, 3: ~0.88` in the logs.
11. Same as current 840 eval: same split and same metric definition; only differences are historical drop_last=True (832) vs current unified eval (840) and batch-16-vs-1 eval batching. **[INFER]** on txt identity, see §2.

## 4. Negative findings

- `~/.bash_history` contains only `train_neu_pid.py --config pid_neu.yml` entries — no historical NEU training commands recoverable. **[FACT]**
- No 0.91x NEU record exists anywhere in `new/`, `new (copy)/`, or `Industrial-Surface-Defect/` logs/reports outside the two `subregion unet/runs/` chains above (pige/yachi/xd hits are different datasets with 5–8 classes). **[FACT]**
- Checkpoint `best_iou` fields cannot be inspected (checkpoints missing); per instructions, no result was inferred from checkpoint metadata.
