# STDC Naming Check

> Date: 2026-06-15  
> Scope: identify the actual STDC model entry for the Leather checkpoint before any paper/table naming.  
> Restrictions followed: no training, no paper edits, no weight overwrite, no full STDC evaluation in this check.

## 1. Checked Weight

```text
/workspace/Industrial Surface Defect/new (copy)/model_savePath/sdtdcnet_pige_stdc2_pige.pkl
```

The filename contains `stdc2`, but filename text alone is not a reliable architecture identifier.

## 2. Historical Script Evidence

The historical Leather STDC test script uses:

```python
model = BiSeNet('STDCNet1446', 8).to(device)
checkpoint = torch.load("./model_savePath/sdtdcnet_pige_stdc2_pige.pkl")
model.load_state_dict(checkpoint["model_state"])
```

Source:

```text
/workspace/Industrial Surface Defect/new (copy)/test_pige_fcn_unet.py
```

The separate historical training script `train_pige_stdc.py` contains:

```python
model = DSMONet(num_classes=n_classes, backbone=STDCNet1446()).to(device)
```

This means the directory contains more than one STDC-related experiment path. For the checked checkpoint, the direct historical evaluation path is the `BiSeNet('STDCNet1446', 8)` path.

## 3. Model Definition Evidence

`/workspace/Industrial Surface Defect/new (copy)/stdc.py` defines `BiSeNet` with selectable backbones:

- `STDCNet1446`
- `STDCNet813`

`/workspace/Industrial Surface Defect/new (copy)/stdcnet.py` defines both backbone classes.

## 4. Load/Forward Check

The checkpoint is an old-format PyTorch checkpoint. With PyTorch 2.6 default loading, `torch.load(..., weights_only=True)` fails, while `torch.load(..., weights_only=False)` succeeds for this known local checkpoint.

Observed checkpoint keys:

```text
['best_iou', 'epoch', 'model_state', 'optimizer_state', 'scheduler_state']
```

Load/forward results:

| Candidate entry | Strict load | Dummy forward |
|---|---|---|
| `BiSeNet('STDCNet1446', 8)` | Success | Success, output shapes `[(1, 8, 768, 768), (1, 8, 768, 768), (1, 8, 768, 768), (1, 1, 96, 96)]` |
| `BiSeNet('STDCNet813', 8)` | Failed | Not run |
| `DSMONet(num_classes=8, backbone=STDCNet1446())` | Not matched to this checkpoint path | Not used for this checkpoint identity decision |

## 5. Naming Decision

Current safest name for experiments:

```text
STDC-Seg (STDCNet1446)
```

Do not force this result to `STDC2-Seg` based only on the checkpoint filename. The actual checked model entry is `BiSeNet('STDCNet1446', 8)`.

If the paper table needs the canonical STDC1/STDC2 label, use one of these only after manual confirmation of the STDC naming convention in the cited baseline implementation:

- `STDC1-Seg (STDCNet1446; checkpoint filename contains stdc2)`
- `STDC-Seg (STDCNet1446)`

Until then, the recommended table label is **STDC-Seg (STDCNet1446)**.

