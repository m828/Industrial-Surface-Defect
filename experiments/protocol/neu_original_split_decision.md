# NEU-Seg Original Split Decision

> Date: 2026-06-16  
> Scope: A2MS-DefectNet-B NEU-Seg retraining under the historical/original NEU-Seg split.

## Decision

The user confirmed that this project will continue using the original NEU-Seg train/test split used by the historical experiments and common comparisons in this direction.

No additional independent validation split will be created.

## Locked Split

| Item | Value |
|---|---|
| Train list | `/workspace/Industrial Surface Defect/new/dataset/train_neu.txt` |
| Test/monitor list | `/workspace/Industrial Surface Defect/new/dataset/test_neu.txt` |
| Train samples | 3630 |
| Test/monitor samples | 840 |
| Dataset root | `/workspace/Industrial Surface Defect/new/dataset` |
| Actual dataset root | symlink to `/workspace/Industrial Surface Defect/new (copy)/dataset` |
| Number of classes | 4 |
| Class order | `0=background`, `1=crazing`, `2=inclusion`, `3=patches` |
| Input size | 200 x 200 |
| mIoU includes background | Yes |
| Evaluation preprocessing | `TestRescale(input_hw=(200, 200)) + ToTensor()` |
| ImageNet Normalize | Not used |

## Training-Time Monitor Rule

`test_neu.txt` may be used during training as the validation/monitor list to follow the historical project protocol.

This should be described in later experiment notes and paper-facing materials as:

```text
按照既有 NEU-Seg 实验划分和统一评估协议进行。
```

It must not be described as an additional independent train/validation/test three-way split.

## Paper-Writing Constraint

Do not claim that an extra independent validation split exists. The correct wording is that the experiment follows the existing NEU-Seg split and uses `test_neu.txt` as the historical validation/monitor and unified evaluation list.
