# Leather/Pige Protocol Final Decision

> Date: 2026-06-15  
> Decision type: Experiment protocol decision, not a paper-text update.

## Final Decision

Leather/Pige unified evaluation will use **468 test samples**.

The authoritative test list for subsequent experiments is:

```text
experiments/protocol/leather_valid_test_list.txt
```

## Audit Result

| Check Item | Result |
|---|---|
| `new (copy)/dataset/pige/test.txt` total rows | 468 |
| Non-empty rows | 468 |
| Empty rows | 0 |
| Duplicate rows | 0 |
| Malformed rows | 0 |
| Missing images | 0 |
| Missing labels | 0 |
| `new/dataset/pige/test.txt` matches `new (copy)/dataset/pige/test.txt` | Yes |
| Final effective test samples | 468 |

## Locked Evaluation Settings

| Item | Value |
|---|---|
| Dataset | Leather/Pige |
| Test split | `experiments/protocol/leather_valid_test_list.txt` |
| Input size | 768 x 768 |
| Classes | 8 |
| Background in mIoU | Included |
| Preprocessing | TestRescale + ToTensor |
| ImageNet Normalize | Not used |
| Metric script | `tools/evaluate_class_iou.py` |

## Practical Consequences

- Previous references to Leather/Pige test=467 should be treated as obsolete for experiment planning.
- Existing historical values remain provenance records only unless re-evaluated under this 468-sample protocol.
- Val-set `best_iou` may be used only for checkpoint selection, not as a paper test result.
- This document does not update the manuscript and does not authorize adding any result to the final paper table.

