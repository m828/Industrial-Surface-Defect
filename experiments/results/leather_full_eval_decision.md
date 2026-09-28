# Leather Full Evaluation Decision

> Date: 2026-06-16  
> Scope: decision after Leather/Pige same-protocol full evaluation.  
> Restrictions followed: no training, no paper edits, no weight overwrite, no update to `fixed_existing_results.md`.

## 1. Can Leather/Pige enter the paper main table now?

**Partially, after manual approval.** The evaluation chain is now traceable for Base-S, Base-B, A2MS-S, A2MS-B, STDC-Seg, DDRNet23slim, and PIDNet-S under the same locked 468-sample test protocol.

However, the current A2MS-B Leather checkpoint underperforms Base-B, so the Leather table cannot be used to claim A2MS-B superiority.

## 2. Should Base-B 0127 be the Leather base-version main result?

**Yes, as the current verified candidate.** Base-B 0127 has mIoU=0.899120, the strongest currently verified Leather/Pige result.

## 3. Should A2MS-B Leather enter the main paper table?

**Not as a positive main-model result.** A2MS-B mIoU=0.860749, lower than Base-B by 0.038371.

It can be recorded in internal experiment tables and failure analysis. It should not support abstract/conclusion claims.

## 4. If A2MS-B does not enter as a main result, how can it be described?

Only as one of the following, after author approval:

- failed checkpoint under the locked Leather/Pige test protocol;
- discussion/ablation caution about checkpoint selection and validation/test mismatch;
- motivation for retraining A2MS-B Leather.

## 5. Is immediate A2MS-B Leather retraining recommended?

**Yes, if Leather is intended as a main dataset for A2MS-DefectNet-B.** The current checkpoint is not competitive with Base-B.

## 6. Can A2MS-B NEU retraining start?

**Preparation can start, but training should still wait for explicit command/path approval.** The NEU A2MS-B historical 91.3 checkpoint remains missing, so retraining is likely necessary. Before launch, confirm output directories and non-overwrite naming.

## 7. Current Same-Protocol Result Snapshot

| Rank | Model | mIoU | Background | Open wound | Scratch | Brand mark | Hole | Skin disease | Rotten surface | Wart |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Base-B | 0.899120 | 0.993250 | 0.764801 | 0.700504 | 0.882528 | 0.990708 | 0.971181 | 0.915895 | 0.974096 |
| 2 | STDC-Seg (STDCNet1446) | 0.882401 | 0.992544 | 0.746916 | 0.629960 | 0.884325 | 0.987338 | 0.952547 | 0.895466 | 0.970113 |
| 3 | A2MS-DefectNet-S | 0.868997 | 0.991825 | 0.709699 | 0.608361 | 0.875100 | 0.988379 | 0.949903 | 0.860228 | 0.968483 |
| 4 | PIDNet-S | 0.862009 | 0.989440 | 0.709984 | 0.597094 | 0.864323 | 0.985162 | 0.943552 | 0.881693 | 0.924821 |
| 5 | A2MS-DefectNet-B | 0.860749 | 0.990210 | 0.732640 | 0.544550 | 0.884139 | 0.988418 | 0.965973 | 0.852424 | 0.927637 |
| 6 | Base-S | 0.858613 | 0.990719 | 0.695297 | 0.591564 | 0.855523 | 0.986517 | 0.942885 | 0.847433 | 0.958964 |
| 7 | DDRNet23slim | 0.842052 | 0.989058 | 0.732579 | 0.395457 | 0.852755 | 0.982948 | 0.939142 | 0.889241 | 0.955232 |

## 8. Failed / Not Evaluated Models

| Model | Decision | Reason |
|---|---|---|
| PP-LiteSeg-B | Not evaluated | No matching Leather checkpoint found; code exists but no traceable weight. |
| BiSeNetV1-L | Not evaluated | No matching Leather checkpoint found; code identity remains ambiguous. |
