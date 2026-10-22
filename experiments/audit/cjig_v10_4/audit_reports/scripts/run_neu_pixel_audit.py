#!/usr/bin/env python3
"""CJIG v10.4 read-only pixel-level verification for cached NEU predictions.

This script reads existing 200x200 PNG masks only. It does not load model weights,
run inference, train, modify source data, or write inside the project repository.
"""
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import PIL
from PIL import Image

PROJECT = Path("/workspace/Industrial Surface Defect")
REPO = PROJECT / "Industrial-Surface-Defect"
MECH = PROJECT / "repro_runs/mechanism_eval_240k"
LONG_BASE_CODE = PROJECT / "repro_runs/long240k_seed1337/base_b/code"
OUT = Path("/workspace/cjig_v10_4_readonly_audit")
OUT.mkdir(parents=True, exist_ok=True)

TEST_LIST = LONG_BASE_CODE / "dataset/test_neu.txt"
DATASET_ROOT = LONG_BASE_CODE / "dataset"
GT_DIR = MECH / "gt"
BASE_DIR = MECH / "base_predictions"
A2_DIR = MECH / "a2_predictions"
CACHE_MANIFEST = MECH / "cache_manifest.json"
ORIGINAL_CASE_CSV = MECH / "paired_case_metrics.csv"
ORIGINAL_SCALE_JSON = MECH / "scale_stratified_results.json"
ORIGINAL_BOUNDARY_JSON = MECH / "boundary_quality_results.json"
ORIGINAL_PAIRED_JSON = MECH / "paired_case_analysis.json"
ORIGINAL_MORPH_JSON = MECH / "morphology_exploratory_results.json"
CASE_SELECTION_JSON = MECH / "case_selection.json"
ASSEMBLY_SCRIPT = PROJECT / "repro_runs/leather_historical_audit/assemble_fig34_v6.py"

CLASSES = {1: "crazing", 2: "inclusion", 3: "patches"}
TOLS = (1, 2, 3)
K3 = np.ones((3, 3), np.uint8)
EXPECTED_MIOU = {"base": 0.9124083744680408, "a2": 0.9145442177671137}
BOOT_SEED = 20261018
N_BOOT = 5000
AREA_BINS = {
    "crazing": (0.027324, 0.06185775),
    "inclusion": (0.1064235, 0.18496475),
    "patches": (0.037379, 0.08505),
}
SPECIFIC_IDS = {
    "crazing": ["000510", "000529", "000597", "000524", "000526"],
    "inclusion": ["000511", "000509"],
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path)).astype(np.uint8)


def contour(mask: np.ndarray) -> np.ndarray:
    return (mask - cv2.erode(mask, K3, iterations=1)).clip(0, 1)


def band(mask: np.ndarray, tol: int) -> np.ndarray:
    return (cv2.dilate(mask, K3, iterations=tol) - cv2.erode(mask, K3, iterations=tol)).clip(0, 1)


def boundary_iou(pred: np.ndarray, gt: np.ndarray, tol: int):
    pb, gb = band(pred, tol), band(gt, tol)
    union = int(np.logical_or(pb, gb).sum())
    if union == 0:
        return None
    return float(np.logical_and(pb, gb).sum() / union)


def bf1_score(pred: np.ndarray, gt: np.ndarray, tol: int):
    pc, gc = contour(pred), contour(gt)
    n_pc, n_gc = int(pc.sum()), int(gc.sum())
    if n_pc == 0 and n_gc == 0:
        return None, None, None
    if n_pc == 0 or n_gc == 0:
        return 0.0, 0.0, 0.0
    dt_g = cv2.distanceTransform((1 - gc).astype(np.uint8), cv2.DIST_C, 3)
    dt_p = cv2.distanceTransform((1 - pc).astype(np.uint8), cv2.DIST_C, 3)
    precision = float((dt_g[pc.astype(bool)] <= tol).mean())
    recall = float((dt_p[gc.astype(bool)] <= tol).mean())
    f1 = 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))
    return precision, recall, f1


def safe_ratio(num, den):
    return float(num / den) if den else None


def region_values(pred: np.ndarray, gt: np.ndarray):
    p = pred.astype(bool)
    g = gt.astype(bool)
    tp = int(np.logical_and(p, g).sum())
    pred_area = int(p.sum())
    gt_area = int(g.sum())
    fp = pred_area - tp
    fn = gt_area - tp
    union = tp + fp + fn
    iou = safe_ratio(tp, union)
    dice = safe_ratio(2 * tp, pred_area + gt_area)
    return tp, fp, fn, gt_area, pred_area, union, iou, dice


def component_count(mask: np.ndarray) -> int:
    if int(mask.sum()) == 0:
        return 0
    n, _labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return int(n - 1)


def fmt(x):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "NA"
    return f"{x:.12g}"


def mean_or_nan(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan


def df_block(df: pd.DataFrame) -> str:
    return "```csv\n" + df.to_csv(index=False).strip() + "\n```"


def size_group(class_name: str, ratio: float) -> str:
    p33, p67 = AREA_BINS[class_name]
    if ratio <= p33:
        return "Small"
    if ratio <= p67:
        return "Medium"
    return "Large"


def parse_test_list():
    lines = [line.strip().split() for line in TEST_LIST.read_text().splitlines() if line.strip()]
    ids = [Path(parts[0]).stem for parts in lines]
    ann_paths = [parts[0] for parts in lines]
    return ids, ann_paths


def synthetic_empty_rule_tests():
    g = np.zeros((20, 20), np.uint8)
    p = np.zeros((20, 20), np.uint8)
    g[5:10, 5:10] = 1
    p[6:11, 5:10] = 1
    empty = np.zeros_like(g)
    full = np.ones_like(g)
    edge = np.zeros_like(g)
    edge[0, 0] = 1
    cases = {
        "gt_nonempty_pred_nonempty": (p, g),
        "gt_nonempty_pred_empty": (empty, g),
        "gt_empty_pred_nonempty": (p, empty),
        "gt_empty_pred_empty": (empty, empty),
        "edge_single_pixel_both": (edge, edge),
        "full_mask_both": (full, full),
    }
    out = {}
    for name, (pred, gt) in cases.items():
        out[name] = {}
        for tol in TOLS:
            bprec, brec, bf1 = bf1_score(pred, gt, tol)
            out[name][f"boundary_iou_{tol}px"] = boundary_iou(pred, gt, tol)
            out[name][f"bf1_{tol}px"] = bf1
            out[name][f"bprec_{tol}px"] = bprec
            out[name][f"brec_{tol}px"] = brec
    return out


def main():
    run_started = datetime.now(timezone.utc).isoformat()
    manifest = json.loads(CACHE_MANIFEST.read_text())
    test_ids, ann_rels = parse_test_list()
    manifest_ids = manifest["sample_ids"]
    dir_sets = {
        "gt": sorted(p.stem for p in GT_DIR.glob("*.png")),
        "base": sorted(p.stem for p in BASE_DIR.glob("*.png")),
        "a2": sorted(p.stem for p in A2_DIR.glob("*.png")),
    }
    identity = {
        "test_list_path": str(TEST_LIST),
        "test_list_sha256": sha256_file(TEST_LIST),
        "n_test_list": len(test_ids),
        "manifest_n": len(manifest_ids),
        "manifest_order_equals_test_list_order": manifest_ids == test_ids,
        "manifest_set_equals_test_list_set": sorted(manifest_ids) == sorted(test_ids),
        "directory_counts": {k: len(v) for k, v in dir_sets.items()},
        "directory_set_equals_test_list": {k: v == sorted(test_ids) for k, v in dir_sets.items()},
        "duplicate_ids": {
            "test_list": sorted([k for k, v in Counter(test_ids).items() if v > 1]),
            "manifest": sorted([k for k, v in Counter(manifest_ids).items() if v > 1]),
        },
        "missing_ids_by_directory": {k: sorted(set(test_ids) - set(v)) for k, v in dir_sets.items()},
        "extra_ids_by_directory": {k: sorted(set(v) - set(test_ids)) for k, v in dir_sets.items()},
        "shape_checks": {},
        "label_values": {},
        "cached_gt_vs_source": {"checked": 0, "pixel_equal": 0, "mismatch_ids": []},
        "cache_script_outputs_main_output": "cache_predictions.py line 69 uses model(img)[0].data.max(1)[1]",
        "postprocessing_observed_in_cache_script": "none; direct argmax followed by PNG uint8 save",
    }

    hists = {"base": np.zeros((4, 4), dtype=np.int64), "a2": np.zeros((4, 4), dtype=np.int64)}
    case_rows = []
    per_image_rows = []
    all_shapes = {k: Counter() for k in ("gt", "base", "a2")}
    all_values = {k: Counter() for k in ("gt", "base", "a2")}
    source_mismatch = []

    for sid, ann_rel in zip(test_ids, ann_rels):
        gt = read_mask(GT_DIR / f"{sid}.png")
        pred_b = read_mask(BASE_DIR / f"{sid}.png")
        pred_a = read_mask(A2_DIR / f"{sid}.png")
        arrays = {"gt": gt, "base": pred_b, "a2": pred_a}
        for tag, arr in arrays.items():
            all_shapes[tag][tuple(arr.shape)] += 1
            all_values[tag].update(int(v) for v in np.unique(arr))
        if gt.shape != (200, 200) or pred_b.shape != (200, 200) or pred_a.shape != (200, 200):
            identity.setdefault("bad_shape_ids", []).append(sid)
        source_gt = read_mask(DATASET_ROOT / ann_rel)
        if source_gt.shape == gt.shape and np.array_equal(source_gt, gt):
            identity["cached_gt_vs_source"]["pixel_equal"] += 1
        else:
            source_mismatch.append(sid)
        identity["cached_gt_vs_source"]["checked"] += 1

        valid = (gt >= 0) & (gt < 4)
        for tag, pred in (("base", pred_b), ("a2", pred_a)):
            hists[tag] += np.bincount(4 * gt[valid].astype(np.int64) + pred[valid], minlength=16).reshape(4, 4)

        img_rec = {"sample_id": sid}
        for tag, pred in (("base", pred_b), ("a2", pred_a)):
            ious = []
            for cid in range(4):
                _tp, _fp, _fn, _ga, _pa, union, iou, _dice = region_values(pred == cid, gt == cid)
                if union > 0:
                    ious.append(iou)
            img_rec[f"{tag}_miou_4class_present_classes"] = float(np.mean(ious))
        img_rec["delta_miou_4class_present_classes"] = img_rec["a2_miou_4class_present_classes"] - img_rec["base_miou_4class_present_classes"]
        per_image_rows.append(img_rec)

        for cid, cname in CLASSES.items():
            g = gt == cid
            pb = pred_b == cid
            pa = pred_a == cid
            rec = {
                "sample_id": sid,
                "class": cname,
                "class_id": cid,
                "gt_area": int(g.sum()),
                "gt_area_ratio": float(g.sum() / 40000.0),
                "gt_positive": bool(g.sum() > 0),
                "gt_component_count": component_count(g.astype(np.uint8)),
            }
            if g.sum() > 0:
                rec["crazing_complexity"] = float(contour(g.astype(np.uint8)).sum() / np.sqrt(g.sum())) if cname == "crazing" else np.nan
            else:
                rec["crazing_complexity"] = np.nan
            for tag, pred in (("base", pb), ("a2", pa)):
                tp, fp, fn, ga, pa_area, union, iou, dice = region_values(pred, g)
                rec[f"{tag}_tp"] = tp
                rec[f"{tag}_fp"] = fp
                rec[f"{tag}_fn"] = fn
                rec[f"{tag}_pred_area"] = pa_area
                rec[f"{tag}_union"] = union
                rec[f"{tag}_iou"] = iou if iou is not None else np.nan
                rec[f"{tag}_dice"] = dice if dice is not None else np.nan
                for tol in TOLS:
                    bi = boundary_iou(pred.astype(np.uint8), g.astype(np.uint8), tol)
                    bprec, brec, bf1 = bf1_score(pred.astype(np.uint8), g.astype(np.uint8), tol)
                    rec[f"{tag}_boundary_iou_{tol}px"] = bi if bi is not None else np.nan
                    rec[f"{tag}_bf1_{tol}px"] = bf1 if bf1 is not None else np.nan
                    rec[f"{tag}_bprec_{tol}px"] = bprec if bprec is not None else np.nan
                    rec[f"{tag}_brec_{tol}px"] = brec if brec is not None else np.nan
            for metric in ("iou", "dice"):
                rec[f"delta_{metric}"] = rec[f"a2_{metric}"] - rec[f"base_{metric}"]
            for tol in TOLS:
                for metric in ("boundary_iou", "bf1"):
                    rec[f"delta_{metric}_{tol}px"] = rec[f"a2_{metric}_{tol}px"] - rec[f"base_{metric}_{tol}px"]
            case_rows.append(rec)

    identity["cached_gt_vs_source"]["mismatch_ids"] = source_mismatch
    identity["shape_checks"] = {k: {";".join(map(str, shape)): n for shape, n in v.items()} for k, v in all_shapes.items()}
    identity["label_values"] = {k: sorted(v) for k, v in all_values.items()}
    identity["all_shapes_200x200"] = all(set(v.keys()) == {(200, 200)} for v in all_shapes.values())
    identity["all_labels_within_0_to_3"] = all(set(v.keys()).issubset({0, 1, 2, 3}) for v in all_values.values())

    confusion_summary = {}
    metric_rows = []
    for tag, hist in hists.items():
        tp = np.diag(hist).astype(float)
        gt_tot = hist.sum(axis=1).astype(float)
        pred_tot = hist.sum(axis=0).astype(float)
        union = gt_tot + pred_tot - tp
        iou = tp / union
        dice = 2 * tp / (gt_tot + pred_tot)
        precision = tp / pred_tot
        recall = tp / gt_tot
        miou = float(np.mean(iou))
        pixel_acc = float(tp.sum() / hist.sum())
        confusion_summary[tag] = {
            "confusion_matrix_rows_gt_columns_pred": hist.tolist(),
            "miou": miou,
            "pixel_accuracy": pixel_acc,
            "expected_miou": EXPECTED_MIOU[tag],
            "absolute_miou_error": abs(miou - EXPECTED_MIOU[tag]),
        }
        for cid, cname in [(0, "background"), *CLASSES.items()]:
            metric_rows.append({
                "model": tag,
                "class_id": cid,
                "class": cname,
                "tp": int(tp[cid]),
                "fp": int(pred_tot[cid] - tp[cid]),
                "fn": int(gt_tot[cid] - tp[cid]),
                "gt_area": int(gt_tot[cid]),
                "pred_area": int(pred_tot[cid]),
                "union": int(union[cid]),
                "iou": float(iou[cid]),
                "dice": float(dice[cid]),
                "precision": float(precision[cid]),
                "recall": float(recall[cid]),
                "miou": miou,
                "pixel_accuracy": pixel_acc,
            })
    identity["miou_recalculation"] = confusion_summary

    case_df = pd.DataFrame(case_rows)
    per_image_df = pd.DataFrame(per_image_rows)
    case_df.to_csv(OUT / "recomputed_case_metrics.csv", index=False)
    per_image_df.to_csv(OUT / "recomputed_per_image_miou.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(OUT / "neu_pixel_metric_reproduction.csv", index=False)
    (OUT / "neu_confusion_matrices.json").write_text(json.dumps(confusion_summary, indent=2), encoding="utf-8")
    (OUT / "neu_identity_checks.json").write_text(json.dumps(identity, indent=2, ensure_ascii=False), encoding="utf-8")

    # Compare independent recomputation with the original case CSV.
    orig = pd.read_csv(ORIGINAL_CASE_CSV)
    orig["sample_id"] = orig["sample_id"].astype(str).str.zfill(6)
    case_df["sample_id"] = case_df["sample_id"].astype(str)
    compare = {"row_count_original": int(len(orig)), "row_count_recomputed": int(len(case_df)), "columns": {}}
    key = ["sample_id", "class"]
    merged = orig.merge(case_df, on=key, suffixes=("_orig", "_new"), validate="one_to_one")
    compare["key_join_rows"] = int(len(merged))
    for col in [c for c in orig.columns if c not in key]:
        new_col = col if col in case_df.columns else None
        if new_col is None:
            compare["columns"][col] = {"status": "missing_in_recomputed"}
            continue
        a = pd.to_numeric(merged[f"{col}_orig"], errors="coerce")
        b = pd.to_numeric(merged[f"{col}_new"], errors="coerce")
        both_na = (a.isna() & b.isna()).sum()
        diff = (a - b).abs()
        compare["columns"][col] = {
            "both_na": int(both_na),
            "max_abs_diff": float(diff.max(skipna=True)) if diff.notna().any() else 0.0,
            "n_diff_gt_1e-10": int((diff > 1e-10).sum()),
        }
    (OUT / "paired_case_reproduction_diff.json").write_text(json.dumps(compare, indent=2), encoding="utf-8")

    # Empty-rule and named-sample audit.
    synthetic = synthetic_empty_rule_tests()
    (OUT / "boundary_empty_rule_tests.json").write_text(json.dumps(synthetic, indent=2), encoding="utf-8")
    named = {}
    for cname, ids in SPECIFIC_IDS.items():
        sub = case_df[(case_df["class"] == cname) & (case_df["sample_id"].isin(ids))]
        named[cname] = {}
        for sid in ids:
            row = sub[sub["sample_id"] == sid]
            if row.empty:
                named[cname][sid] = {"status": "missing"}
                continue
            r = row.iloc[0]
            rec = {
                "gt_area": int(r["gt_area"]),
                "base_pred_area": int(r["base_pred_area"]),
                "a2_pred_area": int(r["a2_pred_area"]),
                "base_valid_boundary2": bool(pd.notna(r["base_boundary_iou_2px"])),
                "a2_valid_boundary2": bool(pd.notna(r["a2_boundary_iou_2px"])),
                "base_boundary_iou_2px": None if pd.isna(r["base_boundary_iou_2px"]) else float(r["base_boundary_iou_2px"]),
                "a2_boundary_iou_2px": None if pd.isna(r["a2_boundary_iou_2px"]) else float(r["a2_boundary_iou_2px"]),
                "base_iou": None if pd.isna(r["base_iou"]) else float(r["base_iou"]),
                "a2_iou": None if pd.isna(r["a2_iou"]) else float(r["a2_iou"]),
            }
            named[cname][sid] = rec
    (OUT / "boundary_named_sample_audit.json").write_text(json.dumps(named, indent=2), encoding="utf-8")

    # A/B/C estimands.
    metrics = ["iou", "dice"] + [f"boundary_iou_{t}px" for t in TOLS] + [f"bf1_{t}px" for t in TOLS]
    estimand_rows = []
    estimand_sets = {}
    for cname in CLASSES.values():
        sub = case_df[case_df["class"] == cname].copy()
        for metric in metrics:
            bcol, acol = f"base_{metric}", f"a2_{metric}"
            b = sub[bcol].to_numpy(float)
            a = sub[acol].to_numpy(float)
            gt_pos = sub["gt_positive"].to_numpy(bool)
            vb, va = np.isfinite(b), np.isfinite(a)
            common = vb & va
            gt_common = common & gt_pos
            sets = {
                "base_valid_ids": sub.loc[vb, "sample_id"].tolist(),
                "a2_valid_ids": sub.loc[va, "sample_id"].tolist(),
                "common_valid_ids": sub.loc[common, "sample_id"].tolist(),
                "gt_positive_common_ids": sub.loc[gt_common, "sample_id"].tolist(),
                "base_only_valid_ids": sub.loc[vb & ~va, "sample_id"].tolist(),
                "a2_only_valid_ids": sub.loc[va & ~vb, "sample_id"].tolist(),
            }
            estimand_sets[f"{cname}:{metric}"] = sets
            point = {
                "A": float(np.nanmean(a) - np.nanmean(b)),
                "B": float(np.nanmean(a[common] - b[common])),
                "C": float(np.nanmean(a[gt_common] - b[gt_common])),
            }
            abs_vals = {
                "A": (float(np.nanmean(b)), float(np.nanmean(a))),
                "B": (float(np.nanmean(b[common])), float(np.nanmean(a[common]))),
                "C": (float(np.nanmean(b[gt_common])), float(np.nanmean(a[gt_common]))),
            }
            ns = {"A_base": int(vb.sum()), "A_a2": int(va.sum()), "B": int(common.sum()), "C": int(gt_common.sum())}
            for est in ("A", "B", "C"):
                estimand_rows.append({
                    "class": cname,
                    "metric": metric,
                    "estimand": est,
                    "base_abs": abs_vals[est][0],
                    "a2_abs": abs_vals[est][1],
                    "delta": point[est],
                    "n": ns[est] if est != "A" else f"{ns['A_base']}/{ns['A_a2']}",
                    "n_base_valid_A": ns["A_base"],
                    "n_a2_valid_A": ns["A_a2"],
                    "n_common_B": ns["B"],
                    "n_gt_positive_common_C": ns["C"],
                    "base_only_ids": ";".join(sets["base_only_valid_ids"]),
                    "a2_only_ids": ";".join(sets["a2_only_valid_ids"]),
                })
    estimand_df = pd.DataFrame(estimand_rows)
    estimand_df.to_csv(OUT / "estimand_abc_pixel_verification.csv", index=False)
    (OUT / "estimand_abc_sample_sets.json").write_text(json.dumps(estimand_sets, indent=2), encoding="utf-8")

    # Common-image bootstrap with the requested v10.3 settings.
    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = rng.integers(0, len(test_ids), size=(N_BOOT, len(test_ids)))
    boot_idx_hash = sha256_bytes(boot_idx.tobytes())
    boot_rows = []
    id_to_pos = {sid: i for i, sid in enumerate(test_ids)}
    for cname in CLASSES.values():
        sub_all = case_df[case_df["class"] == cname].copy()
        sub_all["pos"] = sub_all["sample_id"].map(id_to_pos)
        sub_all = sub_all.sort_values("pos")
        if sub_all["sample_id"].tolist() != test_ids:
            raise RuntimeError(f"case rows not aligned to 840 test IDs for {cname}")
        gt_pos = sub_all["gt_positive"].to_numpy(bool)
        for metric in metrics:
            b = sub_all[f"base_{metric}"].to_numpy(float)
            a = sub_all[f"a2_{metric}"].to_numpy(float)
            vb, va = np.isfinite(b), np.isfinite(a)
            valid_defs = {
                "A_base": vb,
                "A_a2": va,
                "B": vb & va,
                "C": vb & va & gt_pos,
            }
            sampled = {}
            empty_counts = {}
            for name, valid in valid_defs.items():
                vals = b if name == "A_base" else a if name == "A_a2" else a - b
                vidx = valid[boot_idx]
                counts = vidx.sum(axis=1)
                sums = np.where(vidx, vals[boot_idx], 0.0).sum(axis=1)
                sampled[name] = np.divide(sums, counts, out=np.full(N_BOOT, np.nan), where=counts > 0)
                empty_counts[name] = int((counts == 0).sum())
            boot_defs = {
                "A": (sampled["A_a2"] - sampled["A_base"], max(empty_counts["A_base"], empty_counts["A_a2"]), f"{int(vb.sum())}/{int(va.sum())}"),
                "B": (sampled["B"], empty_counts["B"], int((vb & va).sum())),
                "C": (sampled["C"], empty_counts["C"], int((vb & va & gt_pos).sum())),
            }
            point_rows = estimand_df[(estimand_df["class"] == cname) & (estimand_df["metric"] == metric)]
            for est, (vals, empty_n, n) in boot_defs.items():
                pr = point_rows[point_rows["estimand"] == est].iloc[0]
                boot_rows.append({
                    "class": cname,
                    "metric": metric,
                    "estimand": est,
                    "base_abs": pr["base_abs"],
                    "a2_abs": pr["a2_abs"],
                    "delta": pr["delta"],
                    "effective_n": n,
                    "ci95_low": float(np.nanpercentile(vals, 2.5)),
                    "ci95_high": float(np.nanpercentile(vals, 97.5)),
                    "empty_replicates": int(empty_n),
                    "bootstrap_n": N_BOOT,
                    "seed": BOOT_SEED,
                    "resampling_unit": "image",
                    "shared_indices_both_models": True,
                    "interval_method": "percentile",
                    "numpy_version": np.__version__,
                    "rng": "numpy.random.default_rng(PCG64)",
                    "sampling_index_sha256": boot_idx_hash,
                })
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(OUT / "bootstrap_reproduction_summary.csv", index=False)
    (OUT / "bootstrap_runtime_metadata.json").write_text(json.dumps({
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "opencv_version": cv2.__version__,
        "pillow_version": PIL.__version__,
        "python_version": sys.version.split()[0],
        "bootstrap_n": N_BOOT,
        "seed": BOOT_SEED,
        "rng": "numpy.random.default_rng(PCG64)",
        "sampling_index_shape": list(boot_idx.shape),
        "sampling_index_sha256": boot_idx_hash,
    }, indent=2), encoding="utf-8")

    # Dice root-cause and corrected size-stratified values.
    scale_old = json.loads(ORIGINAL_SCALE_JSON.read_text())
    dice_rows = []
    for cname in CLASSES.values():
        sub = case_df[(case_df["class"] == cname) & (case_df["gt_positive"])].copy()
        sub["size_group"] = [size_group(cname, r) for r in sub["gt_area_ratio"]]
        for grp in ("Small", "Medium", "Large"):
            g = sub[sub["size_group"] == grp]
            old = scale_old["groups"][cname][grp]
            old_iou_paired = old["iou"]["paired"]
            old_dice_paired = old["dice"]["paired"]
            d = (g["a2_dice"] - g["base_dice"]).to_numpy(float)
            dice_rows.append({
                "class": cname,
                "size_group": grp,
                "n": int(len(g)),
                "base_dice": float(g["base_dice"].mean()),
                "a2_dice": float(g["a2_dice"].mean()),
                "delta_dice": float(np.mean(d)),
                "paired_mean": float(np.mean(d)),
                "paired_median": float(np.median(d)),
                "paired_sd": float(np.std(d, ddof=1)),
                "old_dice_paired_mean": old_dice_paired["mean"],
                "old_iou_paired_mean": old_iou_paired["mean"],
                "old_dice_equals_old_iou": bool(abs(old_dice_paired["mean"] - old_iou_paired["mean"]) < 1e-15),
            })
    dice_df = pd.DataFrame(dice_rows)
    dice_df.to_csv(OUT / "scale_dice_corrected_recalculation.csv", index=False)

    # RQ1 aggregate / case decomposition.
    rq1_rows = []
    rq1_group_rows = []
    for cname, cid in [(v, k) for k, v in CLASSES.items()]:
        sub = case_df[case_df["class"] == cname].copy()
        full = {}
        for tag in ("base", "a2"):
            tp = sub[f"{tag}_tp"].sum()
            union = sub[f"{tag}_union"].sum()
            full[tag] = float(tp / union)
        gp = sub[sub["gt_positive"]].copy()
        n_gp = len(gp)
        rb = (gp["base_tp"] / gp["base_union"]).to_numpy(float)
        ra = (gp["a2_tp"] / gp["a2_union"]).to_numpy(float)
        ub = gp["base_union"].to_numpy(float)
        ua = gp["a2_union"].to_numpy(float)
        wb = ub / ub.sum()
        wa = ua / ua.sum()
        agg_b_gp = float((wb * rb).sum())
        agg_a_gp = float((wa * ra).sum())
        quality = float((wb * (ra - rb)).sum())
        reweight = float(((wa - wb) * ra).sum())
        case_delta = float((ra - rb).mean())
        bi_delta = gp["delta_boundary_iou_2px"].to_numpy(float)
        di = ra - rb
        sign_i = np.sign(di)
        sign_b = np.sign(bi_delta)
        same_strict = int(((sign_i > 0) & (sign_b > 0) | ((sign_i < 0) & (sign_b < 0))).sum())
        same_including_joint_ties = int((sign_i == sign_b).sum())
        rq1_rows.append({
            "class": cname,
            "n_all_images": len(sub),
            "n_gt_positive": n_gp,
            "aggregate_iou_base_all840": full["base"],
            "aggregate_iou_a2_all840": full["a2"],
            "aggregate_iou_delta_all840": full["a2"] - full["base"],
            "aggregate_iou_base_gt_positive": agg_b_gp,
            "aggregate_iou_a2_gt_positive": agg_a_gp,
            "aggregate_iou_delta_gt_positive": agg_a_gp - agg_b_gp,
            "case_mean_iou_delta_gt_positive": case_delta,
            "boundary_iou_2px_case_delta_gt_positive": float(np.nanmean(bi_delta)),
            "same_direction_strict_count": same_strict,
            "same_direction_including_joint_ties_count": same_including_joint_ties,
            "same_direction_denominator": n_gp,
            "quality_fixed_base_union_weight": quality,
            "union_reweighting_contribution": reweight,
            "quality_plus_reweighting": quality + reweight,
            "aggregate_minus_case_delta": (agg_a_gp - agg_b_gp) - case_delta,
        })
        gp["size_group"] = [size_group(cname, r) for r in gp["gt_area_ratio"]]
        for grp in ("Small", "Medium", "Large"):
            idx = (gp["size_group"] == grp).to_numpy(bool)
            if not idx.any():
                continue
            q = float((wb[idx] * (ra[idx] - rb[idx])).sum())
            rw = float(((wa[idx] - wb[idx]) * ra[idx]).sum())
            rq1_group_rows.append({
                "class": cname,
                "size_group": grp,
                "n": int(idx.sum()),
                "base_union_sum": float(ub[idx].sum()),
                "a2_union_sum": float(ua[idx].sum()),
                "quality_contribution_fixed_base_weight": q,
                "union_reweighting_contribution": rw,
                "total_contribution": q + rw,
                "share_of_gt_positive_aggregate_delta": (q + rw) / (agg_a_gp - agg_b_gp) if abs(agg_a_gp - agg_b_gp) > 0 else np.nan,
            })
    pd.DataFrame(rq1_rows).to_csv(OUT / "rq1_decomposition.csv", index=False)
    pd.DataFrame(rq1_group_rows).to_csv(OUT / "rq1_size_group_contributions.csv", index=False)

    # RQ2 GT-area, morphology and exploratory connected components.
    rq2_rows = []
    morph_rows = []
    component_rows = []
    for cname in CLASSES.values():
        sub = case_df[(case_df["class"] == cname) & (case_df["gt_positive"])].copy()
        ratios = sub["gt_area_ratio"].to_numpy(float)
        p33, p67 = np.percentile(ratios, [33, 67])
        sub["size_group_recomputed"] = [size_group(cname, r) for r in ratios]
        for grp in ("Small", "Medium", "Large"):
            g = sub[sub["size_group_recomputed"] == grp]
            rq2_rows.append({
                "class": cname,
                "size_group": grp,
                "p33_recomputed": float(p33),
                "p67_recomputed": float(p67),
                "p33_frozen": AREA_BINS[cname][0],
                "p67_frozen": AREA_BINS[cname][1],
                "n": int(len(g)),
                "iou_delta": float((g["a2_iou"] - g["base_iou"]).mean()),
                "dice_delta": float((g["a2_dice"] - g["base_dice"]).mean()),
                "boundary_iou_2px_delta": float((g["a2_boundary_iou_2px"] - g["base_boundary_iou_2px"]).mean()),
                "bf1_2px_delta": float((g["a2_bf1_2px"] - g["base_bf1_2px"]).mean()),
            })
        if cname == "crazing":
            t1, t2 = np.percentile(sub["crazing_complexity"].to_numpy(float), [33.333, 66.667])
            sub["complexity_group"] = np.where(sub["crazing_complexity"] <= t1, "low", np.where(sub["crazing_complexity"] <= t2, "mid", "high"))
            for grp in ("low", "mid", "high"):
                g = sub[sub["complexity_group"] == grp]
                morph_rows.append({
                    "complexity_group": grp,
                    "n": int(len(g)),
                    "complexity_min": float(g["crazing_complexity"].min()),
                    "complexity_max": float(g["crazing_complexity"].max()),
                    "iou_delta": float((g["a2_iou"] - g["base_iou"]).mean()),
                    "dice_delta": float((g["a2_dice"] - g["base_dice"]).mean()),
                    "boundary_iou_2px_delta": float((g["a2_boundary_iou_2px"] - g["base_boundary_iou_2px"]).mean()),
                    "bf1_2px_delta": float((g["a2_bf1_2px"] - g["base_bf1_2px"]).mean()),
                    "component_count_mean": float(g["gt_component_count"].mean()),
                    "area_ratio_mean": float(g["gt_area_ratio"].mean()),
                })
            high = sub[sub["complexity_group"] == "high"]
            for grp in ("Small", "Medium", "Large"):
                g = high[high["size_group_recomputed"] == grp]
                component_rows.append({
                    "analysis": "high_complexity_by_coarse_area",
                    "size_group": grp,
                    "n": int(len(g)),
                    "iou_delta": float((g["a2_iou"] - g["base_iou"]).mean()) if len(g) else np.nan,
                    "boundary_iou_2px_delta": float((g["a2_boundary_iou_2px"] - g["base_boundary_iou_2px"]).mean()) if len(g) else np.nan,
                })
            component_rows.append({
                "analysis": "exploratory_correlation",
                "size_group": "all",
                "n": int(len(sub)),
                "pearson_area_vs_complexity": float(np.corrcoef(sub["gt_area_ratio"], sub["crazing_complexity"])[0, 1]),
                "pearson_area_vs_component_count": float(np.corrcoef(sub["gt_area_ratio"], sub["gt_component_count"])[0, 1]),
                "pearson_complexity_vs_component_count": float(np.corrcoef(sub["crazing_complexity"], sub["gt_component_count"])[0, 1]),
                "iou_delta": np.nan,
                "boundary_iou_2px_delta": np.nan,
            })
    pd.DataFrame(rq2_rows).to_csv(OUT / "rq2_area_reproduction.csv", index=False)
    pd.DataFrame(morph_rows).to_csv(OUT / "rq2_complexity_reproduction.csv", index=False)
    pd.DataFrame(component_rows).to_csv(OUT / "rq2_exploratory_confounders.csv", index=False)

    # Figure 3 source audit.
    selection = json.loads(CASE_SELECTION_JSON.read_text())
    candidate_records = []
    for cname, groups in selection.items():
        for group, records in groups.items():
            for rank, rec in enumerate(records, 1):
                sid = rec["sample_id"]
                raw = case_df[(case_df["class"] == cname) & (case_df["sample_id"] == sid)].iloc[0]
                candidate_records.append({
                    "class": cname,
                    "group": group,
                    "rank": rank,
                    "sample_id": sid,
                    "recorded_delta_bf1_2px": rec["delta_bf1_2px"],
                    "recomputed_delta_bf1_2px": float(raw["delta_bf1_2px"]),
                    "absolute_difference": abs(float(raw["delta_bf1_2px"]) - rec["delta_bf1_2px"]),
                })
    fig3_candidates = pd.DataFrame(candidate_records)
    fig3_candidates.to_csv(OUT / "figure3_candidate_trace.csv", index=False)
    fig3_panels = [
        ("crazing", "top3_improved", 1, "000653"),
        ("crazing", "bottom3_worsened", 1, "000722"),
        ("crazing", "zero3", 1, "000058"),
        ("patches", "top3_improved", 1, "000802"),
        ("patches", "bottom3_worsened", 1, "000722"),
        ("patches", "zero3", 1, "000439"),
    ]
    fig3_actual = []
    for cname, group, rank, sid in fig3_panels:
        raw = case_df[(case_df["class"] == cname) & (case_df["sample_id"] == sid)].iloc[0]
        panel = MECH / "visualizations" / f"{cname}_{group}_{rank}_{sid}.png"
        fig3_actual.append({
            "class": cname,
            "group": group,
            "rank": rank,
            "sample_id": sid,
            "panel_path": str(panel),
            "panel_exists": panel.exists(),
            "recomputed_delta_bf1_2px": float(raw["delta_bf1_2px"]),
        })
    (OUT / "figure3_actual_panels.json").write_text(json.dumps(fig3_actual, indent=2), encoding="utf-8")

    # Concise Markdown reports with line-level implementation notes.
    identity_md = [
        "# NEU mask identity verification",
        "",
        f"Audit time UTC: {run_started}",
        f"Test list: `{TEST_LIST}`",
        f"Test-list SHA-256: `{identity['test_list_sha256']}`",
        f"Cached GT vs source annotations: {identity['cached_gt_vs_source']['pixel_equal']}/{identity['cached_gt_vs_source']['checked']} pixel-equal",
        f"Manifest order equals test-list order: {identity['manifest_order_equals_test_list_order']}",
        f"Directory sets equal test set: {identity['directory_set_equals_test_list']}",
        f"All masks 200x200: {identity['all_shapes_200x200']}",
        f"All labels within 0..3: {identity['all_labels_within_0_to_3']}",
        "",
        "## Independent mIoU reproduction",
    ]
    for tag in ("base", "a2"):
        r = confusion_summary[tag]
        identity_md.append(f"- {tag}: mIoU={r['miou']:.15f}, expected={r['expected_miou']:.15f}, absolute error={r['absolute_miou_error']:.3g}")
    identity_md += [
        "",
        "The prediction cache script uses `model(img)[0].data.max(1)[1]` and writes uint8 PNG masks directly; no post-processing branch was observed in that script.",
    ]
    (OUT / "neu_mask_identity_verification.md").write_text("\n".join(identity_md) + "\n", encoding="utf-8")

    boundary_md = [
        "# Boundary metric implementation audit",
        "",
        "Original implementation: `compute_mechanism_metrics.py` lines 17-46.",
        "",
        "- `band(m,t) = cv2.dilate(m,K3,iterations=t) - cv2.erode(m,K3,iterations=t)`, clipped to [0,1].",
        "- `K3` is a 3x3 all-ones uint8 structuring element.",
        "- BF1 uses one-erosion contours, OpenCV distance transforms, and tolerance thresholds 1/2/3 px.",
        "- Boundary IoU returns `None`/NA only when the union of both bands is zero.",
        "- BF1 returns `None`/NA only when both contours are empty; one-sided empty cases return 0.",
        "- No explicit OpenCV border argument is set; OpenCV defaults therefore apply.",
        "",
        "Synthetic empty-rule outputs are stored in `boundary_empty_rule_tests.json`.",
        "Named-sample outputs are stored in `boundary_named_sample_audit.json`.",
        "",
        "## A/B/C centers",
    ]
    for _, r in estimand_df[(estimand_df["class"] == "crazing") & (estimand_df["metric"] == "boundary_iou_2px")].iterrows():
        boundary_md.append(f"- Crazing BIoU@2 {r['estimand']}: delta={r['delta']:.15f}, n={r['n']}, base={r['base_abs']:.15f}, A2={r['a2_abs']:.15f}")
    boundary_md.append("")
    boundary_md.append("Bootstrap intervals use 5000 image-level resamples, seed 20261018, shared indices, percentile intervals, and fixed cached masks; see `bootstrap_reproduction_summary.csv`.")
    (OUT / "boundary_metric_implementation_audit.md").write_text("\n".join(boundary_md) + "\n", encoding="utf-8")

    dice_md = [
        "# Scale Dice root-cause audit",
        "",
        "Root cause found in `analyze_mechanism.py` lines 87-90: `paired_case_metrics.csv` contains `base_dice` and `a2_dice`, but no `delta_dice`; the scale loop falls back to `delta_iou` when `delta_dice` is absent.",
        "This is a downstream paired-field construction/write-out error, not an incorrect per-case Dice formula in `compute_mechanism_metrics.py`.",
        "",
        "Corrected independent Dice values are in `scale_dice_corrected_recalculation.csv`.",
        "",
        dice_df.pipe(df_block),
    ]
    (OUT / "scale_dice_root_cause_audit.md").write_text("\n".join(dice_md) + "\n", encoding="utf-8")

    rq1_md = [
        "# RQ1 pixel decomposition",
        "",
        "This is a statistical decomposition of fixed cached predictions, not a causal module ablation.",
        "",
        pd.DataFrame(rq1_rows).pipe(df_block),
        "",
        "## Union-weight contribution by frozen size group",
        "",
        pd.DataFrame(rq1_group_rows).pipe(df_block),
    ]
    (OUT / "rq1_pixel_decomposition.md").write_text("\n".join(rq1_md) + "\n", encoding="utf-8")

    rq2_md = [
        "# RQ2 raw-mask area and complexity audit",
        "",
        f"Area bins were reproduced from GT with numpy percentile(33/67); frozen thresholds were `{AREA_BINS}`.",
        "The complexity proxy is the original one-erosion contour pixel count divided by sqrt(area), not a geometric perimeter implementation.",
        "Connected-component analyses are exploratory and were not part of the original protocol.",
        "",
        "## Area groups",
        pd.DataFrame(rq2_rows).pipe(df_block),
        "",
        "## Crazing complexity tertiles",
        pd.DataFrame(morph_rows).pipe(df_block),
        "",
        "## Exploratory high-complexity/coarse-area and correlations",
        pd.DataFrame(component_rows).pipe(df_block),
    ]
    (OUT / "rq2_raw_mask_condition_audit.md").write_text("\n".join(rq2_md) + "\n", encoding="utf-8")

    fig_md = [
        "# Figure case source audit",
        "",
        f"Figure 3 candidates: {len(fig3_candidates)} image-class records; unique sample IDs: {fig3_candidates['sample_id'].nunique()}",
        f"Maximum recorded-vs-recomputed ΔBF1@2 error: {fig3_candidates['absolute_difference'].max():.3g}",
        "",
        "Actual six Figure 3 rows are hard-coded in `assemble_fig34_v6.py` lines 39-46 and were traced in `figure3_actual_panels.json`.",
        "Leather Figure 4 full 268-image raw prediction cache was not present under `leather_historical_audit`; only the six-panel manifest and evaluation outputs are currently available.",
    ]
    (OUT / "figure_case_source_audit.md").write_text("\n".join(fig_md) + "\n", encoding="utf-8")

    summary = {
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "identity": {
            "n": len(test_ids),
            "manifest_order_equals_test_list": identity["manifest_order_equals_test_list_order"],
            "all_shapes_200x200": identity["all_shapes_200x200"],
            "all_labels_within_0_to_3": identity["all_labels_within_0_to_3"],
            "gt_source_pixel_equal": f"{identity['cached_gt_vs_source']['pixel_equal']}/{identity['cached_gt_vs_source']['checked']}",
        },
        "miou": {tag: confusion_summary[tag]["miou"] for tag in ("base", "a2")},
        "crazing_boundary2_abc": estimand_df[(estimand_df["class"] == "crazing") & (estimand_df["metric"] == "boundary_iou_2px")][["estimand", "delta", "n"]].to_dict("records"),
        "bootstrap_metadata_sha256": boot_idx_hash,
        "rq1": rq1_rows,
    }
    (OUT / "neu_pixel_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
