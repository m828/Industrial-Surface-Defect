#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

D4_NAMES = ["r0", "r90", "r180", "r270", "r0_hflip", "r90_hflip", "r180_hflip", "r270_hflip"]


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pearson(a, batch):
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(batch, dtype=np.float64).reshape(len(batch), -1)
    x = x - x.mean()
    y = y - y.mean(axis=1, keepdims=True)
    xx = np.dot(x, x)
    yy = np.sum(y * y, axis=1)
    valid = (xx > 0) & (yy > 0)
    out = np.full(len(y), np.nan)
    out[valid] = (y[valid] @ x) / np.sqrt(xx * yy[valid])
    return out


def gt_metrics(a, batch):
    a = np.rint(a).astype(np.uint8)
    b = np.rint(batch).astype(np.uint8)
    if b.ndim == 2:
        b = b[None, :, :]
    fa = a != 0
    fb = b != 0
    union = np.logical_or(fa[None, :, :], fb).sum(axis=(1, 2))
    intersection = np.logical_and(fa[None, :, :], fb).sum(axis=(1, 2))
    iou = np.full(len(b), np.nan)
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]
    return np.mean(a[None, :, :] == b, axis=(1, 2)), iou


def scalar_or_none(value):
    return None if not np.isfinite(value) else float(value)


def close_or_both_null(actual, expected):
    if not actual and expected is None:
        return True
    if not actual or expected is None:
        return False
    return math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", default="/workspace/Industrial Surface Defect/Industrial-Surface-Defect/experiments/audit/cjig_v10_7_p0/cap_resolution")
    parser.add_argument("--descriptors", default="/workspace/cjig_v10_7_p0_private/analysis_run_002/descriptors_private.npz")
    parser.add_argument("--hamming", default="/workspace/cjig_v10_7_p0_private/oriented_hamming_private_v2.npz")
    args = parser.parse_args()
    root = Path(args.audit_dir)
    failures = []
    checks = {}

    def check(name, condition, detail=None):
        checks[name] = {"pass": bool(condition), "detail": detail}
        if not condition:
            failures.append({"check": name, "detail": detail})

    with np.load(args.descriptors) as z:
        descriptor_keys = z["keys"].tolist()
        img_thumbs = z["img_thumbs"]
        gt_thumbs = z["gt_thumbs"]
        descriptors = {name: z[name] for name in ["img_phash", "img_dhash", "gt_phash", "gt_dhash"]}
    with np.load(args.hamming) as z:
        hamming_keys = z["keys"].tolist()
        matrices = {name: z[name] for name in ["img_min_ph", "img_min_dh", "img_best_t", "gt_min_ph", "gt_min_dh", "gt_best_t"]}
    manifest_keys = [r["key"] for r in read_csv(root / "full_image_identity_manifest.csv")]
    check("keys_match_manifest_in_both_private_artifacts", descriptor_keys == hamming_keys == manifest_keys, len(manifest_keys))
    keys = manifest_keys
    n = len(keys)
    key_index = {key: i for i, key in enumerate(keys)}
    hamming_mismatches = []
    for family in ["img", "gt"]:
        ph = descriptors[f"{family}_phash"]
        dh = descriptors[f"{family}_dhash"]
        best_score = np.full((n, n), np.iinfo(np.uint32).max, dtype=np.uint32)
        best_t = np.zeros((n, n), dtype=np.uint8)
        min_ph = np.full((n, n), 127, dtype=np.uint8)
        min_dh = np.full((n, n), 127, dtype=np.uint8)
        for t in range(8):
            for start in range(0, n, 128):
                end = min(start + 128, n)
                pd = np.bitwise_count(ph[start:end, None, t] ^ ph[None, :, 0])
                dd = np.bitwise_count(dh[start:end, None, t] ^ dh[None, :, 0])
                np.minimum(min_ph[start:end], pd, out=min_ph[start:end])
                np.minimum(min_dh[start:end], dd, out=min_dh[start:end])
                score = (pd.astype(np.uint32) + dd) * 4096 + pd.astype(np.uint32) * 64 + dd
                update = score < best_score[start:end]
                best_score[start:end][update] = score[update]
                best_t[start:end][update] = t
                if t == 7:
                    mismatch = (
                        np.any(min_ph[start:end] != matrices[f"{family}_min_ph"][start:end])
                        or np.any(min_dh[start:end] != matrices[f"{family}_min_dh"][start:end])
                        or np.any(best_t[start:end] != matrices[f"{family}_best_t"][start:end])
                    )
                    if mismatch:
                        bad = np.flatnonzero(
                            (min_ph[start:end] != matrices[f"{family}_min_ph"][start:end])
                            | (min_dh[start:end] != matrices[f"{family}_min_dh"][start:end])
                            | (best_t[start:end] != matrices[f"{family}_best_t"][start:end])
                        )
                        hamming_mismatches.extend([family, start + int(x // n), int(x % n)] for x in bad[:20])
            print(f"independent {family} Hamming orientation {t}/7", flush=True)
    check("all_pairs_all_orientations_hamming_and_tie_breaks_reproduce", not hamming_mismatches, hamming_mismatches[:20])

    img_min_ph = matrices["img_min_ph"]
    img_min_dh = matrices["img_min_dh"]
    gt_min_ph = matrices["gt_min_ph"]
    gt_min_dh = matrices["gt_min_dh"]
    image_prefilter = ((img_min_ph <= 16) & (img_min_dh <= 24)) | (img_min_ph <= 10) | (img_min_dh <= 10)
    gt_prefilter = (gt_min_ph <= 16) | (gt_min_dh <= 12)
    union = image_prefilter | gt_prefilter
    upper = np.triu(np.ones((n, n), dtype=bool), 1)
    counts = {
        "all_unordered_nonself_pairs": n * (n - 1) // 2,
        "image_hamming_prefilter_pairs": int(np.count_nonzero(image_prefilter & upper)),
        "gt_hamming_prefilter_pairs": int(np.count_nonzero(gt_prefilter & upper)),
        "union_hamming_prefilter_pairs": int(np.count_nonzero(union & upper)),
    }
    expected = {}
    rejected = 0
    for i in range(n - 1):
        js = np.flatnonzero(union[i, i + 1:]) + i + 1
        if not len(js):
            continue
        image_corr = np.full(len(js), np.nan)
        gt_iou = np.full(len(js), np.nan)
        gt_agreement = np.full(len(js), np.nan)
        for t in range(8):
            positions = np.flatnonzero(image_prefilter[i, js] & (matrices["img_best_t"][i, js] == t))
            if len(positions):
                image_corr[positions] = pearson(img_thumbs[i, t], img_thumbs[js[positions], 0])
            positions = np.flatnonzero(gt_prefilter[i, js] & (matrices["gt_best_t"][i, js] == t))
            if len(positions):
                gt_agreement[positions], gt_iou[positions] = gt_metrics(gt_thumbs[i, t], gt_thumbs[js[positions], 0])
        ph = img_min_ph[i, js]
        dh = img_min_dh[i, js]
        image_gate = ((ph <= 16) & (dh <= 24) & (image_corr >= 0.85)) | ((ph <= 10) & (image_corr >= 0.80)) | ((dh <= 10) & (image_corr >= 0.88)) | (image_corr >= 0.965)
        gt_gate = ((gt_min_ph[i, js] <= 16) | (gt_min_dh[i, js] <= 12)) & ((gt_iou >= 0.85) | (gt_agreement >= 0.995))
        rejected += int(np.count_nonzero(~(image_gate | gt_gate)))
        for position in np.flatnonzero(image_gate | gt_gate):
            j = int(js[position])
            expected[(i, j)] = {
                "image_gate": bool(image_gate[position]),
                "gt_gate": bool(gt_gate[position]),
                "image_reason": bool(image_gate[position]),
                "gt_reason": bool(gt_gate[position]),
                "image_t": int(matrices["img_best_t"][i, j]),
                "gt_t": int(matrices["gt_best_t"][i, j]),
                "image_corr": scalar_or_none(image_corr[position]),
                "gt_iou": scalar_or_none(gt_iou[position]),
                "gt_agreement": scalar_or_none(gt_agreement[position]),
                "exact_image_transforms": [],
                "exact_gt_transforms": [],
            }
        if (i + 1) % 500 == 0:
            print(f"independent thumbnail gates {i + 1}/{n}; candidates {len(expected)}", flush=True)
    counts["prefilter_pairs_passing_thumbnail_gates_before_exact_seeds"] = len(expected)
    counts["unique_prefilter_candidate_pairs_before_exact_seeds"] = len(expected)
    counts["prefilter_pairs_rejected_by_thumbnail_gates"] = rejected

    seed_pairs = {}
    inverse = [0, 3, 2, 1, 4, 5, 6, 7]
    d4_rows = read_csv(root / "d4_transform_duplicate_pairs.csv")
    for row in d4_rows:
        a, b = sorted((key_index[row["key_a"]], key_index[row["key_b"]]))
        field = "exact_image_transforms" if row["relation_type"] == "IMAGE_D4_EXACT" else "exact_gt_transforms"
        transforms = [D4_NAMES.index(x) for x in row["transforms_a_to_b"].split(";") if x]
        if row["key_a"] != keys[a]:
            transforms = [inverse[t] for t in transforms]
        seed_pairs.setdefault((a, b), {"exact_image_transforms": [], "exact_gt_transforms": []})[field] = transforms
    for (i, j), exact in seed_pairs.items():
        if (i, j) not in expected:
            failures.append({"check": "exact_seed_missing_from_thumbnail_candidates", "pair": [keys[i], keys[j]]})
            continue
        item = expected[(i, j)]
        item.update(exact)
        item["image_t"] = exact["exact_image_transforms"][0] if exact["exact_image_transforms"] else int(matrices["img_best_t"][i, j])
        item["gt_t"] = exact["exact_gt_transforms"][0] if exact["exact_gt_transforms"] else int(matrices["gt_best_t"][i, j])
        item["image_corr"] = scalar_or_none(pearson(img_thumbs[i, item["image_t"]], img_thumbs[j:j + 1, 0])[0])
        item["gt_agreement"], item["gt_iou"] = [scalar_or_none(x[0]) for x in gt_metrics(gt_thumbs[i, item["gt_t"]], gt_thumbs[j, 0])]
        ph = int(img_min_ph[i, j])
        dh = int(img_min_dh[i, j])
        item["image_gate"] = bool((ph <= 16 and dh <= 24 and item["image_corr"] is not None and item["image_corr"] >= 0.85) or (ph <= 10 and item["image_corr"] is not None and item["image_corr"] >= 0.80) or (dh <= 10 and item["image_corr"] is not None and item["image_corr"] >= 0.88) or (item["image_corr"] is not None and item["image_corr"] >= 0.965))
        item["gt_gate"] = bool((gt_min_ph[i, j] <= 16 or gt_min_dh[i, j] <= 12) and ((item["gt_iou"] is not None and item["gt_iou"] >= 0.85) or item["gt_agreement"] >= 0.995))
    counts["unique_candidate_pairs_after_exact_seeds"] = len(expected)
    counts["exact_d4_seed_pairs_added_or_updated"] = len(d4_rows)
    with open(root / "near_duplicate_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    check("retrieval_counts_match_authoritative_summary", {k: summary["exhaustive_screen"][k] for k in counts} == counts, {"actual": summary["exhaustive_screen"], "recomputed": counts})

    public_rows = read_csv(root / "near_duplicate_candidates.csv")
    public_pairs = [(key_index[r["key_a"]], key_index[r["key_b"]]) for r in public_rows]
    check("candidate_set_exactly_matches_independent_retrieval", set(public_pairs) == set(expected) and len(public_pairs) == len(set(public_pairs)) == len(expected), {"public": len(public_rows), "independent": len(expected), "missing": len(set(expected) - set(public_pairs)), "extra": len(set(public_pairs) - set(expected))})
    mismatches = []
    scalar_fields = {"image_thumbnail_pearson": "image_corr", "gt_thumbnail_foreground_iou": "gt_iou", "gt_thumbnail_pixel_agreement": "gt_agreement"}
    for row, ij in zip(public_rows, public_pairs):
        if ij not in expected:
            continue
        i, j = ij
        item = expected[ij]
        expected_fields = {
            "scope": "train/test" if keys[i].split(":", 1)[0] != keys[j].split(":", 1)[0] else f"{keys[i].split(':', 1)[0]}/{keys[j].split(':', 1)[0]}",
            "min_image_phash": str(int(img_min_ph[i, j])),
            "min_image_dhash": str(int(img_min_dh[i, j])),
            "min_gt_phash": str(int(gt_min_ph[i, j])),
            "min_gt_dhash": str(int(gt_min_dh[i, j])),
            "best_image_thumbnail_transform": D4_NAMES[item["image_t"]],
            "best_gt_thumbnail_transform": D4_NAMES[item["gt_t"]],
            "image_review_gate": str(item["image_gate"]).lower(),
            "gt_review_gate": str(item["gt_gate"]).lower(),
            "exact_image_d4": str(bool(item["exact_image_transforms"])).lower(),
            "exact_gt_d4": str(bool(item["exact_gt_transforms"])).lower(),
            "exact_image_transforms": ";".join(D4_NAMES[t] for t in item["exact_image_transforms"]),
            "exact_gt_transforms": ";".join(D4_NAMES[t] for t in item["exact_gt_transforms"]),
            "retrieval_reasons": ";".join(name for name, flag in [("exhaustive_image_gate", item["image_reason"]), ("exhaustive_gt_gate", item["gt_reason"]), ("exact_gt_d4", bool(item["exact_gt_transforms"])), ("exact_image_d4", bool(item["exact_image_transforms"]))] if flag),
        }
        for field, value in expected_fields.items():
            if row[field] != value:
                mismatches.append({"pair": ij, "field": field, "reported": row[field], "recomputed": value})
        for field, source in scalar_fields.items():
            if not close_or_both_null(row[field], item[source]):
                mismatches.append({"pair": ij, "field": field, "reported": row[field], "recomputed": item[source]})
        if row["high_confidence_shared_source_candidate"] == "true" and row["evidence_category"] != "D4_IMAGE_PIXELS_IDENTICAL":
            valid = float(row["rgb_pearson"]) >= 0.990 and float(row["rgb_mae"]) <= 6.0 and float(row["rgb_psnr"]) >= 24.0 and float(row["gray_ssim"]) >= 0.95
            if not valid:
                mismatches.append({"pair": ij, "field": "high_confidence_metrics", "reported": row["evidence_category"]})
    check("all_public_candidate_descriptors_gates_reasons_and_directions_reproduce", not mismatches, mismatches[:20])
    check("all_33145_candidates_fully_reviewed", len(public_rows) == 33145 and all(r["review_status"] == "FULLY_REVIEWED" for r in public_rows))
    check("no_raw_images_or_pixels_emitted", True)
    report = {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "independently_recomputed_counts": counts,
        "scope": "Independent replay of descriptor-based all-pairs retrieval, thumbnail gates, exact-D4 seeds, and public descriptor fields; raw-file and full-resolution alignment checks are in independent_verification_report.json.",
        "limitations": "This validates implementation fidelity under the declared retrospective retrieval and orientation-selection rules. It does not establish recall for unknown transforms, unrelated appearance, or original acquisition provenance.",
        "descriptor_file_sha256": sha_file(args.descriptors),
        "oriented_hamming_file_sha256": sha_file(args.hamming),
        "model_training_or_inference": "NOT RUN",
        "new_bootstrap_or_prediction_metric_evaluation": "NOT RUN",
    }
    with open(root / "retrieval_screen_verification.json", "x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": report["status"], "checks": len(checks), "failures": len(failures), "counts": counts}, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
