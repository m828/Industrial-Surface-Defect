#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import re
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

D4_NAMES = ["r0", "r90", "r180", "r270", "r0_hflip", "r90_hflip", "r180_hflip", "r270_hflip"]


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pixel_sha(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(json.dumps(list(a.shape)).encode("ascii"))
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def d4(a, t):
    out = np.rot90(a, t % 4)
    if t >= 4:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json_new(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not math.isfinite(x):
            return str(x)
        return x
    with open(path, "x", encoding="utf-8") as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def pearson(a, b):
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return None if denominator == 0 else float(np.dot(x, y) / denominator)


def close_enough(actual, expected, tolerance=1e-9):
    if actual in [None, ""] and expected in [None, ""]:
        return True
    if actual in [None, ""] or expected in [None, ""]:
        return False
    return math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance)


def pair(a, b):
    return tuple(sorted((a, b)))


def grouped_pairs(manifest, field):
    groups = defaultdict(list)
    for row in manifest:
        groups[row[field]].append(row["key"])
    return {pair(a, b) for members in groups.values() if len(members) > 1 for a, b in combinations(sorted(members), 2)}


WORKER_ARRAYS = {}


def verify_review_row(payload):
    index, row = payload
    t = D4_NAMES.index(row["transform"])
    rgb_a = d4(WORKER_ARRAYS[row["key_a"]][0], t)
    rgb_b = WORKER_ARRAYS[row["key_b"]][0]
    gray_a = np.asarray(Image.fromarray(rgb_a, "RGB").convert("L"), dtype=np.uint8)
    gray_b = np.asarray(Image.fromarray(rgb_b, "RGB").convert("L"), dtype=np.uint8)
    gt_a = d4(WORKER_ARRAYS[row["key_a"]][1], t)
    gt_b = WORKER_ARRAYS[row["key_b"]][1]
    mae = float(np.mean(np.abs(rgb_a.astype(np.float64) - rgb_b.astype(np.float64))))
    mse = float(np.mean((rgb_a.astype(np.float64) - rgb_b.astype(np.float64)) ** 2))
    psnr = None if mse == 0 else float(10 * np.log10(255 ** 2 / mse))
    ssim = float(structural_similarity(gray_a, gray_b, data_range=255, win_size=7, gaussian_weights=False, use_sample_covariance=True, K1=0.01, K2=0.03))
    union = np.logical_or(gt_a != 0, gt_b != 0).sum()
    foreground_iou = None if union == 0 else float(np.logical_and(gt_a != 0, gt_b != 0).sum() / union)
    comparisons = {
        "rgb_pearson": pearson(rgb_a, rgb_b),
        "rgb_mae": mae,
        "rgb_psnr": psnr,
        "gray_ssim": ssim,
        "gt_pixel_agreement": float(np.mean(gt_a == gt_b)),
        "gt_foreground_iou": foreground_iou,
        "gt_diff_pixels": int(np.count_nonzero(gt_a != gt_b)),
    }
    failures = []
    for field, expected in comparisons.items():
        if not close_enough(row[field], expected):
            failures.append({"pair": [row["key_a"], row["key_b"]], "field": field, "reported": row[field], "recomputed": expected})
    return index, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/workspace/Industrial Surface Defect")
    parser.add_argument("--output-dir", default="/workspace/Industrial Surface Defect/Industrial-Surface-Defect/experiments/audit/cjig_v10_7_p0")
    parser.add_argument("--upstream-root", default="/workspace/cjig_v10_7_p0_private/upstream_archive")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    project = Path(args.project_root)
    output = Path(args.output_dir)
    upstream = Path(args.upstream_root)
    failures = []
    checks = {}

    def check(name, condition, detail=None):
        checks[name] = {"pass": bool(condition), "detail": detail}
        if not condition:
            failures.append({"check": name, "detail": detail})

    manifest = read_csv(output / "full_image_identity_manifest.csv")
    exact_rows = read_csv(output / "exact_duplicate_pairs.csv")
    d4_rows = read_csv(output / "d4_transform_duplicate_pairs.csv")
    candidate_rows = read_csv(output / "near_duplicate_candidates.csv")
    with open(output / "positive_control_verification.json", encoding="utf-8") as f:
        controls = json.load(f)
    with open(output / "upstream_identity_summary.json", encoding="utf-8") as f:
        upstream_summary = json.load(f)
    with open(output / "near_duplicate_summary.json", encoding="utf-8") as f:
        near_summary = json.load(f)

    check("manifest_4470_rows", len(manifest) == 4470, len(manifest))
    keys = [r["key"] for r in manifest]
    check("manifest_unique_composite_keys", len(set(keys)) == 4470)
    check("manifest_split_counts", Counter(r["split"] for r in manifest) == {"train": 3630, "test": 840}, dict(Counter(r["split"] for r in manifest)))
    check("known_positive_controls_pass", controls["overall_pass"] is True)
    check("upstream_8940_byte_matches", upstream_summary["all_8940_files_byte_identical"] is True)

    arrays = {}
    source_mtimes = []
    for index, row in enumerate(manifest):
        key = row["key"]
        image_path = project / row["image_relative_path"]
        gt_path = project / row["gt_relative_path"]
        upstream_image = upstream / row["upstream_image_relative_path"]
        upstream_gt = upstream / row["upstream_gt_relative_path"]
        for path in [image_path, gt_path]:
            source_mtimes.append(path.stat().st_mtime)
        if sha_file(image_path) != row["image_file_sha256"] or sha_file(gt_path) != row["gt_file_sha256"]:
            failures.append({"check": "local_file_sha", "key": key})
        if sha_file(upstream_image) != row["upstream_image_file_sha256"] or sha_file(upstream_gt) != row["upstream_gt_file_sha256"]:
            failures.append({"check": "upstream_file_sha", "key": key})
        with Image.open(image_path) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        with Image.open(gt_path) as im:
            gt = np.asarray(im.convert("L"), dtype=np.uint8)
        if pixel_sha(rgb) != row["image_pixel_sha256"] or pixel_sha(gt) != row["gt_pixel_sha256"]:
            failures.append({"check": "decoded_pixel_sha", "key": key})
        arrays[key] = (rgb, gt)
        if (index + 1) % 1000 == 0:
            print(f"verify source files {index + 1}/4470", flush=True)
    check("source_file_and_pixel_failures_empty", not any(x["check"] in ["local_file_sha", "upstream_file_sha", "decoded_pixel_sha"] for x in failures))

    expected_exact = {}
    for relation, field in [
        ("IMAGE_FILE_BYTES_IDENTICAL", "image_file_sha256"),
        ("IMAGE_DECODED_PIXELS_IDENTICAL", "image_pixel_sha256"),
        ("GT_FILE_BYTES_IDENTICAL", "gt_file_sha256"),
        ("GT_DECODED_PIXELS_IDENTICAL", "gt_pixel_sha256"),
    ]:
        expected_exact[relation] = grouped_pairs(manifest, field)
        actual = {pair(r["key_a"], r["key_b"]) for r in exact_rows if r["relation_type"] == relation}
        check(f"exact_relation_{relation}", actual == expected_exact[relation], {"actual": len(actual), "expected": len(expected_exact[relation])})

    for relation in ["IMAGE_FILE_BYTES_IDENTICAL", "IMAGE_DECODED_PIXELS_IDENTICAL", "GT_FILE_BYTES_IDENTICAL", "GT_DECODED_PIXELS_IDENTICAL", "IMAGE_D4_EXACT", "GT_D4_EXACT"]:
        source_rows = d4_rows if relation.endswith("D4_EXACT") else exact_rows
        relation_pairs = [(r["relation_type"], pair(r["key_a"], r["key_b"])) for r in source_rows if r["relation_type"] == relation]
        check(f"no_duplicate_rows_{relation}", len(relation_pairs) == len(set(relation_pairs)), len(relation_pairs))
    candidate_pair_ids = [pair(r["key_a"], r["key_b"]) for r in candidate_rows]
    check("near_candidates_unique_pairs", len(candidate_pair_ids) == len(set(candidate_pair_ids)), len(candidate_pair_ids))

    expected_d4 = {"IMAGE_D4_EXACT": set(), "GT_D4_EXACT": set()}
    transform_lookup = {}
    manifest_by_key = {r["key"]: r for r in manifest}
    canonical_failures = []
    for key, (rgb, gt) in arrays.items():
        image_series = [pixel_sha(d4(rgb, t)) for t in range(8)]
        gt_series = [pixel_sha(d4(gt, t)) for t in range(8)]
        for t in range(8):
            transform_lookup[(key, "IMAGE_D4_EXACT", t)] = image_series[t]
            transform_lookup[(key, "GT_D4_EXACT", t)] = gt_series[t]
        if min(image_series) != manifest_by_key[key]["image_d4_canonical_sha256"] or min(gt_series) != manifest_by_key[key]["gt_d4_canonical_sha256"]:
            canonical_failures.append(key)
    check("all_4470_d4_canonical_hashes_recomputed", not canonical_failures, canonical_failures[:20])
    for relation, pixel_field, canonical_field in [
        ("IMAGE_D4_EXACT", "image_pixel_sha256", "image_d4_canonical_sha256"),
        ("GT_D4_EXACT", "gt_pixel_sha256", "gt_d4_canonical_sha256"),
    ]:
        expected_d4[relation] = grouped_pairs(manifest, canonical_field)
        actual = {pair(r["key_a"], r["key_b"]) for r in d4_rows if r["relation_type"] == relation}
        check(f"d4_relation_{relation}", actual == expected_d4[relation], {"actual": len(actual), "expected": len(expected_d4[relation])})
        row_by_key = {r["key"]: r for r in manifest}
        for row in [r for r in d4_rows if r["relation_type"] == relation]:
            expected_transforms = [D4_NAMES[t] for t in range(8) if transform_lookup[(row["key_a"], relation, t)] == row_by_key[row["key_b"]][pixel_field]]
            actual_transforms = [x for x in row["transforms_a_to_b"].split(";") if x]
            if actual_transforms != expected_transforms:
                failures.append({"check": "d4_transform_direction", "pair": [row["key_a"], row["key_b"]], "relation": relation, "actual": actual_transforms, "expected": expected_transforms})
    check("d4_transform_directions_all_match", not any(x["check"] == "d4_transform_direction" for x in failures))

    reviewed = [r for r in candidate_rows if r["review_status"] == "FULLY_REVIEWED"]
    assert args.workers >= 1
    global WORKER_ARRAYS
    WORKER_ARRAYS = arrays
    metric_failures = []
    payloads = list(enumerate(reviewed))
    if args.workers > 1:
        with mp.get_context("fork").Pool(args.workers) as pool:
            results = pool.imap_unordered(verify_review_row, payloads, chunksize=16)
            for completed, (_, row_failures) in enumerate(results, 1):
                metric_failures.extend(row_failures)
                if completed % 500 == 0:
                    print(f"verify full reviews {completed}/{len(reviewed)}", flush=True)
    else:
        for completed, (_, row_failures) in enumerate(map(verify_review_row, payloads), 1):
            metric_failures.extend(row_failures)
            if completed % 500 == 0:
                print(f"verify full reviews {completed}/{len(reviewed)}", flush=True)
    metric_failures.sort(key=lambda x: (x["pair"], x["field"]))
    check("all_reviewed_alignment_metrics_reproduce", not metric_failures, {"reviewed": len(reviewed), "failures": metric_failures[:20]})
    check("fully_reviewed_count_matches_summary", len(reviewed) == near_summary["fully_reviewed_pairs"], {"actual": len(reviewed), "summary": near_summary["fully_reviewed_pairs"]})
    if near_summary.get("authoritative_merged_result"):
        check("merged_all_candidates_fully_reviewed", len(reviewed) == len(candidate_rows) and all(r["review_status"] == "FULLY_REVIEWED" for r in candidate_rows), {"reviewed": len(reviewed), "candidates": len(candidate_rows)})
        check("merged_cap_resolution_declared_complete", near_summary.get("screen_complete_after_declared_cap_resolution") is True and near_summary["cap_resolution"]["remaining_unreviewed_candidates"] == 0)

    edge_rows = read_csv(output / "source_family_edges.csv")
    with open(output / "source_family_summary.json", encoding="utf-8") as f:
        family_summary = json.load(f)
    actual_high_edges = {pair(r["key_a"], r["key_b"]) for r in edge_rows if r["high_confidence_shared_source_candidate"] == "true"}
    expected_high_edges = {pair(r["key_a"], r["key_b"]) for r in d4_rows if r["relation_type"] == "IMAGE_D4_EXACT"} | {pair(r["key_a"], r["key_b"]) for r in reviewed if r["high_confidence_shared_source_candidate"] == "true"}
    check("source_graph_high_edges_exactly_match_evidence", actual_high_edges == expected_high_edges, {"actual": len(actual_high_edges), "expected": len(expected_high_edges)})
    candidate_high_count = sum(r["high_confidence_shared_source_candidate"] == "true" for r in reviewed)
    check("near_summary_high_count_matches_candidates", candidate_high_count == near_summary["confirmed_high_confidence_pairs_including_exact_image_d4"], candidate_high_count)
    check("family_high_edge_count_matches_summary", len(actual_high_edges) == family_summary["high_confidence_candidate"]["edges_joined"], len(actual_high_edges))
    membership_rows = read_csv(output / "source_family_membership.csv")
    membership_keys = {(r["level"], r["key"]) for r in membership_rows}
    expected_membership_keys = {(level, key) for level in ["digital_identity", "transform_equivalence", "high_confidence_candidate"] for key in keys}
    check("family_membership_complete_and_unique", len(membership_rows) == len(membership_keys) and membership_keys == expected_membership_keys, len(membership_rows))
    split_by_key = {r["key"]: r["split"] for r in manifest}
    for level, flag in [("digital_identity", "digital_identity_verified"), ("transform_equivalence", "transform_equivalence_verified"), ("high_confidence_candidate", "high_confidence_shared_source_candidate")]:
        parent = {key: key for key in keys}
        def find(key):
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key
        selected_edges = [pair(r["key_a"], r["key_b"]) for r in edge_rows if r[flag] == "true"]
        for a, b in selected_edges:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        groups = defaultdict(list)
        for key in keys:
            groups[find(key)].append(key)
        nonsingletons = [members for members in groups.values() if len(members) > 1]
        actual_stats = {
            "components_total_including_singletons": len(groups),
            "nonsingleton_components": len(nonsingletons),
            "nonsingletons_with_train": sum(any(split_by_key[k] == "train" for k in members) for members in nonsingletons),
            "nonsingletons_with_test": sum(any(split_by_key[k] == "test" for k in members) for members in nonsingletons),
            "cross_split_nonsingletons": sum(len({split_by_key[k] for k in members}) == 2 for members in nonsingletons),
            "train_records_in_nonsingletons": sum(split_by_key[k] == "train" for members in nonsingletons for k in members),
            "test_records_in_nonsingletons": sum(split_by_key[k] == "test" for members in nonsingletons for k in members),
            "edges_joined": len(selected_edges),
        }
        mismatches = {field: {"actual": value, "summary": family_summary[level][field]} for field, value in actual_stats.items() if family_summary[level][field] != value}
        check(f"family_components_recomputed_{level}", not mismatches, mismatches)
    priority_mismatches = []
    for component in family_summary["priority_components"]:
        actual_members = sorted(r["key"] for r in membership_rows if r["level"] == "high_confidence_candidate" and r["group_id"] == component["group_id"])
        if actual_members != sorted(component["members"]):
            priority_mismatches.append(component["group_id"])
    check("priority_component_membership_matches", not priority_mismatches, priority_mismatches)

    forbidden_extensions = {".jpg", ".jpeg", ".png", ".pkl", ".pth", ".pt", ".npy", ".npz", ".tif", ".tiff", ".bmp", ".tfevents"}
    public_files = [p for p in output.rglob("*") if p.is_file()]
    forbidden_files = [str(p.relative_to(output)) for p in public_files if p.suffix.lower() in forbidden_extensions or "tfevents" in p.name]
    credential_patterns = [r"-----BEGIN [A-Z ]*PRIVATE KEY-----", r"github_pat_[A-Za-z0-9_]+", r"ghp_[A-Za-z0-9]+", r"AKIA[0-9A-Z]{16}", r"Authorization:\s*Bearer\s+\S+"]
    credential_hits = []
    email_hits = []
    for path in public_files:
        if path.suffix.lower() not in {".md", ".json", ".csv", ".txt", ".py", ".yml", ".yaml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in credential_patterns:
            if re.search(pattern, text):
                credential_hits.append(str(path.relative_to(output)))
        if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text):
            email_hits.append(str(path.relative_to(output)))
    check("public_no_raw_or_checkpoint_files", not forbidden_files, forbidden_files)
    check("public_no_credential_patterns", not credential_hits, credential_hits)
    check("public_no_email_addresses", not email_hits, email_hits)

    temporal = {
        "source_files": len(source_mtimes),
        "minimum_mtime": datetime.fromtimestamp(min(source_mtimes)).astimezone().isoformat(),
        "maximum_mtime": datetime.fromtimestamp(max(source_mtimes)).astimezone().isoformat(),
        "formal_base_training_log_first_timestamp": "2026-09-22T14:42:16+08:00",
        "interpretation": "Filesystem mtimes support but do not cryptographically prove training-time byte identity; resolved paths, pre-run list hashes, runtime logs, and fixed-upstream byte identity are separate evidence.",
    }
    write_json_new(output / "source_temporal_metadata.json", temporal)
    check("all_source_mtimes_precede_formal_training", max(source_mtimes) < datetime(2026, 9, 22, 14, 42, 16).timestamp(), temporal)

    report = {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "reviewed_alignment_pairs_independently_recomputed": len(reviewed),
        "public_files_scanned": len(public_files),
        "raw_or_checkpoint_files_found": forbidden_files,
        "credential_pattern_files": credential_hits,
        "email_address_files": email_hits,
        "model_training_or_inference": "NOT RUN",
        "new_bootstrap_or_prediction_metric_evaluation": "NOT RUN",
    }
    write_json_new(output / "independent_verification_report.json", report)
    print(json.dumps({"status": report["status"], "checks": len(checks), "failures": len(failures), "reviewed_pairs": len(reviewed)}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
