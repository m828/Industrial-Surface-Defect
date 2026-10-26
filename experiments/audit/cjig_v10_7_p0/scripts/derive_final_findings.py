#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

PRIORITY = {
    frozenset(("test:000301", "train:001281")),
    frozenset(("test:000301", "train:002151")),
    frozenset(("test:000301", "train:003021")),
    frozenset(("test:000305", "train:001285")),
    frozenset(("test:000305", "train:002155")),
    frozenset(("test:000305", "train:003025")),
}


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_csv_new(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: encode(row.get(k)) for k in fields})


def encode(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ";".join(str(x) for x in value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/workspace/Industrial Surface Defect/Industrial-Surface-Defect/experiments/audit/cjig_v10_7_p0/cap_resolution")
    args = parser.parse_args()
    root = Path(args.input_dir)
    manifest = read_csv(root / "full_image_identity_manifest.csv")
    d4 = read_csv(root / "d4_transform_duplicate_pairs.csv")
    candidates = read_csv(root / "near_duplicate_candidates.csv")
    near = load_json(root / "near_duplicate_summary.json")
    families = load_json(root / "source_family_summary.json")
    exact = load_json(root / "exact_duplicate_summary.json")
    d4_summary = load_json(root / "d4_duplicate_summary.json")
    assert len(manifest) == 4470
    assert len({(r["key_a"], r["key_b"]) for r in candidates}) == len(candidates)
    assert all(r["review_status"] == "FULLY_REVIEWED" for r in candidates)

    groups = defaultdict(list)
    for row in manifest:
        groups[row["gt_d4_canonical_sha256"]].append(row)
    image_d4_pairs = {frozenset((r["key_a"], r["key_b"])) for r in d4 if r["relation_type"] == "IMAGE_D4_EXACT"}
    high_pairs = {frozenset((r["key_a"], r["key_b"])) for r in candidates if r["high_confidence_shared_source_candidate"] == "true"}
    candidate_map = {frozenset((r["key_a"], r["key_b"])): r for r in candidates}
    group_rows = []
    for canonical, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        keys = sorted(r["key"] for r in members)
        pairs = list(combinations(keys, 2))
        image_exact_n = sum(frozenset(p) in image_d4_pairs for p in pairs)
        high_n = sum(frozenset(p) in high_pairs for p in pairs)
        image_evidence_pairs = {p for p in pairs if frozenset(p) in image_d4_pairs or frozenset(p) in high_pairs}
        group_rows.append({
            "gt_d4_canonical_sha256": canonical,
            "group_size": len(keys),
            "train_records": sum(r["split"] == "train" for r in members),
            "test_records": sum(r["split"] == "test" for r in members),
            "unordered_pairs": len(pairs),
            "image_d4_exact_pairs_within_group": image_exact_n,
            "high_confidence_image_pairs_within_group": high_n,
            "pairs_without_image_d4_or_high_confidence_evidence": len(pairs) - len(image_evidence_pairs),
            "members": keys,
            "interpretation": "GT grouping only; image/acquisition evidence must be evaluated independently",
        })
    write_csv_new(root / "gt_d4_canonical_groups.csv", group_rows)

    priority_rows = [candidate_map[p] for p in sorted(PRIORITY, key=lambda x: tuple(sorted(x))) if p in candidate_map]
    write_csv_new(root / "priority_rotation_pair_metrics.csv", priority_rows)
    exact_cross_d4 = [r for r in d4 if r["relation_type"] == "IMAGE_D4_EXACT" and r["scope"] == "train/test"]
    exact_cross_metrics = [candidate_map[frozenset((r["key_a"], r["key_b"]))] for r in exact_cross_d4]
    write_csv_new(root / "cross_split_exact_image_pair_metrics.csv", exact_cross_metrics)

    by_sample = defaultdict(dict)
    for row in manifest:
        by_sample[row["sample_id"]][row["split"]] = row
    shared_ids = sorted(k for k, v in by_sample.items() if len(v) == 2)
    same_name_file_equal = sum(by_sample[k]["train"]["image_file_sha256"] == by_sample[k]["test"]["image_file_sha256"] for k in shared_ids)
    same_name_pixel_equal = sum(by_sample[k]["train"]["image_pixel_sha256"] == by_sample[k]["test"]["image_pixel_sha256"] for k in shared_ids)
    findings = {
        "record_counts": {"train": 3630, "test": 840, "total": 4470},
        "same_sample_id_across_splits": {"ids": len(shared_ids), "image_file_equal": same_name_file_equal, "image_pixel_equal": same_name_pixel_equal},
        "exact_identity": {
            "image_file_duplicate_pairs": exact["relations"]["IMAGE_FILE_BYTES_IDENTICAL"]["pairs_by_scope"],
            "image_pixel_duplicate_pairs": exact["relations"]["IMAGE_DECODED_PIXELS_IDENTICAL"]["pairs_by_scope"],
            "gt_file_duplicate_pairs": exact["relations"]["GT_FILE_BYTES_IDENTICAL"]["pairs_by_scope"],
            "gt_pixel_duplicate_pairs": exact["relations"]["GT_DECODED_PIXELS_IDENTICAL"]["pairs_by_scope"],
            "same_image_different_gt_pairs": exact["derived"]["same_image_different_gt_pairs"],
            "same_gt_different_image_pairs": exact["derived"]["same_gt_different_image_pairs"],
        },
        "d4": d4_summary,
        "cross_split_exact_image_pairs": [{"key_a": r["key_a"], "key_b": r["key_b"], "transforms": r["transforms_a_to_b"], "gt_diff_pixels": candidate_map[frozenset((r["key_a"], r["key_b"]))]["gt_diff_pixels"], "gt_foreground_iou": candidate_map[frozenset((r["key_a"], r["key_b"]))]["gt_foreground_iou"]} for r in exact_cross_d4],
        "gt_d4_group_evidence": {
            "nonsingleton_groups": len(group_rows),
            "groups_with_image_d4_pair": sum(r["image_d4_exact_pairs_within_group"] > 0 for r in group_rows),
            "groups_with_high_confidence_image_pair": sum(r["high_confidence_image_pairs_within_group"] > 0 for r in group_rows),
            "pairs_without_image_d4_or_high_confidence_evidence": sum(r["pairs_without_image_d4_or_high_confidence_evidence"] for r in group_rows),
            "interpretation": "GT canonical groups are not acquisition families.",
        },
        "near_duplicate_screen": {
            "candidate_pairs": len(candidates),
            "fully_reviewed_pairs": sum(r["review_status"] == "FULLY_REVIEWED" for r in candidates),
            "remaining_cap_not_reviewed": sum(r["review_status"] == "CAP_NOT_REVIEWED" for r in candidates),
            "high_confidence_pairs": near["confirmed_high_confidence_pairs_including_exact_image_d4"],
            "high_confidence_cross_split_pairs": near["high_confidence_cross_split_pairs"],
            "high_confidence_cross_split_test_record_count": len(near["high_confidence_cross_split_test_records"]),
            "other_high_confidence_test_records_beyond_000301_000305_count": len(near["other_high_confidence_test_records_beyond_000301_000305"]),
            "category_counts": near["reviewed_category_counts"],
            "cap_resolution": near["cap_resolution"],
            "exhaustive_screen": near["exhaustive_screen"],
        },
        "source_families": families,
        "priority_six_rotation_pairs": {
            "count": len(priority_rows),
            "categories": sorted({r["evidence_category"] for r in priority_rows}),
            "rgb_pearson_range": [min(float(r["rgb_pearson"]) for r in priority_rows), max(float(r["rgb_pearson"]) for r in priority_rows)],
            "rgb_mae_range": [min(float(r["rgb_mae"]) for r in priority_rows), max(float(r["rgb_mae"]) for r in priority_rows)],
            "gray_ssim_range": [min(float(r["gray_ssim"]) for r in priority_rows), max(float(r["gray_ssim"]) for r in priority_rows)],
            "all_gt_exact": all(int(r["gt_diff_pixels"]) == 0 for r in priority_rows),
            "any_high_confidence": any(r["high_confidence_shared_source_candidate"] == "true" for r in priority_rows),
            "interpretation": "Exact GT alignment plus near image similarity, but the frozen high-confidence image criteria are not met because SSIM is below 0.95.",
        },
        "status_before_independent_verifier": {
            "actual_training_input_identity": "VERIFIED",
            "full_dataset_transform_screen": "PENDING_INDEPENDENT_VERIFICATION",
            "cross_split_source_independence": families["cross_split_source_independence"],
            "original_acquisition_provenance": "PARTIAL",
            "manuscript_scientific_readiness": "HOLD FOR SOURCE EVIDENCE",
        },
    }
    with open(root / "key_findings.json", "x", encoding="utf-8") as f:
        json.dump(findings, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": "COMPLETED", "gt_d4_groups": len(group_rows), "candidates": len(candidates), "priority_rows": len(priority_rows), "exact_cross_image_pairs": len(exact_cross_metrics)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
