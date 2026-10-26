#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from run_neu_source_audit import Audit, read_csv

BOOL_FIELDS = {
    "image_review_gate", "gt_review_gate", "exact_image_d4", "exact_gt_d4", "candidate_for_full_review",
    "near_candidate_excluding_exact_image", "high_confidence_shared_source_candidate", "image_high_similarity_any_orientation",
    "image_near_any_orientation", "gt_near_any_orientation", "gt_exact_any_orientation", "includes_identity_r0",
}
INT_FIELDS = {
    "min_image_phash", "min_image_dhash", "min_gt_phash", "min_gt_dhash", "transform_index", "category_code",
    "gt_diff_pixels", "list_index", "image_file_bytes", "gt_file_bytes", "image_height", "image_width", "image_channels",
    "gt_height", "gt_width",
}
FLOAT_FIELDS = {
    "image_thumbnail_pearson", "gt_thumbnail_foreground_iou", "gt_thumbnail_pixel_agreement", "review_priority", "hash_priority",
    "rgb_pearson", "rgb_mae", "rgb_mse", "rgb_psnr", "gray_pearson", "gray_mae", "gray_psnr", "gray_ssim",
    "gt_pixel_agreement", "gt_foreground_iou", "gt_present_class_macro_iou",
}
LIST_FIELDS = {"retrieval_reasons", "exact_image_transforms", "exact_gt_transforms", "orientations_reviewed", "transforms_a_to_b", "nonidentity_transforms", "rotation_only_nonidentity_transforms", "mirror_transforms"}
COPY_FILES = [
    "full_image_identity_manifest.csv", "exact_duplicate_pairs.csv", "exact_duplicate_summary.json",
    "d4_transform_duplicate_pairs.csv", "d4_duplicate_summary.json", "positive_control_verification.json",
    "detector_self_tests.json", "actual_priority_training_input_identity.csv", "priority_sample_identity.csv",
    "upstream_identity_summary.json", "run_environment.json",
]


def typed(row):
    out = dict(row)
    for field in BOOL_FIELDS:
        if field in out:
            out[field] = out[field] is True or out[field] == "true"
    for field in INT_FIELDS:
        if field in out and out[field] not in [None, ""]:
            out[field] = int(float(out[field]))
    for field in FLOAT_FIELDS:
        if field in out:
            out[field] = None if out[field] in [None, ""] else float(out[field])
    for field in LIST_FIELDS:
        if field in out:
            out[field] = [] if out[field] in [None, ""] else [x for x in out[field].split(";") if x]
    return out


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


BACKLOG_AUDIT = None


def review_backlog_row(payload):
    index, row = payload
    return index, BACKLOG_AUDIT.full_review(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/workspace/Industrial Surface Defect")
    parser.add_argument("--repo-root", default="/workspace/Industrial Surface Defect/Industrial-Surface-Defect")
    parser.add_argument("--upstream-root", default="/workspace/cjig_v10_7_p0_private/upstream_archive")
    parser.add_argument("--output-dir", default="/workspace/Industrial Surface Defect/Industrial-Surface-Defect/experiments/audit/cjig_v10_7_p0/cap_resolution")
    parser.add_argument("--private-dir", default="/workspace/cjig_v10_7_p0_private/analysis_run_007_cap_resolution_parallel")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    audit = Audit(args)
    if audit.private.exists():
        raise FileExistsError(audit.private)
    audit.private.mkdir(parents=True)
    audit.staging.mkdir()
    parent = audit.output.parent / "exhaustive_hamming"
    manifest = [typed(r) for r in read_csv(parent / "full_image_identity_manifest.csv")]
    audit.manifest = manifest
    audit.keys = [r["key"] for r in manifest]
    for row in manifest:
        key = row["key"]
        audit.records[key] = dict(row)
        audit.records[key].update({
            "image_path": (audit.project / row["image_relative_path"]).resolve(),
            "gt_path": (audit.project / row["gt_relative_path"]).resolve(),
            "upstream_image_path": (audit.upstream_checkout / row["upstream_image_relative_path"]).resolve(),
            "upstream_gt_path": (audit.upstream_checkout / row["upstream_gt_relative_path"]).resolve(),
        })
    audit.d4_rows = [typed(r) for r in read_csv(parent / "d4_transform_duplicate_pairs.csv")]
    parent_candidates = [typed(r) for r in read_csv(parent / "near_duplicate_candidates.csv")]
    assert len(parent_candidates) == audit.config["maximum_merged_candidate_rows"]
    pair_ids = [(r["key_a"], r["key_b"]) for r in parent_candidates]
    assert len(pair_ids) == len(set(pair_ids)), "Parent candidate IDs are not unique"
    backlog = [r for r in parent_candidates if r["review_status"] == "CAP_NOT_REVIEWED" and r["filter_reason"] == "REVIEW_CAP"]
    reviewed_parent = [r for r in parent_candidates if r["review_status"] == "FULLY_REVIEWED"]
    assert len(backlog) == audit.config["expected_backlog_count"], len(backlog)
    assert len(backlog) + len(reviewed_parent) == len(parent_candidates)
    required_images = {k for r in backlog for k in [r["key_a"], r["key_b"]]}
    for index, key in enumerate(sorted(required_images), 1):
        r = audit.records[key]
        with Image.open(r["image_path"]) as im:
            audit.img_arrays[key] = np.asarray(im.convert("RGB"), dtype=np.uint8)
        with Image.open(r["gt_path"]) as im:
            audit.gt_arrays[key] = np.asarray(im.convert("L"), dtype=np.uint8)
        if index % 250 == 0:
            print(f"backlog source records loaded {index}/{len(required_images)}", flush=True)
    assert args.workers >= 1
    global BACKLOG_AUDIT
    BACKLOG_AUDIT = audit
    reviewed_by_index = {}
    payloads = list(enumerate(backlog))
    if args.workers > 1:
        with mp.get_context("fork").Pool(args.workers) as pool:
            results = pool.imap_unordered(review_backlog_row, payloads, chunksize=16)
            for completed, (index, reviewed) in enumerate(results, 1):
                reviewed_by_index[index] = reviewed
                if completed % 250 == 0:
                    print(f"backlog full reviews {completed}/{len(backlog)}", flush=True)
    else:
        for completed, (index, reviewed) in enumerate(map(review_backlog_row, payloads), 1):
            reviewed_by_index[index] = reviewed
            if completed % 250 == 0:
                print(f"backlog full reviews {completed}/{len(backlog)}", flush=True)
    reviewed_new = [reviewed_by_index[index] for index in range(len(backlog))]
    merged = reviewed_parent + reviewed_new
    merged.sort(key=lambda r: (r["key_a"], r["key_b"]))
    assert len(merged) == len(parent_candidates)
    assert {(r["key_a"], r["key_b"]) for r in merged} == set(pair_ids)
    assert all(r["review_status"] == "FULLY_REVIEWED" for r in merged)
    assert len({(r["key_a"], r["key_b"]) for r in merged}) == len(merged)
    audit.candidate_rows = merged
    audit.emit_csv("near_duplicate_candidates.csv", merged)
    for name in COPY_FILES:
        shutil.copyfile(parent / name, audit.staging / name)
        audit.generated.append(name)
    parent_summary = load_json(parent / "near_duplicate_summary.json")
    high_cross = [r for r in merged if r["scope"] == "train/test" and r["high_confidence_shared_source_candidate"]]
    category_counts = Counter((r["scope"], r["evidence_category"]) for r in merged)
    reason_counts = Counter(reason for r in merged for reason in r["retrieval_reasons"])
    near_reviewed = [r for r in merged if r["near_candidate_excluding_exact_image"]]
    audit.overflow = dict(parent_summary["retrieval_overflow"])
    audit.overflow.update({
        "first_pass_cap_not_reviewed": len(backlog),
        "second_pass_backlog_reviewed": len(reviewed_new),
        "remaining_cap_not_reviewed": 0,
        "new_candidate_pairs_retrieved_in_second_pass": 0,
    })
    audit.near_summary = {
        "authoritative_merged_result": True,
        "parent_result": "../exhaustive_hamming/near_duplicate_summary.json",
        "retrieved_unique_pairs": len(merged),
        "retrieval_reason_pair_counts": dict(sorted(reason_counts.items())),
        "eligible_for_full_review_before_caps": len(parent_candidates),
        "fully_reviewed_pairs": len(merged),
        "near_candidates_fully_reviewed_excluding_exact_image": len(near_reviewed),
        "confirmed_high_confidence_pairs_including_exact_image_d4": sum(bool(r["high_confidence_shared_source_candidate"]) for r in merged),
        "confirmed_high_confidence_near_pairs_excluding_exact_image": sum(bool(r["high_confidence_shared_source_candidate"] and r["near_candidate_excluding_exact_image"]) for r in merged),
        "reviewed_category_counts": [{"scope": scope, "category": category, "pairs": count} for (scope, category), count in sorted(category_counts.items())],
        "high_confidence_cross_split_pairs": len(high_cross),
        "high_confidence_cross_split_test_records": sorted({r["key_a"] if r["split_a"] == "test" else r["key_b"] for r in high_cross}),
        "other_high_confidence_test_records_beyond_000301_000305": sorted({r["key_a"] if r["split_a"] == "test" else r["key_b"] for r in high_cross} - {"test:000301", "test:000305"}),
        "retrieval_overflow": audit.overflow,
        "screen_complete_under_frozen_config": False,
        "screen_complete_after_declared_cap_resolution": True,
        "exhaustive_screen": parent_summary["exhaustive_screen"],
        "cap_resolution": {
            "first_pass_fully_reviewed": len(reviewed_parent),
            "second_pass_fully_reviewed": len(reviewed_new),
            "total_unique_candidates": len(merged),
            "remaining_unreviewed_candidates": 0,
            "new_pair_ids": 0,
            "duplicate_pair_ids": 0,
            "thresholds_changed": False,
        },
        "interpretation_boundary": "Every candidate was reviewed in two declared passes. A high-confidence candidate is still not acquisition-level proof; GT-only similarity never establishes an image-source family.",
    }
    audit.emit_json("near_duplicate_summary.json", audit.near_summary)
    additional = [r for r in high_cross if not {r["key_a"], r["key_b"]}.issubset({"train:001281", "train:001285", "train:002151", "train:002155", "train:003021", "train:003025", "test:000301", "test:000305"})]
    audit.emit_csv("additional_cross_split_candidates.csv", additional)
    audit.upstream_summary = load_json(parent / "upstream_identity_summary.json")
    audit.positive_controls = load_json(parent / "positive_control_verification.json")
    audit.build_source_graph()
    audit.emit_execution("COMPLETED_CAP_RESOLUTION")
    audit.publish()
    print(json.dumps({"status": "COMPLETED", "parent_reviewed": len(reviewed_parent), "backlog_reviewed": len(reviewed_new), "total": len(merged), "high_cross": len(high_cross)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
