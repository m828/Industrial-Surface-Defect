#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from run_neu_source_audit import Audit, D4_NAMES, scope_of

REASON_LABELS = [("exhaustive_image_gate", 1), ("exhaustive_image_secondary_gate", 2), ("exhaustive_gt_gate", 4), ("exhaustive_gt_secondary_gate", 8), ("exact_gt_d4", 16), ("exact_image_d4", 32)]


def none_if_nan(value):
    return None if value is None or not np.isfinite(value) else float(value)


def batch_pearson(a, batch):
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(batch, dtype=np.float64).reshape(len(batch), -1)
    x = x - x.mean()
    y = y - y.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.dot(x, x) * np.sum(y * y, axis=1))
    out = np.full(len(y), np.nan, dtype=np.float64)
    valid = denominator > 0
    out[valid] = (y[valid] @ x) / denominator[valid]
    return out


def batch_gt_metrics(a, batch):
    x = np.asarray(a, dtype=np.uint8)
    y = np.asarray(batch, dtype=np.uint8)
    agreement = np.mean(x[None, :, :] == y, axis=(1, 2))
    fa = x != 0
    fb = y != 0
    union = np.logical_or(fa[None, :, :], fb).sum(axis=(1, 2))
    intersection = np.logical_and(fa[None, :, :], fb).sum(axis=(1, 2))
    iou = np.full(len(y), np.nan, dtype=np.float64)
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]
    return agreement, iou


class ExhaustiveAudit(Audit):
    def retrieve_candidates(self):
        hamming_path = Path("/workspace/cjig_v10_7_p0_private/oriented_hamming_private_v2.npz")
        with np.load(hamming_path) as z:
            assert np.array_equal(z["keys"], np.array(self.keys))
            img_min_ph, img_min_dh, img_best_t = z["img_min_ph"], z["img_min_dh"], z["img_best_t"]
            gt_min_ph, gt_min_dh, gt_best_t = z["gt_min_ph"], z["gt_min_dh"], z["gt_best_t"]
        n = len(self.keys)
        image_prefilter = (img_min_ph <= 16) & (img_min_dh <= 24) | (img_min_ph <= 10) | (img_min_dh <= 10)
        gt_prefilter = (gt_min_ph <= 16) | (gt_min_dh <= 12)
        union_prefilter = image_prefilter | gt_prefilter
        upper = np.triu(np.ones((n, n), dtype=bool), 1)
        prefilter_count = int(np.count_nonzero(union_prefilter & upper))
        image_prefilter_count = int(np.count_nonzero(image_prefilter & upper))
        gt_prefilter_count = int(np.count_nonzero(gt_prefilter & upper))
        self.descriptor_cache = {}
        self.candidate_reasons = {}
        eligible_prefilter = 0
        for i in range(n - 1):
            js = np.flatnonzero(union_prefilter[i, i + 1:]) + i + 1
            if len(js) == 0:
                continue
            image_positions = np.flatnonzero(image_prefilter[i, js])
            gt_positions = np.flatnonzero(gt_prefilter[i, js])
            image_corr = np.full(len(js), np.nan)
            gt_agreement = np.full(len(js), np.nan)
            gt_iou = np.full(len(js), np.nan)
            if len(image_positions):
                for t in np.unique(img_best_t[i, js[image_positions]]):
                    positions = image_positions[img_best_t[i, js[image_positions]] == t]
                    image_corr[positions] = batch_pearson(self.arrays["img_thumbs"][i, t], self.arrays["img_thumbs"][js[positions], 0])
            if len(gt_positions):
                for t in np.unique(gt_best_t[i, js[gt_positions]]):
                    positions = gt_positions[gt_best_t[i, js[gt_positions]] == t]
                    a = np.rint(self.arrays["gt_thumbs"][i, t]).astype(np.uint8)
                    b = np.rint(self.arrays["gt_thumbs"][js[positions], 0]).astype(np.uint8)
                    gt_agreement[positions], gt_iou[positions] = batch_gt_metrics(a, b)
            ph = img_min_ph[i, js]
            dh = img_min_dh[i, js]
            image_gate = ((ph <= 16) & (dh <= 24) & (image_corr >= 0.85)) | ((ph <= 10) & (image_corr >= 0.80)) | ((dh <= 10) & (image_corr >= 0.88)) | (image_corr >= 0.965)
            gt_gate = ((gt_min_ph[i, js] <= 16) | (gt_min_dh[i, js] <= 12)) & ((gt_iou >= 0.85) | (gt_agreement >= 0.995))
            accepted = image_gate | gt_gate
            eligible_prefilter += int(np.count_nonzero(accepted))
            for position in np.flatnonzero(accepted):
                j = int(js[position])
                pair = (i, j)
                bits = (1 if image_gate[position] else 0) | (4 if gt_gate[position] else 0)
                self.candidate_reasons[pair] = self.candidate_reasons.get(pair, 0) | bits
                self.descriptor_cache[pair] = {
                    "key_a": self.keys[i],
                    "key_b": self.keys[j],
                    "split_a": self.records[self.keys[i]]["split"],
                    "split_b": self.records[self.keys[j]]["split"],
                    "scope": scope_of(self.keys[i], self.keys[j], self.records),
                    "min_image_phash": int(ph[position]),
                    "min_image_dhash": int(dh[position]),
                    "min_gt_phash": int(gt_min_ph[i, j]),
                    "min_gt_dhash": int(gt_min_dh[i, j]),
                    "best_image_thumbnail_transform": D4_NAMES[int(img_best_t[i, j])],
                    "best_gt_thumbnail_transform": D4_NAMES[int(gt_best_t[i, j])],
                    "image_thumbnail_pearson": none_if_nan(image_corr[position]),
                    "gt_thumbnail_foreground_iou": none_if_nan(gt_iou[position]),
                    "gt_thumbnail_pixel_agreement": none_if_nan(gt_agreement[position]),
                    "image_review_gate": bool(image_gate[position]),
                    "gt_review_gate": bool(gt_gate[position]),
                    "exact_image_d4": False,
                    "exact_gt_d4": False,
                    "exact_image_transforms": [],
                    "exact_gt_transforms": [],
                    "review_priority": float(max(image_corr[position] if np.isfinite(image_corr[position]) else -1.0, gt_iou[position] if np.isfinite(gt_iou[position]) else -1.0)),
                    "hash_priority": int(ph[position] + dh[position] + gt_min_ph[i, j] + gt_min_dh[i, j]),
                }
            if (i + 1) % 500 == 0:
                print(f"exhaustive thumbnail gates row {i + 1}/{n}; candidates {len(self.candidate_reasons)}", flush=True)
        prefilter_candidates = set(self.candidate_reasons)
        for row in self.d4_rows:
            a, b = sorted((self.keys.index(row["key_a"]), self.keys.index(row["key_b"])))
            bit = 16 if row["relation_type"] == "GT_D4_EXACT" else 32
            self.candidate_reasons[(a, b)] = self.candidate_reasons.get((a, b), 0) | bit
            self.descriptor_cache[(a, b)] = super().candidate_descriptor_metrics(a, b)
        cap = self.config["candidate_retrieval"]["max_retrieved_unique_pairs"]
        if len(self.candidate_reasons) > cap:
            self.overflow["retrieved_pair_cap"] = True
            raise RuntimeError(f"Eligible candidate count {len(self.candidate_reasons)} exceeds frozen safety cap {cap}; no near conclusion published")
        self.exhaustive_stats = {
            "method": "exhaustive all-pairs oriented Hamming prefilter followed by vectorized thumbnail gates; no LSH",
            "hamming_feature_file": str(hamming_path),
            "all_unordered_nonself_pairs": n * (n - 1) // 2,
            "image_hamming_prefilter_pairs": image_prefilter_count,
            "gt_hamming_prefilter_pairs": gt_prefilter_count,
            "union_hamming_prefilter_pairs": prefilter_count,
            "prefilter_pairs_passing_thumbnail_gates_before_exact_seeds": eligible_prefilter,
            "prefilter_pairs_rejected_by_thumbnail_gates": prefilter_count - eligible_prefilter,
            "unique_prefilter_candidate_pairs_before_exact_seeds": len(prefilter_candidates),
            "exact_d4_seed_pairs_added_or_updated": len(self.d4_rows),
            "unique_candidate_pairs_after_exact_seeds": len(self.candidate_reasons),
            "eligible_candidate_safety_cap": cap,
            "precandidate_storage": "The private oriented distance matrices and aggregate rejection counts preserve the all-pairs screen; only thumbnail-gate candidates and exact-D4 seeds enter the public candidate CSV.",
        }
        print(json.dumps(self.exhaustive_stats, ensure_ascii=False), flush=True)

    def candidate_descriptor_metrics(self, a, b):
        row = dict(self.descriptor_cache[(a, b)])
        bits = self.candidate_reasons[(a, b)]
        row["retrieval_reasons"] = [name for name, bit in REASON_LABELS if bits & bit]
        return row

    def near_screen(self):
        super().near_screen()
        self.near_summary["exhaustive_screen"] = self.exhaustive_stats
        self.near_summary["retrieval_reason_pair_counts"] = {
            name: sum(bool(self.candidate_reasons[(self.keys.index(r["key_a"]), self.keys.index(r["key_b"]))] & bit) for r in self.candidate_rows)
            for name, bit in REASON_LABELS
        }
        self.emit_json("near_duplicate_summary.json", self.near_summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/workspace/Industrial Surface Defect")
    parser.add_argument("--repo-root", default="/workspace/Industrial Surface Defect/Industrial-Surface-Defect")
    parser.add_argument("--upstream-root", default="/workspace/cjig_v10_7_p0_private/upstream_archive")
    parser.add_argument("--output-dir", default="/workspace/Industrial Surface Defect/Industrial-Surface-Defect/experiments/audit/cjig_v10_7_p0/exhaustive_hamming")
    parser.add_argument("--private-dir", default="/workspace/cjig_v10_7_p0_private/analysis_run_005_exhaustive")
    args = parser.parse_args()
    ExhaustiveAudit(args).run()


if __name__ == "__main__":
    main()
