#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np


def compute_family(ph, dh, block_size=128):
    n = ph.shape[0]
    min_ph = np.full((n, n), 127, dtype=np.uint8)
    min_dh = np.full((n, n), 127, dtype=np.uint8)
    best_sum = np.full((n, n), 127, dtype=np.uint8)
    best_ph = np.full((n, n), 127, dtype=np.uint8)
    best_dh = np.full((n, n), 127, dtype=np.uint8)
    best_t = np.zeros((n, n), dtype=np.uint8)
    for t in range(8):
        target_ph = ph[None, :, 0]
        target_dh = dh[None, :, 0]
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            pd = np.bitwise_count(ph[start:end, None, t] ^ target_ph)
            dd = np.bitwise_count(dh[start:end, None, t] ^ target_dh)
            np.minimum(min_ph[start:end], pd, out=min_ph[start:end])
            np.minimum(min_dh[start:end], dd, out=min_dh[start:end])
            total = pd + dd
            current = best_sum[start:end]
            current_ph = best_ph[start:end]
            current_dh = best_dh[start:end]
            update = (total < current) | ((total == current) & (pd < current_ph)) | ((total == current) & (pd == current_ph) & (dd < current_dh))
            current[update] = total[update]
            current_ph[update] = pd[update]
            current_dh[update] = dd[update]
            best_t[start:end][update] = t
        print(f"orientation {t} complete", flush=True)
    return min_ph, min_dh, best_t


def upper_count(mask):
    return int(np.count_nonzero(np.triu(mask, 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors", default="/workspace/cjig_v10_7_p0_private/analysis_run_002/descriptors_private.npz")
    parser.add_argument("--private-output", default="/workspace/cjig_v10_7_p0_private/oriented_hamming_private.npz")
    parser.add_argument("--count-output", default="/workspace/cjig_v10_7_p0_private/oriented_hamming_prefilter_counts.json")
    args = parser.parse_args()
    for path in [args.private_output, args.count_output]:
        if Path(path).exists():
            raise FileExistsError(path)
    started = time.time()
    with np.load(args.descriptors) as z:
        keys = z["keys"]
        img_ph, img_dh = z["img_phash"], z["img_dhash"]
        gt_ph, gt_dh = z["gt_phash"], z["gt_dhash"]
    print("compute image oriented Hamming", flush=True)
    img_min_ph, img_min_dh, img_best_t = compute_family(img_ph, img_dh)
    print("compute GT oriented Hamming", flush=True)
    gt_min_ph, gt_min_dh, gt_best_t = compute_family(gt_ph, gt_dh)
    image_prefilter = (img_min_ph <= 16) & (img_min_dh <= 24) | (img_min_ph <= 10) | (img_min_dh <= 10)
    gt_prefilter = (gt_min_ph <= 16) | (gt_min_dh <= 12)
    counts = {
        "method": "exhaustive all record pairs against all eight D4 orientations using NumPy uint64 bitwise_count; no LSH and no candidate-pair enumeration in this count step",
        "records": int(len(keys)),
        "unordered_nonself_pairs": int(len(keys) * (len(keys) - 1) // 2),
        "image_hamming_prefilter_pairs": upper_count(image_prefilter),
        "gt_hamming_prefilter_pairs": upper_count(gt_prefilter),
        "union_hamming_prefilter_pairs": upper_count(image_prefilter | gt_prefilter),
        "image_rule_before_thumbnail": "(min_ph<=16 AND min_dh<=24) OR min_ph<=10 OR min_dh<=10",
        "gt_rule_before_thumbnail": "min_ph<=16 OR min_dh<=12",
        "thumbnail_and_full_review_rules": "unchanged from the frozen parent configuration",
        "wall_seconds": time.time() - started,
        "numpy_version": np.__version__,
        "raw_pixels_in_output": False,
    }
    np.savez(args.private_output, keys=keys, img_min_ph=img_min_ph, img_min_dh=img_min_dh, img_best_t=img_best_t, gt_min_ph=gt_min_ph, gt_min_dh=gt_min_dh, gt_best_t=gt_best_t)
    with open(args.count_output, "x", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(counts, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
