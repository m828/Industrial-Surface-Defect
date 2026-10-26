#!/usr/bin/env python3
import argparse
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import PIL
from PIL import Image
import scipy
import skimage
from skimage.metrics import structural_similarity

D4_NAMES = ["r0", "r90", "r180", "r270", "r0_hflip", "r90_hflip", "r180_hflip", "r270_hflip"]
PRIORITY_TRAIN = ["001281", "001285", "002151", "002155", "003021", "003025"]
PRIORITY_TEST = ["000301", "000305"]
PRIORITY_KEYS = {f"train:{x}" for x in PRIORITY_TRAIN} | {f"test:{x}" for x in PRIORITY_TEST}
EXPECTED_EXACT_PAIRS = [
    ("train:001281", "train:001285"),
    ("train:002151", "train:002155"),
    ("train:003021", "train:003025"),
    ("test:000301", "test:000305"),
]


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat()


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


def d4(a, index):
    k = index % 4
    out = np.rot90(a, k)
    if index >= 4:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def d4_hashes(a):
    return [pixel_sha(d4(a, t)) for t in range(8)]


def phash64(thumbnail):
    x = np.asarray(thumbnail, dtype=np.float64)
    dct = cv2.dct(x)
    low = dct[:8, :8].copy().reshape(-1)[1:]
    med = np.median(low)
    bits = low > med
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def dhash64(thumbnail):
    x = np.asarray(thumbnail, dtype=np.float32)
    small = cv2.resize(x, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def image_descriptors(rgb):
    gray = np.asarray(Image.fromarray(rgb, "RGB").convert("L"), dtype=np.uint8)
    thumbs, phashes, dhashes = [], [], []
    for t in range(8):
        rotated = d4(gray, t)
        thumb = np.asarray(Image.fromarray(rotated, "L").resize((32, 32), Image.Resampling.BILINEAR))
        thumbs.append(thumb)
        phashes.append(phash64(thumb))
        dhashes.append(dhash64(thumb))
    return gray, np.stack(thumbs), np.array(phashes, dtype=np.uint64), np.array(dhashes, dtype=np.uint64)


def gt_descriptors(gt):
    thumbs, phashes, dhashes = [], [], []
    for t in range(8):
        thumb = cv2.resize(d4(gt, t).astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA)
        thumbs.append(thumb)
        phashes.append(phash64(thumb))
        dhashes.append(dhash64(thumb))
    return np.stack(thumbs), np.array(phashes, dtype=np.uint64), np.array(dhashes, dtype=np.uint64)


def hamming(a, b):
    return (int(a) ^ int(b)).bit_count()


def pearson(a, b):
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size == 0:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denom == 0.0:
        return None
    return float(np.dot(x, y) / denom)


def error_metrics(a, b):
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mae = float(np.mean(np.abs(x - y)))
    mse = float(np.mean((x - y) ** 2))
    psnr = None if mse == 0.0 else float(10.0 * np.log10((255.0 ** 2) / mse))
    return {"mae": mae, "mse": mse, "psnr": psnr}


def gt_metrics(a, b):
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    foreground_a = a != 0
    foreground_b = b != 0
    union = np.logical_or(foreground_a, foreground_b).sum()
    intersection = np.logical_and(foreground_a, foreground_b).sum()
    foreground_iou = None if union == 0 else float(intersection / union)
    class_ious = []
    for class_id in range(4):
        ca = a == class_id
        cb = b == class_id
        class_union = np.logical_or(ca, cb).sum()
        if class_union:
            class_ious.append(float(np.logical_and(ca, cb).sum() / class_union))
    return {
        "pixel_agreement": float(np.mean(a == b)),
        "foreground_iou": foreground_iou,
        "present_class_macro_iou": float(np.mean(class_ious)) if class_ious else None,
        "diff_pixels": int(np.count_nonzero(a != b)),
    }


def image_high(row):
    return (
        row["rgb_pearson"] is not None
        and row["rgb_pearson"] >= 0.990
        and row["rgb_mae"] <= 6.0
        and row["rgb_psnr"] is not None
        and row["rgb_psnr"] >= 24.0
        and row["gray_ssim"] is not None
        and row["gray_ssim"] >= 0.95
    )


def image_near(row):
    return (
        row["rgb_pearson"] is not None
        and row["rgb_pearson"] >= 0.950
        and row["rgb_mae"] <= 12.0
        and row["gray_ssim"] is not None
        and row["gray_ssim"] >= 0.80
    )


def gt_near(row):
    return (
        row["gt_foreground_iou"] is not None
        and row["gt_foreground_iou"] >= 0.95
        and row["gt_pixel_agreement"] >= 0.995
    )


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: encode_cell(row.get(k)) for k in fieldnames})


def encode_cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ";".join(str(x) for x in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else str(value)
    return value


def write_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return "Infinity" if x > 0 else str(x)
        return x
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def scope_of(a, b, records):
    sa = records[a]["split"]
    sb = records[b]["split"]
    return "train/test" if sa != sb else f"{sa}/{sb}"


def ordered_pair(a, b):
    return (a, b) if a < b else (b, a)


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def component_stats(records, edges, level):
    uf = UnionFind(records)
    for a, b in edges:
        uf.union(a, b)
    groups = defaultdict(list)
    for key in records:
        groups[uf.find(key)].append(key)
    ordered = sorted((sorted(v) for v in groups.values()), key=lambda v: (v[0], len(v)))
    memberships = []
    non_singletons = []
    for index, members in enumerate(ordered, 1):
        group_id = f"{level}-{index:04d}"
        splits = {records[k]["split"] for k in members}
        for key in members:
            memberships.append({"level": level, "group_id": group_id, "key": key, "split": records[key]["split"], "sample_id": records[key]["sample_id"], "group_size": len(members), "group_scope": "train/test" if len(splits) == 2 else f"{next(iter(splits))}/{next(iter(splits))}"})
        if len(members) > 1:
            non_singletons.append((group_id, members, splits))
    train_nonsingleton = sum(1 for _, members, _ in non_singletons for k in members if records[k]["split"] == "train")
    test_nonsingleton = sum(1 for _, members, _ in non_singletons for k in members if records[k]["split"] == "test")
    return {
        "level": level,
        "components_total_including_singletons": len(ordered),
        "nonsingleton_components": len(non_singletons),
        "nonsingletons_with_train": sum(any(records[k]["split"] == "train" for k in members) for _, members, _ in non_singletons),
        "nonsingletons_with_test": sum(any(records[k]["split"] == "test" for k in members) for _, members, _ in non_singletons),
        "cross_split_nonsingletons": sum(len(splits) == 2 for _, _, splits in non_singletons),
        "train_records_in_nonsingletons": train_nonsingleton,
        "test_records_in_nonsingletons": test_nonsingleton,
        "train_nonsingleton_coverage": train_nonsingleton / 3630,
        "test_nonsingleton_coverage": test_nonsingleton / 840,
        "edges_joined": len(edges),
    }, memberships, non_singletons


class Audit:
    def __init__(self, args):
        self.args = args
        self.project = Path(args.project_root).resolve()
        self.repo = Path(args.repo_root).resolve()
        self.upstream_checkout = Path(args.upstream_root).resolve()
        self.output = Path(args.output_dir).resolve()
        self.private = Path(args.private_dir).resolve()
        with open(self.output / "analysis_config.json", encoding="utf-8") as f:
            self.config = json.load(f)
        self.started = now_iso()
        self.t0 = time.time()
        self.staging = self.private / "staging"
        self.generated = []
        self.records = {}
        self.keys = []
        self.img_arrays = {}
        self.gt_arrays = {}
        self.exact_rows = []
        self.d4_rows = []
        self.candidate_rows = []
        self.positive_controls = None
        self.overflow = {"retrieved_pair_cap": False, "lsh_buckets": [], "per_record_cap_pairs": 0, "global_cap_pairs": 0}

    def path(self, name):
        return self.staging / name

    def emit_csv(self, name, rows, fields=None):
        write_csv(self.path(name), rows, fields)
        self.generated.append(name)

    def emit_json(self, name, value):
        write_json(self.path(name), value)
        self.generated.append(name)

    def find_upstream_dataset(self):
        for root, dirs, files in os.walk(self.upstream_checkout):
            dirs[:] = [d for d in dirs if d != ".git"]
            p = Path(root)
            if all((p / d).is_dir() for d in ["annotations/training", "annotations/test", "images/training", "images/test"]):
                return p
        raise FileNotFoundError("Upstream dataset root with images/annotations was not found")

    def load_lists(self):
        split_specs = [
            ("train", "training", self.project / "new (copy)/dataset/train_neu.txt", 3630),
            ("test", "test", self.project / "new (copy)/dataset/test_neu.txt", 840),
        ]
        local_root = self.project / "new (copy)/dataset"
        self.upstream_dataset = self.find_upstream_dataset()
        for split, source_dir, list_path, expected_n in split_specs:
            with open(list_path, encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == expected_n, (list_path, len(lines))
            assert len(set(lines)) == expected_n
            for line_index, line in enumerate(lines):
                fields = line.split()
                assert len(fields) == 2, (list_path, line_index + 1, line)
                gt_rel, img_rel = fields
                gt_path = (local_root / gt_rel).resolve()
                img_path = (local_root / img_rel).resolve()
                assert gt_path.is_file() and img_path.is_file()
                assert Path(gt_rel).stem == Path(img_rel).stem
                sample_id = Path(img_rel).stem
                key = f"{split}:{sample_id}"
                assert key not in self.records
                assert Path(gt_rel).parts[2] == source_dir and Path(img_rel).parts[2] == source_dir
                upstream_gt = self.upstream_dataset / Path(*Path(gt_rel).parts[1:])
                upstream_img = self.upstream_dataset / Path(*Path(img_rel).parts[1:])
                self.records[key] = {
                    "key": key,
                    "split": split,
                    "sample_id": sample_id,
                    "source_split_dir": source_dir,
                    "list_index": line_index + 1,
                    "list_path": str(list_path.relative_to(self.project)),
                    "image_path": img_path,
                    "gt_path": gt_path,
                    "image_relative_path": str(img_path.relative_to(self.project)),
                    "gt_relative_path": str(gt_path.relative_to(self.project)),
                    "upstream_image_path": upstream_img,
                    "upstream_gt_path": upstream_gt,
                    "upstream_image_relative_path": str(upstream_img.relative_to(self.upstream_checkout)),
                    "upstream_gt_relative_path": str(upstream_gt.relative_to(self.upstream_checkout)),
                }
                self.keys.append(key)
        assert len(self.keys) == 4470
        for split in ["train", "test"]:
            split_records = [r for r in self.records.values() if r["split"] == split]
            assert len({r["image_path"] for r in split_records}) == len(split_records)
            assert len({r["gt_path"] for r in split_records}) == len(split_records)
        self.local_dataset = local_root / "NEUSeg"
        for subdir in ["images/training", "images/test", "annotations/training", "annotations/test"]:
            local_files = {p.name for p in (self.local_dataset / subdir).iterdir() if p.is_file()}
            listed = {p.name for r in self.records.values() for p in [r["image_path"] if subdir.startswith("images") else r["gt_path"]] if p.parent.name == subdir.split("/")[1] and p.parent.parent.name == subdir.split("/")[0]}
            assert local_files == listed, (subdir, len(local_files), len(listed))
            upstream_files = {p.name for p in (self.upstream_dataset / subdir).iterdir() if p.is_file()}
            assert upstream_files == local_files, (subdir, len(upstream_files), len(local_files))

    def build_manifest(self):
        manifest = []
        arrays = {
            "keys": np.array(self.keys),
            "img_d4_sha": np.empty((4470, 8), dtype="U64"),
            "gt_d4_sha": np.empty((4470, 8), dtype="U64"),
            "img_thumbs": np.empty((4470, 8, 32, 32), dtype=np.uint8),
            "gt_thumbs": np.empty((4470, 8, 32, 32), dtype=np.float32),
            "img_phash": np.empty((4470, 8), dtype=np.uint64),
            "img_dhash": np.empty((4470, 8), dtype=np.uint64),
            "gt_phash": np.empty((4470, 8), dtype=np.uint64),
            "gt_dhash": np.empty((4470, 8), dtype=np.uint64),
        }
        upstream_counts = Counter()
        for index, key in enumerate(self.keys):
            r = self.records[key]
            r["image_file_sha256"] = sha_file(r["image_path"])
            r["gt_file_sha256"] = sha_file(r["gt_path"])
            with Image.open(r["image_path"]) as im:
                image_format, image_mode_original = im.format, im.mode
                rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
            with Image.open(r["gt_path"]) as im:
                gt_format, gt_mode_original = im.format, im.mode
                gt = np.asarray(im.convert("L"), dtype=np.uint8)
            assert rgb.shape == (200, 200, 3) and gt.shape == (200, 200)
            assert set(np.unique(gt)).issubset({0, 1, 2, 3})
            self.img_arrays[key] = rgb
            self.gt_arrays[key] = gt
            r["image_pixel_sha256"] = pixel_sha(rgb)
            r["gt_pixel_sha256"] = pixel_sha(gt)
            img_d4 = d4_hashes(rgb)
            gt_d4 = d4_hashes(gt)
            arrays["img_d4_sha"][index] = img_d4
            arrays["gt_d4_sha"][index] = gt_d4
            r["image_d4_canonical_sha256"] = min(img_d4)
            r["gt_d4_canonical_sha256"] = min(gt_d4)
            gray, img_thumbs, img_ph, img_dh = image_descriptors(rgb)
            gt_thumbs, gt_ph, gt_dh = gt_descriptors(gt)
            arrays["img_thumbs"][index] = img_thumbs
            arrays["gt_thumbs"][index] = gt_thumbs
            arrays["img_phash"][index] = img_ph
            arrays["img_dhash"][index] = img_dh
            arrays["gt_phash"][index] = gt_ph
            arrays["gt_dhash"][index] = gt_dh
            r["image_phases"] = img_ph
            r["gt_phases"] = gt_ph
            for kind, path_key in [("image", "upstream_image_path"), ("gt", "upstream_gt_path")]:
                upstream_path = r[path_key]
                exists = upstream_path.is_file()
                r[f"upstream_{kind}_exists"] = exists
                if exists:
                    upstream_sha = sha_file(upstream_path)
                    r[f"upstream_{kind}_file_sha256"] = upstream_sha
                    with Image.open(upstream_path) as up:
                        decoded = np.asarray(up.convert("RGB" if kind == "image" else "L"), dtype=np.uint8)
                    r[f"upstream_{kind}_pixel_sha256"] = pixel_sha(decoded)
                    byte_equal = upstream_sha == r[f"{kind}_file_sha256"]
                    pixel_equal = r[f"upstream_{kind}_pixel_sha256"] == r[f"{kind}_pixel_sha256"]
                    r[f"upstream_{kind}_file_bytes_equal"] = byte_equal
                    r[f"upstream_{kind}_pixels_equal"] = pixel_equal
                    upstream_counts[f"{kind}_exists"] += 1
                    upstream_counts[f"{kind}_byte_equal"] += int(byte_equal)
                    upstream_counts[f"{kind}_pixel_equal"] += int(pixel_equal)
                else:
                    r[f"upstream_{kind}_file_sha256"] = None
                    r[f"upstream_{kind}_pixel_sha256"] = None
                    r[f"upstream_{kind}_file_bytes_equal"] = False
                    r[f"upstream_{kind}_pixels_equal"] = False
            label_values = sorted(int(x) for x in np.unique(gt))
            manifest.append({
                "key": key,
                "split": r["split"],
                "sample_id": r["sample_id"],
                "list_index": r["list_index"],
                "list_path": r["list_path"],
                "image_relative_path": r["image_relative_path"],
                "gt_relative_path": r["gt_relative_path"],
                "image_file_bytes": r["image_path"].stat().st_size,
                "gt_file_bytes": r["gt_path"].stat().st_size,
                "image_file_sha256": r["image_file_sha256"],
                "gt_file_sha256": r["gt_file_sha256"],
                "image_pixel_sha256": r["image_pixel_sha256"],
                "gt_pixel_sha256": r["gt_pixel_sha256"],
                "image_d4_canonical_sha256": r["image_d4_canonical_sha256"],
                "gt_d4_canonical_sha256": r["gt_d4_canonical_sha256"],
                "image_format": image_format,
                "image_original_mode": image_mode_original,
                "image_decoded_mode": "RGB",
                "image_height": rgb.shape[0],
                "image_width": rgb.shape[1],
                "image_channels": rgb.shape[2],
                "gt_format": gt_format,
                "gt_original_mode": gt_mode_original,
                "gt_decoded_mode": "L",
                "gt_height": gt.shape[0],
                "gt_width": gt.shape[1],
                "gt_label_values": label_values,
                "gt_label_domain_valid": set(label_values).issubset({0, 1, 2, 3}),
                "image_gt_same_hw": rgb.shape[:2] == gt.shape[:2],
                "upstream_image_relative_path": r["upstream_image_relative_path"],
                "upstream_gt_relative_path": r["upstream_gt_relative_path"],
                "upstream_image_file_sha256": r["upstream_image_file_sha256"],
                "upstream_gt_file_sha256": r["upstream_gt_file_sha256"],
                "upstream_image_pixel_sha256": r["upstream_image_pixel_sha256"],
                "upstream_gt_pixel_sha256": r["upstream_gt_pixel_sha256"],
                "upstream_image_file_bytes_equal": r["upstream_image_file_bytes_equal"],
                "upstream_gt_file_bytes_equal": r["upstream_gt_file_bytes_equal"],
                "upstream_image_pixels_equal": r["upstream_image_pixels_equal"],
                "upstream_gt_pixels_equal": r["upstream_gt_pixels_equal"],
            })
            if (index + 1) % 500 == 0:
                print(f"manifest {index + 1}/4470", flush=True)
        self.manifest = manifest
        self.arrays = arrays
        np.savez_compressed(self.private / "descriptors_private.npz", **arrays)
        self.emit_csv("full_image_identity_manifest.csv", manifest)
        expected_commit = self.config["source_layout"]["upstream_commit"]
        repository = self.config["source_layout"]["upstream_repository"]
        remote_main = None
        remote_error = None
        try:
            remote_main = subprocess.check_output(["git", "ls-remote", repository, "refs/heads/main"], text=True, stderr=subprocess.STDOUT).split()[0]
        except (subprocess.CalledProcessError, IndexError) as exc:
            remote_error = str(exc)
        if (self.upstream_checkout / ".git").exists():
            actual_head = subprocess.check_output(["git", "-C", str(self.upstream_checkout), "rev-parse", "HEAD"], text=True).strip()
            verified_commit = actual_head
            verification_method = "git_checkout_HEAD_plus_ls_remote"
            archive_sha256 = None
        else:
            archive_roots = [p for p in self.upstream_checkout.iterdir() if p.is_dir() and expected_commit in p.name]
            assert len(archive_roots) == 1, archive_roots
            actual_head = None
            verified_commit = expected_commit
            verification_method = "immutable_codeload_commit_archive_plus_ls_remote"
            archive_path = self.upstream_checkout.parent / "upstream_neu_seg_da8cebed.tar.gz"
            archive_sha256 = sha_file(archive_path) if archive_path.is_file() else None
        upstream_summary = {
            "upstream_repository": repository,
            "requested_commit": expected_commit,
            "actual_checkout_head": actual_head,
            "verified_upstream_commit": verified_commit,
            "commit_verification_method": verification_method,
            "remote_main_at_verification": remote_main,
            "remote_main_live_match": remote_main == expected_commit if remote_main is not None else None,
            "remote_verification_error": remote_error,
            "commit_archive_sha256": archive_sha256,
            "commit_exact_match": verified_commit == expected_commit,
            "dataset_root_relative_to_checkout": str(self.upstream_dataset.relative_to(self.upstream_checkout)),
            "records": 4470,
            "image_files_existing": upstream_counts["image_exists"],
            "gt_files_existing": upstream_counts["gt_exists"],
            "image_file_byte_matches": upstream_counts["image_byte_equal"],
            "gt_file_byte_matches": upstream_counts["gt_byte_equal"],
            "image_pixel_matches": upstream_counts["image_pixel_equal"],
            "gt_pixel_matches": upstream_counts["gt_pixel_equal"],
            "all_8940_files_byte_identical": upstream_counts["image_byte_equal"] == 4470 and upstream_counts["gt_byte_equal"] == 4470,
            "interpretation_boundary": "Byte identity connects current local inputs to this fixed upstream tree; it does not establish original camera acquisition IDs or how upstream augmentation/splitting was performed.",
        }
        assert upstream_summary["commit_exact_match"]
        self.emit_json("upstream_identity_summary.json", upstream_summary)
        self.upstream_summary = upstream_summary

    def relation_pairs(self, relation, hash_getter):
        groups = defaultdict(list)
        for key in self.keys:
            groups[hash_getter(self.records[key])].append(key)
        rows = []
        group_stats = Counter()
        for digest, members in sorted(groups.items()):
            if len(members) < 2:
                continue
            splits = {self.records[k]["split"] for k in members}
            group_scope = "train/test" if len(splits) == 2 else f"{next(iter(splits))}/{next(iter(splits))}"
            group_stats[group_scope] += 1
            for a, b in combinations(sorted(members), 2):
                rows.append({
                    "relation_type": relation,
                    "shared_sha256": digest,
                    "key_a": a,
                    "key_b": b,
                    "split_a": self.records[a]["split"],
                    "split_b": self.records[b]["split"],
                    "sample_id_a": self.records[a]["sample_id"],
                    "sample_id_b": self.records[b]["sample_id"],
                    "scope": scope_of(a, b, self.records),
                    "image_file_bytes_equal": self.records[a]["image_file_sha256"] == self.records[b]["image_file_sha256"],
                    "image_pixels_equal": self.records[a]["image_pixel_sha256"] == self.records[b]["image_pixel_sha256"],
                    "gt_file_bytes_equal": self.records[a]["gt_file_sha256"] == self.records[b]["gt_file_sha256"],
                    "gt_pixels_equal": self.records[a]["gt_pixel_sha256"] == self.records[b]["gt_pixel_sha256"],
                })
        return rows, groups, group_stats

    def exact_screen(self):
        relation_specs = [
            ("IMAGE_FILE_BYTES_IDENTICAL", lambda r: r["image_file_sha256"]),
            ("IMAGE_DECODED_PIXELS_IDENTICAL", lambda r: r["image_pixel_sha256"]),
            ("GT_FILE_BYTES_IDENTICAL", lambda r: r["gt_file_sha256"]),
            ("GT_DECODED_PIXELS_IDENTICAL", lambda r: r["gt_pixel_sha256"]),
        ]
        summary = {"relations": {}, "derived": {}}
        for relation, getter in relation_specs:
            rows, groups, group_stats = self.relation_pairs(relation, getter)
            self.exact_rows.extend(rows)
            scope_counts = Counter(row["scope"] for row in rows)
            involved = defaultdict(set)
            for row in rows:
                involved[row["scope"]].update([row["key_a"], row["key_b"]])
            summary["relations"][relation] = {
                "nonsingleton_groups_by_scope": dict(sorted(group_stats.items())),
                "pairs_by_scope": {s: scope_counts.get(s, 0) for s in ["train/train", "test/test", "train/test"]},
                "records_involved_by_scope": {s: len(involved[s]) for s in ["train/train", "test/test", "train/test"]},
                "unique_hashes_total": len(groups),
            }
        self.emit_csv("exact_duplicate_pairs.csv", self.exact_rows)
        image_identity_pairs = [(r["key_a"], r["key_b"]) for r in self.exact_rows if r["relation_type"] == "IMAGE_DECODED_PIXELS_IDENTICAL"]
        gt_identity_pairs = [(r["key_a"], r["key_b"]) for r in self.exact_rows if r["relation_type"] == "GT_DECODED_PIXELS_IDENTICAL"]
        summary["derived"]["same_image_different_gt_pairs"] = sum(self.records[a]["gt_pixel_sha256"] != self.records[b]["gt_pixel_sha256"] for a, b in image_identity_pairs)
        summary["derived"]["same_gt_different_image_pairs"] = sum(self.records[a]["image_pixel_sha256"] != self.records[b]["image_pixel_sha256"] for a, b in gt_identity_pairs)
        summary["derived"]["record_count"] = len(self.records)
        summary["derived"]["all_images_rgb_200x200"] = all(r["image_decoded_mode"] == "RGB" and r["image_height"] == 200 and r["image_width"] == 200 for r in self.manifest)
        summary["derived"]["all_gt_l_200x200_labels_0_to_3"] = all(r["gt_decoded_mode"] == "L" and r["gt_height"] == 200 and r["gt_width"] == 200 and r["gt_label_domain_valid"] for r in self.manifest)
        summary["derived"]["all_image_gt_dimensions_match"] = all(r["image_gt_same_hw"] for r in self.manifest)
        self.exact_summary = summary
        self.emit_json("exact_duplicate_summary.json", summary)

    def d4_screen(self):
        all_rows = []
        summary = {"relations": {}}
        for relation, record_field, canonical_field, array_name in [
            ("IMAGE_D4_EXACT", "image_pixel_sha256", "image_d4_canonical_sha256", "img_d4_sha"),
            ("GT_D4_EXACT", "gt_pixel_sha256", "gt_d4_canonical_sha256", "gt_d4_sha"),
        ]:
            groups = defaultdict(list)
            for key in self.keys:
                groups[self.records[key][canonical_field]].append(key)
            relation_rows = []
            group_stats = Counter()
            for canonical, members in sorted(groups.items()):
                if len(members) < 2:
                    continue
                splits = {self.records[k]["split"] for k in members}
                group_scope = "train/test" if len(splits) == 2 else f"{next(iter(splits))}/{next(iter(splits))}"
                group_stats[group_scope] += 1
                for a, b in combinations(sorted(members), 2):
                    ia, ib = self.keys.index(a), self.keys.index(b)
                    hashes_a = self.arrays[array_name][ia]
                    original_b = self.records[b][record_field]
                    transforms = [D4_NAMES[t] for t in range(8) if hashes_a[t] == original_b]
                    assert transforms, (relation, a, b)
                    row = {
                        "relation_type": relation,
                        "canonical_sha256": canonical,
                        "key_a": a,
                        "key_b": b,
                        "split_a": self.records[a]["split"],
                        "split_b": self.records[b]["split"],
                        "sample_id_a": self.records[a]["sample_id"],
                        "sample_id_b": self.records[b]["sample_id"],
                        "scope": scope_of(a, b, self.records),
                        "transforms_a_to_b": transforms,
                        "includes_identity_r0": "r0" in transforms,
                        "nonidentity_transforms": [x for x in transforms if x != "r0"],
                        "rotation_only_nonidentity_transforms": [x for x in transforms if x in ["r90", "r180", "r270"]],
                        "mirror_transforms": [x for x in transforms if "hflip" in x],
                    }
                    relation_rows.append(row)
                    all_rows.append(row)
            counts = Counter(r["scope"] for r in relation_rows)
            nonidentity_counts = Counter(r["scope"] for r in relation_rows if r["nonidentity_transforms"])
            involved = defaultdict(set)
            for row in relation_rows:
                involved[row["scope"]].update([row["key_a"], row["key_b"]])
            summary["relations"][relation] = {
                "nonsingleton_groups_by_scope": dict(sorted(group_stats.items())),
                "pairs_including_identity_by_scope": {s: counts.get(s, 0) for s in ["train/train", "test/test", "train/test"]},
                "pairs_with_nonidentity_transform_by_scope": {s: nonidentity_counts.get(s, 0) for s in ["train/train", "test/test", "train/test"]},
                "records_involved_by_scope": {s: len(involved[s]) for s in ["train/train", "test/test", "train/test"]},
            }
        self.d4_rows = all_rows
        self.emit_csv("d4_transform_duplicate_pairs.csv", all_rows)
        self.d4_summary = summary
        self.emit_json("d4_duplicate_summary.json", summary)

    def verify_positive_controls(self):
        exact_pair_set = {ordered_pair(r["key_a"], r["key_b"]) for r in self.exact_rows if r["relation_type"] == "IMAGE_DECODED_PIXELS_IDENTICAL"}
        exact_checks = []
        for a, b in EXPECTED_EXACT_PAIRS:
            diff = int(np.count_nonzero(self.gt_arrays[a] != self.gt_arrays[b]))
            exact_checks.append({
                "pair": [a, b],
                "image_decoded_pair_detected": ordered_pair(a, b) in exact_pair_set,
                "image_file_bytes_equal": self.records[a]["image_file_sha256"] == self.records[b]["image_file_sha256"],
                "gt_diff_pixels": diff,
                "pass": ordered_pair(a, b) in exact_pair_set and diff == 580,
            })
        gt_rotation_rows = []
        for row in self.d4_rows:
            if row["relation_type"] != "GT_D4_EXACT" or row["scope"] != "train/test" or not row["rotation_only_nonidentity_transforms"]:
                continue
            pair_keys = {row["key_a"], row["key_b"]}
            if len(pair_keys & {f"test:{x}" for x in PRIORITY_TEST}) == 1 and pair_keys.issubset(PRIORITY_KEYS):
                gt_rotation_rows.append(row)
        per_test = {f"test:{x}": [] for x in PRIORITY_TEST}
        for row in gt_rotation_rows:
            test_key = row["key_a"] if row["split_a"] == "test" else row["key_b"]
            per_test[test_key].append(row)
        rotation_pass = len({ordered_pair(r["key_a"], r["key_b"]) for r in gt_rotation_rows}) == 6 and all(len(rows) == 3 for rows in per_test.values())
        self.positive_controls = {
            "four_exact_image_checks": exact_checks,
            "four_exact_image_pairs_pass": all(x["pass"] for x in exact_checks),
            "six_cross_split_rotation_gt_pairs_detected": len({ordered_pair(r["key_a"], r["key_b"]) for r in gt_rotation_rows}),
            "rotation_pairs": gt_rotation_rows,
            "per_test_rotation_partner_count": {k: len(v) for k, v in per_test.items()},
            "six_rotation_gt_pairs_pass": rotation_pass,
            "overall_pass": all(x["pass"] for x in exact_checks) and rotation_pass,
        }
        self.emit_json("positive_control_verification.json", self.positive_controls)
        priority_rows = [r for r in self.manifest if r["key"] in PRIORITY_KEYS]
        self.emit_csv("priority_sample_identity.csv", priority_rows)
        if not self.positive_controls["overall_pass"]:
            raise RuntimeError("Known positive controls failed; near-duplicate stage blocked. Inspect private staging outputs.")

    def detector_self_tests(self):
        key = "test:000301"
        index = self.keys.index(key)
        original = self.img_arrays[key]
        buffer = io.BytesIO()
        Image.fromarray(d4(original, 1), "RGB").save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        with Image.open(buffer) as im:
            transformed = np.asarray(im.convert("RGB"), dtype=np.uint8)
        _, thumbs, ph, dh = image_descriptors(transformed)
        original_thumbs = self.arrays["img_thumbs"][index]
        original_ph = self.arrays["img_phash"][index]
        original_dh = self.arrays["img_dhash"][index]
        min_ph = min(hamming(original_ph[t], ph[0]) for t in range(8))
        min_dh = min(hamming(original_dh[t], dh[0]) for t in range(8))
        best_t = min(range(8), key=lambda t: (hamming(original_ph[t], ph[0]) + hamming(original_dh[t], dh[0]), t))
        thumb_corr = pearson(original_thumbs[best_t], thumbs[0])
        result = {
            "synthetic_jpeg_rotation_detector_check": {
                "source_key": key,
                "operation": "r90 then in-memory JPEG quality=95; no file or dataset modification",
                "min_phash_hamming": min_ph,
                "min_dhash_hamming": min_dh,
                "thumbnail_pearson": thumb_corr,
                "passes_image_review_gate": bool((min_ph <= 16 and min_dh <= 24 and thumb_corr is not None and thumb_corr >= 0.85) or (min_ph <= 10 and thumb_corr is not None and thumb_corr >= 0.80) or (min_dh <= 10 and thumb_corr is not None and thumb_corr >= 0.88) or (thumb_corr is not None and thumb_corr >= 0.965)),
            },
            "d4_synthetic_square_check": {},
            "known_positive_controls": self.positive_controls["overall_pass"],
        }
        x = np.arange(64, dtype=np.uint8).reshape(8, 8)
        rotation_inverse = all(np.array_equal(d4(d4(x, k), (-k) % 4), x) for k in range(4))
        mirror_involution = np.array_equal(d4(d4(x, 4), 4), x)
        eight_unique = len({pixel_sha(d4(x, t)) for t in range(8)}) == 8
        d4_sanity = rotation_inverse and mirror_involution and eight_unique
        result["d4_synthetic_square_check"] = {"eight_unique_orientations_for_asymmetric_square": eight_unique, "rotation_inverse": rotation_inverse, "horizontal_flip_involution": mirror_involution, "pass": d4_sanity}
        self.emit_json("detector_self_tests.json", result)
        assert result["d4_synthetic_square_check"]["pass"]
        assert result["synthetic_jpeg_rotation_detector_check"]["passes_image_review_gate"]

    def retrieve_candidates(self):
        n = len(self.keys)
        candidate_reasons = defaultdict(int)
        cap = self.config["candidate_retrieval"]["max_retrieved_unique_pairs"]
        max_bucket = self.config["candidate_retrieval"]["max_lsh_bucket_entries"]
        reason_bits = {"img_phash": 1, "img_dhash": 2, "gt_phash": 4, "gt_dhash": 8}
        for array_name, bit in reason_bits.items():
            hashes = self.arrays[array_name]
            buckets = defaultdict(list)
            for t in range(8):
                for i in range(n):
                    value = int(hashes[i, t])
                    for band in range(4):
                        buckets[(band, (value >> (16 * band)) & 0xFFFF)].append((i, t))
            for bucket_key, entries in sorted(buckets.items()):
                if len(entries) > max_bucket:
                    self.overflow["lsh_buckets"].append({"array": array_name, "band": bucket_key[0], "band_value": bucket_key[1], "entries": len(entries)})
                    continue
                for (ia, ta), (ib, tb) in combinations(entries, 2):
                    if ia == ib:
                        continue
                    a, b = sorted((ia, ib))
                    if (a, b) not in candidate_reasons and len(candidate_reasons) >= cap:
                        self.overflow["retrieved_pair_cap"] = True
                        continue
                    candidate_reasons[(a, b)] |= bit
            print(f"retrieval {array_name}: cumulative {len(candidate_reasons)} pairs", flush=True)
        for row in self.d4_rows:
            a, b = sorted((self.keys.index(row["key_a"]), self.keys.index(row["key_b"])))
            bit = 16 if row["relation_type"] == "GT_D4_EXACT" else 32
            candidate_reasons[(a, b)] |= bit
        self.candidate_reasons = candidate_reasons

    def candidate_descriptor_metrics(self, a, b):
        img_ph = [hamming(self.arrays["img_phash"][a, t], self.arrays["img_phash"][b, 0]) for t in range(8)]
        img_dh = [hamming(self.arrays["img_dhash"][a, t], self.arrays["img_dhash"][b, 0]) for t in range(8)]
        gt_ph = [hamming(self.arrays["gt_phash"][a, t], self.arrays["gt_phash"][b, 0]) for t in range(8)]
        gt_dh = [hamming(self.arrays["gt_dhash"][a, t], self.arrays["gt_dhash"][b, 0]) for t in range(8)]
        best_img = min(range(8), key=lambda t: (img_ph[t] + img_dh[t], img_ph[t], img_dh[t], t))
        best_gt = min(range(8), key=lambda t: (gt_ph[t] + gt_dh[t], gt_ph[t], gt_dh[t], t))
        exact_img_t = [t for t in range(8) if self.arrays["img_d4_sha"][a, t] == self.records[self.keys[b]]["image_pixel_sha256"]]
        exact_gt_t = [t for t in range(8) if self.arrays["gt_d4_sha"][a, t] == self.records[self.keys[b]]["gt_pixel_sha256"]]
        if exact_img_t:
            best_img = exact_img_t[0]
        if exact_gt_t:
            best_gt = exact_gt_t[0]
        img_corr = pearson(self.arrays["img_thumbs"][a, best_img], self.arrays["img_thumbs"][b, 0])
        gt_a = np.rint(self.arrays["gt_thumbs"][a, best_gt]).astype(np.uint8)
        gt_b = np.rint(self.arrays["gt_thumbs"][b, 0]).astype(np.uint8)
        gt_thumb = gt_metrics(gt_a, gt_b)
        image_gate = bool(
            (min(img_ph) <= 16 and min(img_dh) <= 24 and img_corr is not None and img_corr >= 0.85)
            or (min(img_ph) <= 10 and img_corr is not None and img_corr >= 0.80)
            or (min(img_dh) <= 10 and img_corr is not None and img_corr >= 0.88)
            or (img_corr is not None and img_corr >= 0.965)
        )
        gt_gate = bool((min(gt_ph) <= 16 or min(gt_dh) <= 12) and ((gt_thumb["foreground_iou"] is not None and gt_thumb["foreground_iou"] >= 0.85) or gt_thumb["pixel_agreement"] >= 0.995))
        return {
            "key_a": self.keys[a],
            "key_b": self.keys[b],
            "split_a": self.records[self.keys[a]]["split"],
            "split_b": self.records[self.keys[b]]["split"],
            "scope": scope_of(self.keys[a], self.keys[b], self.records),
            "retrieval_reasons": [name for name, bit in [("image_phash_lsh", 1), ("image_dhash_lsh", 2), ("gt_phash_lsh", 4), ("gt_dhash_lsh", 8), ("exact_gt_d4", 16), ("exact_image_d4", 32)] if self.candidate_reasons[(a, b)] & bit],
            "min_image_phash": min(img_ph),
            "min_image_dhash": min(img_dh),
            "min_gt_phash": min(gt_ph),
            "min_gt_dhash": min(gt_dh),
            "best_image_thumbnail_transform": D4_NAMES[best_img],
            "best_gt_thumbnail_transform": D4_NAMES[best_gt],
            "image_thumbnail_pearson": img_corr,
            "gt_thumbnail_foreground_iou": gt_thumb["foreground_iou"],
            "gt_thumbnail_pixel_agreement": gt_thumb["pixel_agreement"],
            "image_review_gate": image_gate,
            "gt_review_gate": gt_gate,
            "exact_image_d4": bool(exact_img_t),
            "exact_gt_d4": bool(exact_gt_t),
            "exact_image_transforms": [D4_NAMES[t] for t in exact_img_t],
            "exact_gt_transforms": [D4_NAMES[t] for t in exact_gt_t],
            "review_priority": max(img_corr if img_corr is not None else -1.0, gt_thumb["foreground_iou"] if gt_thumb["foreground_iou"] is not None else -1.0),
            "hash_priority": min(img_ph) + min(img_dh) + min(gt_ph) + min(gt_dh),
        }

    def full_review(self, candidate):
        a_key, b_key = candidate["key_a"], candidate["key_b"]
        a, b = self.keys.index(a_key), self.keys.index(b_key)
        orientations = {D4_NAMES.index(candidate["best_image_thumbnail_transform"]), D4_NAMES.index(candidate["best_gt_thumbnail_transform"])}
        orientations.update(D4_NAMES.index(x) for x in candidate["exact_image_transforms"])
        orientations.update(D4_NAMES.index(x) for x in candidate["exact_gt_transforms"])
        rows = []
        for t in sorted(orientations):
            rgb_a = d4(self.img_arrays[a_key], t)
            rgb_b = self.img_arrays[b_key]
            gray_a = np.asarray(Image.fromarray(rgb_a, "RGB").convert("L"), dtype=np.uint8)
            gray_b = np.asarray(Image.fromarray(rgb_b, "RGB").convert("L"), dtype=np.uint8)
            rgb_error = error_metrics(rgb_a, rgb_b)
            gray_error = error_metrics(gray_a, gray_b)
            ssim = float(structural_similarity(gray_a, gray_b, data_range=255, win_size=7, gaussian_weights=False, use_sample_covariance=True, K1=0.01, K2=0.03))
            gt = gt_metrics(d4(self.gt_arrays[a_key], t), self.gt_arrays[b_key])
            rows.append({
                "transform_index": t,
                "transform": D4_NAMES[t],
                "rgb_pearson": pearson(rgb_a, rgb_b),
                "rgb_mae": rgb_error["mae"],
                "rgb_mse": rgb_error["mse"],
                "rgb_psnr": rgb_error["psnr"],
                "gray_pearson": pearson(gray_a, gray_b),
                "gray_mae": gray_error["mae"],
                "gray_psnr": gray_error["psnr"],
                "gray_ssim": ssim,
                "gt_pixel_agreement": gt["pixel_agreement"],
                "gt_foreground_iou": gt["foreground_iou"],
                "gt_present_class_macro_iou": gt["present_class_macro_iou"],
                "gt_diff_pixels": gt["diff_pixels"],
            })
        raw_identical = self.records[a_key]["image_file_sha256"] == self.records[b_key]["image_file_sha256"]
        decoded_identical = self.records[a_key]["image_pixel_sha256"] == self.records[b_key]["image_pixel_sha256"]
        exact_image_rows = [r for r in rows if r["rgb_mse"] == 0.0]
        joint_exact_gt_high = [r for r in rows if r["gt_diff_pixels"] == 0 and image_high(r)]
        high_rows = [r for r in rows if image_high(r)]
        gt_near_rows = [r for r in rows if gt_near(r)]
        near_rows = [r for r in rows if image_near(r)]
        if raw_identical:
            category, code = "FILE_BYTES_IDENTICAL", 1
            eligible = [r for r in rows if r["transform"] == "r0"] or rows
        elif decoded_identical:
            category, code = "DECODED_PIXELS_IDENTICAL", 2
            eligible = [r for r in rows if r["transform"] == "r0"] or rows
        elif exact_image_rows:
            category, code = "D4_IMAGE_PIXELS_IDENTICAL", 3
            eligible = exact_image_rows
        elif joint_exact_gt_high:
            category, code = "D4_GT_EXACT_AND_IMAGE_HIGH_SIMILARITY", 4
            eligible = joint_exact_gt_high
        elif high_rows:
            category, code = "IMAGE_NEAR_HIGH_CONFIDENCE_SHARED_SOURCE_CANDIDATE", 5
            eligible = high_rows
        elif gt_near_rows and not near_rows:
            category, code = "GT_SIMILAR_ONLY_SOURCE_UNRESOLVED", 6
            eligible = gt_near_rows
        elif near_rows:
            category, code = "IMAGE_NEAR_ONLY_SOURCE_UNRESOLVED", 5
            eligible = near_rows
        else:
            category, code = "REVIEWED_BELOW_NEAR_THRESHOLD", 7
            eligible = rows
        selected = sorted(eligible, key=lambda r: (-(r["rgb_pearson"] if r["rgb_pearson"] is not None else -2.0), r["rgb_mae"], r["transform_index"]))[0]
        result = dict(candidate)
        result.pop("review_priority", None)
        result.pop("hash_priority", None)
        result.update(selected)
        result.update({
            "candidate_for_full_review": True,
            "review_status": "FULLY_REVIEWED",
            "filter_reason": "",
            "category_code": code,
            "evidence_category": category,
            "image_high_similarity_any_orientation": any(image_high(r) for r in rows),
            "image_near_any_orientation": any(image_near(r) for r in rows),
            "gt_near_any_orientation": any(gt_near(r) for r in rows),
            "gt_exact_any_orientation": any(r["gt_diff_pixels"] == 0 for r in rows),
            "high_confidence_shared_source_candidate": code in [3, 4] or (code == 5 and any(image_high(r) for r in rows)),
            "near_candidate_excluding_exact_image": code not in [1, 2, 3],
            "orientations_reviewed": [r["transform"] for r in rows],
        })
        return result

    def near_screen(self):
        self.retrieve_candidates()
        descriptors = []
        for number, (a, b) in enumerate(sorted(self.candidate_reasons), 1):
            descriptors.append(self.candidate_descriptor_metrics(a, b))
            if number % 10000 == 0:
                print(f"descriptor candidates {number}/{len(self.candidate_reasons)}", flush=True)
        eligible = [r for r in descriptors if r["image_review_gate"] or r["gt_review_gate"] or r["exact_gt_d4"] or r["exact_image_d4"] or ({r["key_a"], r["key_b"]}.issubset(PRIORITY_KEYS) and r["key_a"] != r["key_b"])]
        eligible.sort(key=lambda r: (-r["review_priority"], r["hash_priority"], r["key_a"], r["key_b"]))
        selected_ids = set()
        per_record = Counter()
        cap_hits = 0
        global_cap = self.config["review_gate"]["global_full_reviews"]
        per_cap = self.config["review_gate"]["per_record_full_reviews"]
        for row in eligible:
            mandatory = row["exact_gt_d4"] or row["exact_image_d4"] or {row["key_a"], row["key_b"]}.issubset(PRIORITY_KEYS)
            if mandatory:
                selected_ids.add((row["key_a"], row["key_b"]))
                continue
            if len(selected_ids) >= global_cap:
                self.overflow["global_cap_pairs"] += 1
                cap_hits += 1
                continue
            if per_record[row["key_a"]] >= per_cap or per_record[row["key_b"]] >= per_cap:
                self.overflow["per_record_cap_pairs"] += 1
                cap_hits += 1
                continue
            selected_ids.add((row["key_a"], row["key_b"]))
            per_record[row["key_a"]] += 1
            per_record[row["key_b"]] += 1
        selected_order = {(r["key_a"], r["key_b"]): i for i, r in enumerate(eligible) if (r["key_a"], r["key_b"]) in selected_ids}
        eligible_by_pair = {(r["key_a"], r["key_b"]): r for r in eligible}
        reviewed = {}
        for number, pair in enumerate(sorted(selected_ids, key=lambda p: selected_order[p]), 1):
            reviewed[pair] = self.full_review(eligible_by_pair[pair])
            if number % 250 == 0:
                print(f"full reviews {number}/{len(selected_ids)}", flush=True)
        final_rows = []
        for row in descriptors:
            pair = (row["key_a"], row["key_b"])
            if pair in reviewed:
                final_rows.append(reviewed[pair])
            else:
                out = dict(row)
                out.pop("review_priority", None)
                out.pop("hash_priority", None)
                was_eligible = row["image_review_gate"] or row["gt_review_gate"] or row["exact_gt_d4"] or row["exact_image_d4"] or {row["key_a"], row["key_b"]}.issubset(PRIORITY_KEYS)
                out.update({
                    "candidate_for_full_review": bool(was_eligible),
                    "review_status": "CAP_NOT_REVIEWED" if was_eligible else "FILTERED_BEFORE_FULL_REVIEW",
                    "filter_reason": "REVIEW_CAP" if was_eligible else "HASH_OR_THUMBNAIL_GATE_NOT_MET",
                    "evidence_category": "NOT_FULLY_REVIEWED" if was_eligible else "RETRIEVAL_COLLISION_ONLY",
                    "high_confidence_shared_source_candidate": False,
                    "near_candidate_excluding_exact_image": bool(was_eligible and not row["exact_image_d4"]),
                })
                final_rows.append(out)
        final_rows.sort(key=lambda r: (r["key_a"], r["key_b"]))
        self.candidate_rows = final_rows
        self.emit_csv("near_duplicate_candidates.csv", final_rows)
        reviewed_rows = list(reviewed.values())
        category_counts = Counter((r["scope"], r["evidence_category"]) for r in reviewed_rows)
        high_cross = [r for r in reviewed_rows if r["scope"] == "train/test" and r["high_confidence_shared_source_candidate"]]
        near_reviewed = [r for r in reviewed_rows if r["near_candidate_excluding_exact_image"]]
        summary = {
            "retrieved_unique_pairs": len(descriptors),
            "retrieval_reason_pair_counts": {name: sum(any(reason == name for reason in r["retrieval_reasons"]) for r in final_rows) for name in ["image_phash_lsh", "image_dhash_lsh", "gt_phash_lsh", "gt_dhash_lsh", "exact_gt_d4", "exact_image_d4"]},
            "eligible_for_full_review_before_caps": len(eligible),
            "fully_reviewed_pairs": len(reviewed_rows),
            "near_candidates_fully_reviewed_excluding_exact_image": len(near_reviewed),
            "confirmed_high_confidence_pairs_including_exact_image_d4": sum(bool(r["high_confidence_shared_source_candidate"]) for r in reviewed_rows),
            "confirmed_high_confidence_near_pairs_excluding_exact_image": sum(bool(r["high_confidence_shared_source_candidate"] and r["near_candidate_excluding_exact_image"]) for r in reviewed_rows),
            "reviewed_category_counts": [{"scope": scope, "category": category, "pairs": count} for (scope, category), count in sorted(category_counts.items())],
            "high_confidence_cross_split_pairs": len(high_cross),
            "high_confidence_cross_split_test_records": sorted({r["key_a"] if r["split_a"] == "test" else r["key_b"] for r in high_cross}),
            "other_high_confidence_test_records_beyond_000301_000305": sorted({r["key_a"] if r["split_a"] == "test" else r["key_b"] for r in high_cross} - {f"test:{x}" for x in PRIORITY_TEST}),
            "retrieval_overflow": self.overflow,
            "screen_complete_under_frozen_config": not self.overflow["retrieved_pair_cap"] and not self.overflow["lsh_buckets"] and self.overflow["per_record_cap_pairs"] == 0 and self.overflow["global_cap_pairs"] == 0,
            "interpretation_boundary": "A high-confidence candidate is not acquisition-level proof. Retrieval collisions below the full-review gate are not evidence of unrelated acquisition, and GT-only similarity never establishes an image-source family.",
        }
        self.near_summary = summary
        self.emit_json("near_duplicate_summary.json", summary)
        additional = [r for r in high_cross if not ({r["key_a"], r["key_b"]}.issubset(PRIORITY_KEYS))]
        self.emit_csv("additional_cross_split_candidates.csv", additional)

    def build_source_graph(self):
        edges_by_level = {"digital_identity": set(), "transform_equivalence": set(), "high_confidence_candidate": set()}
        evidence_rows = {}
        for row in self.d4_rows:
            pair = ordered_pair(row["key_a"], row["key_b"])
            current = evidence_rows.setdefault(pair, {"key_a": pair[0], "key_b": pair[1], "scope": scope_of(*pair, self.records), "evidence": set(), "image_relation": False, "gt_relation": False})
            current["evidence"].add(row["relation_type"] + "[" + ";".join(row["transforms_a_to_b"]) + "]")
            if row["relation_type"] == "IMAGE_D4_EXACT":
                edges_by_level["digital_identity"].add(pair) if row["includes_identity_r0"] else None
                edges_by_level["transform_equivalence"].add(pair)
                edges_by_level["high_confidence_candidate"].add(pair)
                current["image_relation"] = True
            else:
                current["gt_relation"] = True
        for row in self.candidate_rows:
            if not row.get("review_status") == "FULLY_REVIEWED":
                continue
            pair = ordered_pair(row["key_a"], row["key_b"])
            current = evidence_rows.setdefault(pair, {"key_a": pair[0], "key_b": pair[1], "scope": scope_of(*pair, self.records), "evidence": set(), "image_relation": False, "gt_relation": False})
            current["evidence"].add(row["evidence_category"])
            if row["high_confidence_shared_source_candidate"]:
                edges_by_level["high_confidence_candidate"].add(pair)
                current["image_relation"] = True
            if row["gt_exact_any_orientation"]:
                current["gt_relation"] = True
        edge_output = []
        for (a, b), row in sorted(evidence_rows.items()):
            pair = (a, b)
            if pair_in(pair, edges_by_level["digital_identity"]):
                source_status = "DIGITAL_IDENTITY_VERIFIED"
            elif pair_in(pair, edges_by_level["transform_equivalence"]):
                source_status = "TRANSFORM_EQUIVALENCE_VERIFIED"
            elif pair_in(pair, edges_by_level["high_confidence_candidate"]):
                source_status = "HIGH_CONFIDENCE_SHARED_SOURCE_CANDIDATE"
            else:
                source_status = "SOURCE_UNRESOLVED"
            edge_output.append({
                "key_a": a,
                "key_b": b,
                "scope": row["scope"],
                "evidence": sorted(row["evidence"]),
                "image_relation": row["image_relation"],
                "gt_relation": row["gt_relation"],
                "digital_identity_verified": pair_in(pair, edges_by_level["digital_identity"]),
                "transform_equivalence_verified": pair_in(pair, edges_by_level["transform_equivalence"]),
                "high_confidence_shared_source_candidate": pair_in(pair, edges_by_level["high_confidence_candidate"]),
                "acquisition_source_verified": False,
                "source_status": source_status,
            })
        self.emit_csv("source_family_edges.csv", edge_output)
        summaries = {}
        memberships = []
        nonsingletons_by_level = {}
        for level in ["digital_identity", "transform_equivalence", "high_confidence_candidate"]:
            stats, member_rows, non_singletons = component_stats(self.records, sorted(edges_by_level[level]), level)
            summaries[level] = stats
            memberships.extend(member_rows)
            nonsingletons_by_level[level] = non_singletons
        summaries["acquisition_proven"] = {
            "explicit_acquisition_ids_found": 0,
            "nonsingleton_components_with_direct_acquisition_proof": 0,
            "components_total_including_singletons": None,
            "train_records_in_nonsingletons": None,
            "test_records_in_nonsingletons": None,
            "status": "SOURCE_UNRESOLVED",
            "reason": "No upstream acquisition-ID map or derivative-generation record was found; digital and candidate components must not be relabeled as acquisition-proven families.",
        }
        known_components = []
        for group_id, members, splits in nonsingletons_by_level["high_confidence_candidate"]:
            if set(members) & PRIORITY_KEYS:
                known_components.append({"group_id": group_id, "members": members, "size": len(members), "train_records": sum(self.records[k]["split"] == "train" for k in members), "test_records": sum(self.records[k]["split"] == "test" for k in members), "contains_all_eight_priority_records": set(members).issuperset(PRIORITY_KEYS)})
        summaries["priority_components"] = known_components
        summaries["independence_derivation"] = {
            "cross_split_digital_identity_pairs": sum(scope_of(a, b, self.records) == "train/test" for a, b in edges_by_level["digital_identity"]),
            "cross_split_transform_equivalence_pairs": sum(scope_of(a, b, self.records) == "train/test" for a, b in edges_by_level["transform_equivalence"]),
            "cross_split_high_confidence_pairs": sum(scope_of(a, b, self.records) == "train/test" for a, b in edges_by_level["high_confidence_candidate"]),
        }
        if summaries["independence_derivation"]["cross_split_transform_equivalence_pairs"] > 0:
            status = "COMPROMISED"
        elif summaries["independence_derivation"]["cross_split_high_confidence_pairs"] > 0:
            status = "HIGH-RISK CANDIDATE"
        else:
            status = "UNRESOLVED"
        summaries["cross_split_source_independence"] = status
        summaries["unreviewed_or_filtered_candidate_pairs"] = sum(r["review_status"] != "FULLY_REVIEWED" for r in self.candidate_rows)
        summaries["cap_or_overflow_pending_review"] = self.overflow
        self.family_summary = summaries
        self.emit_csv("source_family_membership.csv", memberships)
        self.emit_json("source_family_summary.json", summaries)

    def emit_priority_identity(self):
        rows = []
        for key in sorted(PRIORITY_KEYS):
            r = self.records[key]
            rows.append({
                "key": key,
                "actual_resolved_image_path": str(r["image_path"]),
                "actual_resolved_gt_path": str(r["gt_path"]),
                "list_path": r["list_path"],
                "list_index": r["list_index"],
                "image_file_sha256": r["image_file_sha256"],
                "gt_file_sha256": r["gt_file_sha256"],
                "image_pixel_sha256": r["image_pixel_sha256"],
                "gt_pixel_sha256": r["gt_pixel_sha256"],
                "upstream_image_relative_path": r["upstream_image_relative_path"],
                "upstream_gt_relative_path": r["upstream_gt_relative_path"],
                "upstream_image_file_bytes_equal": r["upstream_image_file_bytes_equal"],
                "upstream_gt_file_bytes_equal": r["upstream_gt_file_bytes_equal"],
                "upstream_image_pixels_equal": r["upstream_image_pixels_equal"],
                "upstream_gt_pixels_equal": r["upstream_gt_pixels_equal"],
                "present_in_runtime_train_or_test_list": True,
                "image_gt_basename_match": r["image_path"].stem == r["gt_path"].stem,
            })
        self.emit_csv("actual_priority_training_input_identity.csv", rows)

    def emit_execution(self, status):
        head = subprocess.check_output(["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "-C", str(self.repo), "branch", "--show-current"], text=True).strip()
        remote = subprocess.check_output(["git", "-C", str(self.repo), "remote", "get-url", "origin"], text=True).strip()
        status_short = subprocess.check_output(["git", "-C", str(self.repo), "status", "--short"], text=True).splitlines()
        runtime_links = {}
        for model in ["base_b", "a2_full"]:
            code = self.project / f"repro_runs/long240k_seed1337/{model}/code"
            runtime_links[model] = {
                "dataset_symlink": str(code / "dataset"),
                "resolved_dataset": str((code / "dataset").resolve()),
                "resolved_train_list": str((code / "dataset/train_neu.txt").resolve()),
                "resolved_test_list": str((code / "dataset/test_neu.txt").resolve()),
                "train_script_sha256": sha_file(code / f"train_long240k_{model}.py"),
                "datagenerator_sha256": sha_file(code / "datagenerator_neu.py"),
                "data_augmentation_sha256": sha_file(code / "dataAug_new.py"),
                "config_sha256": sha_file(code.parent / "config/config.yml"),
            }
        execution = {
            "audit_id": self.config["audit_id"],
            "status": status,
            "started": self.started,
            "ended": now_iso(),
            "wall_seconds": time.time() - self.t0,
            "command_arguments": vars(self.args),
            "repo_root": str(self.repo),
            "repo_head": head,
            "repo_branch": branch,
            "repo_remote_origin": remote,
            "repo_status_short_at_execution_end": status_short,
            "runtime_dataset_links": runtime_links,
            "train_list_sha256": sha_file(self.project / "new (copy)/dataset/train_neu.txt"),
            "test_list_sha256": sha_file(self.project / "new (copy)/dataset/test_neu.txt"),
            "actual_runtime_log_train_list_line": "2026-09-22 14:42:21,065 INFO Train list: ./dataset/train_neu.txt (3630 samples)",
            "actual_runtime_log": "repro_runs/long240k_seed1337/base_b/logs/base_b_s1337_long240k_codex1/run_2026_09_22_14_42_16.log",
            "online_augmentation": "Compose([Transforms_PIL(200,200),ToTensor]); one of random_blur/random_flip/random_noise/random_rotate per item; train DataLoader shuffle=True, drop_last=True, batch_size=16",
            "offline_augmentation_at_runtime": "none in the verified DataGenerator; raw files are opened directly before online transforms",
            "upstream_identity": self.upstream_summary,
            "positive_controls_pass": self.positive_controls["overall_pass"] if self.positive_controls else False,
            "near_summary": getattr(self, "near_summary", None),
            "family_independence_status": getattr(self, "family_summary", {}).get("cross_split_source_independence"),
            "not_run": self.config["not_run"],
            "private_artifacts": "descriptors_private.npz and staging are outside the public repository and must not be committed",
        }
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "opencv": cv2.__version__,
            "scipy": scipy.__version__,
            "skimage": skimage.__version__,
            "analysis_libraries_used_for_pixels_only": True,
            "model_framework_imported": False,
            "model_inference_executed": False,
        }
        self.emit_json("run_environment.json", environment)
        self.emit_json("execution_record.json", execution)

    def publish(self):
        self.generated = sorted(set(self.generated))
        missing = [name for name in self.generated if not (self.staging / name).is_file()]
        assert not missing, missing
        occupied = [name for name in self.generated if (self.output / name).exists()]
        if occupied:
            raise FileExistsError(f"Refusing to overwrite public audit outputs: {occupied}")
        for name in self.generated:
            target = self.output / name
            with open(self.staging / name, "rb") as src, open(target, "xb") as dst:
                shutil.copyfileobj(src, dst)
            assert sha_file(self.staging / name) == sha_file(target)
        write_json(self.private / "publish_verification.json", {"status": "PASS", "files": self.generated, "sha256": {name: sha_file(self.output / name) for name in self.generated}, "published_at": now_iso()})

    def run(self):
        if self.private.exists():
            raise FileExistsError(f"Refusing to overwrite private run directory: {self.private}")
        self.private.mkdir(parents=True)
        self.staging.mkdir()
        self.load_lists()
        self.build_manifest()
        self.exact_screen()
        self.d4_screen()
        self.verify_positive_controls()
        self.detector_self_tests()
        self.near_screen()
        self.build_source_graph()
        self.emit_priority_identity()
        self.emit_execution("COMPLETED")
        self.publish()
        print(json.dumps({"status": "COMPLETED", "records": len(self.records), "exact": self.exact_summary, "d4": self.d4_summary, "near": self.near_summary, "families": self.family_summary}, ensure_ascii=False, default=str))


def pair_in(pair, edge_set):
    return ordered_pair(*pair) in edge_set


def synthetic_self_test():
    x = np.arange(64, dtype=np.uint8).reshape(8, 8)
    assert len({pixel_sha(d4(x, t)) for t in range(8)}) == 8
    assert pixel_sha(d4(x, 0)) == pixel_sha(x)
    rgb = np.stack([x, x, x], axis=-1)
    _, thumbs, ph, dh = image_descriptors(rgb)
    assert thumbs.shape == (8, 32, 32) and ph.shape == (8,) and dh.shape == (8,)
    assert gt_metrics(x, x)["diff_pixels"] == 0
    print(json.dumps({"status": "PASS", "d4_orientations": 8, "descriptor_shape": list(thumbs.shape), "model_inference": False}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/workspace/Industrial Surface Defect")
    parser.add_argument("--repo-root", default="/workspace/Industrial Surface Defect/Industrial-Surface-Defect")
    parser.add_argument("--upstream-root", default="/workspace/cjig_v10_7_p0_private/upstream_NEU_Seg")
    parser.add_argument("--output-dir", default="/workspace/Industrial Surface Defect/Industrial-Surface-Defect/experiments/audit/cjig_v10_7_p0")
    parser.add_argument("--private-dir", default="/workspace/cjig_v10_7_p0_private/analysis_run_001")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        synthetic_self_test()
    else:
        Audit(args).run()


if __name__ == "__main__":
    main()
