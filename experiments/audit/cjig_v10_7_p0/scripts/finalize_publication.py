#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

EXCLUDED_ROOT = {
    "near_duplicate_candidates.csv": "ORIGINAL_LSH_IMPLEMENTATION_DEFECT_AND_GITHUB_FILE_SIZE",
    "near_duplicate_summary.json": "ORIGINAL_LSH_IMPLEMENTATION_DEFECT",
    "source_family_edges.csv": "ORIGINAL_LSH_INCOMPLETE_DEDUPLICATED_DIAGNOSTIC",
    "source_family_membership.csv": "ORIGINAL_LSH_INCOMPLETE_DEDUPLICATED_DIAGNOSTIC",
    "source_family_summary.json": "ORIGINAL_LSH_INCOMPLETE_DEDUPLICATED_DIAGNOSTIC",
    "additional_cross_split_candidates.csv": "ORIGINAL_LSH_IMPLEMENTATION_DEFECT",
    "execution_record.json": "CONTAINS_ORIGINAL_LSH_NEAR_AND_FAMILY_FIELDS",
}
EXCLUDED_DIRS = {"overflow_resolution": "CANCELLED_LSH_CAPACITY_EXPANSION_NO_FINAL_RESULT"}
PRIVATE_FILES = [
    "/workspace/cjig_v10_7_p0_private/upstream_neu_seg_da8cebed.tar.gz",
    "/workspace/cjig_v10_7_p0_private/analysis_run_002/descriptors_private.npz",
    "/workspace/cjig_v10_7_p0_private/oriented_hamming_private_v2.npz",
    "/workspace/cjig_v10_7_p0_private/oriented_hamming_prefilter_counts_v2.json",
    "/workspace/cjig_v10_7_p0_private/analysis_run_001.log",
    "/workspace/cjig_v10_7_p0_private/analysis_run_002.log",
    "/workspace/cjig_v10_7_p0_private/analysis_run_003_overflow.log",
    "/workspace/cjig_v10_7_p0_private/analysis_run_004_overflow.log",
    "/workspace/cjig_v10_7_p0_private/analysis_run_005_exhaustive.log",
    "/workspace/cjig_v10_7_p0_private/analysis_run_006_cap_resolution.log",
    "/workspace/cjig_v10_7_p0_private/analysis_run_007_cap_resolution_parallel.log",
    "/workspace/cjig_v10_7_p0_private/independent_verification.log",
    "/workspace/cjig_v10_7_p0_private/retrieval_screen_verification_attempt1.log",
    "/workspace/cjig_v10_7_p0_private/retrieval_screen_verification_attempt1.json",
    "/workspace/cjig_v10_7_p0_private/retrieval_screen_verification_retry1.log",
    "/workspace/cjig_v10_7_p0_private/oriented_hamming.log",
    "/workspace/cjig_v10_7_p0_private/upstream_archive_members.txt",
    "/workspace/cjig_v10_7_p0_private/analysis_run_005_exhaustive/publish_verification.json",
    "/workspace/cjig_v10_7_p0_private/analysis_run_007_cap_resolution_parallel/publish_verification.json",
]


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json_new(path, value):
    with open(path, "x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", default="/workspace/Industrial Surface Defect/Industrial-Surface-Defect/experiments/audit/cjig_v10_7_p0")
    parser.add_argument("--repo-root", default="/workspace/Industrial Surface Defect/Industrial-Surface-Defect")
    args = parser.parse_args()
    root = Path(args.audit_root).resolve()
    repo = Path(args.repo_root).resolve()
    verification_status = {}
    for name in ["independent_verification_report.json", "retrieval_screen_verification.json"]:
        with open(root / "cap_resolution" / name, encoding="utf-8") as f:
            verification_status[name] = json.load(f)["status"]
    if set(verification_status.values()) != {"PASS"}:
        raise SystemExit(json.dumps({"status": "BLOCKED", "verification": verification_status}, ensure_ascii=False))
    all_files = sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    candidates = []
    excluded = []
    for path in all_files:
        rel = path.relative_to(root)
        if rel.parts[0] in EXCLUDED_DIRS:
            excluded.append({"path": str(rel), "reason": EXCLUDED_DIRS[rel.parts[0]]})
        elif len(rel.parts) == 1 and rel.name in EXCLUDED_ROOT:
            excluded.append({"path": str(rel), "reason": EXCLUDED_ROOT[rel.name]})
        else:
            candidates.append(path)
    forbidden_extensions = {".jpg", ".jpeg", ".png", ".pkl", ".pth", ".pt", ".npy", ".npz", ".tif", ".tiff", ".bmp", ".tfevents", ".rar"}
    forbidden = [str(p.relative_to(root)) for p in all_files if p.suffix.lower() in forbidden_extensions or "tfevents" in p.name]
    credential_patterns = [r"-----BEGIN [A-Z ]*PRIVATE KEY-----", r"github_pat_[A-Za-z0-9_]+", r"ghp_[A-Za-z0-9]+", r"AKIA[0-9A-Z]{16}", r"Authorization:\s*Bearer\s+\S+", "neu" + "123"]
    credential_hits = []
    email_hits = []
    for path in all_files:
        if path.suffix.lower() not in {".md", ".json", ".csv", ".txt", ".py", ".yml", ".yaml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(re.search(pattern, text) for pattern in credential_patterns):
            credential_hits.append(str(path.relative_to(root)))
        if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text):
            email_hits.append(str(path.relative_to(root)))
    oversized = [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size} for p in candidates if p.stat().st_size > 90 * 1024 * 1024]
    private_manifest = []
    for name in PRIVATE_FILES:
        path = Path(name)
        if path.is_file():
            private_manifest.append({"path": name, "bytes": path.stat().st_size, "sha256": sha_file(path), "git_delivery": "NOT_UPLOADED"})
        else:
            private_manifest.append({"path": name, "status": "MISSING", "git_delivery": "NOT_UPLOADED"})
    for path in [root / name for name in EXCLUDED_ROOT]:
        if path.is_file():
            private_manifest.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha_file(path), "role": "server-only original LSH diagnostic", "git_delivery": "NOT_UPLOADED"})
    private_manifest.append({"path": "/workspace/cjig_v10_7_p0_private/upstream_archive", "type": "directory", "role": "fixed upstream extracted source; raw data not uploaded", "git_delivery": "NOT_UPLOADED"})
    private_manifest.append({"path": "/workspace/cjig_v10_7_p0_private/analysis_run_005_exhaustive", "type": "directory", "role": "private staging and descriptors", "git_delivery": "NOT_UPLOADED"})
    private_manifest.append({"path": "/workspace/cjig_v10_7_p0_private/analysis_run_007_cap_resolution_parallel", "type": "directory", "role": "private backlog staging", "git_delivery": "NOT_UPLOADED"})
    write_json_new(root / "server_private_artifact_manifest.json", {"artifacts": private_manifest, "note": "Server paths and hashes are recorded for traceability; no raw data, archive, checkpoint, descriptor NPZ, or full log is uploaded."})
    candidates.append(root / "server_private_artifact_manifest.json")
    candidates = sorted(set(candidates))
    candidate_rel_repo = sorted([str(p.relative_to(repo)) for p in candidates] + [
        "experiments/audit/cjig_v10_7_p0/git_commit_candidate_list.txt",
        "experiments/audit/cjig_v10_7_p0/publication_scan.json",
        "experiments/audit/cjig_v10_7_p0/SHA256SUMS.txt",
    ])
    with open(root / "git_commit_candidate_list.txt", "x", encoding="utf-8") as f:
        f.write("\n".join(candidate_rel_repo) + "\n")
    candidates.append(root / "git_commit_candidate_list.txt")
    status = "PASS" if not forbidden and not credential_hits and not email_hits and not oversized else "FAIL"
    scan = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "all_output_files_scanned": len(all_files),
        "candidate_file_count_before_sha": len(candidates) + 2,
        "candidate_bytes_before_sha": sum(p.stat().st_size for p in candidates),
        "forbidden_raw_or_checkpoint_files": forbidden,
        "credential_or_dataset_password_pattern_files": credential_hits,
        "email_address_files": email_hits,
        "oversized_candidate_files_over_90MiB": oversized,
        "excluded_server_only_files": excluded,
        "public_path_policy": "Only non-sensitive metadata, reports, scripts, hashes, and aggregate statistics are candidates. Generic /workspace paths used for reproducibility are retained; no personal home paths or email addresses are included.",
        "candidate_files": sorted(str(p.relative_to(root)) for p in candidates) + ["publication_scan.json", "SHA256SUMS.txt"],
    }
    write_json_new(root / "publication_scan.json", scan)
    candidates.append(root / "publication_scan.json")
    candidates = sorted(set(candidates))
    if status != "PASS":
        raise SystemExit(json.dumps(scan, ensure_ascii=False, indent=2))
    lines = [f"{sha_file(p)}  {p.relative_to(root)}" for p in candidates]
    with open(root / "SHA256SUMS.txt", "x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"status": status, "candidate_files_including_sha": len(candidates) + 1, "sha_entries": len(lines), "candidate_bytes": sum(p.stat().st_size for p in candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
