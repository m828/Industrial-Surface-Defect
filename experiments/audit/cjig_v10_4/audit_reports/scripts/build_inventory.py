#!/usr/bin/env python3
"""Build read-only source inventory and SHA-256 manifest for CJIG v10.4."""
import hashlib
import json
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path("/workspace/Industrial Surface Defect")
REPO = PROJECT / "Industrial-Surface-Defect"
OUT = Path("/workspace/cjig_v10_4_readonly_audit")
MECH = PROJECT / "repro_runs/mechanism_eval_240k"
LEATHER_AUDIT = PROJECT / "repro_runs/leather_historical_audit"
LONG = PROJECT / "repro_runs/long240k_seed1337"
LEATHER60 = PROJECT / "repro_runs/leather_60k_seed1337"

ROOTS = [MECH, LEATHER_AUDIT]
EXTRA_FILES = [
    LONG / "base_b/base_b_pre_run_manifest.json",
    LONG / "a2_full/a2_full_pre_run_manifest.json",
    LONG / "base_b/code/train_long240k_base_b.py",
    LONG / "a2_full/code/train_long240k_a2_full.py",
    LONG / "base_b/code/dataset/test_neu.txt",
    LONG / "a2_full/code/dataset/test_neu.txt",
    REPO / "tools/evaluate_class_iou.py",
    REPO / "experiments/protocol/leather_valid_test_list.txt",
    LEATHER60 / "base_b/code/train_leather60k_base_b.py",
    LEATHER60 / "a2_full/code/train_leather60k_a2_full.py",
    LEATHER60 / "base_b/analysis/base_b_s1337_leather60k_codex1_train_summary.json",
    LEATHER60 / "a2_full/analysis/a2_full_s1337_leather60k_codex1_train_summary.json",
    LEATHER60 / "leather_results.json",
    PROJECT / "repro_runs/build_leather_run.sh",
]

RESULT_TO_SCRIPT = {
    "paired_case_metrics.csv": "compute_mechanism_metrics.py",
    "boundary_quality_results.json": "analyze_mechanism.py + addendum_aggregate.py",
    "paired_case_analysis.json": "analyze_mechanism.py + addendum_aggregate.py",
    "scale_stratified_results.json": "analyze_mechanism.py",
    "morphology_exploratory_results.json": "analyze_mechanism.py",
    "case_selection.json": "analyze_mechanism.py",
    "gt_area_distribution.json": "gt_area_stats.py",
    "cache_manifest.json": "cache_predictions.py",
    "final_viz_manifest.json": "leather_viz_final_v6.py",
}

CHECKPOINT_METADATA = {
    "hist_0127_baseb_78k_reverify.json": "run_evals.sh; historical 0127 checkpoint metadata",
    "hist_a2_160k_fixedfwd.json": "run_evals.sh; historical A2MS fixed-forward checkpoint metadata",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_type(path: Path) -> str:
    try:
        with path.open("rb") as f:
            head = f.read(16)
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png-image"
    except OSError:
        pass
    ext = path.suffix.lower().lstrip(".")
    if ext in {"py", "sh", "json", "csv", "md", "txt", "log"}:
        return ext
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or (ext if ext else "unknown")


def generation_record(path: Path) -> str:
    name = path.name
    parts = set(path.parts)
    if {"base_predictions", "a2_predictions", "gt"} & parts:
        return "cache_predictions.py; cache_manifest.json"
    if name in RESULT_TO_SCRIPT:
        return RESULT_TO_SCRIPT[name]
    if name in CHECKPOINT_METADATA:
        return CHECKPOINT_METADATA[name]
    if path.parent.name == "evals" and name.endswith((".csv", ".log")):
        return "run_evals.sh; evaluate_class_iou.py"
    if path.parent.name == "final_viz" and name.endswith(".png"):
        return "leather_viz_final_v6.py"
    if path.parent.name == "visualizations" and name.endswith(".png"):
        return "visualize_cases_v6.py (or earlier visualize_cases.py for v1 artifacts)"
    if name.startswith("train_") and name.endswith(".py"):
        return "training launch script (not executed in this audit)"
    if name.endswith("_train_summary.json"):
        return "training script summary (not regenerated in this audit)"
    return "filesystem metadata only; no explicit generation record found in this pass"


def entry_for(path: Path, collection: str) -> dict:
    st = path.stat()
    return {
        "path": str(path),
        "collection": collection,
        "file_name": path.name,
        "file_type": file_type(path),
        "size_bytes": st.st_size,
        "mtime_iso": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256(path),
        "readable": os.access(path, os.R_OK),
        "generation_record": generation_record(path),
        "availability": "readable_file",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for root in ROOTS:
        collection = root.name
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            filenames.sort()
            for name in filenames:
                path = Path(dirpath) / name
                if path.is_file() and not path.is_symlink():
                    entries.append(entry_for(path, collection))
    for path in EXTRA_FILES:
        if path.exists() and path.is_file():
            entries.append(entry_for(path, "external_protocol_or_run_record"))
        else:
            entries.append({
                "path": str(path),
                "collection": "external_protocol_or_run_record",
                "file_name": path.name,
                "file_type": "missing",
                "size_bytes": None,
                "mtime_iso": None,
                "sha256": None,
                "readable": False,
                "generation_record": "candidate path checked",
                "availability": "missing",
            })

    manifest = {
        "audit_id": "cjig_v10_4_readonly_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT),
        "git_toplevel": str(REPO),
        "task_book_path_expected": str(REPO / "docs/paper/cjig_v10_3_server_readonly_audit_task.md"),
        "task_book_status": "missing_at_expected_path",
        "scope_roots": [str(p) for p in ROOTS],
        "entries": entries,
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {}
    for e in entries:
        key = (e["collection"], e["file_type"])
        cur = summary.setdefault(key, {"count": 0, "size_bytes": 0, "missing": 0})
        cur["count"] += 1
        cur["size_bytes"] += e["size_bytes"] or 0
        cur["missing"] += 0 if e["readable"] else 1
    lines = [
        "# CJIG v10.4 source inventory",
        "",
        f"Created UTC: {manifest['created_utc']}",
        f"Project root: `{PROJECT}`",
        f"Git worktree: `{REPO}`",
        "Task book status: expected local file was not found; this audit proceeded from the v10.4 prompt.",
        "",
        "## Collections",
    ]
    for (collection, ftype), s in sorted(summary.items()):
        lines.append(f"- `{collection}` / `{ftype}`: {s['count']} files, {s['size_bytes']} bytes, missing={s['missing']}")
    lines += [
        "",
        "The complete per-file path/type/size/SHA-256/availability/generation-record table is in `source_manifest.json`.",
        "No raw masks, source images, or model weights were copied into this audit directory.",
    ]
    (OUT / "source_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT), "entries": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
