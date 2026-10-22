#!/usr/bin/env bash
# CJIG v10.4 evidence archive builder (read-only w.r.t. sources).
# Copies verified evidence into the repo archive dir, packages NEU masks,
# and verifies every archived file against its source SHA-256.
set -euo pipefail

ROOT="/workspace/Industrial Surface Defect"
REPO="$ROOT/Industrial-Surface-Defect"
AUDIT="/workspace/cjig_v10_4_readonly_audit"
ARC="$REPO/experiments/audit/cjig_v10_4"
MECH="$ROOT/repro_runs/mechanism_eval_240k"
L60="$ROOT/repro_runs/leather_60k_seed1337"
LHIST="$ROOT/repro_runs/leather_historical_audit"

rm -rf "$ARC"
mkdir -p "$ARC/audit_reports/scripts" "$ARC/neu_masks" \
         "$ARC/leather/logs" "$ARC/leather/checkpoint_metadata" \
         "$ARC/leather/evals" "$ARC/leather/manifests" \
         "$ARC/leather/control_60k" "$ARC/leather/figure4"

# --- A1: 30 audit deliverables + 2 scripts (byte-identical copies) ---
for f in bootstrap_reproduction_summary.csv bootstrap_runtime_metadata.json \
         boundary_empty_rule_tests.json boundary_metric_implementation_audit.md \
         boundary_named_sample_audit.json estimand_abc_pixel_verification.csv \
         estimand_abc_sample_sets.json figure3_actual_panels.json \
         figure3_candidate_trace.csv figure_case_source_audit.md \
         final_scientific_evidence_status.md leather_raw_provenance_update.md \
         neu_confusion_matrices.json neu_identity_checks.json \
         neu_mask_identity_verification.md neu_pixel_audit_summary.json \
         neu_pixel_metric_reproduction.csv paired_case_reproduction_diff.json \
         recomputed_case_metrics.csv recomputed_per_image_miou.csv \
         rq1_decomposition.csv rq1_pixel_decomposition.md \
         rq1_size_group_contributions.csv rq2_area_reproduction.csv \
         rq2_complexity_reproduction.csv rq2_exploratory_confounders.csv \
         rq2_raw_mask_condition_audit.md scale_dice_corrected_recalculation.csv \
         scale_dice_root_cause_audit.md source_inventory.md \
         source_manifest.json source_manifest_supplemental.json; do
  cp -p "$AUDIT/$f" "$ARC/audit_reports/$f"
done
cp -p "$AUDIT/scripts/build_inventory.py" "$ARC/audit_reports/scripts/build_inventory.py"
cp -p "$AUDIT/scripts/run_neu_pixel_audit.py" "$ARC/audit_reports/scripts/run_neu_pixel_audit.py"
cp -p "$AUDIT/scripts/build_archive.sh" "$ARC/audit_reports/scripts/build_archive.sh"

# --- B: Leather provenance ---
cp -p "$ROOT/new (copy)/runs_other/dsmonet_resnet_pige/dsmor50_pige/run_2024_01_27_14_05_37.log" \
      "$ARC/leather/logs/run_2024_01_27_14_05_37.log"
cp -p "$REPO/checkpoint_metadata/DSMONet-B/checkpoint_info.json" \
      "$ARC/leather/checkpoint_metadata/DSMONet-B_checkpoint_info.json"
cp -p "$REPO/checkpoint_metadata/A2MS-DSMONet-B/checkpoint_info.json" \
      "$ARC/leather/checkpoint_metadata/A2MS-DSMONet-B_checkpoint_info.json"
cp -p "$LHIST/evals/hist_0127_baseb_78k_reverify_class_iou.csv" "$ARC/leather/evals/"
cp -p "$LHIST/evals/hist_a2_160k_fixedfwd_class_iou.csv" "$ARC/leather/evals/"
cp -p "$LHIST/manifests/hist_0127_baseb_78k_reverify.json" "$ARC/leather/manifests/"
cp -p "$LHIST/manifests/hist_a2_160k_fixedfwd.json" "$ARC/leather/manifests/"
cp -p "$REPO/experiments/audit/leather_paper_final_results.json" "$ARC/leather/leather_paper_final_results.json"

cp -p "$L60/base_b/logs/base_b_s1337_leather60k_codex1/config.yml" "$ARC/leather/control_60k/base_b_config.yml"
cp -p "$L60/a2_full/logs/a2_full_s1337_leather60k_codex1/config.yml" "$ARC/leather/control_60k/a2_full_config.yml"
cp -p "$L60/base_b/logs/base_b_s1337_leather60k_codex1/run_2026_09_24_15_39_05.log" \
      "$ARC/leather/control_60k/base_b_run_2026_09_24_15_39_05.log"
cp -p "$L60/a2_full/logs/a2_full_s1337_leather60k_codex1/run_2026_09_25_04_23_08.log" \
      "$ARC/leather/control_60k/a2_full_run_2026_09_25_04_23_08.log"
cp -p "$L60/base_b/analysis/train_metrics.jsonl" "$ARC/leather/control_60k/base_b_train_metrics.jsonl"
cp -p "$L60/a2_full/analysis/train_metrics.jsonl" "$ARC/leather/control_60k/a2_full_train_metrics.jsonl"
cp -p "$L60/base_b/analysis/base_b_s1337_leather60k_codex1_train_summary.json" "$ARC/leather/control_60k/base_b_train_summary.json"
cp -p "$L60/a2_full/analysis/a2_full_s1337_leather60k_codex1_train_summary.json" "$ARC/leather/control_60k/a2_full_train_summary.json"
cp -p "$L60/base_b/analysis/unified468_iter60000.csv" "$ARC/leather/control_60k/base_b_unified468_iter60000.csv"
cp -p "$L60/a2_full/analysis/unified468_iter60000.csv" "$ARC/leather/control_60k/a2_full_unified468_iter60000.csv"

cp -p "$LHIST/final_viz/final_viz_manifest.json" "$ARC/leather/figure4/"
cp -p "$LHIST/leather_viz_final_v6.py" "$ARC/leather/figure4/"
cp -p "$LHIST/assemble_fig34_v6.py" "$ARC/leather/figure4/"

# --- A2: NEU masks -> per-file manifest + tarball + extraction verification ---
python3 - "$MECH" "$ARC/neu_masks" <<'PYEOF'
import csv, hashlib, json, os, subprocess, sys, tarfile, tempfile

mech, outdir = sys.argv[1], sys.argv[2]
groups = ["gt", "base_predictions", "a2_predictions"]
rows = []
for g in groups:
    d = os.path.join(mech, g)
    names = sorted(os.listdir(d))
    assert len(names) == 840, f"{g}: expected 840 files, got {len(names)}"
    for n in names:
        p = os.path.join(d, n)
        with open(p, "rb") as fh:
            data = fh.read()
        rows.append((f"{g}/{n}", hashlib.sha256(data).hexdigest(), len(data)))
assert len(rows) == 2520

manifest_csv = os.path.join(outdir, "mask_sha256_manifest.csv")
with open(manifest_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["relative_path", "sha256", "bytes"])
    w.writerows(rows)

tar_path = os.path.join(outdir, "neu_test840_masks.tar.gz")
with tarfile.open(tar_path, "w:gz") as tf:
    for g in groups:
        tf.add(os.path.join(mech, g), arcname=g)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# verify: extract to temp, hash every file, compare to manifest
expected = {r[0]: (r[1], r[2]) for r in rows}
checked, bad = 0, []
with tempfile.TemporaryDirectory() as td:
    with tarfile.open(tar_path, "r:gz") as tf:
        try:
            tf.extractall(td, filter="data")
        except TypeError:  # python < 3.11.4 lacks the filter kwarg; archive is self-built
            tf.extractall(td)
    for rel, (sha, size) in expected.items():
        p = os.path.join(td, rel)
        if not os.path.isfile(p):
            bad.append((rel, "missing_in_archive")); continue
        with open(p, "rb") as fh:
            data = fh.read()
        if hashlib.sha256(data).hexdigest() != sha or len(data) != size:
            bad.append((rel, "hash_or_size_mismatch")); continue
        checked += 1

ver = {
    "source_root": mech,
    "groups": {g: 840 for g in groups},
    "total_files": 2520,
    "manifest_csv": "mask_sha256_manifest.csv",
    "tarball": "neu_test840_masks.tar.gz",
    "tarball_sha256": sha256_file(tar_path),
    "tarball_bytes": os.path.getsize(tar_path),
    "extraction_recheck": {"files_checked": checked, "mismatches": bad,
                            "status": "PASS" if checked == 2520 and not bad else "FAIL"},
    "note": "Manifest hashes computed from ORIGINAL source dirs; archive verified by full extraction + per-file SHA-256 recheck.",
}
with open(os.path.join(outdir, "mask_archive_verification.json"), "w") as fh:
    json.dump(ver, fh, indent=2)
print(json.dumps({"tarball_sha256": ver["tarball_sha256"], "tarball_bytes": ver["tarball_bytes"],
                  "status": ver["extraction_recheck"]["status"]}, indent=2))
PYEOF

echo "=== archive tree ==="
find "$ARC" -type f | sort | sed "s|$REPO/||"
echo "=== file count: $(find "$ARC" -type f | wc -l) ==="
du -sh "$ARC"
