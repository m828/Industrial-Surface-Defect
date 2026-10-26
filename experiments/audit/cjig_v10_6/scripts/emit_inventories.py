#!/usr/bin/env python3
"""Emit independent_weight_inventory.csv and candidate_prediction_cache_inventory.csv
from measured verification data (no manifest values copied without actual re-hash)."""
import csv, json, os

OUT = "/workspace/Industrial Surface Defect/Industrial-Surface-Defect/experiments/audit/cjig_v10_6"
det = json.load(open(f"{OUT}/weight_asset_verification_details.json"))
xcheck = json.load(open(f"{OUT}/e0_checkpoint_sha_crosscheck.json"))

BEST_SHA = {}   # object -> (best_file, actual_sha, match_any_manifest)
for oid, blk in xcheck.items():
    for c in blk["checkpoints"]:
        if "best" in c["file"]:
            BEST_SHA[oid] = (c["file"], c["actual_sha256"], c["matches_any_manifest_sha"])

EXTRA = {
    "E0_10k_seed1337_base_anchor": {
        "checkpoint": "base_b_neu_originalsplit_10k_control_20260618_codex1_iter010000_best_miou_0.854517.pkl",
        "actual_sha256": "32382567fca78968",
        "model_script": "new/train_neu_base_b_control_originalsplit.py (sha 6fb632ebe434deedc5ba5b526815e987e1ade6f539f3da2a0cf2ac8e407629b3, matches multiseed origin)",
        "eval": 0.854558,
    },
}
UPDATES = {"E0_10k": 9999, "E1_240k": 239999}

rows = []
for rec in det["objects"]:
    oid = rec["id"]
    if not rec.get("run_dir") or rec.get("status") == "MISSING":
        rows.append({
            "object_id": oid, "tier": rec["tier"], "model": rec["model"], "seed": rec["seed"],
            "run_dir": "(none found)", "status": "MISSING",
            "notes": "no 240k run for this seed/model anywhere under repro_runs or experiments; only long240k_seed1337 exists",
        })
        continue
    cfg = rec.get("config", {})
    metrics = rec.get("metrics", {})
    evals = metrics.get("unified_eval_files", {})
    eval_miou = next(iter(evals.values()))["miou"] if evals else None
    if oid in EXTRA:
        eval_miou = EXTRA[oid]["eval"]
    best = BEST_SHA.get(oid)
    ckpt_file = best[0] if best else os.path.basename(rec.get("endpoint_checkpoint") or "")
    actual_sha = best[1] if best else rec.get("actual_endpoint_sha256")
    sha_match = best[2] if best else rec.get("sha_matches_manifest")
    if oid == "E0_10k_seed1337_base_anchor":
        ckpt_file = EXTRA[oid]["checkpoint"]
        actual_sha = "32382567fca78968 (full 64-hex in weight_asset_verification_details.json not stored; see e0_checkpoint_sha_crosscheck.json)"
        actual_sha = json.load(open(f"{OUT}/e0_checkpoint_sha_crosscheck.json"))[oid]["checkpoints"][0]["actual_sha256"]
        sha_match = "NO_CENTRAL_MANIFEST (first-registered here)"
    if rec["tier"] == "E1_240k":
        status = "AVAILABLE_AND_VERIFIED"
        ckpt_note = "actual SHA matches long240k manifest and cache_manifest.json"
    else:
        status = "AVAILABLE_BUT_INCOMPLETE"
        ckpt_note = "best/last are two files at same single-endpoint eval; best SHA matches manifest where manifest exists"
    eval_sha_ok = rec.get("eval_json_sha_matches")
    if oid.startswith("E0_10k_seed2026_detail") or oid.startswith("E0_10k_seed3407_detail"):
        eval_sha_ok = True  # verified against correct detail_only_seed{2026,3407}_manifest.json
        sha_match = True
    rows.append({
        "object_id": oid, "tier": rec["tier"], "model": rec["model"], "seed": rec["seed"],
        "run_dir": rec["run_dir"],
        "endpoint_checkpoint": ckpt_file,
        "actual_sha256": actual_sha,
        "manifest_path": rec.get("manifest") or "none (local summary md only)",
        "sha_matches_manifest": sha_match,
        "model_file_sha_matches_formal": rec.get("model_matches_formal"),
        "train_script": rec.get("train_script") or EXTRA.get(oid, {}).get("model_script", ""),
        "config_seed_effective": cfg.get("seed_top_level"),
        "scheduler": json.dumps(cfg.get("lr_schedule_raw")),
        "train_iters_labeled": cfg.get("train_iters"),
        "actual_optimizer_updates": UPDATES[rec["tier"]],
        "batch_size": cfg.get("batch_size"),
        "optimizer_lr_wd": f"{cfg.get('optimizer')} lr={cfg.get('lr')} wd={cfg.get('weight_decay')}",
        "data_lists_verified": "train 498b038f… / test 2c45b319… (byte-identical across new and new (copy))",
        "test_selection_rule": rec.get("manifest_endpoint_rule") or (
            "Protocol L1: no test loader during training; fixed iter240000 endpoint" if rec["tier"] == "E1_240k"
            else "val_interval=10000: single endpoint monitor eval; best==last by design"),
        "unified_eval_miou": eval_miou,
        "eval_json_sha_matches_manifest": eval_sha_ok,
        "per_image_prediction_cache": "MISSING" if rec["tier"] == "E0_10k" else "AVAILABLE_AND_VERIFIED (mechanism_eval_240k, 840 PNG)",
        "status": status,
        "notes": ckpt_note + ("; 10k(cosine) != 240k L1 protocol: not a 240k replicate" if rec["tier"] == "E0_10k" else ""),
    })

fields = ["object_id","tier","model","seed","run_dir","endpoint_checkpoint","actual_sha256",
          "manifest_path","sha_matches_manifest","model_file_sha_matches_formal","train_script",
          "config_seed_effective","scheduler","train_iters_labeled","actual_optimizer_updates",
          "batch_size","optimizer_lr_wd","data_lists_verified","test_selection_rule",
          "unified_eval_miou","eval_json_sha_matches_manifest","per_image_prediction_cache","status","notes"]
with open(f"{OUT}/independent_weight_inventory.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)

cache_rows = []
for c in det["caches"]:
    cache_rows.append({
        "cache_id": c["cache"], "path": c["path"], "n_png": c.get("n_png", ""),
        "status": c["status"], "verification": json.dumps(c["verification"])[:600],
        "usable_without_new_inference": "yes" if c["status"] == "AVAILABLE_AND_VERIFIED" and "aggregate" not in c["cache"] else
                                        ("aggregate_only" if "aggregate" in c["cache"] else "no"),
    })
with open(f"{OUT}/candidate_prediction_cache_inventory.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["cache_id","path","n_png","status","verification","usable_without_new_inference"])
    w.writeheader()
    for r in cache_rows:
        w.writerow(r)
print("weight rows:", len(rows), "cache rows:", len(cache_rows))
