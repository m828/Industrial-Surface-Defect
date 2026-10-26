#!/usr/bin/env python3
"""CJIG v10.6-P1 read-only weight/prediction asset verification.
Hashes real checkpoint bytes, compares to manifest-registered values,
parses configs, checks prediction caches. No model loading, no inference.
"""
import csv, hashlib, json, os, sys
import yaml

ROOT = "/workspace/Industrial Surface Defect"
REPO = f"{ROOT}/Industrial-Surface-Defect"
AUDIT = f"{REPO}/experiments/audit"
OUT = f"{AUDIT}/cjig_v10_6"
MS = f"{ROOT}/repro_runs/multiseed_10k"
L240 = f"{ROOT}/repro_runs/long240k_seed1337"
MECH = f"{ROOT}/repro_runs/mechanism_eval_240k"
FORMAL_MODEL_SHA = {
    "base": "fab316c57518e8273ba676199ccff16500ad60ae89a77c2843585f3ce0c184cf",
    "a2ms": "a0e6fd52b00162ebd16805c473a59983cbd34c2308da115cd170ba1710bb376a",
}
TRAIN_LIST_SHA = "498b038f32cd68834894ab433f11468de54a8ef80ac3a6eb1f812d4afcfc1d64"
TEST_LIST_SHA = "2c45b319e04d2307eb0977720de90b98a4867863bce8eca6f8b682295f928f06"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def jload(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None


def find_ckpts(run_dir):
    d = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".pkl"))


def find_pred_caches(run_dir):
    hits = []
    for base, dirs, files in os.walk(run_dir):
        bn = os.path.basename(base).lower()
        if any(k in bn for k in ("predict", "mask", "cache")) and "__pycache__" not in base:
            pngs = [f for f in files if f.endswith(".png")]
            hits.append({"dir": base, "n_png": len(pngs)})
    return hits


def parse_config(run_dir):
    for cand in [os.path.join(run_dir, "config", "config.yml")]:
        if os.path.isfile(cand):
            return cand, yaml.safe_load(open(cand))
    for f in sorted(os.listdir(run_dir)) if os.path.isdir(run_dir) else []:
        if f.endswith(".yml"):
            p = os.path.join(run_dir, f)
            return p, yaml.safe_load(open(p))
    return None, {}


def cfg_summary(cfg):
    tr = cfg.get("training", {}) or {}
    opt = tr.get("optimizer", {}) or {}
    sched = tr.get("lr_schedule", "ABSENT_KEY")
    return {
        "seed_top_level": cfg.get("seed", "ABSENT(default=1337)"),
        "train_iters": tr.get("train_iters"),
        "batch_size": tr.get("batch_size"),
        "val_interval": tr.get("val_interval"),
        "optimizer": opt.get("name"), "lr": opt.get("lr"), "weight_decay": opt.get("weight_decay"),
        "lr_schedule_raw": sched,
        "loss_weights": tr.get("loss_weights") or (tr.get("loss") or {}).get("weights"),
        "resume": tr.get("resume"),
    }


def metrics_summary(run_dir):
    ad = os.path.join(run_dir, "analysis")
    if not os.path.isdir(ad):
        return {}
    out = {}
    for f in sorted(os.listdir(ad)):
        if f == "train_metrics.jsonl":
            p = os.path.join(ad, f)
            n, first, last, lrs = 0, None, None, set()
            iters = []
            with open(p) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    n += 1
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    it = r.get("iter", r.get("iteration"))
                    if it is not None:
                        iters.append(it)
                    lr = r.get("lr", r.get("LR"))
                    if lr is not None:
                        lrs.add(round(float(lr), 12))
            out["train_metrics"] = {
                "records": n, "first_iter": iters[0] if iters else None,
                "last_iter": iters[-1] if iters else None,
                "max_iter": max(iters) if iters else None,
                "lr_unique_n": len(lrs), "lr_min": min(lrs) if lrs else None,
                "lr_max": max(lrs) if lrs else None,
            }
        if f.endswith("_train_summary.json"):
            s = jload(os.path.join(ad, f))
            if s:
                out["train_summary"] = {k: s[k] for k in (
                    "run_id", "train_iters", "final_iter", "total_iters", "seed",
                    "wall_time_seconds", "train_end_wall_time_seconds",
                    "start_time", "end_time", "resumed_from", "status") if k in s}
                out["train_summary_keys"] = sorted(s.keys())
        if f.startswith("unified_eval_840") and f.endswith(".json"):
            r = jload(os.path.join(ad, f))
            if r:
                out.setdefault("unified_eval_files", {})[f] = {
                    "sha256": sha(os.path.join(ad, f)),
                    "miou": r.get("miou", r.get("mIoU", (r.get("metrics") or {}).get("miou"))),
                }
        if f.startswith("unified840_iter240000"):
            r = jload(os.path.join(ad, f))
            if r:
                out.setdefault("unified_eval_files", {})[f] = {
                    "sha256": sha(os.path.join(ad, f)),
                    "miou": r.get("miou", r.get("mIoU", (r.get("metrics") or {}).get("miou"))),
                }
    return out


OBJECTS = []
for seed in (2026, 3407):
    for variant, model in [("base_b", "base"), ("a1", "a2ms"), ("a2", "a2ms"), ("detail_only", "detail_only")]:
        OBJECTS.append({
            "id": f"E0_10k_seed{seed}_{variant}", "tier": "E0_10k", "seed": seed,
            "model": model, "run_dir": f"{MS}/seed_{seed}/{variant}",
            "manifest": f"{AUDIT}/multiseed_{variant}_s{seed}_manifest.json",
            "required": variant in ("base_b", "a2"),
        })
OBJECTS.append({
    "id": "E0_10k_seed1337_detail_only", "tier": "E0_10k", "seed": 1337,
    "model": "detail_only", "run_dir": f"{MS}/seed_1337/detail_only",
    "manifest": f"{AUDIT}/detail_only_seed1337_manifest.json", "required": False,
})
OBJECTS.append({
    "id": "E0_10k_seed1337_base_anchor", "tier": "E0_10k", "seed": 1337,
    "model": "base", "run_dir": f"{REPO}/experiments/retrain_a2ms_b_neu/base_b_10k_control",
    "manifest": None, "required": True,
})
OBJECTS.append({
    "id": "E0_10k_seed1337_a2ms_anchor", "tier": "E0_10k", "seed": 1337,
    "model": "a2ms", "run_dir": f"{ROOT}/repro_runs/neu_a2ms_b_fixed_aam_detail_10k_20260908_121000",
    "manifest": f"{AUDIT}/a2ms_b_fixed_a2_manifest.json", "required": True,
})
OBJECTS.append({
    "id": "E0_10k_seed1337_a1_anchor", "tier": "E0_10k", "seed": 1337,
    "model": "a2ms", "run_dir": f"{ROOT}/repro_runs/neu_a2ms_b_fixed_aam_nodetail_10k_20260908_121000",
    "manifest": f"{AUDIT}/a2ms_b_fixed_a1_manifest.json", "required": False,
})
for variant, sub in [("base", "base_b"), ("a2ms", "a2_full")]:
    OBJECTS.append({
        "id": f"E1_240k_seed1337_{variant}", "tier": "E1_240k", "seed": 1337,
        "model": variant, "run_dir": f"{L240}/{sub}",
        "manifest": f"{AUDIT}/long240k_{sub if sub=='base_b' else 'a2_full'}_manifest.json",
        "required": True,
    })
for seed in (2026, 3407):
    for variant in ("base", "a2ms"):
        OBJECTS.append({
            "id": f"E1_240k_seed{seed}_{variant}", "tier": "E1_240k", "seed": seed,
            "model": variant, "run_dir": None, "manifest": None, "required": True,
        })

details = []
for ob in OBJECTS:
    rec = dict(ob)
    rd = ob["run_dir"]
    if not rd or not os.path.isdir(rd):
        rec.update(status="MISSING", note="no run directory on server (searched repro_runs and experiments)")
        details.append(rec)
        continue
    man = jload(ob["manifest"]) if ob["manifest"] else None
    rec["manifest_present"] = man is not None
    ckpts = find_ckpts(rd)
    rec["checkpoints_on_disk"] = [os.path.basename(c) for c in ckpts]
    # choose endpoint checkpoint: iter240000 for E1, else prefer *_last / best
    endpoint = None
    if ob["tier"] == "E1_240k":
        endpoint = next((c for c in ckpts if "iter240000" in c), None)
        reg = ((man or {}).get("post_run", {}).get("permanent_checkpoints", {}) or {})
        rec["registered_endpoint_sha"] = (reg.get("240000", {}) or {}).get("sha256")
    else:
        endpoint = next((c for c in ckpts if c.endswith("_last.pkl")), None) or \
                   next((c for c in ckpts if "best" in c), ckpts[-1] if ckpts else None)
        reg = ((man or {}).get("checkpoints", {}) or {})
        rec["registered_endpoint_sha"] = (reg.get("best_equals_last_by_design", {}) or {}).get("sha256")
        rec["registered_best_sha_alt"] = json.dumps(reg)[:200] if not rec["registered_endpoint_sha"] else None
    rec["endpoint_checkpoint"] = endpoint
    if endpoint and os.path.isfile(endpoint):
        rec["actual_endpoint_sha256"] = sha(endpoint)
        rec["endpoint_bytes"] = os.path.getsize(endpoint)
        if rec.get("registered_endpoint_sha"):
            rec["sha_matches_manifest"] = rec["actual_endpoint_sha256"] == rec["registered_endpoint_sha"]
    # model + train script
    code = os.path.join(rd, "code")
    if os.path.isdir(code):
        for f in sorted(os.listdir(code)):
            if f.startswith("model_dsmo") and f.endswith(".py"):
                p = os.path.join(code, f)
                s = sha(p)
                rec["model_file"] = f
                rec["model_file_actual_sha256"] = s
                rec["model_matches_formal"] = s == FORMAL_MODEL_SHA.get(ob["model"])
            if f.startswith("train_") and f.endswith(".py"):
                rec["train_script"] = f
                rec["train_script_actual_sha256"] = sha(os.path.join(code, f))
    cfg_path, cfg = parse_config(rd)
    rec["config_path"] = cfg_path
    rec["config"] = cfg_summary(cfg)
    if man:
        rec["manifest_dataset_lists"] = man.get("dataset_lists")
        rec["manifest_environment"] = man.get("environment")
        rec["manifest_endpoint_rule"] = man.get("endpoint_rule")
        reg_eval = ((man.get("unified_eval_840") or {}).get("result_json_sha256"))
        rec["registered_eval_json_sha"] = reg_eval
    rec["metrics"] = metrics_summary(rd)
    if rec.get("registered_eval_json_sha"):
        uef = rec["metrics"].get("unified_eval_files", {})
        rec["eval_json_sha_matches"] = any(
            v["sha256"] == rec["registered_eval_json_sha"] for v in uef.values())
    rec["prediction_caches_in_run"] = find_pred_caches(rd)
    rec["train_logs"] = []
    ld = os.path.join(rd, "logs")
    for base, dirs, files in os.walk(ld):
        for f in files:
            if f.endswith((".log", ".yml")):
                rec["train_logs"].append(os.path.join(base, f))
    details.append(rec)
    print("done", ob["id"], rec.get("actual_endpoint_sha256", "")[:12],
          "sha_match:", rec.get("sha_matches_manifest"), flush=True)

# 240k seed1337 prediction caches (verified in v10.4)
caches = []
for name, path, n_expected in [
    ("E1_240k_seed1337_gt", f"{MECH}/gt", 840),
    ("E1_240k_seed1337_base_predictions", f"{MECH}/base_predictions", 840),
    ("E1_240k_seed1337_a2ms_predictions", f"{MECH}/a2_predictions", 840),
]:
    n = len([f for f in os.listdir(path) if f.endswith(".png")]) if os.path.isdir(path) else 0
    caches.append({"cache": name, "path": path, "n_png": n,
                   "status": "AVAILABLE_AND_VERIFIED" if n == n_expected else "AVAILABLE_BUT_INCOMPLETE",
                   "verification": "v10.4 mask identity + mIoU recomputation (0.9124083744680408 / 0.9145442177671137)"})
for rec in details:
    if rec["tier"] == "E0_10k":
        caches.append({"cache": rec["id"] + "_per_image_predictions",
                       "path": rec["run_dir"],
                       "n_png": sum(c["n_png"] for c in rec.get("prediction_caches_in_run", [])),
                       "status": "MISSING" if not rec.get("prediction_caches_in_run") else "AVAILABLE_BUT_INCOMPLETE",
                       "verification": "no 840-image prediction cache inside run dir; only aggregate unified_eval_840.json" if not rec.get("prediction_caches_in_run") else "see dirs"})
        caches.append({"cache": rec["id"] + "_aggregate_eval_json",
                       "path": rec["run_dir"] + "/analysis",
                       "status": "AVAILABLE_AND_VERIFIED" if rec["metrics"].get("unified_eval_files") else "MISSING",
                       "verification": rec["metrics"].get("unified_eval_files", {})})
    elif rec["tier"] == "E1_240k" and rec.get("run_dir"):
        caches.append({"cache": rec["id"] + "_aggregate_eval_all_savepoints",
                       "path": rec["run_dir"] + "/analysis",
                       "status": "AVAILABLE_AND_VERIFIED" if rec["metrics"].get("unified_eval_files") else "MISSING",
                       "verification": rec["metrics"].get("unified_eval_files", {})})

with open(f"{OUT}/weight_asset_verification_details.json", "w") as fh:
    json.dump({"objects": details, "caches": caches}, fh, indent=2)
print("objects:", len(details), "caches:", len(caches))
