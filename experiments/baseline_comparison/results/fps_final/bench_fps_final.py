#!/usr/bin/env python3
"""CJIG v7 unified 10-model FPS benchmark (NEU-Seg inference path).

Protocol (frozen for this run):
- GPU: NVIDIA A100-PCIE-40GB, batch=1, FP32, model.eval(), torch.no_grad()
- Entry input: 200x200 for every model; models with structural divisibility
  constraints keep their internal padding (U-Net/LETNet 200->208, BiSeNetV2
  200->224) via the same _PadCropWrapper used by tools/evaluate_class_iou.py.
- Warmup: 100 iterations per model (round-robin 10x10).
- Measurement: 10 interleaved blocks x 100 per-iteration timings per model,
  models cycled round-robin within each block (equal exposure to any GPU
  frequency/temperature drift or transient contention).
- FPS = 1000 / median(per-iteration latency in ms).

Deployment-path consistency with complexity_audit_final.md:
- BiSeNetV2 is benchmarked in aux_mode='eval' (aux2/3/4/5_4 heads removed);
  MACs 2.36G refer to this form. Checkpoint conversion: drop aux* keys.
- FDSNet is benchmarked with aux=False (edge/semantic aux layers removed);
  MACs 0.17G refer to this form. Checkpoint conversion: drop aux layer keys.
- Both conversions are verified to leave the main segmentation output
  bit-identical to the as-evaluated form before timing.
- DDRNet-23/PIDNet-S aux branches are self.training-gated (skipped by eval());
  STDC-Seg executes all heads unconditionally and is timed as-is;
  DSMONet-B/A2MS-DSMONet-B detail-supervision heads are self.training-gated.
"""
import datetime
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path("/workspace/Industrial Surface Defect")
REPO = WORKSPACE / "Industrial-Surface-Defect"
EVAL_TOOL = REPO / "tools" / "evaluate_class_iou.py"
OUT_DIR = REPO / "experiments" / "baseline_comparison" / "results" / "fps_final"

WARMUP = 100          # per model
BLOCKS = 10           # interleaved blocks
ITERS_PER_BLOCK = 100 # per-iteration timings per model per block

# Frozen reference values from final_baseline_table_v3.md (never recomputed here)
FROZEN = {
    "U-Net":           {"params_m": 31.04, "macs_g": 36.10},
    "DeepLabv3+":      {"params_m": 59.34, "macs_g": 14.29},
    "BiSeNetV2":       {"params_m": 5.20,  "macs_g": 2.36},
    "STDC-Seg":        {"params_m": 16.07, "macs_g": 6.02},
    "PIDNet-S":        {"params_m": 7.72,  "macs_g": 0.97},
    "DDRNet-23":       {"params_m": 20.30, "macs_g": 2.90},
    "LETNet":          {"params_m": 0.95,  "macs_g": 1.21},
    "FDSNet":          {"params_m": 2.59,  "macs_g": 0.17},
    "DSMONet-B":       {"params_m": 29.37, "macs_g": 6.65},
    "A2MS-DSMONet-B":  {"params_m": 29.46, "macs_g": 6.65},
}

JOBS = [
    {"name": "U-Net", "alias": "unet",
     "ckpt": WORKSPACE / "repro_runs/baseline_neu_240k_classical/unet/checkpoints/neu_unet_240k_iter240000.pkl",
     "internal_input": 208},
    {"name": "DeepLabv3+", "alias": "deeplabv3p",
     "ckpt": WORKSPACE / "repro_runs/baseline_neu_240k_classical/deeplabv3p/checkpoints/neu_deeplabv3p_240k_iter240000.pkl",
     "internal_input": 200},
    {"name": "BiSeNetV2", "alias": "bisenetv2",
     "ckpt": WORKSPACE / "repro_runs/baseline_neu_240k_classical/bisenetv2/checkpoints/neu_bisenetv2_240k_iter240000.pkl",
     "internal_input": 224, "deployment": "aux_mode_eval"},
    {"name": "STDC-Seg", "alias": "stdc",
     "ckpt": WORKSPACE / "repro_runs/baseline_neu_240k/stdc/checkpoints/neu_stdc_bisenet_240k_iter240000.pkl",
     "internal_input": 200},
    {"name": "PIDNet-S", "alias": "pidnet-s",
     "ckpt": WORKSPACE / "repro_runs/baseline_neu_240k/pidnet/checkpoints/neu_pidnet_240k_iter240000.pkl",
     "internal_input": 200},
    {"name": "DDRNet-23", "alias": "ddrnet",
     "ckpt": WORKSPACE / "repro_runs/baseline_neu_240k/ddrnet/checkpoints/neu_ddrnet_240k_iter240000.pkl",
     "internal_input": 200},
    {"name": "LETNet", "alias": "letnet",
     "ckpt": WORKSPACE / "repro_runs/baseline_neu_240k_industrial/letnet/checkpoints/neu_letnet_240k_iter240000.pkl",
     "internal_input": 208},
    {"name": "FDSNet", "alias": "fdsnet",
     "ckpt": WORKSPACE / "repro_runs/baseline_neu_240k_industrial/fdsnet/checkpoints/neu_fdsnet_240k_iter240000.pkl",
     "internal_input": 200, "deployment": "aux_false"},
    {"name": "DSMONet-B", "alias": "base_b",
     "ckpt": WORKSPACE / "repro_runs/long240k_seed1337/base_b/checkpoints/base_b_s1337_long240k_codex1_iter240000.pkl",
     "internal_input": 200},
    {"name": "A2MS-DSMONet-B", "alias": "a2ms_b",
     "ckpt": WORKSPACE / "repro_runs/long240k_seed1337/a2_full/checkpoints/a2_full_s1337_long240k_codex1_iter240000.pkl",
     "model_file": WORKSPACE / "new/model_dsmo_rs50_eSE_adapt_detailloss_fixed.py",
     "internal_input": 200},
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_eval_tool():
    spec = importlib.util.spec_from_file_location("ev", EVAL_TOOL)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def build_standard(ev, job, device):
    key = ev.normalize_model_name(job["alias"])
    model_file = Path(job.get("model_file") or ev.DEFAULT_MODEL_FILES[key])
    model, desc = ev.MODEL_BUILDERS[key](model_file, 4, device)
    ev.load_state(model, job["ckpt"], device, strict=True)
    return model, desc, model_file


def build_bisenetv2_deploy(ev, job, device):
    """Deployment form: aux_mode='eval' (aux heads absent, MACs 2.36G form).
    Conversion rule: drop aux2./aux3./aux4./aux5_4. keys from the 240k checkpoint."""
    model_file = Path(ev.DEFAULT_MODEL_FILES["BiSeNetV2"])
    module = ev.load_module_from_file(model_file, "fps_bisenetv2_module")
    module.BiSeNetV2.load_pretrain = lambda self: None
    inner = module.BiSeNetV2(4, aux_mode="eval")
    model = ev._PadCropWrapper(inner, pad_to=224).to(device)
    state = ev.strip_module_prefix(ev.extract_state_dict(ev.torch_load_checkpoint(job["ckpt"], device)))
    dropped = [k for k in state if k.split(".")[0] in ("aux2", "aux3", "aux4", "aux5_4")]
    state = {k: v for k, v in state.items() if k.split(".")[0] not in ("aux2", "aux3", "aux4", "aux5_4")}
    model.load_state_dict(state, strict=True)
    desc = "BiSeNetV2(aux_mode='eval', aux heads stripped) with 200->224 pad / 224->200 crop wrapper"
    return model, desc, model_file, dropped


def build_fdsnet_deploy(ev, job, device):
    """Deployment form: aux=False (edge/semantic aux layers absent, MACs 0.17G form).
    Conversion rule: drop edge_auxlayer./semantic_auxlayer. keys from the checkpoint."""
    model_file = Path(ev.DEFAULT_MODEL_FILES["FDSNet"])
    import types
    ev._install_mmcv_shim()
    pkg = types.ModuleType("fdsnet_models")
    pkg.__path__ = [str(model_file.parent)]
    sys.modules.setdefault("fdsnet_models", pkg)
    module = ev.load_module_from_file(model_file, "fdsnet_models.fdsnet_fps")
    model = module.FDSNet(num_classes=4, aux=False).to(device)
    state = ev.strip_module_prefix(ev.extract_state_dict(ev.torch_load_checkpoint(job["ckpt"], device)))
    dropped = [k for k in state if k.split(".")[0] in ("edge_auxlayer", "semantic_auxlayer")]
    state = {k: v for k, v in state.items() if k.split(".")[0] not in ("edge_auxlayer", "semantic_auxlayer")}
    model.load_state_dict(state, strict=True)
    desc = "FDSNet(num_classes=4, aux=False, aux layers stripped) [deployment form]"
    return model, desc, model_file, dropped


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ev = load_eval_tool()
    # Pre-bind vendored utils/model_resnet from new/ into sys.modules. Otherwise
    # loading deeplabv3.py prepends "new (copy)/model" to sys.path and its local
    # utils.py shadows the correct utils when model_resnet is imported later.
    # (new/ and "new (copy)/" copies of both files are byte-identical.)
    sys.path.insert(0, str(WORKSPACE / "new"))
    import utils  # noqa: F401
    import model_resnet  # noqa: F401
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False  # fixed input size; keep clocks stable

    results = {}
    models = {}
    meta = {}

    # ---- load all models (deployment conversion where required) ----
    for job in JOBS:
        name = job["name"]
        dropped = []
        if job.get("deployment") == "aux_mode_eval":
            model, desc, model_file, dropped = build_bisenetv2_deploy(ev, job, device)
            # verify prediction identity vs as-evaluated form
            ref, _, _ = build_standard(ev, {"name": name, "alias": job["alias"], "ckpt": job["ckpt"]}, device)
            ref.eval()
        elif job.get("deployment") == "aux_false":
            model, desc, model_file, dropped = build_fdsnet_deploy(ev, job, device)
            ref, _, _ = build_standard(ev, {"name": name, "alias": job["alias"], "ckpt": job["ckpt"]}, device)
            ref.eval()
        else:
            model, desc, model_file = build_standard(ev, job, device)
            ref = None
        model.eval()
        models[name] = model
        meta[name] = {
            "desc": desc, "model_file": str(model_file), "dropped_keys": len(dropped),
            "ref_model": ref,
        }
        print(f"[load] {name}: {desc} | dropped_aux_keys={len(dropped)}", flush=True)

    # ---- deployment conversion verification (main output must be identical) ----
    x_verify = torch.randn(2, 3, 200, 200, device=device)
    with torch.no_grad():
        for name, m in models.items():
            ref = meta[name].pop("ref_model")
            if ref is None:
                meta[name]["conversion_max_abs_diff"] = None
                continue
            out_new = m(x_verify)
            out_ref = ref(x_verify)
            if isinstance(out_new, (list, tuple)):
                out_new = out_new[0]
            if isinstance(out_ref, (list, tuple)):
                out_ref = out_ref[0]
            diff = (out_new - out_ref).abs().max().item()
            meta[name]["conversion_max_abs_diff"] = diff
            assert diff == 0.0, f"{name}: deployment conversion changed predictions (max|d|={diff})"
            del ref
            torch.cuda.empty_cache()
            print(f"[verify] {name}: deployment output identical to as-evaluated form", flush=True)

    # ---- fixed input ----
    gen = torch.Generator(device=device).manual_seed(1337)
    x = torch.randn(1, 3, 200, 200, device=device, generator=gen)

    # ---- in-process foreign-GPU-process poller (2s interval) ----
    own_pid = str(os.getpid())
    foreign_samples = []  # (epoch, [foreign pids])
    poll_stop = threading.Event()

    def _poller():
        while not poll_stop.is_set():
            try:
                pids = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                    capture_output=True, text=True).stdout.split()
                pids = [p for p in pids if p != own_pid]
                if pids:
                    foreign_samples.append((time.time(), pids))
            except Exception:
                pass
            poll_stop.wait(2.0)

    poll_thread = threading.Thread(target=_poller, daemon=True)
    poll_thread.start()

    # ---- warmup: 100 per model, round-robin 10x10 ----
    with torch.no_grad():
        for _ in range(WARMUP // 10):
            for m in models.values():
                for _ in range(10):
                    m(x)
        torch.cuda.synchronize()
        print("[warmup] done", flush=True)

        # ---- interleaved measurement ----
        block_med = {n: [] for n in models}
        block_meta = []
        for b in range(BLOCKS):
            block_start = time.time()
            t_block = time.perf_counter()
            block_samples = {}
            for name, m in models.items():
                samples = []
                for _ in range(ITERS_PER_BLOCK):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    m(x)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - t0) * 1000.0)
                block_samples[name] = samples
                block_med[name].append(float(np.median(samples)))
            block_end = time.time()
            block_meta.append({
                "block": b + 1,
                "start_epoch": block_start,
                "end_epoch": block_end,
                "wall_s": round(time.perf_counter() - t_block, 2),
                "samples": block_samples,
            })
            print(f"[block] {b + 1}/{BLOCKS} done wall={block_meta[-1]['wall_s']}s", flush=True)

    poll_stop.set()
    poll_thread.join(timeout=5)

    # ---- per-block contamination judgement ----
    def block_foreign(bm):
        lo, hi = bm["start_epoch"] - 1.0, bm["end_epoch"] + 1.0
        hits = [p for ep, pids in foreign_samples if lo <= ep <= hi for p in pids]
        return sorted(set(hits))

    for bm in block_meta:
        bm["foreign_pids"] = block_foreign(bm)
        bm["contaminated"] = bool(bm["foreign_pids"])
    clean_blocks = [bm["block"] for bm in block_meta if not bm["contaminated"]]
    contaminated_blocks = [bm["block"] for bm in block_meta if bm["contaminated"]]
    usable = len(clean_blocks) >= 6  # stats over clean blocks only, >=600 samples/model
    print(f"[judge] clean_blocks={clean_blocks} contaminated={contaminated_blocks} usable={usable}", flush=True)

    gpu_name = torch.cuda.get_device_name(0)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    for name in models:
        pooled = [s for bm in block_meta if not bm["contaminated"] for s in bm["samples"][name]]
        if not pooled:  # fully contaminated fallback: all samples (flagged unusable anyway)
            pooled = [s for bm in block_meta for s in bm["samples"][name]]
        arr = np.array(pooled)
        med = float(np.median(arr))
        job = next(j for j in JOBS if j["name"] == name)
        results[name] = {
            "input_size": "200x200 (entry)",
            "internal_input_size": f'{job["internal_input"]}x{job["internal_input"]}',
            "latency_ms_mean": round(float(arr.mean()), 4),
            "latency_ms_median": round(med, 4),
            "latency_std": round(float(arr.std()), 4),
            "fps": round(1000.0 / med, 2),
            "gpu": gpu_name,
            "precision": "FP32",
            "batch": 1,
            "timestamp": timestamp,
            "checkpoint": str(job["ckpt"]),
            "checkpoint_sha256": sha256_of(job["ckpt"]),
            "model_file": meta[name]["model_file"],
            "model_desc": meta[name]["desc"],
            "deployment_conversion_dropped_keys": meta[name]["dropped_keys"],
            "conversion_max_abs_diff": meta[name]["conversion_max_abs_diff"],
            "params_m_measured_form": round(sum(p.numel() for p in models[name].parameters()) / 1e6, 4),
            "params_m_frozen_ref": FROZEN[name]["params_m"],
            "macs_g_frozen_ref": FROZEN[name]["macs_g"],
            "block_medians_ms": [round(v, 4) for v in block_med[name]],
            "blocks_used": clean_blocks,
            "n_samples": int(arr.size),
            "protocol": f"interleaved {BLOCKS}x{ITERS_PER_BLOCK} per-iteration timings, warmup {WARMUP}, batch1 FP32 eval no_grad, stats over clean blocks only, FPS=1000/median",
        }
        print(f"[result] {name:15s} median {med:7.3f} ms  mean {arr.mean():7.3f}  std {arr.std():6.3f}  FPS {1000.0 / med:8.2f}", flush=True)

    with open(OUT_DIR / "fps_final_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    clean = not contaminated_blocks
    with open(OUT_DIR / "fps_final_run_meta.json", "w") as f:
        json.dump({
            "gpu": gpu_name, "timestamp": timestamp,
            "clean_run": clean, "usable": usable,
            "clean_blocks": clean_blocks, "contaminated_blocks": contaminated_blocks,
            "foreign_samples": [(round(ep, 1), pids) for ep, pids in foreign_samples],
            "blocks": [{k: v for k, v in bm.items() if k != "samples"} for bm in block_meta],
            "torch": torch.__version__,
            "cudnn_benchmark": False,
            "protocol": f"interleaved {BLOCKS}x{ITERS_PER_BLOCK} per-iteration timings, warmup {WARMUP}, batch1 FP32 eval no_grad, stats over clean blocks only, FPS=1000/median",
        }, f, indent=2, ensure_ascii=False)
    print(f"[done] clean_run={clean} usable={usable} wrote results + run meta", flush=True)
    sys.exit(0 if usable else 3)


if __name__ == "__main__":
    main()
