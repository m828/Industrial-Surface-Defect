#!/usr/bin/env python3
"""v6 relabel of leather_viz_final.py: same historical val-selected checkpoints
(DSMONet-B 0127@78k, A2MS 160k@130k), locked test468, same fixed selection rule.
Only column titles / suptitle renamed (Base-B -> DSMONet-B). Inference only; no training.
Consistency gate vs audited aggregate mIoU retained."""
import os, sys, json, importlib
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont("/usr/share/fonts/truetype/arphic/uming.ttc")
plt.rcParams["font.sans-serif"] = ["AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = "/workspace/Industrial Surface Defect"
NEW = f"{ROOT}/new"
PIGE = f"{ROOT}/new (copy)/dataset/pige"
TEST = f"{ROOT}/Industrial-Surface-Defect/experiments/protocol/leather_valid_test_list.txt"
OUT = f"{ROOT}/repro_runs/leather_historical_audit/final_viz"
os.makedirs(OUT, exist_ok=True)

RUNS = {"base": {"code": NEW, "module": "model_dsmo_rs50",
                 "ckpt": f"{ROOT}/new (copy)/model_savePath/dsmonet_resnet_pascal_pige_dsmor50_0127_80000.pkl",
                 "expect_miou": 0.899120},
        "a2": {"code": NEW, "module": "model_dsmo_rs50_eSE_adapt_detailloss_fixed",
               "ckpt": f"{NEW}/dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl",
               "expect_miou": 0.909713}}

lines = [l.strip().split() for l in open(TEST) if l.strip()]
assert len(lines) == 468
ids = [os.path.splitext(os.path.basename(a))[0] for a, _ in lines]

def agg_miou(preds, gts):
    inter = np.zeros(8); union = np.zeros(8)
    for sid in ids:
        p, g = preds[sid], gts[sid]
        for c in range(8):
            pc, gc = (p == c), (g == c)
            inter[c] += np.logical_and(pc, gc).sum()
            union[c] += np.logical_or(pc, gc).sum()
    return float(np.mean(inter / union))

gts = {}
for i, (_, rel_lab) in enumerate(lines):
    gts[ids[i]] = np.array(Image.open(os.path.join(PIGE, rel_lab)).resize((768, 768), Image.NEAREST)).astype(np.uint8)

preds = {}
for tag, r in RUNS.items():
    os.chdir(r["code"]); sys.path.insert(0, r["code"])
    for m in list(sys.modules):
        if m.startswith(("model_dsmo", "model_resnet")):
            del sys.modules[m]
    from model_resnet import resnet50
    mod = importlib.import_module(r["module"])
    model = mod.DSMONet(num_classes=8, backbone=resnet50()).cuda()
    model.load_state_dict(torch.load(r["ckpt"], map_location="cuda", weights_only=False)["model_state"])
    model.eval()
    preds[tag] = {}
    with torch.no_grad():
        for i, (rel_img, _) in enumerate(lines):
            img = Image.open(os.path.join(PIGE, rel_img)).convert("RGB").resize((768, 768), Image.BILINEAR)
            x = torch.from_numpy(np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0)[None].cuda()
            preds[tag][ids[i]] = model(x)[0].data.max(1)[1].cpu().numpy()[0].astype(np.uint8)
    del model; torch.cuda.empty_cache()
    m = agg_miou(preds[tag], gts)
    print(tag, "aggregate mIoU from cached preds:", m, "expected:", r["expect_miou"])
    assert abs(m - r["expect_miou"]) < 2e-4, f"consistency gate FAILED for {tag}: {m} vs {r['expect_miou']}"
    print(tag, "consistency gate PASSED")

def img_miou(pred, gt):
    ious = []
    for c in range(8):
        g = gt == c; p = pred == c
        u = np.logical_or(g, p).sum()
        if u > 0:
            ious.append(np.logical_and(g, p).sum() / u)
    return float(np.mean(ious))

delta = {}
for sid in ids:
    gt = gts[sid]
    if int((gt > 0).sum()) == 0:
        continue
    mb = img_miou(preds["base"][sid], gt)
    ma = img_miou(preds["a2"][sid], gt)
    delta[sid] = {"base": mb, "a2": ma, "delta": ma - mb}

srt = sorted(delta.items(), key=lambda kv: kv[1]["delta"])
sel = [("worsened", k) for k, _ in srt[:2]] + [("improved", k) for k, _ in srt[-2:][::-1]] + \
      [("near_zero", k) for k, _ in sorted(delta.items(), key=lambda kv: abs(kv[1]["delta"]))[:2]]

PALETTE = np.array([[40, 40, 40], [220, 60, 60], [60, 160, 60], [60, 100, 220],
                    [220, 180, 40], [180, 60, 200], [60, 200, 200], [230, 130, 40]], np.uint8)
CLSN = ["background", "open_wound", "scratch", "brand_mark", "hole", "skin_disease", "rotten_surface", "wart"]

manifest = []
for group, sid in sel:
    gt = gts[sid]
    img = np.array(Image.open(os.path.join(PIGE, f"images/{sid}.jpg")).convert("RGB").resize((768, 768), Image.BILINEAR))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, data, title in zip(axes,
            [img, PALETTE[gt], PALETTE[preds["base"][sid]], PALETTE[preds["a2"][sid]]],
            ["原图", "标注 GT", "DSMONet-B 预测", "A2MS-DSMONet-B 预测"]):
        ax.imshow(data); ax.set_title(title, fontsize=11); ax.axis("off")
    d = delta[sid]
    present = sorted(set(np.unique(gt).tolist()) - {0})
    fig.suptitle(f"{group} | {sid} | classes={[CLSN[c] for c in present]} | ΔmIoU(A2MS-DSMONet)={d['delta']:+.4f} (DSMONet-B {d['base']:.4f} / A2MS-B {d['a2']:.4f})", fontsize=9)
    fig.tight_layout()
    fp = f"{OUT}/leather_final_{group}_{sid}.png"
    fig.savefig(fp, dpi=150); plt.close(fig)
    manifest.append({"group": group, "sample_id": sid, "gt_classes": present, **d, "file": fp})
    print("saved", fp)

n_defect = len(delta)
pos = sum(1 for v in delta.values() if v["delta"] > 1e-6)
neg = sum(1 for v in delta.values() if v["delta"] < -1e-6)
json.dump({"checkpoints": {"base": RUNS["base"]["ckpt"], "a2": RUNS["a2"]["ckpt"]},
           "base_actual_iter": 78000, "a2_actual_iter": 130000,
           "consistency_gate": {"base": 0.899120, "a2": 0.909713, "status": "passed"},
           "rule": "defect-bearing subset of test468 ranked by per-image 8-class mIoU delta (A2MS-DSMONet-B minus DSMONet-B); top2/bottom2/2-nearest-zero",
           "n_defect_bearing": n_defect, "n_improved": pos, "n_worsened": neg, "n_tie": n_defect - pos - neg,
           "label_version": "v6 (DSMONet-B naming)",
           "cases": manifest}, open(f"{OUT}/final_viz_manifest.json", "w"), indent=2)
print("DONE", len(manifest), "panels; defect-bearing:", n_defect, "improved:", pos, "worsened:", neg)
