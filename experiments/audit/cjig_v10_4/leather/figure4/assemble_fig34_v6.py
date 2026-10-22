#!/usr/bin/env python3
"""v6 relabel assembly of Fig.3 / Fig.4 — identical panels and row order as v5,
only suptitle model names updated (Base-B -> DSMONet-B).
Outputs PNG(600dpi)+PDF+SVG to docs/paper/figures/."""
import os, json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont("/usr/share/fonts/truetype/arphic/uming.ttc")
plt.rcParams["font.sans-serif"] = ["AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = "/workspace/Industrial Surface Defect"
FIG = f"{ROOT}/Industrial-Surface-Defect/docs/paper/figures"

def stack(panels, row_labels, out_base, title):
    imgs = [Image.open(p) for p in panels]
    w = max(i.width for i in imgs)
    imgs = [i.resize((w, int(i.height * w / i.width)), Image.BILINEAR) for i in imgs]
    n = len(imgs)
    fig, axes = plt.subplots(n, 1, figsize=(w / 260, n * imgs[0].height / 260 + 0.4))
    for ax, im, lab in zip(axes, imgs, row_labels):
        ax.imshow(im); ax.axis("off")
        ax.set_ylabel(lab, fontsize=11, rotation=0, ha="right", va="center", labelpad=68)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    for ext, kw in (("png", dict(dpi=600)), ("pdf", {}), ("svg", {})):
        fp = f"{FIG}/{out_base}.{ext}"
        fig.savefig(fp, bbox_inches="tight", **kw)
        print("saved", fp)
    plt.close(fig)

# ---- Fig.3 NEU ----
V = f"{ROOT}/repro_runs/mechanism_eval_240k/visualizations"
fig3_panels = [
    (f"{V}/crazing_top3_improved_1_000653.png", "裂纹·改善"),
    (f"{V}/crazing_bottom3_worsened_1_000722.png", "裂纹·退化"),
    (f"{V}/crazing_zero3_1_000058.png", "裂纹·相当"),
    (f"{V}/patches_top3_improved_1_000802.png", "斑块·改善"),
    (f"{V}/patches_bottom3_worsened_1_000722.png", "斑块·退化"),
    (f"{V}/patches_zero3_1_000439.png", "斑块·相当"),
]
for p, _ in fig3_panels:
    assert os.path.exists(p), p
stack([p for p, _ in fig3_panels], [l for _, l in fig3_panels], "Fig3_NEU_Qualitative",
      "NEU-Seg 可视化（每行：原图 / GT / DSMONet-B / A2MS-DSMONet-B / DSMONet-B 误差 / A2MS 误差）")

# ---- Fig.4 Leather ----
FV = f"{ROOT}/repro_runs/leather_historical_audit/final_viz"
m4 = json.load(open(f"{FV}/final_viz_manifest.json"))
order = ["improved", "worsened", "near_zero"]
lab_map = {"improved": "改善", "worsened": "退化", "near_zero": "相当"}
panels4, labels4 = [], []
for g in order:
    for c in [c for c in m4["cases"] if c["group"] == g]:
        panels4.append(c["file"]); labels4.append(f"{lab_map[g]}·{c['sample_id']}")
stack(panels4, labels4, "Fig4_Leather_Qualitative",
      "皮革缺陷可视化（每行：原图 / GT / DSMONet-B / A2MS-DSMONet-B）")
print("DONE")
