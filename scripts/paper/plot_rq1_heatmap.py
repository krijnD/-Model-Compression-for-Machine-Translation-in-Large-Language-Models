"""Paper figure for RQ1: XCOMET-XXL change vs fp16 per direction and bit-width, one ACL column wide.

Reads results/scores/<run>.tsv and writes paper/figures/rq1_xcomet_delta.pdf.
Colour saturates at -40 so the uneven w3 row stays readable next to w2; every cell is annotated.
Usage: python scripts/paper/plot_rq1_heatmap.py
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from mtcompress.paths import FIGURES, SCORES

OUT = FIGURES / "rq1_xcomet_delta.pdf"
RUNS = {"fp16": "ours-beam", "w8": "gptq-w8g128", "w4": "gptq-w4g128", "w3": "gptq-w3g128", "w2": "gptq-w2g128"}
PAIRS = "de-en cs-en is-en zh-en ru-en en-de en-cs en-is en-zh en-ru".split()
METRIC = "xcomet-xxl"


def load(run):
    rows = csv.DictReader(open(SCORES / f"{run}.tsv"), delimiter="\t")
    return {r["pair"]: float(r[run]) for r in rows if r["metric"] == METRIC and r[run] not in ("", "-")}


fp16 = load(RUNS["fp16"])
bits = ["w8", "w4", "w3", "w2"]
delta = np.array([[load(RUNS[b])[p] - fp16[p] for p in PAIRS] for b in bits])

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "font.size": 7, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.03, 1.35))
ax.imshow(np.clip(delta, -40, 0), cmap="Reds_r", vmin=-40, vmax=0, aspect="auto")
for i in range(len(bits)):
    for j in range(len(PAIRS)):
        v = delta[i, j]
        label = "0.0" if abs(v) < 0.05 else (f"{v:+.1f}" if abs(v) < 10 else f"{v:+.0f}")
        ax.text(j, i, label, ha="center", va="center", fontsize=5.6,
                color="white" if v < -25 else "black")
# Icelandic at 3 bits: the uneven loss the text points to
for j in (PAIRS.index("is-en"), PAIRS.index("en-is")):
    ax.add_patch(Rectangle((j - 0.5, bits.index("w3") - 0.5), 1, 1, fill=False, lw=1.0, ec="black"))
ax.axvline(4.5, color="white", lw=2)
ax.set_xticks(range(len(PAIRS)))
ax.set_xticklabels([p.replace("-", "→") for p in PAIRS], rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(bits)))
ax.set_yticklabels(bits)
ax.tick_params(length=0, pad=2)
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout(pad=0.1)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.01)
print(OUT)
