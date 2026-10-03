"""Paper figure for RQ2: peak GPU memory vs average XCOMET-XXL, with arrows from each plain low-bit model to its
distilled-adapter version. One ACL column wide.

Memory: peak_alloc at batch 4 from results/json/probe16_<tag>.json (de-en, beam 5, source 256), with the
adapter counted in bf16 (measured fp32 peak minus 2 bytes per adapter parameter, the paper's convention).
Quality: mean XCOMET-XXL over the 10 directions from results/scores/<run>.tsv.
Writes paper/figures/rq2_tradeoff.pdf. Usage: python scripts/paper/plot_rq2_tradeoff.py
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mtcompress.paths import FIGURES, JSON, SCORES

OUT = FIGURES / "rq2_tradeoff.pdf"
PAIRS = "de-en cs-en is-en zh-en ru-en en-de en-cs en-is en-zh en-ru".split()
GIB = 1024 ** 3
ADAPTER_BF16_SAVING = {"w3kd": 62_586_880 * 2 / GIB, "w2kd": 250_347_520 * 2 / GIB}  # fp32 -> bf16
# label: (probe tag, score run)
MODELS = {"fp16": ("fp16", "ours-beam"), "w8": ("w8", "gptq-w8g128"), "w4": ("w4", "gptq-w4g128"),
          "w3": ("w3", "gptq-w3g128"), "w2": ("w2", "gptq-w2g128"),
          "w3+KD": ("w3kd", "gptq-w3g128-as4-kd"), "w2+KD": ("w2kd", "gptq-w2g128-as4-kd-r64-cont")}


def peak(tag):
    d = json.load(open(JSON / f"probe16_{tag}.json"))
    return d["per_batch"]["4"]["peak_alloc_mib"] / 1024 - ADAPTER_BF16_SAVING.get(tag, 0.0)


def xcomet(run):
    rows = [r for r in csv.DictReader(open(SCORES / f"{run}.tsv"), delimiter="\t")
            if r["metric"] == "xcomet-xxl" and r["pair"] in PAIRS]
    assert len(rows) == 10, f"{run}: {len(rows)} directions scored"
    return sum(float(r[run]) for r in rows) / 10


LABEL = {"w3+KD": (0.7, 2.5, "left"), "w2+KD": (-0.6, -6.5, "right"), "w4": (0.6, -6.5, "left"),
         "w3": (0.6, -3.5, "left")}  # (dx, dy, ha) per label, so the crowded corner stays readable
pts = {name: (peak(tag), xcomet(run)) for name, (tag, run) in MODELS.items()}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.03, 2.1))
ax.axhline(pts["fp16"][1], color="#999999", lw=0.8, ls="--", zorder=0)
ax.text(22, pts["fp16"][1] + 2, "fp16 quality", fontsize=6.5, color="#777777", ha="right")
for name, (x, y) in pts.items():
    kd = name.endswith("KD")
    ax.scatter(x, y, s=30 if kd else 22, zorder=3, color="#c0392b" if kd else "#555555",
               marker="*" if kd else "o", edgecolor="none")
    dx, dy, ha = LABEL.get(name, (0.6, -3.5, "left"))
    ax.annotate(name, (x, y), xytext=(x + dx, y + dy), fontsize=7, ha=ha,
                color="#c0392b" if kd else "#333333", fontweight="bold" if kd else "normal")
for plain, rec in (("w3", "w3+KD"), ("w2", "w2+KD")):
    ax.annotate("", xy=pts[rec], xytext=pts[plain],
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.0, shrinkA=4, shrinkB=5))
ax.set_xlabel("peak GPU memory, batch 4 (GiB)")
ax.set_ylabel("XCOMET-XXL (avg. 10 directions)")
ax.set_xlim(5, 31)
ax.set_ylim(15, 100)
ax.grid(alpha=0.3, lw=0.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.1)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.01)
for k, v in pts.items():
    print(f"{k:6s} peak {v[0]:5.2f} GiB  XCOMET {v[1]:5.2f}")
print(OUT)
