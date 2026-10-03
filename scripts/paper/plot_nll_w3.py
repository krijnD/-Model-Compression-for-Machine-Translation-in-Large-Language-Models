"""Appendix figure: at 3 bits, does the model lose the language or the translation mapping?

x = monolingual NLL increase over fp16 per language, y = translation NLL increase for both directions
(into English and out of English), joined per language. One ACL column wide.
Reads results/json/lang_nll_{fp16,w3}.json, writes paper/figures/nll_w3.pdf.
Usage: python scripts/paper/plot_nll_w3.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mtcompress.paths import FIGURES, JSON as J

OUT = FIGURES / "nll_w3.pdf"
LANGS = {"is": "Icelandic", "de": "German", "cs": "Czech", "ru": "Russian", "zh": "Chinese"}

fp16 = json.load(open(J / "lang_nll_fp16.json"))["nll"]
w3 = json.load(open(J / "lang_nll_w3.json"))["nll"]
d = {k: w3[k] - fp16[k] for k in w3}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "pdf.fonttype": 42, "mathtext.fontset": "stix"})
fig, ax = plt.subplots(figsize=(3.03, 2.0))
for lang, name in LANGS.items():
    x = d[f"mono:{lang}"]
    into, out = d[f"trans:{lang}-en"], d[f"trans:en-{lang}"]
    c = "#c0392b" if lang == "is" else "#555555"
    ax.plot([x, x], [into, out], color=c, lw=1, zorder=1)
    ax.scatter([x], [into], marker="^", s=22, color=c, zorder=2)
    ax.scatter([x], [out], marker="v", s=22, color=c, zorder=2)
    ax.annotate(name, (x, max(into, out)), xytext=(4, 2), textcoords="offset points", fontsize=7, color=c,
                fontweight="bold" if lang == "is" else "normal")
ax.scatter([], [], marker="^", color="#555555", label=r"into English")
ax.scatter([], [], marker="v", color="#555555", label=r"out of English")
ax.legend(loc="upper right", fontsize=6.5, frameon=False, handletextpad=0.2)
ax.set_xlabel(r"monolingual $\Delta$NLL (nats/token)")
ax.set_ylabel(r"translation $\Delta$NLL")
ax.set_xlim(0.15, 0.95)
ax.set_ylim(0, 0.75)
ax.grid(alpha=0.3, lw=0.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.1)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.01)
print(OUT)
