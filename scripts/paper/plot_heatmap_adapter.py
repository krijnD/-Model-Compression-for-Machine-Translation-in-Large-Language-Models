"""Direction x precision heatmap with the distilled adapter, in the style of fig2_heatmap (make_report.py).

Columns: FP16 (absolute score, grey: the reference), then the change vs FP16 for W4G128, W3G128 and W3G128 + adapter.
Rows: the four directions the adapter was generated for (is-en, en-is, de-en, en-de).
Metrics: BLEU, chrF++ (sacrebleu on outputs/*/wmt22) and XCOMET-XXL (outputs/xcomet-xxl/<run>/<pair>.txt).
MetricX-24 (in fig2) was not run for the adapter outputs, so chrF++ takes its place.
W3 is the repacked run on the same ExllamaV2 kernel as the adapter run (docs/04-kernel-repack.md §8), so the two
W3 columns differ only by the adapter. Colour scale as fig2: RdYlGn_r, symmetric per metric, red = worse.

  ../venv/bin/python francesco/analysis/plot_heatmap_adapter.py   # -> results/figures/fig10_heatmap_adapter.{png,pdf}
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sacrebleu.metrics import BLEU, CHRF  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO.parent / "outputs"
TESTSET = REPO / "third_party/ALMA/outputs/wmt22_outputs/wmt-testset"
FIG = REPO / "francesco/results/figures/fig10_heatmap_adapter"

PAIRS = ["is-en", "en-is", "de-en", "en-de"]
# column label -> (generation folder in outputs/, xcomet folder in outputs/xcomet-xxl/)
RUNS = {"FP16": ("alma-13b-r-beam", "ours-beam"), "W4G128": ("gptq-w4g128", "gptq-w4g128"),
        "W3G128": ("gptq-w3g128-as4", "gptq-w3g128-as4"),
        "W3G128\n+ adapter": ("gptq-w3g128-as4-kd", "gptq-w3g128-as4-kd")}
METRICS = ["bleu", "chrf++", "xcomet-xxl"]
PRETTY = {"bleu": "BLEU", "chrf++": "chrF++", "xcomet-xxl": "XCOMET-XXL"}


def score(metric, run, xc, pair):
    s, t = pair.split("-")
    if metric == "xcomet-xxl":
        f = OUT / "xcomet-xxl" / xc / f"{pair}.txt"
        m = re.findall(r"score: ([0-9.]+)", f.read_text()) if f.exists() else []
        return 100 * float(m[-1]) if m else np.nan
    ref = (TESTSET / f"{s}{t}/test.{pair}.{t}").read_text(encoding="utf-8").splitlines()
    f = OUT / run / "wmt22" / f"test-{pair}"
    if not f.exists():
        return np.nan
    hyp = f.read_text(encoding="utf-8").splitlines()
    if len(hyp) != len(ref):
        return np.nan
    if metric == "bleu":
        return BLEU(tokenize="zh" if t == "zh" else "13a").corpus_score(hyp, [ref]).score
    return CHRF(word_order=2).corpus_score(hyp, [ref]).score


def main():
    cols = list(RUNS)
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.6))
    for ax, metric in zip(axes, METRICS):
        raw = np.array([[score(metric, *RUNS[c], p) for c in cols] for p in PAIRS])
        delta = raw[:, 1:] - raw[:, :1]                     # vs FP16; all metrics here: higher is better
        orient = -delta                                     # positive == worse, as fig2
        vmax = np.nanmax(np.abs(orient)) if np.isfinite(orient).any() else 1.0
        img = np.full(raw.shape, np.nan)
        img[:, 1:] = orient
        ax.imshow(img, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, len(PAIRS), color="#e6e5e0", zorder=0.5))  # FP16: reference, grey
        ax.set_xlim(-0.5, len(cols) - 0.5)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, fontsize=7)
        ax.set_yticks(range(len(PAIRS)))
        ax.set_yticklabels([p.replace("-", r"$\rightarrow$") for p in PAIRS])
        ax.set_title(PRETTY[metric])
        ax.grid(False)
        ax.axvline(0.5, color="white", lw=2)
        for i in range(raw.shape[0]):
            ax.text(0, i, f"{raw[i, 0]:.1f}", ha="center", va="center", fontsize=6.5, color="#1f1f1e")
            for j in range(1, raw.shape[1]):
                if np.isfinite(delta[i, j - 1]):
                    v = 0.0 if abs(delta[i, j - 1]) < 0.05 else delta[i, j - 1]
                    dark = orient[i, j - 1] > 0.6 * vmax  # white text on the darkest reds
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6.5,
                            color="white" if dark else "black",
                            fontweight="bold" if j == raw.shape[1] - 1 else "normal")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}.{ext}", dpi=200)
    print(f"wrote {FIG}.png/.pdf")


if __name__ == "__main__":
    main()
