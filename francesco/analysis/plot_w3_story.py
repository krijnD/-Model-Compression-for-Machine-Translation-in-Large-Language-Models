"""One-figure story of the 3-bit result (README "The story in six steps", docs/06-distillation.md).

  A  size vs is-en BLEU: the GPTQ curve (fp16, w8, w4, w3, w2) and w3 + distilled adapter moved back onto it
  B  translation NLL increase over fp16 per direction, all 10 directions: plain w3 vs w3 + adapter
  C  BLEU per generated direction: plain w3 -> w3 + adapter, with fp16 and w4 as references

Everything is read from files: outputs/*/wmt22 (BLEU via sacrebleu), results/json/lang_nll_{fp16,w3,w3kd}.json.
w3 in A and C is the repacked run on the same ExllamaV2 kernel as the adapter run (docs/04-kernel-repack.md §8);
the other precisions are the scored grid. Directions without both generations yet are left out of C.

  ../venv/bin/python francesco/analysis/plot_w3_story.py    # -> results/figures/fig8_w3_story.{png,pdf}
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sacrebleu.metrics import BLEU  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO.parent / "outputs"
TESTSET = REPO / "third_party/ALMA/outputs/wmt22_outputs/wmt-testset"
J = REPO / "francesco/results/json"
FIG = REPO / "francesco/results/figures/fig8_w3_story"

# dataviz reference palette: categorical slots 1-2 (validated pair) + ink/neutral tokens
FIX, DAMAGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE, REF = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb", "#9a9890"

# checkpoint sizes on disk, GiB (../quantization_sizes.txt); adapter in bf16 (docs/06-distillation.md §1)
SIZE = {"fp16": 24.24, "w8": 12.72, "w4": 6.76, "w3": 5.27, "w2": 3.78}
ADAPTER_GIB = 0.12
RUN = {"fp16": "alma-13b-r-beam", "w8": "gptq-w8g128", "w4": "gptq-w4g128", "w3": "gptq-w3g128-as4",
       "w2": "gptq-w2g128", "w3+KD": "gptq-w3g128-as4-kd"}
PAIRS_ALL = ["is-en", "en-is", "zh-en", "en-zh", "cs-en", "en-cs", "ru-en", "en-ru", "de-en", "en-de"]


def bleu(run, pair):
    s, t = pair.split("-")
    ref = (TESTSET / f"{s}{t}/test.{pair}.{t}").read_text(encoding="utf-8").splitlines()
    f = OUT / run / "wmt22" / f"test-{pair}"
    if not f.exists():
        return None
    hyp = f.read_text(encoding="utf-8").splitlines()
    if len(hyp) != len(ref):
        return None
    return BLEU(tokenize="zh" if t == "zh" else "13a").corpus_score(hyp, [ref]).score


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    plt.rcParams.update({"font.size": 10, "axes.labelcolor": INK})
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16, 5.2), facecolor=SURFACE,
                                  gridspec_kw={"width_ratios": [1.15, 1.1, 1]})
    for ax in (a, b, c):
        style(ax)

    # A: size vs is-en BLEU
    pts = {k: (SIZE[k], bleu(RUN[k], "is-en")) for k in ["fp16", "w8", "w4", "w3", "w2"]}
    xs, ys = zip(*[pts[k] for k in ["w2", "w3", "w4", "w8", "fp16"]])
    a.plot(xs, ys, color=REF, lw=2, zorder=1)
    for k, (x, y) in pts.items():
        col = DAMAGE if k == "w3" else REF
        a.scatter(x, y, s=70, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        a.annotate(f"{k}  {y:.1f}", (x, y), xytext=(8, -12 if k in ("w8", "fp16") else 4),
                   textcoords="offset points", color=INK, fontsize=9)
    kd = (SIZE["w3"] + ADAPTER_GIB, bleu(RUN["w3+KD"], "is-en"))
    a.annotate("", xy=kd, xytext=pts["w3"], arrowprops=dict(arrowstyle="-|>", color=FIX, lw=2.2,
                                                           shrinkA=6, shrinkB=6))
    a.scatter(*kd, s=110, color=FIX, edgecolor=SURFACE, linewidth=2, zorder=4)
    a.annotate(f"w3 + adapter  {kd[1]:.1f}\n5.4 GiB, on par with w4 (6.8 GiB)", kd, xytext=(8.5, 27),
               textcoords="data", color=INK, fontsize=9, fontweight="bold",
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=2, shrinkB=7))
    a.set_xscale("log")
    a.set_xticks([4, 6, 12, 24]); a.set_xticklabels(["4", "6", "12", "24"])
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlim(3.2, 30)
    a.set_xlabel("model size on disk (GiB, log scale)")
    a.set_ylabel("BLEU, Icelandic → English")
    a.set_title("A. The adapter puts 3-bit back on the curve", loc="left", color=INK, fontsize=11)

    # B: translation dNLL per direction, w3 vs w3 + adapter
    fp, w3, kd_ = (json.load(open(J / f"lang_nll_{t}.json"))["nll"] for t in ("fp16", "w3", "w3kd"))
    y = range(len(PAIRS_ALL))
    for i, p in enumerate(PAIRS_ALL):
        d3, dk = w3[f"trans:{p}"] - fp[f"trans:{p}"], kd_[f"trans:{p}"] - fp[f"trans:{p}"]
        b.plot([dk, d3], [i, i], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        b.scatter(d3, i, s=60, color=DAMAGE, edgecolor=SURFACE, linewidth=2, zorder=3,
                  label="plain w3" if i == 0 else None)
        b.scatter(dk, i, s=60, color=FIX, edgecolor=SURFACE, linewidth=2, zorder=3,
                  label="w3 + adapter" if i == 0 else None)
        if p in ("is-en", "en-is"):
            b.annotate(f"+{d3:.2f}", (d3, i), xytext=(7, -3), textcoords="offset points", color=INK, fontsize=9)
    b.axvline(0, color=MUTED, lw=1)
    b.set_yticks(list(y)); b.set_yticklabels(PAIRS_ALL); b.invert_yaxis()
    b.set_xlabel("translation NLL increase over fp16 (nats/token; 0 = fp16)")
    b.set_title("B. Damage per direction: gone everywhere", loc="left", color=INK, fontsize=11)
    b.legend(frameon=False, loc="lower right")

    # C: BLEU dumbbells for generated directions
    rows = []
    for p in ["is-en", "en-is", "de-en", "en-de"]:
        v = {k: bleu(RUN[k], p) for k in ("fp16", "w4", "w3", "w3+KD")}
        if None not in v.values():
            rows.append((p, v))
    for i, (p, v) in enumerate(rows):
        c.plot([v["w3"], v["w3+KD"]], [i, i], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        c.plot([v["fp16"]] * 2, [i - 0.22, i + 0.22], color=INK, lw=2, zorder=2,
               label="fp16" if i == 0 else None)
        c.scatter(v["w4"], i, marker="|", s=250, color=REF, linewidth=2, zorder=2,
                  label="w4" if i == 0 else None)
        c.scatter(v["w3"], i, s=60, color=DAMAGE, edgecolor=SURFACE, linewidth=2, zorder=3)
        c.scatter(v["w3+KD"], i, s=60, color=FIX, edgecolor=SURFACE, linewidth=2, zorder=3)
        rec = (v["w3+KD"] - v["w3"]) / (v["fp16"] - v["w3"]) * 100
        # label above the row at the left edge, clear of the fp16/w4 ticks
        c.text(0.02, i - 0.3, f"{p}: {v['w3']:.1f} → {v['w3+KD']:.1f} BLEU, {rec:.0f} % recovered",
               transform=matplotlib.transforms.blended_transform_factory(c.transAxes, c.transData),
               color=INK, fontsize=9, va="bottom")
    c.set_yticks(range(len(rows))); c.set_yticklabels([p for p, _ in rows]); c.invert_yaxis()
    c.set_ylim(len(rows) - 0.4, -0.75)
    c.set_xlabel("BLEU on the WMT test set")
    c.set_title("C. Real translations: back to fp16", loc="left", color=INK, fontsize=11)
    c.legend(frameon=False, loc="lower left")

    fig.suptitle("ALMA-13B-R at 3 bits: a 125 MB adapter distilled from fp16 recovers the loss "
                 "(quantizer unchanged)", x=0.01, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}.{ext}", dpi=170, facecolor=SURFACE)
    print(f"wrote {FIG}.png/.pdf ({len(rows)} directions in C)")


if __name__ == "__main__":
    main()
