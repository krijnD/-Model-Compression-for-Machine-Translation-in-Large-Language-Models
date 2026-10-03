"""Two report figures for the 2-bit recovery (docs/06-distillation.md §5), BLEU on the WMT test sets.

  fig11_w2_recovery_by_direction  share of fp16 BLEU kept, per direction: plain w3 vs w2 + distilled adapter
                                  (plain w2 keeps ~0 % everywhere). Directions still generating are left out.
  fig13_vram_quality              like fig12 but x = peak VRAM during generation (de-en, batch 4, beam 5, H100;
                                  results/json/probe16_*.json). Adapter points are ESTIMATES (hollow): the measured
                                  fast-kernel base (as4, ExllamaV2) + the adapter as held in memory (fp32).
  fig12_size_quality              model size vs share of fp16 BLEU kept, averaged over the directions that EVERY
                                  plotted system has (so all points use the same sentences), with the two adapters
                                  drawn as arrows from their base checkpoint.

w2 + adapter = the continuation adapter (gptq-w2g128-as4-kd-r64-cont). Plain w3 = the scored grid run in fig 11, and
the repacked run (same kernel as w3 + adapter) in fig 12, as everywhere the adapter is compared.
  ../venv/bin/python francesco/analysis/plot_w2_recovery.py
"""
import functools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sacrebleu.metrics import BLEU  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO.parent / "outputs"
TESTSET = REPO / "third_party/ALMA/outputs/wmt22_outputs/wmt-testset"
FIGS = REPO / "francesco/results/figures"
PAIRS = ["is-en", "zh-en", "cs-en", "ru-en", "de-en", "en-is", "en-zh", "en-cs", "en-ru", "en-de"]

# dataviz reference palette, categorical slots in order (validated), + ink/neutral tokens (same as figs 8-10)
FIX, DAMAGE, W3REF = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE, REF = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb", "#9a9890"
W2KD = "gptq-w2g128-as4-kd-r64-cont"
SIZE = {"fp16": 24.24, "w8": 12.72, "w4": 6.76, "w3": 5.27, "w2": 3.78, "w3+KD": 5.39, "w2+KD": 4.25}


@functools.lru_cache(maxsize=None)
def bleu(run, pair):
    s, t = pair.split("-")
    ref = (TESTSET / f"{s}{t}/test.{pair}.{t}").read_text(encoding="utf-8").splitlines()
    f = OUT / run / "wmt22" / f"test-{pair}"
    if not f.exists():
        return None
    hyp = f.read_text(encoding="utf-8").splitlines()
    return BLEU(tokenize="zh" if t == "zh" else "13a").corpus_score(hyp, [ref]).score if len(hyp) == len(ref) else None


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.set_axisbelow(True)


# Report style for fig 11: no in-figure title (the caption carries the message), serif text, paired colours
# (light = plain checkpoint, dark = + distilled adapter; blue = 2-bit, orange = 3-bit), vector PDF.
PAIRED = {"w2": "#a6cee3", "w2kd": "#1f78b4", "w3": "#fdbf6f", "w3kd": "#ff7f00"}  # ColorBrewer "Paired"
PAPER_RC = {"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.5, "axes.labelsize": 8.5,
            "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.6,
            "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42}


def fig11():
    """Grouped bars: BLEU as % of fp16 per direction for plain w2, w2 + adapter, plain w3, w3 + adapter.
    Missing bars = not evaluated (w3 + adapter: 4 directions; w2 + adapter: still generating)."""
    order = ["is-en", "zh-en", "cs-en", "ru-en", "de-en", "en-is", "en-zh", "en-cs", "en-ru", "en-de"]
    series = [("w2", "gptq-w2g128", "w2 (3.78 GiB)"), ("w2kd", W2KD, "w2 + adapter (4.25 GiB)"),
              ("w3", "gptq-w3g128", "w3 (5.27 GiB)"), ("w3kd", "gptq-w3g128-as4-kd", "w3 + adapter (5.39 GiB)")]
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(6.5, 2.4))
        w = 0.19
        xs = [k + (0.5 if k >= 5 else 0) for k in range(len(order))]  # small gap between the two groups
        for n, (key, run, label) in enumerate(series):
            vals = [None if bleu(run, p) is None else 100 * bleu(run, p) / bleu("alma-13b-r-beam", p) for p in order]
            pos = [x + (n - 1.5) * w for x, v in zip(xs, vals) if v is not None]
            ax.bar(pos, [v for v in vals if v is not None], width=w, color=PAIRED[key], label=label,
                   edgecolor="white", linewidth=0.4, zorder=2)
        ax.axhline(100, color="black", lw=0.8, ls=(0, (4, 2)), zorder=3, label="fp16 (24.24 GiB)")
        ax.set_xticks(xs)
        ax.set_xticklabels([p.replace("-", "→") for p in order])
        ax.set_ylabel("BLEU (% of fp16)")
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
        ax.grid(axis="y", color="0.88", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="x", length=0)
        h, l = ax.get_legend_handles_labels()
        h, l = h[1:] + h[:1], l[1:] + l[:1]  # bars first, fp16 line last
        ax.legend(h, l, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.4,
                  columnspacing=1.6, fontsize=7.5)
        fig.tight_layout(pad=0.3)
        for ext in ("png", "pdf"):
            fig.savefig(FIGS / f"fig11_w2_recovery_by_direction.{ext}", dpi=300)
        plt.close(fig)
    return sum(bleu(W2KD, p) is not None for p in order)


def _quality():
    """Mean BLEU as % of fp16 over the directions every system has (same sentences for every point)."""
    runs = {"fp16": "alma-13b-r-beam", "w8": "gptq-w8g128", "w4": "gptq-w4g128", "w3": "gptq-w3g128-as4",
            "w2": "gptq-w2g128", "w3kd": "gptq-w3g128-as4-kd", "w2kd": W2KD}
    common = [p for p in PAIRS if all(bleu(r, p) is not None for r in runs.values())]
    return {k: 100 * sum(bleu(r, p) / bleu(runs["fp16"], p) for p in common) / len(common) for k, r in runs.items()}, common


def _tradeoff(x, est, xlabel, fname, note, xlog, xlim):
    """Shared report-style scatter for figs 12/13: grey = plain GPTQ curve, light = plain w2/w3, dark = + adapter
    (fig 11 colours), arrows plain -> + adapter; hollow = estimated x."""
    q, common = _quality()
    col = {"fp16": "0.35", "w8": "0.35", "w4": "0.35", "w3": PAIRED["w3"], "w2": PAIRED["w2"],
           "w3kd": PAIRED["w3kd"], "w2kd": PAIRED["w2kd"]}
    name = {"fp16": "fp16", "w8": "w8", "w4": "w4", "w3": "w3", "w2": "w2", "w3kd": "w3 + adapter", "w2kd": "w2 + adapter"}
    off = {"fp16": (-4, -11), "w8": (-3, -11), "w4": (5, -9), "w3": (5, -3), "w2": (5, 2),
           "w3kd": (0, 8), "w2kd": (-7, 0)}
    ha = {"w3kd": "center", "w2kd": "right", "fp16": "right", "w8": "center"}
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        order = ["w2", "w3", "w4", "w8", "fp16"]
        ax.plot([x[k] for k in order], [q[k] for k in order], color="0.75", lw=0.9, zorder=1)
        for base, kd in [("w3", "w3kd"), ("w2", "w2kd")]:
            ax.annotate("", xy=(x[kd], q[kd]), xytext=(x[base], q[base]),
                        arrowprops=dict(arrowstyle="-|>,head_width=0.18,head_length=0.35", color="0.45", lw=0.7,
                                        shrinkA=3.5, shrinkB=4.5), zorder=2)
        for k in ["fp16", "w8", "w4", "w3", "w2", "w3kd", "w2kd"]:
            hollow = k in est
            ax.scatter(x[k], q[k], s=30 if k.endswith("kd") else 20, zorder=3, linewidth=1.1,
                       facecolor="white" if hollow else col[k], edgecolor=col[k])
            ax.annotate(f"{name[k]}", (x[k], q[k]), xytext=off[k], textcoords="offset points",
                        fontsize=7, ha=ha.get(k, "left"), va="center")
        if xlog:
            ax.set_xscale("log")
            ax.set_xticks([4, 6, 12, 24]); ax.set_xticklabels(["4", "6", "12", "24"])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("BLEU (% of fp16)")
        ax.set_ylim(-6, 114); ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(color="0.9", lw=0.5, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.text(0.01, 0.005, note + f" Quality: mean over {', '.join(common)}.", fontsize=5.6, color="0.35", wrap=True)
        fig.tight_layout(pad=0.3, rect=(0, 0.11, 1, 1))
        for ext in ("png", "pdf"):
            fig.savefig(FIGS / f"{fname}.{ext}", dpi=300)
        plt.close(fig)
    return common


def fig12():
    size = {"fp16": 24.24, "w8": 12.72, "w4": 6.76, "w3": 5.27, "w2": 3.78, "w3kd": 5.39, "w2kd": 4.25}
    return _tradeoff(size, set(), "model size (GiB, log scale)", "fig12_size_quality",
                     "Packed checkpoints; adapters in bf16 included.", xlog=True, xlim=(2.6, 30))


FIG13_CAPTION = (
    "Peak GPU memory during generation vs. translation quality (BLEU as % of fp16, mean over the directions every "
    "system was evaluated on). Packed checkpoints (w3: Torch kernel; w2, w8: Triton; w4: ExLlamaV2), de-en, beam 5, "
    "source length 256, batch 4 unless marked. Adapter memory is counted in bf16 (measured with the adapter in fp32, "
    "minus half the adapter's bytes); quality was measured with the fp32 adapter. Plain w2 generates every output to "
    "256 tokens, which inflates its cache. Dashed lines: usable memory of an 8 GB GPU (~7.4 GiB) and fp16 quality. "
    "Hatched: w2 + adapter translating one sentence at a time. Measured on an A100/H100.")


def fig13():
    """Report-style, two aligned horizontal bar panels (one row per model): peak GPU memory (packed checkpoints,
    adapters counted in bf16, 8 GB line) and BLEU as % of fp16. Caption: FIG13_CAPTION (written next to the PNG)."""
    import json
    J = REPO / "francesco/results/json"
    pk = lambda t, b="4": json.load(open(J / f"probe16_{t}.json"))["per_batch"][b]["peak_alloc_mib"] / 1024  # noqa: E731
    save = {"w3kd": 62_586_880 * 2 / 2**30, "w2kd": 250_347_520 * 2 / 2**30}  # fp32 -> bf16 adapter
    q, common = _quality()
    rows = [("fp16", pk("fp16"), q["fp16"], "0.35"),
            ("w8", pk("w8"), q["w8"], "0.55"),
            ("w4", pk("w4"), q["w4"], "0.55"),
            ("w3", pk("w3"), q["w3"], PAIRED["w3"]),
            ("w3 + adapter", pk("w3kd") - save["w3kd"], q["w3kd"], PAIRED["w3kd"]),
            ("w2", pk("w2"), q["w2"], PAIRED["w2"]),
            ("w2 + adapter", pk("w2kd") - save["w2kd"], q["w2kd"], PAIRED["w2kd"])]
    if (J / "probe16_w2kd_b1.json").exists():
        rows.append(("w2 + adapter, batch 1", pk("w2kd_b1", "1") - save["w2kd"], q["w2kd"], PAIRED["w2kd"]))
    with plt.rc_context(PAPER_RC):
        fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.7), sharey=True, gridspec_kw={"width_ratios": [1.35, 1]})
        y = list(range(len(rows)))
        for yy, (name, mem, qual, c) in zip(y, rows):
            hatch = "////" if "batch 1" in name else None
            a.barh(yy, mem, color=c, height=0.62, hatch=hatch, edgecolor="white" if hatch is None else PAIRED["w2"],
                   linewidth=0.4, zorder=2)
            a.text(mem + 0.4, yy, f"{mem:.1f}", va="center", fontsize=7)
            b.barh(yy, qual, color=c, height=0.62, hatch=hatch, edgecolor="white" if hatch is None else PAIRED["w2"],
                   linewidth=0.4, zorder=2)
            if qual >= 90:  # inside the bar, clear of the fp16 line
                b.text(qual - 1.5, yy, f"{qual:.0f}", va="center", ha="right", fontsize=7, color="white",
                       fontweight="bold" if "adapter" in name else "normal")
            else:
                b.text(max(qual, 0) + 1.5, yy, f"{qual:.0f}", va="center", fontsize=7)
        a.axvline(7.45, color="black", lw=0.8, ls=(0, (3, 2)), zorder=3)
        a.text(7.45 + 0.4, -0.95, "8 GB GPU", ha="left", va="center", fontsize=7)
        b.axvline(100, color="black", lw=0.8, ls=(0, (3, 2)), zorder=3)
        a.set_yticks(y)
        a.set_yticklabels([r[0] for r in rows])
        a.invert_yaxis()
        a.set_ylim(len(rows) - 0.5, -1.2)
        a.set_xlim(0, 33)
        b.set_xlim(0, 112)
        a.set_xlabel("peak GPU memory (GiB)")
        b.set_xlabel("BLEU (% of fp16)")
        b.set_xticks([0, 25, 50, 75, 100])
        for ax in (a, b):
            ax.grid(axis="x", color="0.9", lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.tight_layout(pad=0.3, w_pad=1.2)
        for ext in ("png", "pdf"):
            fig.savefig(FIGS / f"fig13_vram_quality.{ext}", dpi=300)
        plt.close(fig)
    (FIGS / "fig13_vram_quality.caption.txt").write_text(FIG13_CAPTION + f" Quality directions: {', '.join(common)}.\n")


if __name__ == "__main__":
    n = fig11()
    c = fig12()
    fig13()
    print(f"fig11: {n} directions; fig12 common directions: {c}")
