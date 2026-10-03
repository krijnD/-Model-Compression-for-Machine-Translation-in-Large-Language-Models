"""NLL story of the 2-bit + distilled adapter run (docs/06-distillation.md §8), companion to plot_w3_story.py.

  A  size vs mean translation NLL (10 directions, log scale): fp16, w8, w4, w3, w2 and both adapters
  B  translation NLL increase over fp16 per direction: plain w2 -> w2 + adapter, plain w3 as reference
  C  BLEU on the WMT test set per generated direction: plain w2 -> w2 + adapter, with fp16, plain w3 and
     w3 + adapter as references (directions without a complete w2 + adapter file are left out)
  D  language vs mapping damage per language: plain-text dNLL (x) vs mean translation dNLL of both
     directions (y), for w2 + adapter and plain w3

NLL: results/json/lang_nll_{fp16,w8,w4,w3,w2,w3kd,<tag>}.json (teacher-forced, WMT test sets, 500 sentences per
item). BLEU: sacrebleu on outputs/*/wmt22 (w2 + adapter generated on an A100, the rest on H100s).

  ../venv/bin/python francesco/analysis/plot_w2_story.py [--tag w2kd-r64 --run gptq-w2g128-as4-kd-r64]
  # -> results/figures/fig9_w2_story.{png,pdf}
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sacrebleu.metrics import BLEU  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
J = REPO / "francesco/results/json"
OUT = REPO.parent / "outputs"
TESTSET = REPO / "third_party/ALMA/outputs/wmt22_outputs/wmt-testset"
FIG = REPO / "francesco/results/figures/fig9_w2_story"

# dataviz reference palette, categorical slots 1-3 in order (validated), plus ink/neutral tokens
FIX, DAMAGE, W3REF = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE, REF = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb", "#9a9890"

SIZE = {"fp16": 24.24, "w8": 12.72, "w4": 6.76, "w3": 5.27, "w2": 3.78}   # GiB on disk
ADAPTER = {"w3kd": 0.12, "w2kd": 0.47}                                    # GiB in bf16 (r16 / r64)
PAIRS = ["is-en", "en-is", "zh-en", "en-zh", "cs-en", "en-cs", "ru-en", "en-ru", "de-en", "en-de"]
LANGS = ["is", "zh", "cs", "ru", "de"]


def load(tag):
    return json.load(open(J / f"lang_nll_{tag}.json"))["nll"]


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
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="w2kd-r64")
    ap.add_argument("--run", default="gptq-w2g128-as4-kd-r64", help="w2 + adapter output folder in outputs/")
    a_ = ap.parse_args()
    n = {t: load(t) for t in ["fp16", "w8", "w4", "w3", "w2", "w3kd"]}
    n["w2kd"] = load(a_.tag)
    fp = n["fp16"]
    d = {t: {k: v[k] - fp[k] for k in fp} for t, v in n.items()}
    mean_tr = {t: sum(v[f"trans:{p}"] for p in PAIRS) / len(PAIRS) for t, v in n.items()}

    plt.rcParams.update({"font.size": 10, "axes.labelcolor": INK})
    fig, ((a, b), (cb, c)) = plt.subplots(2, 2, figsize=(15, 10.5), facecolor=SURFACE)
    for ax in (a, b, cb, c):
        style(ax)

    # A: size vs mean translation NLL. Linear y zoomed on the recovered region; w2 is drawn off-scale at the top.
    TOP = 1.22
    yv = lambda k: min(mean_tr[k], TOP)  # noqa: E731
    order = ["w2", "w3", "w4", "w8", "fp16"]
    a.plot([SIZE[k] for k in order], [yv(k) for k in order], color=REF, lw=2, zorder=1, clip_on=False)
    lab_off = {"w2": (8, -4), "w3": (8, 2), "w4": (6, 6), "w8": (-6, 8), "fp16": (-30, 8)}
    for k in order:
        col = DAMAGE if k == "w2" else (W3REF if k == "w3" else REF)
        a.scatter(SIZE[k], yv(k), s=70, color=col, edgecolor=SURFACE, linewidth=2, zorder=3, clip_on=False)
        txt = f"{k}  {mean_tr[k]:.2f}" + ("  (off scale)" if mean_tr[k] > TOP else "")
        a.annotate(txt, (SIZE[k], yv(k)), xytext=lab_off[k], textcoords="offset points", color=INK, fontsize=9)
    for base, kd, lab, off in [("w3", "w3kd", "w3 + r16 adapter", (14, -14)), ("w2", "w2kd", "w2 + r64 adapter", (-8, -34))]:
        p0 = (SIZE[base], yv(base))
        p1 = (SIZE[base] + ADAPTER[kd], mean_tr[kd])
        a.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=FIX, lw=2.2, shrinkA=6, shrinkB=6))
        a.scatter(*p1, s=100, color=FIX, edgecolor=SURFACE, linewidth=2, zorder=4)
        if base == "w3":
            a.annotate(f"{lab}  {mean_tr[kd]:.2f}  ({p1[0]:.2f} GiB)", p1, xytext=off, textcoords="offset points",
                       color=INK, fontsize=9, fontweight="bold")
        else:  # empty lower-left corner, with a pointer, clear of the w3 arrow and the y-axis
            a.annotate(f"{lab}\n{mean_tr[kd]:.2f}  ({p1[0]:.2f} GiB)", p1, xytext=(3.35, 0.86), textcoords="data",
                       color=INK, fontsize=9, fontweight="bold",
                       arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=2, shrinkB=7))
    a.set_xscale("log")
    a.set_xticks([4, 6, 12, 24]); a.set_xticklabels(["4", "6", "12", "24"])
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlim(3.2, 30); a.set_ylim(0.72, TOP)
    a.set_xlabel("model size (GiB, log scale)")
    a.set_ylabel("mean translation NLL, 10 directions (nats/token)")
    a.set_title("A. Both adapters pull the model back toward fp16", loc="left", color=INK, fontsize=11)

    # B: per-direction dumbbells, log x
    for i, p in enumerate(PAIRS):
        k = f"trans:{p}"
        b.plot([d["w2kd"][k], d["w2"][k]], [i, i], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        b.scatter(d["w2"][k], i, s=60, color=DAMAGE, edgecolor=SURFACE, linewidth=2, zorder=3,
                  label="plain w2" if i == 0 else None)
        b.scatter(d["w2kd"][k], i, s=60, color=FIX, edgecolor=SURFACE, linewidth=2, zorder=3,
                  label="w2 + adapter" if i == 0 else None)
        b.scatter(d["w3"][k], i, marker="|", s=220, color=W3REF, linewidth=2.5, zorder=2,
                  label="plain w3 (reference)" if i == 0 else None)
    b.set_xscale("log")
    b.set_yticks(range(len(PAIRS))); b.set_yticklabels(PAIRS); b.invert_yaxis()
    b.set_xlabel("translation NLL increase over fp16 (nats/token, log)")
    b.set_title("B. From broken to the plain-w3 zone, per direction", loc="left", color=INK, fontsize=11)
    b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9)

    # C: BLEU per generated direction
    runs = {"fp16": "alma-13b-r-beam", "w2": "gptq-w2g128", "w2kd": a_.run, "w3": "gptq-w3g128-as4",
            "w3kd": "gptq-w3g128-as4-kd"}
    rows = []
    for p in ["is-en", "en-is", "de-en", "en-de"]:
        v = {k: bleu(r, p) for k, r in runs.items()}
        if v["w2kd"] is not None and v["fp16"] is not None and v["w2"] is not None:
            rows.append((p, v))
    for i, (p, v) in enumerate(rows):
        cb.plot([v["w2"], v["w2kd"]], [i, i], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        cb.plot([v["fp16"]] * 2, [i - 0.22, i + 0.22], color=INK, lw=2, zorder=2, label="fp16" if i == 0 else None)
        if v["w3"] is not None:
            cb.scatter(v["w3"], i, marker="|", s=220, color=W3REF, linewidth=2.5, zorder=2,
                       label="plain w3" if i == 0 else None)
        if v["w3kd"] is not None:
            cb.scatter(v["w3kd"], i, marker="|", s=220, color=REF, linewidth=2.5, zorder=2,
                       label="w3 + adapter" if i == 0 else None)
        cb.scatter(v["w2"], i, s=60, color=DAMAGE, edgecolor=SURFACE, linewidth=2, zorder=3,
                   label="plain w2" if i == 0 else None)
        cb.scatter(v["w2kd"], i, s=80, color=FIX, edgecolor=SURFACE, linewidth=2, zorder=4,
                   label="w2 + adapter" if i == 0 else None)
        rec = (v["w2kd"] - v["w2"]) / (v["fp16"] - v["w2"]) * 100
        cb.text(0.02, i - 0.3, f"{p}: {v['w2']:.1f} → {v['w2kd']:.1f} BLEU, {rec:.0f} % of fp16 recovered"
                + (f"  (plain w3: {v['w3']:.1f})" if v["w3"] is not None else ""),
                transform=matplotlib.transforms.blended_transform_factory(cb.transAxes, cb.transData),
                color=INK, fontsize=9, va="bottom")
    cb.set_yticks(range(len(rows))); cb.set_yticklabels([p for p, _ in rows]); cb.invert_yaxis()
    cb.set_ylim(len(rows) - 0.4, -0.75)
    cb.set_xlim(-1, max(v["fp16"] for _, v in rows) + 3)
    cb.set_xlabel("BLEU on the WMT test set")
    cb.set_title("C. Real translations: from BLEU 0 to well above plain w3", loc="left", color=INK, fontsize=11)
    cb.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=9)

    # C: language (mono) vs mapping (trans) damage
    for t, col, lab in [("w3", W3REF, "plain w3"), ("w2kd", FIX, "w2 + adapter")]:
        xs = [d[t][f"mono:{L}"] for L in LANGS]
        ys = [(d[t][f"trans:{L}-en"] + d[t][f"trans:en-{L}"]) / 2 for L in LANGS]
        c.scatter(xs, ys, s=70, color=col, edgecolor=SURFACE, linewidth=2, zorder=3, label=lab)
        off = {"cs": (-16, 5), "ru": (7, -9), "de": (-17, -11)} if t == "w2kd" else {}
        for L, x, y in zip(LANGS, xs, ys):
            c.annotate(L, (x, y), xytext=off.get(L, (6, 3)), textcoords="offset points", color=INK, fontsize=9)
    lim = max(max(d[t][f"mono:{L}"] for L in LANGS) for t in ("w3", "w2kd")) * 1.1
    c.plot([0, lim], [0, lim], color=MUTED, lw=1, ls=":")
    c.text(lim * 0.62, lim * 0.66, "equal damage", color=MUTED, fontsize=8, rotation=0)
    c.set_xlim(0, lim); c.set_ylim(0, lim * 0.75)
    c.set_xlabel("plain-text NLL increase (knows the language?)")
    c.set_ylabel("mean translation NLL increase (maps between them?)")
    c.set_title("D. Mirror images: w3 loses the mapping,\n    w2 + adapter keeps it but not the language",
                loc="left", color=INK, fontsize=11)
    c.legend(frameon=False, loc="upper left", fontsize=9)

    fig.suptitle("ALMA-13B-R at 2 bits: a distilled rank-64 adapter (0.47 GiB) takes GPTQ from broken (BLEU 0)\n"
                 "to well above plain 3-bit, at 4.25 GiB (quantizer unchanged)", x=0.01, ha="left", color=INK,
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}.{ext}", dpi=170, facecolor=SURFACE)
    print(f"wrote {FIG}.png/.pdf")


if __name__ == "__main__":
    main()
