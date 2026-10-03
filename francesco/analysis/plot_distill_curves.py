"""Training curves of the distillation runs (docs/06-distillation.md): KD loss per step and held-out KL per direction.

Reads the {"step": ...} JSON lines that distill_lora.py prints into the Slurm log, so it also works on a
run that is still going. Log y-axis: w2 starts ~20x higher than w3.

  python francesco/analysis/plot_distill_curves.py            # -> results/figures/fig7_distill_curves.{png,pdf}
  python francesco/analysis/plot_distill_curves.py --runs "w3 r16=logs/slurm-distill-27400299.out" ...
  ("label=log@N" shifts a --init-adapter continuation by N steps; missing logs are skipped)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNS = ["w3, r=16=logs/slurm-distill-27400299.out", "w2, r=64=logs/slurm-distill-27405387.out",
        "w2, r=64 continued=logs/slurm-distill-27409780.out@936"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 of the dataviz reference palette, in order
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def read(log, offset=0):
    """offset: added to every step, so a --init-adapter continuation plots after the run it continues."""
    train, held = [], []
    for line in open(REPO / log, encoding="utf-8"):
        if not line.startswith('{"step"'):
            continue
        r = json.loads(line)
        if "heldout" in r:
            held.append((r["step"] + offset, r["heldout"]))
        elif "kd" in r:
            train.append((r["step"] + offset, r["kd"]))
    return train, held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=RUNS, help='"label=log path" per run')
    ap.add_argument("--pairs", default="is-en,de-en")
    ap.add_argument("--out", default=str(REPO / "francesco/results/figures/fig7_distill_curves"))
    a = ap.parse_args()
    pairs = a.pairs.split(",")

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#fcfcfb")
    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
        ax.set_yscale("log")
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("optimizer step (32 sentences each)")

    for idx, (spec, color) in enumerate(zip(a.runs, COLORS)):
        dy = [0, -9, 9][idx % 3]  # stack end labels of runs that finish close together
        label, log = spec.rsplit("=", 1)  # labels may contain "=" (r=16)
        log, _, off = log.partition("@")    # "log@936": continuation, steps shifted by 936
        if not (REPO / log).exists():
            continue
        train, held = read(log, int(off or 0))
        if train:
            s, k = zip(*train)
            ax1.plot(s, k, color=color, lw=2, label=label)
            ax1.annotate(f"{label}: {k[-1]:.3f}", (s[-1], k[-1]), xytext=(6, dy), textcoords="offset points",
                         color=INK, fontsize=9, va="center")
        for pair, style in zip(pairs, ["-", "--"]):
            pts = [(st, h[pair]["kl"]) for st, h in held if pair in h]
            if pts:
                s, k = zip(*pts)
                ax2.plot(s, k, style, color=color, lw=2, marker="o", ms=5,
                         markeredgecolor="#fcfcfb", markeredgewidth=2, label=f"{label}, {pair}")
                ax2.annotate(f"{k[-1]:.3f}", (s[-1], k[-1]), xytext=(6, dy), textcoords="offset points",
                             color=INK, fontsize=9, va="center")

    ax1.set_ylabel("KD loss, nats per target token")
    ax1.set_title("Training loss: KL(fp16 teacher || student)", color=INK, loc="left", fontsize=11)
    ax2.set_ylabel("held-out KL vs fp16 (32 rows / pair)")
    ax2.set_title("Held-out KL (solid is-en, dashed de-en)", color=INK, loc="left", fontsize=11)
    ax1.legend(frameon=False, loc="upper right")
    ax2.legend(frameon=False, loc="upper right", fontsize=9, handlelength=4)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=160, facecolor=fig.get_facecolor())
    print(f"wrote {a.out}.png/.pdf")


if __name__ == "__main__":
    main()
