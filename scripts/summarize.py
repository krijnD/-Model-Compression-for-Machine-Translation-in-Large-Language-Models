"""Collect all scores of one run and compare them with the ALMA-R paper.

Reads <project dir>/outputs/<metric>/<run>/summary.tsv (written by score_comet.slurm,
score_metricx.slurm and score_lexical.py); missing metrics are shown as "-".
Writes <project dir>/outputs/baseline/<run>.tsv.

Usage (from the repo root): python scripts/summarize.py --run ours --vs paper
"""
import argparse
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parents[2] / "outputs"
PAIRS = "de-en,cs-en,is-en,zh-en,ru-en,en-de,en-cs,en-is,en-zh,en-ru".split(",")
ROWS = PAIRS + ["avg_en-xx", "avg_xx-en"]
METRICS = ["bleu", "chrf", "comet-22", "kiwi-22", "kiwi-xxl", "xcomet-xxl", "metricx-24", "halluc"]

# ALMA-13B-R in the paper (arXiv 2401.08417, Tables 9 and 10): BLEU, COMET-22, KIWI-22, KIWI-XXL, XCOMET
PAPER_METRICS = ["bleu", "comet-22", "kiwi-22", "kiwi-xxl", "xcomet-xxl"]
PAPER = {
    "en-de": [27.72, 86.40, 83.28, 84.25, 97.48], "en-cs": [26.32, 90.29, 84.99, 87.06, 93.61],
    "en-is": [22.88, 86.85, 82.18, 85.68, 91.93], "en-zh": [34.06, 86.86, 82.25, 84.32, 92.03],
    "en-ru": [24.15, 88.30, 83.98, 87.37, 95.22], "avg_en-xx": [27.03, 87.74, 83.34, 85.74, 94.05],
    "de-en": [30.89, 84.95, 81.50, 83.97, 94.20], "cs-en": [44.39, 86.85, 82.63, 83.75, 88.03],
    "is-en": [39.67, 87.14, 81.57, 85.73, 80.49], "zh-en": [23.23, 81.64, 79.24, 77.17, 91.65],
    "ru-en": [39.06, 85.45, 81.72, 81.54, 91.18], "avg_xx-en": [35.45, 85.21, 81.33, 82.43, 89.11],
}
REPORTED = {(row, m): v for row, vals in PAPER.items() for m, v in zip(PAPER_METRICS, vals)}


def load(run):
    """{(row, metric): score}, with the en-xx / xx-en averages recomputed from the pairs."""
    scores = {}
    for metric in METRICS:
        path = OUTPUTS / metric / run / "summary.tsv"
        if not path.exists():
            continue
        for line in path.read_text().splitlines()[1:]:
            pair, value = line.split("\t")[:2]
            if pair in PAIRS:
                scores[pair, metric] = float(value)
        for avg, from_en in [("avg_en-xx", True), ("avg_xx-en", False)]:
            vals = [scores.get((p, metric)) for p in PAIRS if p.startswith("en-") == from_en]
            if None not in vals:
                scores[avg, metric] = sum(vals) / len(vals)
    return scores


def table(title, cells, metrics):
    print(f"\n{title}")
    print(f"{'':10}" + "".join(f"{m:>11}" for m in metrics))
    for row in ROWS:
        print(f"{row:10}" + "".join(f"{cells(row, m):>11}" for m in metrics))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=["paper", "ours", "ours-beam"])
    parser.add_argument("--vs", choices=["paper", "ours", "ours-beam"], help="another run to compare with")
    args = parser.parse_args()

    ours = load(args.run)
    other = load(args.vs) if args.vs else {}
    fmt = lambda v, sign="": "-" if v is None else f"{v:{sign}.2f}"
    diff = lambda a, b: None if a is None or b is None else a - b

    table(f"{args.run}  (metricx-24: lower is better; halluc: % of sentences)",
          lambda r, m: fmt(ours.get((r, m))), METRICS)
    table(f"{args.run} minus paper reported",
          lambda r, m: fmt(diff(ours.get((r, m)), REPORTED[r, m]), "+"), PAPER_METRICS)
    if args.vs:
        table(f"{args.run} minus {args.vs}",
              lambda r, m: fmt(diff(ours.get((r, m)), other.get((r, m))), "+"), METRICS)

    out = OUTPUTS / "baseline" / f"{args.run}.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ["pair", "metric", args.run, "paper_reported"] + ([args.vs] if args.vs else [])
    lines = ["\t".join(header)]
    for row in ROWS:
        for m in METRICS:
            vals = [ours.get((row, m)), REPORTED.get((row, m))] + ([other.get((row, m))] if args.vs else [])
            lines.append("\t".join([row, m] + ["" if v is None else f"{v:.2f}" for v in vals]))
    out.write_text("\n".join(lines) + "\n")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
