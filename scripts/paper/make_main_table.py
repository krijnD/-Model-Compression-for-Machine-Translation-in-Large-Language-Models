"""Paper Table 1: memory, time and average quality of every model, rebuilt from results/ and checked against the paper.

  Disk          weights_gib from results/json/quant_cost_<tag>.json (+ the adapter in bf16 for the KD rows)
  Peak b=1, b=4 peak_alloc of the fixed de-en workload (beam 5, source 256), results/json/probe16_<tag>[_b1].json;
                the KD rows were measured with the adapter in fp32 and are reported in bf16 (paper footnote 1)
  Time          s per sentence at b=4 on one H100 (the KD rows: probe16_<tag>_h100.json; the other adapter
                probes ran on an A100, whose times are not comparable)
  Quality       mean over the ten directions of results/scores/<run>.tsv

Prints the LaTeX rows and compares every cell with Table 1 in paper/acl_latex.tex; exits 1 on a mismatch.
Usage: python scripts/paper/make_main_table.py
"""
import csv
import json
import re
import sys

from mtcompress.alma_prompt import WMT22_PAIRS
from mtcompress.paths import JSON, PAPER, SCORES

GIB_MIB = 1024
ADAPTER_PARAMS = {"w3kd": 62_586_880, "w2kd": 250_347_520}  # rank 16 / rank 64 on all 7 Linears of 40 blocks
# label: (weights tag, probe tag, time probe tag, score run)
ROWS = {"fp16": ("fp16", "fp16", "fp16", "ours-beam"), "w8": ("w8", "w8", "w8", "gptq-w8g128"),
        "w4": ("w4", "w4", "w4", "gptq-w4g128"), "w3": ("w3", "w3", "w3", "gptq-w3g128"),
        "w2": ("w2", "w2", "w2", "gptq-w2g128"),
        "w3+KD": ("w3", "w3kd", "w3kd_h100", "gptq-w3g128-as4-kd"),
        "w2+KD": ("w2", "w2kd", "w2kd_h100", "gptq-w2g128-as4-kd-r64-cont")}
METRICS = [("bleu", "{:.1f}"), ("xcomet-xxl", "{:.1f}"), ("metricx-24", "{:.2f}")]


def load_json(name):
    return json.load(open(JSON / f"{name}.json"))


def peak_gib(tag, batch, suffix=""):
    mib = load_json(f"probe16_{tag}{suffix}")["per_batch"][str(batch)]["peak_alloc_mib"]
    return (mib - ADAPTER_PARAMS.get(tag, 0) * 2 / 1024 ** 2) / GIB_MIB  # fp32 -> bf16: 2 bytes per parameter


def quality(run):
    rows = [r for r in csv.DictReader(open(SCORES / f"{run}.tsv"), delimiter="\t") if r["pair"] in WMT22_PAIRS]
    return [sum(float(r[run]) for r in rows if r["metric"] == m) / len(WMT22_PAIRS) for m, _ in METRICS]


def row(weights, probe, time_probe, run):
    disk = (load_json(f"quant_cost_{weights}")["weights_mib"] + ADAPTER_PARAMS.get(probe, 0) * 2 / 1024 ** 2) / GIB_MIB
    time = load_json(f"probe16_{time_probe}")["per_batch"]["4"]["s_per_seq_sent"]
    cells = [f"{disk:.1f}", f"{peak_gib(probe, 1, '_b1'):.1f}", f"{peak_gib(probe, 4):.1f}", f"{time:.2f}"]
    return cells + [fmt.format(v) for v, (_, fmt) in zip(quality(run), METRICS)]


def paper_rows():
    tex = (PAPER / "acl_latex.tex").read_text()
    table = tex[tex.index(r"\label{tab:main}") - 2000:tex.index(r"\label{tab:main}")]
    out = {}
    for line in table.splitlines():
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
        if cells[0] in ROWS and len(cells) == 8:
            out[cells[0]] = cells[1:]
    return out


def main():
    ours = {label: row(*spec) for label, spec in ROWS.items()}
    paper = paper_rows()
    bad = 0
    for label, cells in ours.items():
        print(f"{label:6s}& " + " & ".join(f"{c:>5s}" for c in cells) + r" \\")
        if label not in paper:
            print(f"  row {label} not found in paper/acl_latex.tex"); bad += 1; continue
        for name, a, b in zip(["disk", "peak b=1", "peak b=4", "time", "BLEU", "XCOMET", "MetricX"], cells, paper[label]):
            if re.sub(r"\s", "", a) != re.sub(r"\s", "", b):
                print(f"  MISMATCH {label} {name}: computed {a}, paper {b}"); bad += 1
    print("Table 1 matches the paper." if not bad else f"{bad} cell(s) differ from the paper.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
