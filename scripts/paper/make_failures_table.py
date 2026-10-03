"""Paper Table 2: share of outputs (%) with at least one failure, rebuilt from results/json/failures.json
(written by scripts/analysis/score_failures.py) and checked against the paper.

Prints the LaTeX rows and compares every cell with Table 2 in paper/acl_latex.tex (bold marks ignored);
exits 1 on a mismatch.
Usage: python scripts/paper/make_failures_table.py
"""
import json
import re
import sys

from mtcompress.paths import JSON, PAPER

PAIRS = "de-en cs-en is-en zh-en ru-en en-de en-cs en-is en-zh en-ru".split()  # column order of Table 2
# label: (run, format); the collapsed plain 2-bit model is shown in whole percent
ROWS = {"fp16": ("ours-beam", "{:.1f}"), "w3": ("gptq-w3g128", "{:.1f}"), "w2": ("gptq-w2g128", "{:.0f}"),
        "w3+KD": ("gptq-w3g128-as4-kd", "{:.1f}"), "w2+KD": ("gptq-w2g128-as4-kd-r64-cont", "{:.1f}")}


def paper_rows():
    tex = (PAPER / "acl_latex.tex").read_text()
    table = tex[tex.index(r"\label{tab:failures}") - 2000:tex.index(r"\label{tab:failures}")]
    out = {}
    for line in table.splitlines():
        cells = [re.sub(r"\\textbf\{(.*?)\}", r"\1", c).strip() for c in line.rstrip("\\ ").split("&")]
        if cells[0] in ROWS and len(cells) == 11:
            out[cells[0]] = cells[1:]
    return out


def main():
    failures = json.load(open(JSON / "failures.json"))
    paper = paper_rows()
    bad = 0
    for label, (run, fmt) in ROWS.items():
        cells = [fmt.format(failures[run][p]["any"]) for p in PAIRS]
        print(f"{label:6s} & " + " & ".join(cells) + r" \\")
        if label not in paper:
            print(f"  row {label} not found in paper/acl_latex.tex"); bad += 1; continue
        for pair, a, b in zip(PAIRS, cells, paper[label]):
            if a != b:
                print(f"  MISMATCH {label} {pair}: computed {a}, paper {b}"); bad += 1
    print("Table 2 matches the paper." if not bad else f"{bad} cell(s) differ from the paper.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
