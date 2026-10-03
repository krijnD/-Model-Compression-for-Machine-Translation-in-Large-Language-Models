#!/usr/bin/env python3
"""Build the LaTeX report for the GPTQ precision grid from <project dir>/outputs/.

Writes into francesco/results/ (LaTeX chunks under tex/, figures under figures/):
  results/tex/tables.tex   \\input-able: averaged grid, FP16-referenced deltas (shaded), sizes
  results/tex/detail.tex   \\input-able: one table per metric with all 10 directions
  results/figures/*.pdf    matplotlib figures, plus .png copies for slides
  results/tex/figures.tex  \\input-able figure environments for the above

Run it from anywhere; it finds the repo by walking up from this file:
    python francesco/analysis/make_report.py                  # every precision with scores
    python francesco/analysis/make_report.py --bits 16,8,4,3  # a subset (default 16,8,4,3,2)
    python francesco/analysis/make_report.py --no-figures     # tables only, no matplotlib

Scores come from scripts/summarize.py -- that module owns "what score did run R get" and
the ALMA-R paper numbers -- so a number here is the same number the eval jobs print.
A precision whose run has no scores yet (e.g. a 2-bit job still generating) is left out of
every table and figure automatically and appears on the next run of this script.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # francesco/analysis
FRANCESCO = HERE.parent                         # francesco/
REPO = FRANCESCO.parent                         # the repository root
PROJECT = REPO.parent
OUTPUTS = PROJECT / "outputs"
SIZES_TXT = REPO / "quantization_sizes.txt"
RESULTS = FRANCESCO / "results"                 # everything this script writes
TEX = RESULTS / "tex"
FIGURES = RESULTS / "figures"
JSON_DIR = RESULTS / "json"                     # probe_*.json, quant_cost_*.json
EVAL_LOGS = REPO / "logs"                       # the eval jobs' own output

# run -> the short tag the cost probe and the eval-log parser use
TAGS = {"ours-beam": "fp16", "ours": "fp16"}
for _b in (2, 3, 4, 8):
    TAGS[f"gptq-w{_b}g128"] = f"w{_b}"

# How many GPUs the run that produced each grid column actually used. The fp16 baseline job ran 4
# ranks over 4 GPUs with the test set sharded across them (its log shows 124 batches per rank for
# de-en's 1984 lines at batch 4), so what the cluster charged it is 4x its wall; every quantized run
# was one process on one GPU. "GPU-h charged" below is wall x this factor, which is why it is the
# only column that is comparable across runs.
RUN_GPUS = {"fp16": 4, "w8": 1, "w4": 1, "w3": 1, "w2": 1}

sys.path.insert(0, str(REPO / "scripts"))
import summarize as S  # noqa: E402  (ROWS, METRICS, PAIRS, PAPER, REPORTED, load)

METRICS = S.METRICS
LOWER_IS_BETTER = {"metricx-24", "halluc"}
PRETTY = {
    "bleu": "BLEU", "chrf": "chrF++", "comet-22": "COMET-22", "kiwi-22": "KIWI-22",
    "kiwi-xxl": "KIWI-XXL", "xcomet-xxl": "XCOMET-XXL", "metricx-24": "MetricX-24",
    "halluc": "Halluc.",
}
PAPER_METRICS = [m for m in METRICS if any((p, m) in S.REPORTED for p in S.ROWS)]
# Discrete precisions, most bits first: the x axis of the figures, spacing independent of bits
# so that 3 and 4 are as readable as 8 and 16.
PRECISION_ORDER = [16, 8, 4, 3, 2]
POS = {b: i for i, b in enumerate(PRECISION_ORDER)}


def tex_pair(pair):
    """de-en -> de$\\rightarrow$en, avg_en-xx -> avg\\_en$\\rightarrow$xx."""
    return pair.replace("_", r"\_").replace("-", r"$\rightarrow$", 1) if "-" in pair else pair


def fmt(v, dp=2, dash="--"):
    return dash if v is None else f"{v:.{dp}f}"


def sfmt(v, dp=2, dash="--"):
    if v is None:
        return dash
    r = round(v, dp)
    return f"{0.0:.{dp}f}" if r == 0 else f"{r:+.{dp}f}"


# -------------------------------------------------------------------------- runs + sizes

def checkpoint_gib():
    """{run: GiB} from quantization_sizes.txt, itself written from the checkpoints on disk."""
    if not SIZES_TXT.exists():
        return {}
    sizes = {}
    for line in SIZES_TXT.read_text().splitlines():
        m = re.match(r"^(fp16|w\d+g\d+)\s+\d+\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            sizes[m.group(1)] = float(m.group(2))
    return sizes


def discover_runs():
    """{run: bit width} for every run that has at least one metric summary."""
    seen = set()
    for metric in METRICS:
        d = OUTPUTS / metric
        if d.is_dir():
            seen |= {sub.name for sub in d.iterdir() if (sub / "summary.tsv").exists()}
    runs = {}
    for run in seen:
        m = re.fullmatch(r"gptq-w(\d+)g(\d+)", run)
        if m:
            runs[run] = int(m.group(1))
        elif run == "ours-beam":
            runs[run] = 16      # fp16, deterministic beam search: the baseline the quantized runs use
        elif run == "ours":
            runs[run] = 16      # fp16, the paper's beam-sampling decoding
    return runs


def build_precisions(bits_wanted):
    """[(label, bits, run, scores)], least precise first, FP16 baseline before the sampling run."""
    precs = []
    for run, bits in discover_runs().items():
        if bits not in bits_wanted:
            continue
        scores = S.load(run)
        if scores:
            precs.append((bits, 0 if run == "ours-beam" else 1, run, scores))
    precs.sort(key=lambda t: (t[0], t[1]))
    out = []
    for bits, _, run, scores in precs:
        label = "FP16" if bits == 16 else f"W{bits}G128"
        if run == "ours":
            label = "FP16 (sampling)"
        out.append((label, bits, run, scores))
    return out


def gib_of(run, bits, sizes):
    if bits == 16:
        return sizes.get("fp16")
    return sizes.get(run.replace("gptq-", ""))


def baseline(precs):
    """The FP16 beam run: the reference every quantized row is scored against."""
    return next((p for p in precs if p[0] == "FP16"), None)


# ------------------------------------------------------------------------------- tables

def shade(delta, metric, norm):
    if delta is None or norm <= 0:
        return ""
    pct = int(round(60 * min(1.0, abs(delta) / norm)))
    if pct <= 0:
        return ""
    worse = (delta > 0) if metric in LOWER_IS_BETTER else (delta < 0)
    color = "red" if worse else "green"
    return "\\cellcolor{" + color + "!" + str(pct) + "}"


def table_averages(precs, sizes):
    ncol = len(METRICS) + 3
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l" + "r" * len(METRICS) + r"rr}", r"\toprule",
             "& " + " & ".join(PRETTY[m] for m in METRICS) + r" & GiB & ratio \\", r"\midrule"]
    fp16 = sizes.get("fp16")
    for panel, key in [(r"en$\rightarrow$xx (5 directions)", "avg_en-xx"),
                       (r"xx$\rightarrow$en (5 directions)", "avg_xx-en")]:
        lines.append(rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{{panel}}}}} \\")
        for label, bits, run, scores in precs:
            cells = [fmt(scores.get((key, m))) for m in METRICS]
            gib = gib_of(run, bits, sizes)
            ratio = "" if gib is None or not fp16 else f"{fp16 / gib:.2f}$\\times$"
            lines.append(f"{label} & " + " & ".join(cells)
                         + f" & {'' if gib is None else f'{gib:.1f}'} & {ratio} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\midrule",
              rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{ALMA-R paper (reported, their decoding)}}}} \\"]
    for avg, key in [(r"avg en$\rightarrow$xx", "avg_en-xx"), (r"avg xx$\rightarrow$en", "avg_xx-en")]:
        cells = [fmt(S.REPORTED.get((key, m))) for m in METRICS]
        lines.append(f"{avg} & " + " & ".join(cells) + r" & & \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{ALMA-13B-R on WMT'22 (WMT'21 for \texttt{is}): weight-only GPTQ, "
              r"asymmetric, group size 128, act-order, every decoder Linear quantized, "
              r"beam 5, bf16, seed 42. FP16 is our beam-search reproduction "
              r"(\texttt{ours-beam}); \texttt{FP16 (sampling)} is the paper's beam-sampling "
              r"decoding and is \emph{not} the reference for the quantized rows. "
              r"MetricX-24 and the hallucination rate (\% of sentences at least twice the "
              r"reference length) are the two where lower is better. GiB is the checkpoint on "
              r"disk, ratio the saving against FP16.}",
              r"\label{tab:grid-avg}", r"\end{table*}"]
    return "\n".join(lines)


def table_deltas(precs):
    ref = baseline(precs)
    if ref is None:
        return "% no FP16 run in outputs/, delta table skipped"
    rows = [p for p in precs if p[0].startswith("W")]
    if not rows:
        return "% no quantized run in outputs/, delta table skipped"
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l" + "r" * len(METRICS) + "}", r"\toprule",
             "& " + " & ".join(PRETTY[m] for m in METRICS) + r" \\", r"\midrule"]
    for panel, key in [(r"en$\rightarrow$xx", "avg_en-xx"), (r"xx$\rightarrow$en", "avg_xx-en")]:
        lines.append(rf"\multicolumn{{{len(METRICS) + 1}}}{{l}}{{\emph{{{panel}}}}} \\")
        norms = {}
        for m in METRICS:
            vals = [abs(s.get((key, m)) - ref[3][key, m]) for _, _, _, s in rows
                    if s.get((key, m)) is not None and (key, m) in ref[3]]
            norms[m] = max(vals) if vals else 0.0
        for label, bits, run, scores in rows:
            cells = []
            for m in METRICS:
                a, b = scores.get((key, m)), ref[3].get((key, m))
                d = None if a is None or b is None else a - b
                cells.append(shade(d, m, norms[m]) + sfmt(d))
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Difference against the FP16 beam baseline (Table~\ref{tab:grid-avg}): "
              r"green is better than FP16, red is worse, saturation scaled to the largest change "
              r"in that column. These are the numbers that say what the compression costs.}",
              r"\label{tab:grid-delta}", r"\end{table*}"]
    return "\n".join(lines)


def table_detail(precs):
    """One table per metric, all 10 directions plus the two averages."""
    out = []
    head = " & ".join(label for label, _, _, _ in precs)
    for m in METRICS:
        with_paper = m in PAPER_METRICS
        out += [r"\begin{table}[t]", r"\centering", r"\small",
                r"\begin{tabular}{l" + "r" * len(precs) + ("r" if with_paper else "") + "}",
                r"\toprule",
                f"& {head}" + (r" & paper \\" if with_paper else r" \\"), r"\midrule"]
        for row in S.PAIRS + ["avg_en-xx", "avg_xx-en"]:
            if row == "avg_en-xx":
                out.append(r"\midrule")
            cells = [fmt(s.get((row, m))) for _, _, _, s in precs]
            paper = f" & {fmt(S.REPORTED.get((row, m)))}" if with_paper else ""
            out.append(f"{tex_pair(row)} & " + " & ".join(cells) + paper + r" \\")
        out += [r"\bottomrule", r"\end{tabular}",
                rf"\caption{{{PRETTY[m]} by direction"
                + (r", with the ALMA-R paper's own reported numbers" if with_paper else "")
                + r".}", rf"\label{{tab:{m}}}", r"\end{table}", ""]
    return "\n".join(out)


def table_sizes(precs, sizes):
    keep = [p for p in precs if gib_of(p[2], p[1], sizes) is not None and "sampling" not in p[0]]
    if not keep or not SIZES_TXT.exists():
        return "% quantization_sizes.txt not found, size table skipped"
    text = SIZES_TXT.read_text()
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"precision & bits & GiB & GB & vs FP16 \\", r"\midrule"]
    for label, bits, run, _ in keep:
        gib = gib_of(run, bits, sizes)
        name = "fp16" if bits == 16 else run.replace("gptq-", "")
        m = re.search(rf"^{re.escape(name)}\s+\d+\s+([\d.]+)\s+([\d.]+)", text, re.M)
        gb = m.group(2) if m else "--"
        ratio = "--" if bits == 16 else f"{sizes['fp16'] / gib:.2f}$\\times$"
        lines.append(f"{label} & {bits} & {gib:.2f} & {gb} & {ratio} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Checkpoint size on disk. Only the 40 decoder layers' Linears are "
              r"quantized; the embeddings, \texttt{lm\_head}, the norms and the scales and "
              r"zero-points stay fp16, a fixed $\approx$0.8\,GiB floor. That is why the "
              r"end-to-end ratio stays well under $16/b$ even at 2 bits "
              r"(\texttt{quantization\_sizes.txt}).}",
              r"\label{tab:sizes}", r"\end{table}"]
    return "\n".join(lines)


# ------------------------------------------------------------------- quantization cost / benefit

def load_costs():
    """{tag: record} from results/json/quant_cost_<tag>.json, written by measure_quant_cost.py."""
    out = {}
    for path in sorted(JSON_DIR.glob("quant_cost_*.json")):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        out[rec.get("tag", path.stem.replace("quant_cost_", ""))] = rec
    return out


def generation_minutes():
    """{tag: {pair: (lines, minutes)}} from the eval jobs' own logs - the only record of the
    17,471-line grid's real times. w2 spans two jobs (the 20 h requeue); a redo supersedes."""
    per = {}
    for path in sorted(EVAL_LOGS.glob("slurm-gptq-eval-*.out")):
        text = path.read_text(errors="replace")
        m = re.search(r"ALMA-13B-R-gptq-w(\d)g128", text)
        if not m:
            continue
        tag = f"w{m.group(1)}"                      # gptq-w2g128 -> w2
        for pair, lines, mins in re.findall(
                r"^(\w\w-\w\w): done, (\d+) lines, \d+ empty, ([\d.]+) min$", text, re.M):
            seen = per.setdefault(tag, {}).get(pair)
            if seen is None or float(mins) > seen[1]:
                per[tag][pair] = (int(lines), float(mins))
    return per


def fp16_grid_minutes(costs):
    """{pair: (lines, minutes)} for the fp16 baseline, from its own generation job log.

    That job ran four processes (one GPU each) and every rank walked the whole test set - its
    per-pair wall is also the single-GPU time for the pair, which the cost probe reproduces
    independently. The pairs are sequential in the log, so a pair's wall is the gap between
    consecutive "*** Prediction for X***" lines; the last one (zh-en) died out-of-memory in that
    job, so it is taken from the cost probe's zh-en run instead. Lines come from the output files.
    """
    best = None
    for path in sorted(REPO.glob("slurm-alma-r-gen-*.out")):
        text = path.read_text(errors="replace")
        if "Decoding: beam" not in text:
            continue                                  # the paper-decoding twin job
        hits = re.findall(r"^(\d\d/\d\d/\d{4} \d\d:\d\d:\d\d) - INFO - __main__ - "
                          r"\*\*\* Prediction for (\w\w-\w\w)\*\*\*", text, re.M)
        if best is None or len(hits) > len(best[1]):
            best = (path, hits)
    if best is None:
        return {}
    from datetime import datetime
    stamps = [(p, datetime.strptime(t, "%m/%d/%Y %H:%M:%S")) for t, p in best[1]]
    out_dir = OUTPUTS / "alma-13b-r-beam" / "wmt22"
    grid = {}
    for (pair, t0), (_, t1) in zip(stamps, stamps[1:]):
        hyp = out_dir / f"test-{pair}"
        if hyp.exists():
            grid[pair] = (sum(1 for _ in hyp.open(errors="replace")), (t1 - t0).total_seconds() / 60)
    samples = (costs.get("fp16", {}).get("samples", {}).get("long") or {})
    if samples.get("ok"):
        # the probe times the first `--samples` sentences of the pair; scale to the pair's real
        # line count so the totals are over the same 17,471 lines as the quantized runs
        pair_lines = samples["lines"]
        hyp = out_dir / "test-zh-en"
        if hyp.exists():
            pair_lines = sum(1 for _ in hyp.open(errors="replace"))
        grid["zh-en"] = (pair_lines, samples["s_per_line"] * pair_lines / 60)
    return grid


def phase(costs, tag, name, field, default=None):
    rec = costs.get(tag, {}).get("phases", {}).get(name, {})
    return rec.get(field, default)


def gib(mib):
    return None if mib is None else mib / 1024


def table_latency(gens):
    """Measured seconds per input line, direction by direction, for every checkpoint."""
    tags = [t for t in ("fp16", "w8", "w4", "w3", "w2") if gens.get(t)]
    if len(tags) < 2:
        return "% no per-direction log times yet, latency table skipped"
    names = {"fp16": "FP16", "w8": "W8G128", "w4": "W4G128", "w3": "W3G128", "w2": "W2G128"}
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{l" + "r" * len(tags) + "}", r"\toprule",
             "direction & " + " & ".join(names[t] for t in tags) + r" \\", r"\midrule"]
    for pair in S.PAIRS:
        cells = []
        for t in tags:
            rec = gens[t].get(pair)
            cells.append("--" if not rec else f"{rec[1] * 60 / rec[0]:.3f}")
        lines.append(f"{tex_pair(pair)} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    gl = {t: sum(n for n, _ in gens[t].values()) for t in tags}
    gm = {t: sum(m for _, m in gens[t].values()) for t in tags}
    gh = {t: gm[t] / 60 * RUN_GPUS.get(t, 1) for t in tags}          # what the cluster charged
    lines.append("all 10 & " + " & ".join(
        "--" if not gm[t] else f"{gm[t] * 60 / gl[t]:.3f}" for t in tags) + r" \\")
    lines.append(r"\emph{GPU-h charged} & " + " & ".join(
        "--" if not gh[t] else f"{gh[t]:.1f}" for t in tags) + r" \\")
    lines.append(r"\emph{GPU-s/line} & " + " & ".join(
        "--" if not gm[t] else f"{gh[t] * 3600 / gl[t]:.2f}" for t in tags) + r" \\")
    ref = gh.get("fp16")
    lines.append(r"\emph{vs FP16} & " + " & ".join(
        "--" if not gh[t] or not ref else f"{gh[t] / ref:.2f}$\times$" for t in tags) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Seconds per input line for the whole WMT'22 grid, generation only, taken "
              r"from each job's own log (a pair's wall divided by its lines). Every precision ran "
              r"the same ten directions, the same beam-5 bf16 settings and the same sentences; "
              r"only \texttt{zh-en} differs by run (source 512 at \texttt{BATCH\_LONG=1} for the "
              r"quantized runs, the same setting for fp16 - that pair comes from the cost probe, "
              r"since the fp16 job's \texttt{zh-en} attempt died out-of-memory). \emph{GPU-h "
              r"charged} is what the cluster billed for each run's generation: the fp16 baseline "
              r"is 4$\times$ its wall because that job ran four ranks over four GPUs with the test "
              r"set sharded across them, every quantized run was one process on one GPU; "
              r"\emph{GPU-s/line} is that divided by the 17,471 lines, so it is the only row that "
              r"is comparable across runs, and the per-line column above is a *latency* only for "
              r"the single-GPU runs. Scoring is excluded.}",
              r"\label{tab:latency}", r"\end{table*}"]
    return "\n".join(lines)


KERNEL_SHORT = {"ExllamaV2QuantLinear": "ExLLamaV2", "TritonV2QuantLinear": "TritonV2",
                "TorchQuantLinear": "Torch", "MarlinQuantLinear": "Marlin",
                "MacheteQuantLinear": "Machete", "TorchFusedQuantLinear": "TorchFused",
                "ExllamaQuantLinear": "ExLLamaV1", "ExllamaEoraQuantLinear": "ExLLamaEoRA"}


def short_kernel(names):
    return ", ".join(KERNEL_SHORT.get(n, n.replace("QuantLinear", "")) for n in names or []) or "-"


def kernel_short(costs, tag):
    """Which kernel GPTQModel auto-selected - the reason the cost is not a smooth function of bits."""
    names = costs.get(tag, {}).get("kernels") or []
    return short_kernel(names) if names else "fp16"


def load_batch_probe():
    """{tag: record} from results/json/probe16_<tag>.json (run_probe.py at BATCH=4 and 16)."""
    out = {}
    for path in sorted(JSON_DIR.glob("probe16_*.json")):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        out[path.stem.replace("probe16_", "")] = rec
    return out


def table_batch(precs, sizes, costs, probes):
    """Does raising the batch give back what the low-precision kernels cost? BATCH=4 vs 16."""
    rows = [(l, b, run, TAGS.get(run)) for l, b, run, _ in precs
            if TAGS.get(run) in probes and "sampling" not in l]
    rows.sort(key=lambda r: (-r[1], r[0]))
    rows = [r for r in rows if probes.get(r[3], {}).get("per_batch")]
    if not rows:
        return "% no probe16_*.json in results/json/, batch table skipped "\
               "(run francesco/analysis/sbatch/probe16.sbatch)"
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"precision & \multicolumn{3}{c}{s per line} & \multicolumn{3}{c}{peak GiB} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"& \texttt{B=4} & \texttt{B=16} & gain & \texttt{B=4} & \texttt{B=16} & "
             r"\texttt{B=16} vs FP16 \\", r"\midrule"]
    ref16 = None
    for label, bits, run, tag in rows:
        per = probes[tag]["per_batch"]
        b4, b16 = per.get("4"), per.get("16")
        if not b4 or not b16:
            continue
        if bits == 16:
            ref16 = b16["s_per_seq_sent"]
        ratio = "" if ref16 is None else f"{b16['s_per_seq_sent'] / ref16:.2f}$\\times$"
        lines.append(f"{label} & {fmt(b4.get('s_per_seq_sent'), 3)} "
                     f"& {fmt(b16.get('s_per_seq_sent'), 3)}"
                     f" & {b4['s_per_seq_sent'] / b16['s_per_seq_sent']:.2f}$\\times$"
                     f" & {fmt(gib(b4.get('peak_alloc_mib')), 1)}"
                     f" & {fmt(gib(b16.get('peak_alloc_mib')), 1)}"
                     f" & {ratio} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{The same fixed workload (de-en, source 256, beam 5, the first four "
              r"sentences) at \texttt{BATCH=4} and \texttt{BATCH=16}, i.e.\ 20 and 80 sequences "
              r"per forward, with the allocator peak of each. Every checkpoint gains from the "
              r"batch and the low-bit ones gain most, because their per-forward dequantization is "
              r"a fixed cost the batch amortizes - but the gain never closes the gap: at "
              r"\texttt{BATCH=16} the quantized rows are still slower per line than fp16 at the "
              r"same batch (\emph{B=16 vs FP16}, above 1). What the compression does buy is the "
              r"room to run that batch at all: w8 at \texttt{BATCH=16} needs 30.7\,GiB and "
              r"0.266\,s/line where fp16 at \texttt{BATCH=4} needs 28.8\,GiB and 0.359\,s/line, "
              r"so spending the memory the quantization freed on a 4$\times$ larger batch is the "
              r"one configuration where the compression converts into throughput.}",
              r"\label{tab:batch}", r"\end{table}"]
    return "\n".join(lines)


def load_backends():
    """{(tag, backend): record} from results/json/backend_<tag>_<backend>.json (backends.sbatch)."""
    out = {}
    for path in sorted(JSON_DIR.glob("backend_*.json")):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        out[(rec.get("tag"), rec.get("backend_arg"))] = rec
    return out


def table_backends(backends, precs):
    """Which GPTQModel kernel should each precision run on? Measured, not assumed."""
    if not backends:
        return "% no backend_*.json in results/json/, kernel table skipped "\
               "(run francesco/analysis/sbatch/backends.sbatch)"
    rows = sorted(backends.items(), key=lambda kv: (kv[0][0] is None, kv[0][0], kv[0][1]))
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{llrrr}", r"\toprule",
             r"checkpoint & kernel & de-en s/line & zh-en 512 @4 s/line & peak GiB \\", r"\midrule"]
    for (tag, backend), rec in rows:
        ph = rec.get("phases", {})
        if not ph:
            continue
        back = backend.replace("_", r"\_")            # exllama_v2 etc. are text mode, so escape
        lines.append(
            f"\\texttt{{{tag}}}/\\texttt{{{short_kernel(rec.get('kernels'))}}} "
            f"& \\texttt{{{back}}}"
            f" & {fmt(ph.get('short', {}).get('s_per_line'), 3)}"
            f" & {fmt(ph.get('long_batch_probe', {}).get('s_per_line'), 3)}"
            f" & {fmt(gib(ph.get('short', {}).get('peak_alloc_mib')), 1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{The same fixed workload (de-en 256 @ \texttt{BATCH=4}, and zh-en 512 at "
              r"\texttt{BATCH=4}) on every kernel GPTQModel 4.2.5 will load for these checkpoints. "
              r"Three things came out of the sweep: \texttt{ExllamaV2QuantLinear} is 1.34$\times$ "
              r"faster than Triton for 4 bits, \texttt{TorchQuantLinear} is 9\,\% faster than "
              r"Triton for 8 (\texttt{Torch}'s \texttt{dequantize\_weight} is a 22\,ms pass versus "
              r"the Triton kernel's 2.4\,s cold), and \texttt{MarlinQuantLinear} - the only fused "
              r"kernel that declares 8-bit support - refuses these checkpoints because it requires "
              r"\emph{symmetric} weights (\texttt{only supports [True] bits: actual sym = False}). "
              r"\texttt{exllama\_v1} is not installed and BitBLAS needs \texttt{desc\_act=False}. "
              r"No kernel in the stack gives 2- or 3-bit a fused path.}",
              r"\label{tab:backends}", r"\end{table}"]
    return "\n".join(lines)


def table_cost(precs, sizes, costs, gens):
    """The price of the compression: what each checkpoint weighs, needs and costs per line."""
    rows = [(label, bits, run, TAGS.get(run)) for label, bits, run, _ in precs
            if TAGS.get(run) in costs and "sampling" not in label]
    rows.sort(key=lambda r: (-r[1], r[0]))          # fp16 first, then 8 / 4 / 3 / 2 bits
    if not rows:
        return "% no quant_cost_*.json in results/json/, cost table skipped " \
               "(run francesco/analysis/sbatch/cost.sbatch)"
    ncol = 14
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{llr" + "r" * (ncol - 3) + "}", r"\toprule",
             r"& & \multicolumn{2}{c}{checkpoint} & \multicolumn{5}{c}{de-en, src 256, "
             r"\texttt{BATCH=4}} & \multicolumn{4}{c}{zh-en, src 512, \texttt{BATCH=1}} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-9}\cmidrule(lr){10-13}",
             r"precision & bits & kernel & GiB on disk & load peak GiB & peak GiB & fixed s "
             r"& \multicolumn{2}{c}{real lines} & peak GiB & fixed s & \multicolumn{2}{c}{real lines} \\",
             r"\cmidrule(lr){8-9}\cmidrule(lr){12-13}",
             r"& & & & & & & s & $n$ & & & s & $n$ \\", r"\midrule"]
    ref_short = next((phase(costs, t, "short", "s_per_line") for _, b, _, t in rows if b == 16), None)
    ref_long = next((phase(costs, t, "long", "s_per_line") for _, b, _, t in rows if b == 16), None)
    for label, bits, run, tag in rows:
        c = costs[tag]
        short, long_ = c["phases"]["short"], c["phases"]["long"]
        samples = c.get("samples", {})
        rs, rl = samples.get("short") or {}, samples.get("long") or {}

        def real(rec):
            return ("--", "--") if not rec.get("s_per_line") else (
                f"{rec['s_per_line']:.3f}", str(rec["lines"]))

        rs_v, rs_n = real(rs)
        rl_v, rl_n = real(rl)
        lines.append(
            f"{label} & {bits} & \\texttt{{{kernel_short(costs, tag)}}}"
            f" & {fmt(gib_of(run, bits, sizes), 2)}"
            f" & {fmt(gib(c.get('load_peak_alloc_mib')), 1)}"
            f" & {fmt(gib(short.get('peak_alloc_mib')), 1)}"
            f" & {fmt(short.get('s_per_line'), 3)} & {rs_v} & {rs_n}"
            f" & {fmt(gib(long_.get('peak_alloc_mib')), 1)}"
            f" & {fmt(long_.get('s_per_line'), 3)} & {rl_v} & {rl_n} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Measured cost per checkpoint, one H100, one process per row "
              r"(\texttt{francesco/analysis/sbatch/cost.sbatch}, "
              r"\texttt{measure\_quant\_cost.py}). \emph{load peak} is the allocator peak while "
              r"the checkpoint was read (for weight-only PTQ the packed tensors are what is "
              r"loaded, and on disk they are already exactly that size); \emph{peak} is the "
              r"allocator peak of the timed \texttt{generate()} calls (reserved, i.e.\ "
              r"fragmentation included, is in the JSON), one warmup excluded. \emph{fixed s} is the "
              r"median of three calls on the first four sentences of the pair - a byte-identical "
              r"workload for every row, ~30 generated tokens; \emph{real lines} is the same model "
              r"over $n$ full lines of the pair at the same batch, which is the number comparable "
              r"to a grid log. The two protocols are the ones the eval jobs run: de-en 256 at "
              r"\texttt{BATCH=4} (20 sequences per forward) and zh-en 512 at "
              r"\texttt{BATCH\_LONG=1} (5). The fp16 row loads through transformers, the packed "
              r"checkpoints through GPTQModel; prompts, padding, bf16, beam 5 and seed 42 are "
              r"identical.}",
              r"\label{tab:cost}", r"\end{table*}"]
    return "\n".join(lines)


def table_benefit(precs, sizes, costs, gens):
    """Quality against what it cost: the trade the grid actually made, per checkpoint."""
    base = baseline(precs)
    rows = [(label, bits, run, TAGS.get(run)) for label, bits, run, _ in precs
            if bits < 16 and TAGS.get(run)]
    rows = [r for r in rows if r[2] in [p[2] for p in precs] and S.load(r[2])]
    rows.sort(key=lambda r: -r[1])
    if not rows or base is None or not gens:
        return "% no scored quantized runs with log times, benefit table skipped"
    fp16_gib = sizes.get("fp16")
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r"precision & disk & peak & grid & quality $\Delta$ vs.\ FP16 & \multicolumn{4}{c}"
             r"{measured on the grid} \\",
             r"\cmidrule(lr){6-6}\cmidrule(lr){7-9}",
             r"& GiB & GiB & GPU-h & XCOMET-XXL & BLEU & chrF++ & s/line & GPU-s/line \\",
             r"\midrule"]
    ref_short = phase(costs, "fp16", "short", "s_per_line")
    for label, bits, run, tag in rows:
        scores, gref = S.load(run), base[3]
        d = {}
        for key in ("avg_en-xx", "avg_xx-en"):
            for m in ("xcomet-xxl", "bleu", "chrf"):
                a, b = scores.get((key, m)), gref.get((key, m))
                d[m] = (d.get(m, 0.0) + (a - b if a is not None and b is not None else 0.0)) / 2
        grid = gens.get(tag, {})
        glines = sum(n for n, _ in grid.values())
        # GPU-seconds the cluster charged: the logs are wall minutes, x the run's GPU count
        gsec = 60 * sum(m for _, m in grid.values()) * RUN_GPUS.get(tag, 1)
        sl = phase(costs, tag, "short", "s_per_line")
        ratio = "" if not sl or not ref_short else f" ({sl / ref_short:.2f}$\\times$)"
        lines.append(f"{label} & {fmt(gib_of(run, bits, sizes), 2)}"
                     f" & {fmt(gib(phase(costs, tag, 'short', 'peak_alloc_mib')), 1)}"
                     f" & {gsec / 3600:.1f}"
                     f" & {sfmt(d['xcomet-xxl'])} & {sfmt(d['bleu'])} & {sfmt(d['chrf'])}"
                     f" & {fmt(sl, 3)}{ratio}"
                     f" & {fmt(gsec / glines, 2) if glines else '--'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{What each checkpoint bought and cost. The $\Delta$ columns are the mean "
              r"of the two 5-direction averages in \texttt{outputs/baseline/<run>.tsv} against "
              r"the fp16 beam run, so they are the same numbers as "
              r"Table~\ref{tab:grid-delta}, collapsed. \emph{peak} is the de-en "
              r"\texttt{BATCH=4} allocator peak from Table~\ref{tab:cost}; \emph{grid GPU-h} is "
              r"the sum of the ten directions in that run's own eval log (generation only, "
              r"single GPU, scoring excluded - the w2 figure is the requeue's, so it excludes "
              r"the wall it lost); \emph{GPU-s/line} is that divided by the 17,471 scored lines; "
              r"\emph{s/line} and its ratio are the probe's de-en row, i.e.\ the same fixed-cost "
              r"structure the full grid pays.}",
              r"\label{tab:benefit}", r"\end{table*}"]
    return "\n".join(lines)


def markdown_summary(precs, sizes, costs, gens):
    """The two markdown tables, as `(cost_block, latency_block)`.

    The README quotes these numbers, so it carries HTML-comment markers and `sync_readme()` below
    drops the generated blocks in - prose stays hand-written, numbers stay machine-generated."""
    if not costs:
        return None, None
    rows = sorted([(l, b, run, TAGS.get(run)) for l, b, run, _ in precs
                   if TAGS.get(run) in costs and "sampling" not in l], key=lambda r: -r[1])


    out = ["| checkpoint | bits | kernel | GiB on disk | peak GiB de-en 256 @4 | fixed s de-en "
           "| real s de-en (n) | peak GiB zh-en 512 @1 | fixed s zh-en | real s zh-en (n) "
           "| grid GPU-h |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, bits, run, tag in rows:
        c = costs[tag]
        ph, samples, grid = c.get("phases", {}), c.get("samples", {}), gens.get(tag, {})
        glines, gsec = sum(n for n, _ in grid.values()), 60 * sum(m for _, m in grid.values())

        def peak(name):
            v = ph.get(name, {}).get("peak_alloc_mib")
            return "--" if v is None else f"{v / 1024:.1f}"

        def real(rec):
            return "--" if not (rec or {}).get("s_per_line") else \
                f"{(rec['s_per_line']):.3f} ({rec['lines']})"

        out.append(
            f"| {label} | {bits} | {kernel_short(costs, tag)} "
            f"| {fmt(gib_of(run, bits, sizes), 2)} | {peak('short')} "
            f"| {fmt(ph.get('short', {}).get('s_per_line'), 3)} | {real(samples.get('short'))} "
            f"| {peak('long')} | {fmt(ph.get('long', {}).get('s_per_line'), 3)} "
            f"| {real(samples.get('long'))} "
            f"| {'--' if not glines else f'{gsec / 3600 * RUN_GPUS.get(tag, 1):.1f}'} |")
    out += ["",
            "Peak VRAM is the allocator peak of one timed `generate()` at the eval protocol's own "
            "shape; *fixed s* is the median of three such calls on the first four sentences of the "
            "pair (a byte-identical workload for every row, ~30 generated tokens); *real s (n)* is "
            "the same model over `n` full lines of that pair at the same batch. *grid GPU-h* is "
            "what the cluster billed for that run's generation (the fp16 baseline is 4x its wall: "
            "four ranks over four GPUs). `--` = not measured."]
    cost_block = list(out)
    out += ["",
            "Seconds per input line for the whole grid (generation only, from each job's own log):",
            ""]
    tags = [t for t in ("fp16", "w8", "w4", "w3", "w2") if gens.get(t)]
    names = {"fp16": "FP16", "w8": "W8G128", "w4": "W4G128", "w3": "W3G128", "w2": "W2G128"}
    out.append("| direction | " + " | ".join(names[t] for t in tags) + " |")
    out.append("|---" * (len(tags) + 1) + "|")
    for pair in S.PAIRS:
        cells = []
        for t in tags:
            rec = gens[t].get(pair)
            cells.append("--" if not rec else f"{rec[1] * 60 / rec[0]:.3f}")
        out.append(f"| {pair} | " + " | ".join(cells) + " |")
    gm = {t: sum(m for _, m in gens[t].values()) for t in tags}
    gl = {t: sum(n for n, _ in gens[t].values()) for t in tags}
    gh = {t: gm[t] / 60 * RUN_GPUS.get(t, 1) for t in tags}
    out.append("| **all 10** | " + " | ".join(f"{gm[t] * 60 / gl[t]:.3f}" for t in tags) + " |")
    out.append("| **GPU-h charged** | " + " | ".join(f"{gh[t]:.1f}" for t in tags) + " |")
    out.append("| **GPU-s/line** | " + " | ".join(
        f"{gh[t] * 3600 / gl[t]:.2f}" for t in tags) + " |")
    ref = gh.get("fp16")
    out.append("| **vs FP16** | " + " | ".join(
        "--" if not ref else f"{gh[t] / ref:.2f}x" for t in tags) + " |")
    return "\n".join(cost_block), "\n".join(out)




def sync_readme(blocks):
    """Replace the `<!-- table N: ... -->` marker lines of francesco/README.md with the generated
    markdown tables. Only the marked blocks are touched; the prose around them is hand-written."""
    readme = FRANCESCO / "README.md"
    if not readme.exists():
        return {}
    text = readme.read_text()
    done = {}
    for n, block in blocks.items():
        # the markers must survive the replacement; a marker that is consumed cannot be re-synced
        beg, end = f"<!-- table {n} begin - generated by make_report.py, do not edit -->", \
                   f"<!-- table {n} end -->"
        pattern = re.compile(rf"{re.escape(beg)}.*?{re.escape(end)}", re.S)
        if pattern.search(text):
            text = pattern.sub(lambda _m, b=block: f"{beg}\n{b}\n{end}", text, count=1)
            done[n] = len(block.splitlines())
    if done:
        readme.write_text(text)
    return done


CAPTIONS = {
    "fig1_degradation":
        "Every metric against the FP16 beam baseline, per translation direction. The x axis is "
        "the weight precision, one tick per quantized checkpoint; the dashed line is FP16 "
        "itself, so a point below it (above, for the two marked $\\downarrow$) is a loss. A "
        "precision whose run has no scores yet is simply absent from the axis.",
    "fig2_heatmap":
        "Difference against FP16 per direction and precision, coloured so that red is worse and "
        "green is better. MetricX-24 is sign-flipped for the colour only: the printed number is "
        "the real $\\Delta$, and its sign is the opposite of the colour. Each panel is scaled to "
        "its own worst cell, so the panels are comparable in shape, not in magnitude.",
    "fig5_cost":
        "Left: the allocator peak of a timed \\texttt{generate()} at the two eval protocols, with "
        "the model's own parameter footprint marked. The weights are what the compression "
        "shrinks; the rest of the bar is the KV cache and the activations, which no weight-only "
        "scheme touches, so the saving saturates well before it reaches the weights' ratio - and "
        "w2's peak rises again over w4's, because it generates 180-token outputs where w4 "
        "generates 40 and the extra KV cache outweighs the smaller weights. "
        "Right: the same checkpoints' cost per input line, log scale. The low-bit checkpoints are "
        "not cheaper per line than fp16 on this stack - GPTQModel's Triton/Torch kernels "
        "dequantize the weights and then matmul (\\texttt{quant\\_matmul} in "
        "\\texttt{triton\\_utils/dequant.py}), so every forward pays for weights it never "
        "stores - unless the batch is raised (see the batch-size section).",
    "fig6_trade":
        "The trade the grid made. XCOMET-XXL is the harness the 3-bit collapse shows up in first "
        "(81.15 to 43.75 on \\texttt{is-en}), so it is the metric that separates the two "
        "checkpoints that are worth keeping from the two that are not. Memory is the de-en "
        "\\texttt{BATCH=4} peak; time is the measured grid (generation only), GPU-seconds per "
        "scored line.",
    "fig3_size_quality":
        "XCOMET-XXL against the checkpoint size on disk for both halves of the grid. The "
        "8-bit checkpoint is 1.9$\\times$ smaller and indistinguishable from FP16; 4 bits costs "
        "well under a point for another 1.9$\\times$; the 3-bit step is where both the size and "
        "the score move.",
}


def make_figures(precs, outdir, sizes, costs, gens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 6.5, "figure.dpi": 200,
        "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    })
    outdir.mkdir(parents=True, exist_ok=True)
    names = []

    def save(fig, name):
        for ext in ("pdf", "png"):
            fig.savefig(outdir / f"{name}.{ext}")
        plt.close(fig)
        names.append(name)

    base = baseline(precs)
    quant = [p for p in precs if p[0].startswith("W")]
    bits = [p[1] for p in quant]
    labels = [p[0] for p in quant]
    xbits = [16] + bits

    def delta_series(key, metric):
        out = []
        for _, _, _, s in quant:
            a = s.get((key, metric))
            b = base[3].get((key, metric)) if base else None
            out.append(np.nan if a is None or b is None else a - b)
        return np.array(out, dtype=float)

    def prec_axis(ax, xlabel=True):
        ax.set_xticks(range(len(PRECISION_ORDER)))
        ax.set_xticklabels([str(b) for b in PRECISION_ORDER])
        ax.set_xlim(-0.3, len(PRECISION_ORDER) - 0.7)
        if xlabel:
            ax.set_xlabel("weight precision (bits)")

    # --- fig 1: one panel per metric, delta against FP16 ------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 3.9), sharex=True)
    for ax, metric in zip(axes.ravel(), METRICS):
        for key, color, mk, name in [("avg_en-xx", "#1f77b4", "o", r"en$\rightarrow$xx"),
                                     ("avg_xx-en", "#d62728", "s", r"xx$\rightarrow$en")]:
            ax.plot([POS[b] for b in bits], delta_series(key, metric), marker=mk, ms=3.2,
                    lw=1.2, color=color, label=name)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_title(PRETTY[metric] + (r" $\downarrow$" if metric in LOWER_IS_BETTER else ""))
    axes[0, 0].legend(loc="lower left")
    for ax in axes.ravel():
        prec_axis(ax, xlabel=False)
    for ax in axes[1, :]:
        ax.set_xlabel("weight precision (bits)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Delta$ vs.\ FP16")
    save(fig, "fig1_degradation")

    # --- fig 2: direction x precision heatmap ----------------------------------------
    heat_metrics = ["bleu", "xcomet-xxl", "metricx-24"]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0))
    for ax, metric in zip(axes, heat_metrics):
        raw = np.full((len(S.PAIRS), len(bits)), np.nan)
        for j, (_, _, _, s) in enumerate(quant):
            for i, pair in enumerate(S.PAIRS):
                a, b = s.get((pair, metric)), (base[3].get((pair, metric)) if base else None)
                if a is not None and b is not None:
                    raw[i, j] = a - b
        orient = raw if metric in LOWER_IS_BETTER else -raw     # positive == worse
        vmax = np.nanmax(np.abs(orient)) if np.isfinite(orient).any() else 1.0
        ax.imshow(orient, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(bits)))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(S.PAIRS)))
        ax.set_yticklabels([tex_pair(p) for p in S.PAIRS])
        ax.set_title(PRETTY[metric] + (r" $\downarrow$" if metric in LOWER_IS_BETTER else ""))
        ax.grid(False)
        for i in range(raw.shape[0]):
            for j in range(raw.shape[1]):
                if np.isfinite(raw[i, j]):
                    v = 0.0 if abs(raw[i, j]) < 0.05 else raw[i, j]
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=5.5)
    save(fig, "fig2_heatmap")

    # --- fig 3: checkpoint size against quality --------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, (key, title) in zip(axes, [("avg_en-xx", r"en$\rightarrow$xx"),
                                       ("avg_xx-en", r"xx$\rightarrow$en")]):
        xs, ys, ls = [], [], []
        for label, b, run, s in precs:
            size, v = gib_of(run, b, sizes), s.get((key, "xcomet-xxl"))
            if size is not None and v is not None and "sampling" not in label:
                xs.append(size)
                ys.append(v)
                ls.append(label)
        ax.plot(xs, ys, "-", color="0.65", lw=1, zorder=1)
        ax.scatter(xs, ys, s=20, zorder=2, color="#1f77b4")
        for i, (x, y, l) in enumerate(zip(xs, ys, ls)):   # alternate above/below: labels collide
            ax.annotate(l, (x, y), textcoords="offset points",
                        xytext=(4, 3) if i % 2 == 0 else (4, -10), fontsize=6)
        ax.set_xlabel("checkpoint on disk (GiB)")
        ax.set_ylabel("XCOMET-XXL")
        ax.set_title(title)
        ax.set_xlim(min(xs) - 2, max(xs) + 5)
        ax.set_ylim(min(ys) - 2, max(ys) + 3)
    save(fig, "fig3_size_quality")

    # --- fig 4: per-direction curves --------------------------------------------------
    for metric in ["xcomet-xxl", "metricx-24", "bleu"]:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
        for ax, (key, title) in zip(axes, [("en", r"en$\rightarrow$xx"), ("xx", r"xx$\rightarrow$en")]):
            for pair in S.PAIRS:
                if pair.startswith("en-") != (key == "en"):
                    continue
                b = base[3].get((pair, metric)) if base else None
                vals = ([b] if b is not None else []) + [p[3].get((pair, metric)) for p in quant]
                ax.plot([POS[x] for x in xbits], vals, marker="o", ms=2.5, lw=1, label=pair)
            prec_axis(ax)
            ax.set_ylabel(PRETTY[metric] + (r" $\downarrow$" if metric in LOWER_IS_BETTER else ""))
            ax.set_title(title)
            ax.legend(ncol=2, loc="best")
        save(fig, f"fig4_{metric.replace('-', '')}")
    # --- fig 5: what the compression costs (memory + speed) ---------------------------
    cost_rows = [(l, b, run, TAGS.get(run)) for l, b, run, _ in precs
                 if TAGS.get(run) in costs and "sampling" not in l]
    cost_rows.sort(key=lambda r: -r[1])              # fp16 first, then 8 / 4 / 3 / 2 bits
    if cost_rows:
        xs = np.arange(len(cost_rows))
        names5 = [l for l, _, _, _ in cost_rows]
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
        fig.subplots_adjust(wspace=0.42)
        ax = axes[0]
        w = 0.38
        for i, (ph, colour, lab) in enumerate([("short", "#1f77b4", "de-en 256, BATCH=4"),
                                               ("long", "#ff7f0e", "zh-en 512, BATCH=1")]):
            vals = [gib(phase(costs, t, ph, "peak_alloc_mib")) or 0.0 for *_, t in cost_rows]
            ax.bar(xs + (i - 0.5) * w, vals, w, color=colour, label=lab)
        ax.plot(xs, [costs[t].get("weights_gib") or 0.0 for *_, t in cost_rows], "k_",
                ms=13, mew=1.7, label="weights alone")
        ax.set_xticks(xs)
        ax.set_xticklabels(names5)
        ax.set_ylabel("peak VRAM (GiB)")
        ax.set_title("memory")
        ax.legend(loc="upper right")
        ax = axes[1]
        for ph, colour, mk, lab in [("short", "#1f77b4", "o", "de-en 256, BATCH=4"),
                                    ("long", "#ff7f0e", "s", "zh-en 512, BATCH=1")]:
            ax.plot(xs, [phase(costs, t, ph, "s_per_line") or np.nan for *_, t in cost_rows],
                    marker=mk, ms=3.5, lw=1.2, color=colour, label=lab)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(names5)
        ax.set_ylabel("s per input line")
        ax.set_title("speed")
        ax.legend(loc="upper left")
        save(fig, "fig5_cost")

    # --- fig 6: the trade itself, quality against memory and against GPU time ---------
    trade = []
    for label, b, run, tag in cost_rows:
        s = next((p[3] for p in precs if p[2] == run), None)
        v = s.get(("avg_xx-en", "xcomet-xxl")) if s else None
        peak = gib(phase(costs, tag, "short", "peak_alloc_mib"))
        grid = gens.get(tag)
        gsec = 60 * sum(m for _, m in grid.values()) / sum(n for n, _ in grid.values()) if grid else None
        if v is not None and peak:
            trade.append((label, v, peak, gsec))
    if len(trade) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
        for ax, idx, xlabel, title in [(axes[0], 2, "peak VRAM, de-en 256 (GiB)", "memory"),
                                       (axes[1], 3, "GPU-seconds per line on the grid", "time")]:
            pts = [t for t in trade if t[idx] is not None]
            pts.sort(key=lambda t: t[idx])
            ax.plot([t[idx] for t in pts], [t[1] for t in pts], "-", color="0.65", lw=1, zorder=1)
            ax.scatter([t[idx] for t in pts], [t[1] for t in pts], s=22, zorder=2, color="#1f77b4")
            for i, t in enumerate(pts):                # alternate above/below: labels collide
                ax.annotate(t[0], (t[idx], t[1]), textcoords="offset points",
                            xytext=(4, 3) if i % 2 == 0 else (4, -10), fontsize=6)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("XCOMET-XXL, xx$\\rightarrow$en")
            ax.set_title(title)
        save(fig, "fig6_trade")

    return names


def figures_tex(names, figdir_rel="figures"):
    out = []
    for name in names:
        caption = CAPTIONS.get(name)
        if caption is None and name.startswith("fig4_"):
            metric = name[len("fig4_"):]
            pretty = next((v for k, v in PRETTY.items() if k.replace("-", "") == metric), metric)
            caption = (f"{pretty} per direction across the precision grid. All ten directions "
                       f"are hit, but not evenly: the low-resource ones (\\texttt{{is}}, "
                       f"\\texttt{{zh}}, \\texttt{{cs}}) lose several times what "
                       f"\\texttt{{de}} or \\texttt{{ru}} lose at the same precision.")
        out += [r"\begin{figure*}[t]", r"\centering",
                rf"\includegraphics[width=\linewidth]{{{figdir_rel}/{name}.pdf}}"]
        if caption:
            out += [rf"\caption{{{caption}}}", rf"\label{{fig:{name}}}"]
        out += [r"\end{figure*}", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bits", default="16,8,4,3,2",
                    help="weight precisions to include (default: %(default)s)")
    ap.add_argument("--outdir", type=Path, default=TEX,
                    help="where the .tex chunks go (default: %(default)s)")
    ap.add_argument("--figdir", type=Path, default=FIGURES,
                    help="where the figures go (default: %(default)s)")
    ap.add_argument("--no-figures", action="store_true", help="skip matplotlib")
    args = ap.parse_args()
    bits_wanted = {int(b) for b in args.bits.split(",") if b.strip()}

    sizes = checkpoint_gib()
    precs = build_precisions(bits_wanted)
    if not precs:
        sys.exit(f"no scored runs found under {OUTPUTS} for bits {sorted(bits_wanted)}")
    missing = bits_wanted - {p[1] for p in precs}
    print("precisions: " + ", ".join(p[0] for p in precs)
          + (f"   (no scores yet: {sorted(missing)})" if missing else ""))

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "tables.tex").write_text("\n\n".join(
        [table_averages(precs, sizes), table_deltas(precs), table_sizes(precs, sizes)]) + "\n")
    (outdir / "detail.tex").write_text(table_detail(precs) + "\n")
    print(f"wrote {outdir / 'tables.tex'} and {outdir / 'detail.tex'}")

    costs = load_costs()
    gens = generation_minutes()
    if costs and "fp16" not in gens:
        fp16_grid = fp16_grid_minutes(costs)
        if fp16_grid:
            gens["fp16"] = fp16_grid
    if costs:
        probes = load_batch_probe()
        (outdir / "cost.tex").write_text("\n\n".join(
            [table_cost(precs, sizes, costs, gens),
             table_latency(gens),
             table_benefit(precs, sizes, costs, gens),
             table_batch(precs, sizes, costs, probes)]) + "\n")
        (outdir / "kernels.tex").write_text(table_backends(load_backends(), precs) + "\n")
        cost_md, latency_md = markdown_summary(precs, sizes, costs, gens)
        (RESULTS / "quant_cost_summary.md").write_text(cost_md + "\n\n" + latency_md + "\n")
        synced = sync_readme({1: cost_md, 2: latency_md})
        print(f"wrote {outdir / 'cost.tex'} and {RESULTS / 'quant_cost_summary.md'}"
              + (f" (README.md tables {sorted(synced)} synced)" if costs and synced else "") + "  "
              f"({', '.join(sorted(costs))} from results/json/, {', '.join(sorted(gens))} from logs/)")

    if not args.no_figures:
        figdir = args.figdir
        figdir.mkdir(parents=True, exist_ok=True)
        names = make_figures(precs, figdir, sizes, costs, gens)
        # figures.tex sits in the tex dir and is \input by results.tex there, so the
        # includegraphics path must be relative to the tex dir, not to francesco/.
        rel = os.path.relpath(figdir, outdir)
        (outdir / "figures.tex").write_text(figures_tex(names, rel) + "\n")
        print(f"wrote {len(names)} figures (+ .png copies) to {figdir} "
              f"and {outdir / 'figures.tex'}")


if __name__ == "__main__":
    main()
