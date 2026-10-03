"""Collect every result of francesco/ into ONE readable file: francesco/results/RESULTS.md.

Nothing is computed that isn't already on disk except BLEU / chrF++ of existing generations (sacrebleu, seconds).
Sources, all read fresh on every run, so rerun this after any job finishes:
  outputs/baseline/<run>.tsv                     grid runs, all 8 metrics, per direction + averages (summarize.py)
  results/json/lang_nll_*.json                   teacher-forced NLL per language (lang_nll.py)
  outputs/<run>/wmt22, outputs/xcomet-xxl/<run>  adapter generations and their XCOMET-XXL
  results/json/bootstrap_ci.json                 paired bootstrap 95 % CIs (bootstrap_ci.py)
  results/distill_runs/<adapter>/train_log.jsonl training / held-out KL (copied from /scratch-shared)
  results/json/failures.json, audit_<run>.json   failure taxonomy and recovery audit (score_failures.py, audit_recovery.py)
Missing inputs are shown as "–" (e.g. a job still running).

  ../venv/bin/python francesco/analysis/make_results_md.py
"""
import csv
import functools
import json
import re
from datetime import datetime
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

REPO = Path(__file__).resolve().parents[2]
OUT = REPO.parent / "outputs"
TESTSET = REPO / "third_party/ALMA/outputs/wmt22_outputs/wmt-testset"
RES = REPO / "francesco/results"
J = RES / "json"
PAIRS = ["de-en", "cs-en", "is-en", "zh-en", "ru-en", "en-de", "en-cs", "en-is", "en-zh", "en-ru"]
SIZE = {"fp16": 24.24, "w8": 12.72, "w4": 6.76, "w3": 5.27, "w2": 3.78, "w2+KD": 4.25}
DASH = "–"


def f(x, d=2, sign=False):
    if x is None:
        return DASH
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def baseline(run):
    p = OUT / "baseline" / f"{run}.tsv"
    if not p.exists():
        return {}
    return {(r["pair"], r["metric"]): float(r[run]) for r in csv.DictReader(open(p), delimiter="\t") if r[run]}


@functools.lru_cache(maxsize=None)  # each file is scored once, not once per metric table
def gen_scores(run, pair):
    s, t = pair.split("-")
    ref = (TESTSET / f"{s}{t}/test.{pair}.{t}").read_text(encoding="utf-8").splitlines()
    fp = OUT / run / "wmt22" / f"test-{pair}"
    if not fp.exists():
        return None
    hyp = fp.read_text(encoding="utf-8").splitlines()
    if len(hyp) != len(ref):
        return None
    return {"bleu": BLEU(tokenize="zh" if t == "zh" else "13a").corpus_score(hyp, [ref]).score,
            "chrf": CHRF(word_order=2).corpus_score(hyp, [ref]).score,
            "halluc": 100 * sum(len(h) >= 2 * len(r) for h, r in zip(hyp, ref)) / len(hyp)}


def xcomet(run, pair):
    p = OUT / "xcomet-xxl" / run / f"{pair}.txt"
    m = re.findall(r"score: ([0-9.]+)", p.read_text()) if p.exists() else []
    return 100 * float(m[-1]) if m else None


def nll(tag):
    p = J / f"lang_nll_{tag}.json"
    return json.load(open(p))["nll"] if p.exists() else None


def main():
    md = [f"# Results — ALMA-13B-R compression (francesco/)\n",
          f"*Generated {datetime.now():%Y-%m-%d %H:%M} by `analysis/make_results_md.py` from the files on disk. "
          f"Rerun it after any job finishes. {DASH} = not available yet. Story and interpretation: `../README.md`; "
          f"method and job history: `docs/06-distillation.md`.*\n"]

    # 1. grid
    md.append("## 1. GPTQ grid (fp16, w8, w4, w3, w2) + w2 with the distilled r64 adapter: averages over the 5 directions each way\n")
    grid = [("fp16", "ours-beam"), ("w8", "gptq-w8g128"), ("w4", "gptq-w4g128"), ("w3", "gptq-w3g128"), ("w2", "gptq-w2g128"),
            ("w2+KD", "gptq-w2g128-as4-kd-r64-cont")]
    mets = ["bleu", "chrf", "comet-22", "xcomet-xxl", "kiwi-22", "kiwi-xxl", "metricx-24", "halluc"]
    rows = []
    for name, run in grid:
        b = baseline(run)
        for side in ("avg_xx-en", "avg_en-xx"):
            rows.append([name, f"{SIZE[name]:.2f}", side.replace("avg_", "")] + [f(b.get((side, m))) for m in mets])
    md.append(table(["model", "GiB", "dir"] + [m if m != "chrf" else "chrF++" for m in mets], rows))
    md.append("\nMetricX-24: lower is better. halluc = % of outputs at least 2× the reference length. "
              "Per-direction numbers: `outputs/baseline/<run>.tsv`.\n")

    # 2. NLL
    md.append("## 2. Teacher-forced NLL increase over fp16 (nats/token, 500 test sentences per item)\n")
    tags = [("w8", "w8"), ("w4", "w4"), ("w3", "w3"), ("w2", "w2"), ("w3+KD r16", "w3kd"),
            ("w2+KD r64", "w2kd-r64"), ("w2+KD r64 cont.", "w2kd-r64-cont")]
    fp = nll("fp16")
    ns = {lab: nll(t) for lab, t in tags}
    items = [f"trans:{p}" for p in PAIRS] + [f"mono:{L}" for L in ["is", "de", "cs", "zh", "ru", "en"]]
    rows = [[it] + [f(ns[lab][it] - fp[it], 3, True) if ns[lab] else DASH for lab, _ in tags] + [f(fp[it], 3)]
            for it in items]
    md.append(table(["item"] + [lab for lab, _ in tags] + ["fp16 NLL"], rows))
    md.append("\n`trans:` = reference translation given the prompt (translation skill); `mono:` = plain text in the "
              "language (knows the language). 0 = as good as fp16.\n")

    # 3. adapter generations
    md.append("## 3. Adapter generations: BLEU / chrF++ / XCOMET-XXL per direction\n")
    md.append("w3 and w3+KD are on the same repacked ExllamaV2 kernel; fp16, w4 and w2 are the grid runs. "
              "Recovered = (KD − plain) / (fp16 − plain).\n")
    systems = [("fp16", "alma-13b-r-beam", "ours-beam"), ("w4", "gptq-w4g128", "gptq-w4g128"),
               ("w3", "gptq-w3g128-as4", "gptq-w3g128-as4"), ("w3+KD", "gptq-w3g128-as4-kd", "gptq-w3g128-as4-kd"),
               ("w2", "gptq-w2g128", "gptq-w2g128"), ("w2+KD r64", "gptq-w2g128-as4-kd-r64", "gptq-w2g128-as4-kd-r64"),
               ("w2+KD cont.", "gptq-w2g128-as4-kd-r64-cont", "gptq-w2g128-as4-kd-r64-cont")]
    for metric, lab in [("bleu", "BLEU"), ("chrf", "chrF++"), ("xcomet", "XCOMET-XXL")]:
        rows = []
        for p in PAIRS:
            v = {}
            for name, run, xc in systems:
                if metric == "xcomet":
                    v[name] = xcomet(xc, p)
                else:
                    g = gen_scores(run, p)
                    v[name] = g[metric] if g else None
            if v["w3+KD"] is None and v["w2+KD r64"] is None and v["w2+KD cont."] is None:
                continue
            rec = lambda kd, base: (f"{(v[kd] - v[base]) / (v['fp16'] - v[base]) * 100:.0f} %"  # noqa: E731
                                    if None not in (v[kd], v[base], v["fp16"]) and v["fp16"] != v[base] else DASH)
            rows.append([p] + [f(v[n]) for n, _, _ in systems]
                        + [rec("w3+KD", "w3"), rec("w2+KD r64", "w2"), rec("w2+KD cont.", "w2")])
        md.append(f"**{lab}**\n")
        md.append(table(["pair"] + [n for n, _, _ in systems] + ["rec. w3+KD", "rec. w2+KD", "rec. w2+KD cont."], rows))
        md.append("")

    # 4. bootstrap
    md.append("## 4. Paired bootstrap 95 % confidence intervals (1,000 resamples)\n")
    bp = J / "bootstrap_ci.json"
    if bp.exists():
        bs = json.load(open(bp))
        rows = []
        for study, pairs in bs.items():
            for p, r in pairs.items():
                for metric in ("bleu", "xcomet"):
                    m = r.get(metric)
                    if not isinstance(m, dict):
                        continue
                    kd = [k for k in m if k.endswith("kd")][0]
                    comps = "; ".join(f"{k}: {v['value']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]"
                                      + (" *" if v["ci95"][0] > 0 or v["ci95"][1] < 0 else "")
                                      for k, v in m.items() if "-" in k)
                    rc = m["recovered"]
                    rows.append([study, p, metric, f"{m[kd]['score']:.2f} [{m[kd]['ci95'][0]:.2f}, {m[kd]['ci95'][1]:.2f}]",
                                 f"{rc['value']*100:.0f} % [{rc['ci95'][0]*100:.0f}, {rc['ci95'][1]*100:.0f}]", comps])
        md.append(table(["study", "pair", "metric", "KD score [95 % CI]", "recovered [95 % CI]", "differences (* = CI excludes 0)"], rows))
        md.append("")
    else:
        md.append(f"{DASH} (run `analysis/bootstrap_ci.py`)\n")

    # 5. training
    md.append("## 5. Distillation runs: training loss and held-out KL vs fp16 (32 held-out train rows per direction)\n")
    rows = []
    for d in sorted((RES / "distill_runs").glob("*")):
        log = [json.loads(line) for line in open(d / "train_log.jsonl")]
        held = [r for r in log if "heldout" in r]
        tr = [r for r in log if "kd" in r and "heldout" not in r]
        cfg = json.loads((d / "adapter_config.json").read_text()) if (d / "adapter_config.json").exists() else {}
        h0, h1 = (held[0]["heldout"], held[-1]["heldout"]) if held else ({}, {})
        kl = lambda h, p: f(h[p]["kl"], 3) if p in h else DASH  # noqa: E731
        rows.append([d.name.replace("ALMA-13B-R-gptq-", ""), cfg.get("r", DASH), cfg.get("steps", DASH),
                     f(tr[-1]["kd"], 3) if tr else DASH]
                    + [f"{kl(h0, p)} → {kl(h1, p)}" for p in ("is-en", "en-is", "de-en", "zh-en")])
    md.append(table(["run", "rank", "steps", "final train KD", "is-en KL", "en-is KL", "de-en KL", "zh-en KL"], rows))
    md.append("\nKL start → end of each run (a continuation starts where its parent ended). Full logs: "
              "`results/distill_runs/<run>/train_log.jsonl`.\n")

    # 6. size and speed
    md.append("## 6. Size and speed\n")
    md.append(table(["model", "packed checkpoint GiB", "+ adapter (bf16) GiB", "total GiB", "fast-kernel container GiB"],
                    [["fp16", "24.24", "–", "24.24", "–"], ["w4", "6.76", "–", "6.76", "6.76"],
                     ["w3 + KD r16", "5.27", "0.12", "5.39", "6.76"], ["w2 + KD r64", "3.78", "0.47", "4.25", "6.76"]]))
    md.append("\nGeneration speed (de-en, batch 4, beam 5, H100): fp16 0.359 s/line; w4 0.461; w3 on Torch 1.188; "
              "w3 repacked on ExllamaV2 0.470 (2.5×). Details: `docs/04-kernel-repack.md` §8.\n")

    # 7. failure modes and recovery audit
    md.append("## 7. Failure modes and recovery audit (`docs/06-distillation.md` §5.8)\n")
    fp = J / "failures.json"
    if fp.exists():
        fl = json.load(open(fp))
        names = {"reference": "reference (LID noise floor)", "ours-beam": "fp16", "gptq-w8g128": "w8",
                 "gptq-w4g128": "w4", "gptq-w3g128": "w3", "gptq-w3g128-as4-kd": "w3 + KD r16",
                 "gptq-w2g128": "w2", "gptq-w2g128-as4-kd-r64": "w2 + KD r64",
                 "gptq-w2g128-as4-kd-r64-cont": "w2 + KD r64-cont"}
        for kind, title in [("any", "any failure"), ("osc", "oscillation (TNG, 4-gram, t = 2)"),
                            ("off_target", "off-target"), ("copy", "source copy"), ("trunc", "truncated (< 0.5× reference)")]:
            md.append(f"**{title}**, % of segments\n")
            rows = [[names.get(run, run)] + [f(r[p].get(kind)) if p in r else DASH for p in PAIRS]
                    for run, r in fl.items() if kind in next(iter(r.values()))]
            md.append(table(["run"] + PAIRS, rows))
            md.append("")
        md.append("Empty outputs: 0 for every run except plain w2 (≤ 0.16 %). Off-target = fastText NLLB LID (lid218e) "
                  "label ≠ target; for a Chinese target, fewer than half of the letters are Han (the LID mislabels "
                  "unsegmented Chinese). Copy = output equals the source after normalisation, and the reference doesn't.\n")
    else:
        md.append(f"{DASH} (run `analysis/score_failures.py`)\n")
    for ap in sorted(J.glob("audit_*.json")):
        au = json.load(open(ap))
        run = ap.stem.removeprefix("audit_")
        md.append(f"**Recovery audit of `{run}`** (`analysis/audit_recovery.py`; fp16 / w2 + KD / plain w3)\n")
        g = lambda p, k, m: au[p]["sys"].get(k, {}).get(m)  # noqa: E731
        rows = []
        for p in PAIRS:
            if p not in au:
                continue
            q = au[p]["sys"]["w2+KD"].get("delta_vs_fp16_by_srclen_quartile")
            rows.append([p, f(au[p]["overlap_8gram_src_pct"], 1),
                         " / ".join(f(g(p, k, "xcomet")) for k in ("fp16", "w2+KD", "w3")),
                         " / ".join(f(g(p, k, "critical_spans"), 0) for k in ("fp16", "w2+KD", "w3")),
                         " / ".join(f(g(p, k, "xcomet_lt_0.5_pct"), 1) for k in ("fp16", "w2+KD", "w3")),
                         " / ".join(f(v, 1, sign=True) for v in q) if q else DASH,
                         f(g(p, "w2+KD", "bleu_vs_fp16_output"), 1), f(g(p, "w4", "bleu_vs_fp16_output"), 1)])
        md.append(table(["pair", "test/train 8-gram overlap %", "XCOMET-XXL", "critical spans", "XCOMET < 0.5 %",
                         "w2 + KD − fp16 XCOMET by source-length quartile (short → long)",
                         "BLEU vs fp16 output: w2 + KD", "w4"], rows))
        md.append("\nWorst 40 segments vs fp16 per direction: `results/audit/<pair>.worst_vs_fp16.tsv`. "
                  "FLORES dev/devtest is inside ALMA's training data, so it isn't an out-of-distribution test.\n")

    (RES / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {RES / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
