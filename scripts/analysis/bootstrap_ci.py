"""Paired bootstrap confidence intervals (Koehn 2004) for the distillation results (docs/06-distillation.md §9-11).

For each direction: resample the test sentences with replacement B times, using the SAME indices for every system
(paired), and recompute each score on every resample. The 2.5/97.5 percentiles give the 95 % CI.
  BLEU      from summed per-sentence n-gram statistics (sacrebleu's own corpus BLEU from the resampled counts,
            not a mean of sentence BLEU)
  XCOMET    mean of the per-segment scores in outputs/xcomet-xxl/<run>/<pair>.json (x100)
Derived quantities, also computed inside every resample:
  recovered = (kd - base) / (fp16 - base)     base = plain quantized model on the same kernel (w3) / plain w2
  kd - w4, kd - plain w3                       with P(diff <= 0) = share of resamples where the gain vanishes
CPU only (login node is fine): ../venv/bin/python scripts/analysis/bootstrap_ci.py [--B 1000]
Output: results/json/bootstrap_ci.json and a printed table.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sacrebleu.metrics import BLEU

REPO = Path(__file__).resolve().parents[2]
OUT = REPO.parent / "outputs"
TESTSET = REPO / "third_party/ALMA/outputs/wmt22_outputs/wmt-testset"
J = REPO / "results/json"

# system name -> (generation folder, xcomet folder)
SYS = {"fp16": ("alma-13b-r-beam", "ours-beam"), "w4": ("gptq-w4g128", "gptq-w4g128"),
       "w3": ("gptq-w3g128-as4", "gptq-w3g128-as4"), "w3kd": ("gptq-w3g128-as4-kd", "gptq-w3g128-as4-kd"),
       "w2": ("gptq-w2g128", "gptq-w2g128"), "w2kd": ("gptq-w2g128-as4-kd-r64", "gptq-w2g128-as4-kd-r64"),
       "w3grid": ("gptq-w3g128", "gptq-w3g128"),  # plain w3 as scored in the grid (all 10 directions)
       "w2kdc": ("gptq-w2g128-as4-kd-r64-cont", "gptq-w2g128-as4-kd-r64-cont")}  # final 2-bit adapter
ALL10 = ["de-en", "cs-en", "is-en", "zh-en", "ru-en", "en-de", "en-cs", "en-is", "en-zh", "en-ru"]
STUDIES = {  # study -> (pairs, systems, kd, base, extra comparisons kd - x)
    "w3+KD": (["is-en", "en-is", "de-en", "en-de"], ["fp16", "w4", "w3", "w3kd"], "w3kd", "w3", ["w4"]),
    "w2+KD r64": (["is-en", "de-en"], ["fp16", "w3", "w2", "w2kd"], "w2kd", "w2", ["w3"]),
    # paper: all ten directions, plain w3 from the grid (the repacked w3 exists for 4 directions only)
    "w3+KD all": (ALL10, ["fp16", "w4", "w3grid", "w3kd"], "w3kd", "w3grid", ["w4"]),
    "w2+KD all": (ALL10, ["fp16", "w3grid", "w2", "w2kdc"], "w2kdc", "w2", ["w3grid"]),
}


def lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def bleu_stats(bleu, hyps, refs):
    return np.array(bleu._extract_corpus_statistics(hyps, [refs]), dtype=np.int64)  # (n_sent, 2 + 4*2)


def bleu_from(bleu, summed):
    return bleu._compute_score_from_stats(list(summed)).score


def xcomet_seg(xc, pair):
    f = OUT / "xcomet-xxl" / xc / f"{pair}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    return 100 * np.array([s["COMET"] for s in next(iter(d.values()))])


def ci(v):
    v = np.asarray(v, dtype=float)
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    res = {}
    for study, (pairs, systems, kd, base, extra) in STUDIES.items():
        for pair in pairs:
            s, t = pair.split("-")
            bleu = BLEU(tokenize="zh" if t == "zh" else "13a")  # as scripts/reproduce/score_lexical.py
            refs = lines(TESTSET / f"{s}{t}/test.{pair}.{t}")
            n = len(refs)
            stats = {k: bleu_stats(bleu, lines(OUT / SYS[k][0] / "wmt22" / f"test-{pair}"), refs) for k in systems}
            xc = {k: xcomet_seg(SYS[k][1], pair) for k in systems}
            idx = rng.integers(0, n, size=(a.B, n))  # paired: the same resamples for every system
            r = {"n": n, "B": a.B}
            for metric in ("bleu", "xcomet"):
                if metric == "xcomet" and any(v is None or len(v) != n for v in xc.values()):
                    r[metric] = "missing for " + ",".join(k for k, v in xc.items() if v is None)
                    continue
                point = {k: (bleu_from(bleu, stats[k].sum(0)) if metric == "bleu" else float(xc[k].mean())) for k in systems}
                boot = {k: np.array([bleu_from(bleu, stats[k][i].sum(0)) for i in idx]) if metric == "bleu"
                        else xc[k][idx].mean(1) for k in systems}
                m = {k: {"score": round(point[k], 2), "ci95": ci(boot[k])} for k in systems}
                denom = boot["fp16"] - boot[base]
                rec = (boot[kd] - boot[base]) / np.where(np.abs(denom) < 1e-9, np.nan, denom)
                m["recovered"] = {"value": round((point[kd] - point[base]) / (point["fp16"] - point[base]), 3),
                                  "ci95": ci(rec[np.isfinite(rec)])}
                for x in [base] + extra:
                    d = boot[kd] - boot[x]
                    m[f"{kd}-{x}"] = {"value": round(point[kd] - point[x], 2), "ci95": ci(d),
                                      "p_diff_le_0": round(float((d <= 0).mean()), 4)}
                d = boot[kd] - boot["fp16"]
                m[f"{kd}-fp16"] = {"value": round(point[kd] - point["fp16"], 2), "ci95": ci(d)}
                r[metric] = m
            res.setdefault(study, {})[pair] = r
            for metric in ("bleu", "xcomet"):
                m = r[metric]
                if isinstance(m, str):
                    print(f"{study:10s} {pair} {metric:6s} {m}")
                    continue
                rc = m["recovered"]
                ex = "  ".join(f"{k}: {v['value']:+.2f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]"
                               for k, v in m.items() if "-" in k)
                print(f"{study:10s} {pair} {metric:6s} {kd} {m[kd]['score']:.2f} {m[kd]['ci95']}  "
                      f"recovered {rc['value']*100:.0f}% [{rc['ci95'][0]*100:.0f},{rc['ci95'][1]*100:.0f}]  {ex}", flush=True)
    json.dump(res, open(J / "bootstrap_ci.json", "w"), indent=1)
    print(f"wrote {J / 'bootstrap_ci.json'}")


if __name__ == "__main__":
    main()
