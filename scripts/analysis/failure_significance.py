"""Failure rates of every run in the paper's failure tables, and which ones exceed full precision (CPU, ~3 min).

Uses the detectors of score_failures.py (Section 2.4) for every run, so Tables 2, 7 and 9 count the same thing.
A run is marked as failing significantly more often than full precision when an exact McNemar test on the same
segments gives p < 0.05 (two-sided) and the run fails on more of the discordant segments.

- wmt22: fp16, w3, w2, w3+KD, w2+KD and w2+CE (Tables 2 and 7).
- wmt24pp: fp16, w2 and w2+KD on WMT24++ (Table 9). The test set is read from --wmt24pp, laid out like the
  WMT'22 one (<src><tgt>/test.<src>-<tgt>.<src|tgt>, 960 segments per direction; xx-en reverses the en-xx pairs),
  and the translations from $OUTPUTS_DIR/<run>/wmt24pp/test-<src>-<tgt>. Skipped if the folder is missing.
Usage: python scripts/analysis/failure_significance.py [--wmt24pp DIR] [--lid /path/lid218e.bin]
Writes results/json/failure_significance.json.
"""
import argparse
import json
from pathlib import Path

import fasttext
from scipy.stats import binomtest

from mtcompress.paths import ARTIFACTS, JSON as J, OUTPUTS, PROJECT_DIR, TESTSET
from score_failures import HAN, LETTER, LID_LABEL, PAIRS, hyp_path, norm, read, toks, top_ngram

SETS = {
    "wmt22": ["ours-beam", "gptq-w3g128", "gptq-w2g128", "gptq-w3g128-as4-kd",
              "gptq-w2g128-as4-kd-r64-cont", "gptq-w2g128-as4-sft-r64-cont"],
    "wmt24pp": ["ood24-fp16", "ood24-w2", "ood24-w2kd"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wmt24pp", default=str(PROJECT_DIR / "wmt24pp/testset"))
    ap.add_argument("--lid", default=str(ARTIFACTS / "lid/lid218e.bin"))
    a = ap.parse_args()
    lid = fasttext.load_model(a.lid)
    roots = {"wmt22": TESTSET, "wmt24pp": Path(a.wmt24pp)}

    def off(texts, tgt):
        if tgt == "zh":
            return [len(HAN.findall(x)) < 0.5 * len(LETTER.findall(x)) for x in texts]
        labels, _ = lid.predict([t.replace("\n", " ") for t in texts])
        return [l[0].removeprefix("__label__") not in LID_LABEL[tgt] for l in labels]

    def hyp_of(ts, run, pair):
        return hyp_path(run, *pair.split("-")) if ts == "wmt22" else OUTPUTS / run / ts / f"test-{pair}"

    result = {}
    for ts, runs in SETS.items():
        if not roots[ts].exists():
            print(f"skipping {ts}: {roots[ts]} not found")
            continue
        flags = {}
        for pair in PAIRS:
            s, t = pair.split("-")
            src = read(roots[ts] / f"{s}{t}/test.{pair}.{s}")
            ref = read(roots[ts] / f"{s}{t}/test.{pair}.{t}")
            src_top = [top_ngram(toks(z, s)) for z in src]
            ref_len = [len(toks(r, t)) for r in ref]
            for run in runs:
                hyp = read(hyp_of(ts, run, pair))
                assert len(hyp) == len(ref), f"{ts} {run} {pair}: {len(hyp)} vs {len(ref)}"
                is_off = off(hyp, t)
                f = []
                for i, h in enumerate(hyp):
                    ht = toks(h, t)
                    f.append(not ht or top_ngram(ht) - src_top[i] >= 2 or (bool(ht) and is_off[i])
                             or (bool(ht) and norm(h) == norm(src[i]) and norm(ref[i]) != norm(src[i]))
                             or 0 < len(ht) / max(1, ref_len[i]) < 0.5)
                flags.setdefault(run, {})[pair] = f

        base = runs[0]
        result[ts] = {}
        for run in runs:
            result[ts][run] = {}
            for pair in PAIRS:
                x, y = flags[base][pair], flags[run][pair]
                more = sum(b and not a for a, b in zip(x, y))  # run fails, fp16 does not
                fewer = sum(a and not b for a, b in zip(x, y))
                p = float(binomtest(more, more + fewer, 0.5).pvalue) if run != base and more + fewer else 1.0
                result[ts][run][pair] = {"any": 100 * sum(y) / len(y), "p_mcnemar": p,
                                         "worse": run != base and more > fewer and p < 0.05}

        print(f"\n== {ts}: % of segments with any failure (* = significantly more than {base})")
        print(f"{'run':30s}" + "".join(f"{p:>8s}" for p in PAIRS))
        for run, r in result[ts].items():
            print(f"{run:30s}" + "".join(f"{r[p]['any']:7.2f}{'*' if r[p]['worse'] else ' '}" for p in PAIRS))

    J.mkdir(parents=True, exist_ok=True)
    (J / "failure_significance.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
