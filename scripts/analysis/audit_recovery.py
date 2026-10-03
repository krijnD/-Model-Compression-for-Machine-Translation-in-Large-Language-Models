"""CPU sanity audit of a recovered model's generations: is the recovery real, or a metric / data artefact?

Per direction and system: train/test overlap, length ratio, repetition loops, off-target language, copied
sources, number preservation, XCOMET-XXL error spans (critical / major), length buckets, similarity to the fp16
teacher, and a dump of the worst segments for manual reading.
Usage: python scripts/analysis/audit_recovery.py [--kd gptq-w2g128-as4-kd-r64-cont]
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import sacrebleu

REPO = Path(__file__).resolve().parents[2]
OUT = REPO.parent / "outputs"
TESTSET = REPO / "third_party/ALMA/outputs/wmt22_outputs/wmt-testset"
ALMA_DATA = REPO / "third_party/ALMA/human_written_data"
J = REPO / "results/json"
PAIRS = ["is-en", "en-is", "de-en", "en-de", "cs-en", "en-cs", "ru-en", "en-ru", "zh-en", "en-zh"]

# Tiny stopword LID over the six languages; scripts first for ru / zh.
STOP = {
    "en": "the of and to is in that for it with was on are as this be by not have from",
    "de": "der die und das ist nicht zu den von mit sich des auf für ein eine dem im auch",
    "cs": "a je se na že v to s z do jsou o by k ale jako pro také není jsem",
    "is": "og að er á í sem til við um það ekki með hann var en þá fyrir hún eru",
}
STOP = {k: set(v.split()) for k, v in STOP.items()}


def lid(text):
    cyr = len(re.findall(r"[Ѐ-ӿ]", text))
    han = len(re.findall(r"[一-鿿]", text))
    lat = len(re.findall(r"[A-Za-zÀ-ž]", text))
    if han > max(cyr, lat) * 0.5 and han > 0:
        return "zh"
    if cyr > lat:
        return "ru"
    toks = re.findall(r"\w+", text.lower())
    if re.search(r"[þðæ]", text.lower()):
        return "is"
    if re.search(r"[ěščřžůň]", text.lower()):
        return "cs"
    scores = {k: sum(t in s for t in toks) for k, s in STOP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "?"


def units(text, lang):
    return list(text.replace(" ", "")) if lang == "zh" else text.split()


def has_loop(text, lang, n=3, k=3):
    u = units(text, lang)
    n = 4 if lang == "zh" else n
    c = Counter(tuple(u[i:i + n]) for i in range(len(u) - n + 1))
    return bool(c) and max(c.values()) >= k


NUM = re.compile(r"\d+(?:[.,]\d+)*")


def nums(text):
    return Counter(re.sub(r"[.,]", "", m) for m in NUM.findall(text))


def norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def shingles(s, n=8):
    t = norm(s).split()
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def lines(p):
    with open(p, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def xcomet(run, pair):
    p = OUT / "xcomet-xxl" / run / f"{pair}.json"
    if not p.exists():
        return None
    segs = list(json.load(open(p)).values())[0]
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kd", default="gptq-w2g128-as4-kd-r64-cont")
    ap.add_argument("--dump", default=str(REPO / "results/audit"))
    a = ap.parse_args()
    systems = {"fp16": ("alma-13b-r-beam", "ours-beam"), "w4": ("gptq-w4g128", "gptq-w4g128"),
               "w3": ("gptq-w3g128", "gptq-w3g128"), "w2+KD": (a.kd, a.kd)}
    dump = Path(a.dump)
    dump.mkdir(parents=True, exist_ok=True)
    report = {}
    chrf = sacrebleu.CHRF()
    for pair in PAIRS:
        s, t = pair.split("-")
        x = s if s != "en" else t
        src = lines(TESTSET / f"{s}{t}/test.{pair}.{s}")
        ref = lines(TESTSET / f"{s}{t}/test.{pair}.{t}")
        # train/test overlap (exact normalised source, and any shared 8-gram)
        train = [ex["translation"] for ex in map(json.loads, open(ALMA_DATA / f"{x}en/train.{x}-en.json"))]
        tr_src = {norm(ex[s]) for ex in train}
        tr_sh = set().union(*(shingles(ex[s]) for ex in train))
        tr_ref = {norm(ex[t]) for ex in train}
        r = {"n": len(src),
             "overlap_exact_src_pct": 100 * np.mean([norm(z) in tr_src for z in src]),
             "overlap_exact_ref_pct": 100 * np.mean([norm(z) in tr_ref for z in ref]),
             "overlap_8gram_src_pct": 100 * np.mean([bool(shingles(z) & tr_sh) for z in src]),
             "sys": {}}
        hyps = {k: lines(OUT / v[0] / "wmt22" / f"test-{pair}") for k, v in systems.items()}
        slen = np.array([len(units(z, s)) for z in src])
        qs = np.quantile(slen, [0.25, 0.5, 0.75])
        bucket = np.digitize(slen, qs)
        xc_fp16 = xcomet(systems["fp16"][1], pair)
        for k, hyp in hyps.items():
            assert len(hyp) == len(ref), (k, pair, len(hyp), len(ref))
            ratio = np.array([len(units(h, t)) / max(1, len(units(rf, t))) for h, rf in zip(hyp, ref)])
            lids = [lid(h) for h in hyp]
            src_n = [nums(z) for z in src]
            miss = [sum((sn - nums(h)).values()) for sn, h in zip(src_n, hyp)]
            tot = sum(sum(sn.values()) for sn in src_n)
            copy = [chrf.sentence_score(h, [z]).score > 70 for h, z in zip(hyp, src)] if s != t else []
            d = {
                "bleu": sacrebleu.corpus_bleu(hyp, [ref], tokenize="zh" if t == "zh" else "13a").score,
                "len_ratio_mean": ratio.mean(), "len_ratio_std": ratio.std(),
                "short_pct(<0.6)": 100 * np.mean(ratio < 0.6), "long_pct(>1.6)": 100 * np.mean(ratio > 1.6),
                "empty_pct": 100 * np.mean([h.strip() == "" for h in hyp]),
                "loop_pct": 100 * np.mean([has_loop(h, t) for h in hyp]),
                "off_target_pct": 100 * np.mean([l not in (t, "?") for l in lids]),
                "copy_src_pct": 100 * np.mean(copy),
                "num_missing_pct_of_src_numbers": 100 * sum(miss) / max(1, tot),
                "segs_missing_a_number_pct": 100 * np.mean([m > 0 for m in miss]),
                "bleu_vs_fp16_output": None if k == "fp16" else
                    sacrebleu.corpus_bleu(hyp, [hyps["fp16"]], tokenize="zh" if t == "zh" else "13a").score,
            }
            xc = xcomet(systems[k][1], pair)
            if xc and len(xc) == len(ref):
                sc = np.array([z["COMET"] for z in xc])
                sev = Counter(e["severity"] for z in xc for e in z.get("errors", []))
                d.update({"xcomet": 100 * sc.mean(), "xcomet_lt_0.5_pct": 100 * np.mean(sc < 0.5),
                          "xcomet_p5": 100 * np.quantile(sc, 0.05),
                          "critical_spans": sev["critical"], "major_spans": sev["major"], "minor_spans": sev["minor"],
                          "segs_with_critical_pct": 100 * np.mean(
                              [any(e["severity"] == "critical" for e in z.get("errors", [])) for z in xc]),
                          "xcomet_by_srclen_quartile": [100 * sc[bucket == b].mean() for b in range(4)]})
                if k == "w2+KD" and xc_fp16:
                    f = np.array([z["COMET"] for z in xc_fp16])
                    delta = sc - f
                    d["delta_vs_fp16_by_srclen_quartile"] = [100 * delta[bucket == b].mean() for b in range(4)]
                    worst = np.argsort(delta)[:40]
                    d["worst5pct_also_fp16_worst5pct"] = 100 * np.mean(
                        np.isin(np.argsort(sc)[:len(sc) // 20], np.argsort(f)[:len(f) // 20]))
                    with open(dump / f"{pair}.worst_vs_fp16.tsv", "w", encoding="utf-8") as fo:
                        fo.write("idx\tdelta\tkd\tfp16\tsrc\tref\tfp16_hyp\tkd_hyp\tkd_errors\n")
                        for i in worst:
                            errs = "; ".join(f"{e['severity']}:{e['text']}" for e in xc[i].get("errors", []))
                            fo.write(f"{i}\t{100*delta[i]:.1f}\t{100*sc[i]:.1f}\t{100*f[i]:.1f}\t{src[i]}\t{ref[i]}"
                                     f"\t{hyps['fp16'][i]}\t{hyp[i]}\t{errs}\n")
            r["sys"][k] = d
        report[pair] = r
    J.mkdir(parents=True, exist_ok=True)
    (J / f"audit_{a.kd}.json").write_text(json.dumps(report, indent=1, default=float))
    keys = ["bleu", "xcomet", "xcomet_p5", "xcomet_lt_0.5_pct", "segs_with_critical_pct", "critical_spans",
            "len_ratio_mean", "len_ratio_std", "short_pct(<0.6)", "long_pct(>1.6)", "loop_pct", "off_target_pct",
            "copy_src_pct", "segs_missing_a_number_pct", "bleu_vs_fp16_output"]
    for pair, r in report.items():
        print(f"\n== {pair}  n={r['n']}  overlap exact src {r['overlap_exact_src_pct']:.1f}% ref "
              f"{r['overlap_exact_ref_pct']:.1f}% 8-gram {r['overlap_8gram_src_pct']:.1f}%")
        print(f"{'metric':32s}" + "".join(f"{k:>9s}" for k in r["sys"]))
        for m in keys:
            vals = [r["sys"][k].get(m) for k in r["sys"]]
            print(f"{m:32s}" + "".join(f"{'-':>9s}" if v is None else f"{v:9.2f}" for v in vals))
        kd = r["sys"]["w2+KD"]
        if "delta_vs_fp16_by_srclen_quartile" in kd:
            print("w2+KD - fp16 XCOMET by source-length quartile:",
                  " ".join(f"{v:+.2f}" for v in kd["delta_vs_fp16_by_srclen_quartile"]),
                  f"| worst-5% shared with fp16: {kd['worst5pct_also_fp16_worst5pct']:.0f}%")


if __name__ == "__main__":
    main()
