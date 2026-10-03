"""Is `is` fragile, or are *hard sentences* fragile and `is` just has more of them?  CPU only.

Uses the per-sentence scores the grid already wrote (<root>/outputs/<metric>/<run>/<pair>.json):
damage = XCOMET-XXL(w3) - XCOMET-XXL(fp16 beam), per sentence; "difficulty" = the fp16 output's
reference-free KIWI-XXL score (a different metric than the damage one, so a shared metric error
does not create the correlation by itself). w8 is the null run (no quality change), so the same
statistic computed on w8 gives the regression-to-the-mean floor the w3 number must beat.

  python francesco/analysis/fragility_by_confidence.py [--bits 3] [--bins 5]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PAIRS = ["de-en", "cs-en", "is-en", "zh-en", "ru-en", "en-de", "en-cs", "en-is", "en-zh", "en-ru"]


def load(root, metric, run, pair):
    x = json.load(open(root / "outputs" / metric / run / f"{pair}.json"))
    if isinstance(x, dict):
        x = next(iter(x.values()))
    return np.array([r["COMET"] for r in x], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results/json/fragility_by_confidence.json"))
    a = ap.parse_args()
    root = Path(a.root)
    run = f"gptq-w{a.bits}g128"
    rows, res = [], {"bits": a.bits, "pairs": {}}
    print(f"{'pair':6s} {'n':>5s} {'fp16 kiwi':>9s} {'dXCOMET':>8s} | rho(kiwi_fp16, d) w{a.bits}  w8-null")
    for p in PAIRS:
        x16 = load(root, "xcomet-xxl", "ours-beam", p)
        xq = load(root, "xcomet-xxl", run, p)
        x8 = load(root, "xcomet-xxl", "gptq-w8g128", p)
        k16 = load(root, "kiwi-xxl", "ours-beam", p)
        d, d8 = xq - x16, x8 - x16
        r, _ = spearmanr(k16, d)
        r8, _ = spearmanr(k16, d8)
        res["pairs"][p] = {"n": len(d), "kiwi_fp16": k16.mean(), "d_mean": d.mean(), "rho": r, "rho_w8": r8}
        print(f"{p:6s} {len(d):5d} {k16.mean():9.3f} {d.mean():8.3f} | {r:+.3f}          {r8:+.3f}")
        rows += [(p, k, dd) for k, dd in zip(k16, d)]

    # Pooled difficulty bins (quantiles of fp16 KIWI-XXL over all 10 pairs): is `is` still the
    # worst direction *within* a bin of equally-hard sentences?
    k = np.array([r[1] for r in rows]); d = np.array([r[2] for r in rows])
    lang = np.array([("is" if "is" in r[0] else "other") for r in rows])
    edges = np.quantile(k, np.linspace(0, 1, a.bins + 1))
    b = np.clip(np.searchsorted(edges, k, side="right") - 1, 0, a.bins - 1)
    print(f"\npooled fp16-KIWI-XXL quintiles: mean damage, `is` pairs vs the other 8 (n_is / n_other)")
    res["bins"] = []
    for i in range(a.bins):
        m_is, m_ot = (b == i) & (lang == "is"), (b == i) & (lang == "other")
        row = {"lo": edges[i], "hi": edges[i + 1], "n_is": int(m_is.sum()), "n_other": int(m_ot.sum()),
               "d_is": d[m_is].mean() if m_is.any() else None, "d_other": d[m_ot].mean()}
        res["bins"].append(row)
        print(f"  [{edges[i]:.3f},{edges[i+1]:.3f}]  is {row['d_is']:+.3f}  other {row['d_other']:+.3f}"
              f"   ({row['n_is']}/{row['n_other']})")
    # difficulty-matched gap: reweight the `other` pool to the `is` difficulty histogram
    w = np.array([r["n_is"] for r in res["bins"]], dtype=float); w /= w.sum()
    gap_raw = d[lang == "is"].mean() - d[lang == "other"].mean()
    gap_matched = sum(wi * (r["d_is"] - r["d_other"]) for wi, r in zip(w, res["bins"]) if r["d_is"] is not None)
    res.update(gap_raw=gap_raw, gap_matched=gap_matched)
    print(f"\n`is` minus other, XCOMET-XXL damage: raw {gap_raw:+.3f}, difficulty-matched {gap_matched:+.3f}")
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
