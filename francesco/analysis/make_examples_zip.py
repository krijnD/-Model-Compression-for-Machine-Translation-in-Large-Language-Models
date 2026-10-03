"""Bundle candidate translation examples for hand-picking the qualitative table (paper Section 3.4).

For every direction: source, reference and the outputs of fp16, w3, w2, w3+KD and w2+KD side by side, with
segment-level XCOMET-XXL where it exists and the failure flags of score_failures.py for w2+KD. Writes one CSV
per case study (short sources first, since they fit a paper table) plus all rows per direction, and a README.
Usage: python francesco/analysis/make_examples_zip.py [--out ~/translation_examples.zip]
"""
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

OUT_ROOT = Path("/gpfs/home6/scur0517/outputs")
TESTSET = Path("/gpfs/home6/scur0517/Model-Compression-MT/third_party/ALMA/outputs/wmt22_outputs/wmt-testset")
PAIRS = "de-en cs-en is-en zh-en ru-en en-de en-cs en-is en-zh en-ru".split()
SYSTEMS = {"fp16": ("alma-13b-r-beam", "ours-beam"), "w3": ("gptq-w3g128", "gptq-w3g128"),
           "w2": ("gptq-w2g128", "gptq-w2g128"), "w3kd": ("gptq-w3g128-as4-kd", "gptq-w3g128-as4-kd"),
           "w2kd": ("gptq-w2g128-as4-kd-r64-cont", "gptq-w2g128-as4-kd-r64-cont")}  # name: (output dir, score run)
KINDS = ["osc", "off_target", "copy", "empty", "trunc"]


def lines(path):
    return path.read_text(encoding="utf-8").split("\n")


def load_pair(pair):
    src_l, tgt_l = pair.split("-")
    folder = TESTSET / f"{src_l}{tgt_l}"
    src, ref = lines(folder / f"test.{pair}.{src_l}"), lines(folder / f"test.{pair}.{tgt_l}")
    n = len(src) - (src[-1] == "")
    rows = [{"id": i, "pair": pair, "src_words": len(src[i].split()), "source": src[i], "reference": ref[i]}
            for i in range(n)]
    for name, (out_dir, run) in SYSTEMS.items():
        hyp = lines(OUT_ROOT / out_dir / "wmt22" / f"test-{pair}")
        xc = OUT_ROOT / "xcomet-xxl" / run / f"{pair}.json"
        seg = list(json.load(open(xc)).values())[0] if xc.exists() else None
        for i, r in enumerate(rows):
            r[name] = hyp[i] if i < len(hyp) else ""
            r[f"xcomet_{name}"] = round(100 * seg[i]["COMET"], 1) if seg else ""
    flags = json.load(open(OUT_ROOT / "failures" / SYSTEMS["w2kd"][1] / f"{pair}.json"))
    for r in rows:
        r["w2kd_failure"] = ",".join(k for k in KINDS if r["id"] in set(flags.get(k, [])))
    return rows


COLS = ["id", "pair", "src_words", "source", "reference", "fp16", "w3", "w2", "w3kd", "w2kd", "w2kd_failure",
        "xcomet_fp16", "xcomet_w3", "xcomet_w2", "xcomet_w3kd", "xcomet_w2kd"]


def to_csv(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return "﻿" + buf.getvalue()  # BOM so Excel opens UTF-8 (Icelandic, Chinese, Russian) correctly


def num(x):
    return x if isinstance(x, (int, float)) else None


README = """# Candidate examples for the qualitative table (Section 3.4)

Goal: pick 3-4 SHORT sentences that make the failure taxonomy concrete. Ideal final table: one oscillation
(en-zh), one source copy (en-de, "Tap Settings."-style), one Icelandic case where plain w3 fails and w3+KD
repairs it, and optionally one long en-is sentence where w2+KD is fluent but wrong.

Systems (columns):
- fp16 : full-precision ALMA-13B-R (the reference model)
- w3 / w2 : plain GPTQ 3-bit / 2-bit
- w3kd / w2kd : the same + distilled LoRA adapter (our recovered models)
- w2kd_failure : automatic flags for w2kd (osc = repetition loop, off_target = wrong language,
  copy = source copied, trunc = output < half the reference)
- xcomet_* : XCOMET-XXL x100 per sentence (higher is better). Empty = not scored (yet).

Files (each sorted with the shortest sources first; the first rows are the best table candidates):
- 01_oscillation_en-zh.csv   w2kd repetition loops into Chinese
- 02_source_copy_en-de.csv   w2kd leaves the English source untranslated
- 03_oscillation_en-is.csv   w2kd repetition loops into Icelandic
- 04_long_gap_en-is.csv      long sentences (top quarter) with no flag, where w2kd loses most XCOMET vs fp16
- 05_icelandic_w3_fixed.csv  is-en / en-is: fp16 >= 80, plain w3 < 50, w3kd within 5 points of fp16
- 06_w2_collapse.csv         plain w2 on short sentences (what "collapse" looks like)
- all/<pair>.csv             every sentence of every direction, for free browsing

Open in Excel / Google Sheets / LibreOffice (UTF-8). Please send back the pair + id of your picks, and a one-line
note on what each example shows (e.g. "w2kd repeats 'light' 40 times; fp16 and w3kd fine"). Check that the
fp16 output is actually good: we want examples where only the compressed model fails.

Note: XCOMET-XXL is run without the reference, so it barely penalises an untranslated copy (e.g. en-de id 21,
"3) Tap Settings." copied: 94.9 vs fp16 96.0). Judge copies and loops by reading, not by the score.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "translation_examples.zip"))
    args = ap.parse_args()
    data = {p: load_pair(p) for p in PAIRS}
    by_len = lambda rows: sorted(rows, key=lambda r: r["src_words"])
    flagged = lambda pair, kind: [r for r in data[pair] if kind in r["w2kd_failure"].split(",")]

    long_isl = [r for r in data["en-is"] if not r["w2kd_failure"] and num(r["xcomet_fp16"]) is not None]
    q3 = sorted(r["src_words"] for r in long_isl)[int(0.75 * len(long_isl))]
    long_gap = sorted((r for r in long_isl if r["src_words"] >= q3 and num(r["xcomet_w2kd"]) is not None),
                      key=lambda r: r["xcomet_w2kd"] - r["xcomet_fp16"])[:40]
    fixed = [r for p in ("is-en", "en-is") for r in data[p]
             if num(r["xcomet_w3"]) is not None and num(r["xcomet_w3kd"]) is not None
             and r["xcomet_fp16"] >= 80 and r["xcomet_w3"] < 50 and r["xcomet_w3kd"] >= r["xcomet_fp16"] - 5]
    collapse = [r for p in ("de-en", "en-de", "is-en") for r in data[p] if 4 <= r["src_words"] <= 12][:30]

    files = {"01_oscillation_en-zh.csv": by_len(flagged("en-zh", "osc"))[:60],
             "02_source_copy_en-de.csv": by_len(flagged("en-de", "copy"))[:60],
             "03_oscillation_en-is.csv": by_len(flagged("en-is", "osc"))[:60],
             "04_long_gap_en-is.csv": long_gap,
             "05_icelandic_w3_fixed.csv": by_len(fixed)[:60],
             "06_w2_collapse.csv": collapse}
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("translation_examples/README.md", README)
        for name, rows in files.items():
            z.writestr(f"translation_examples/{name}", to_csv(rows))
            print(f"{name}: {len(rows)} rows")
        for p, rows in data.items():
            z.writestr(f"translation_examples/all/{p}.csv", to_csv(rows))
    print(args.out)


if __name__ == "__main__":
    main()
