"""Translation failure modes per run and direction (CPU, ~2 min incl. loading the LID model).

- osc (TNG, Raunak et al. 2021 as used by Guerreiro et al. 2023, EACL, §3): the top repeated 4-gram of the
  translation occurs at least t = 2 more times than the top repeated 4-gram of the source. The paper also
  requires the segment to be in the 1 % lowest quality; we report the plain TNG flag.
- off-target (Zhang et al. 2020, ACL): fastText NLLB LID (lid218e) label of the output is not the target language.
  The same LID run on the references gives the noise floor (short headlines, names, URLs). For a Chinese target
  the LID is unreliable (it labels unsegmented Chinese such as "他否认了这项指控。" as Irish on 25 % of the
  references), so there off-target = fewer than half of the letters are Han characters.
- copy: output equals the source after lowercasing and dropping non-word characters, and the reference doesn't.
- empty: no tokens. trunc: 0 < output tokens / reference tokens < 0.5.
Tokens are sacrebleu's (13a; "zh" for Chinese, which splits CJK characters).
Usage: python scripts/analysis/score_failures.py [--runs a b c] [--lid /path/lid218e.bin]
Writes outputs/failures/<run>/summary.tsv, flagged segment ids in outputs/failures/<run>/<pair>.json and
results/json/failures.json.
"""
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import fasttext
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO.parent / "outputs"
TESTSET = REPO / "third_party/ALMA/outputs/wmt22_outputs/wmt-testset"
J = REPO / "results/json"
PAIRS = "de-en,cs-en,is-en,zh-en,ru-en,en-de,en-cs,en-is,en-zh,en-ru".split(",")
RUNS = ["ours-beam", "gptq-w8g128", "gptq-w4g128", "gptq-w3g128", "gptq-w3g128-as4-kd", "gptq-w2g128",
        "gptq-w2g128-as4-kd-r64", "gptq-w2g128-as4-kd-r64-cont"]
LID_LABEL = {"en": {"eng_Latn"}, "de": {"deu_Latn"}, "cs": {"ces_Latn"}, "is": {"isl_Latn"}, "ru": {"rus_Cyrl"},
             "zh": {"zho_Hans", "zho_Hant", "yue_Hant"}}
KINDS = ["osc", "off_target", "copy", "empty", "trunc"]
HAN = re.compile(r"[\u4e00-\u9fff]")
LETTER = re.compile(r"[^\W\d_]")
TOK = {"zh": TokenizerZh(), "other": Tokenizer13a()}


def toks(text, lang):
    return TOK["zh" if lang == "zh" else "other"](text.strip()).split()


def top_ngram(tokens, n=4):
    c = Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return max(c.values()) if c else 0


def norm(s):
    return re.sub(r"\W+", "", s.lower())


def hyp_path(run, src, tgt):  # same mapping as scripts/reproduce/score_lexical.py
    fixed = {"ours": OUTPUTS / f"alma-13b-r/wmt22/test-{src}-{tgt}",
             "ours-beam": OUTPUTS / f"alma-13b-r-beam/wmt22/test-{src}-{tgt}"}
    return fixed.get(run, OUTPUTS / f"{run}/wmt22/test-{src}-{tgt}")


def read(p):
    return p.read_text(encoding="utf-8").splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--lid", default=f"/scratch-shared/{os.environ['USER']}/models/lid/lid218e.bin")
    a = ap.parse_args()
    lid = fasttext.load_model(a.lid)

    def lang(texts):
        labels, _ = lid.predict([t.replace("\n", " ") for t in texts])
        return [l[0].removeprefix("__label__") for l in labels]

    def off(texts, tgt):
        if tgt == "zh":
            return [len(HAN.findall(x)) < 0.5 * len(LETTER.findall(x)) for x in texts]
        return [l not in LID_LABEL[tgt] for l in lang(texts)]

    data, result = {}, {}
    for pair in PAIRS:
        s, t = pair.split("-")
        src = read(TESTSET / f"{s}{t}/test.{pair}.{s}")
        ref = read(TESTSET / f"{s}{t}/test.{pair}.{t}")
        data[pair] = (src, ref, [top_ngram(toks(z, s)) for z in src], [len(toks(r, t)) for r in ref])
        # noise floor of the LID: references flagged as off-target
        result.setdefault("reference", {})[pair] = {
            "off_target": 100 * sum(off(ref, t)) / len(ref)}

    for run in ["reference"] + a.runs:
        if run == "reference":
            continue
        result[run] = {}
        out = OUTPUTS / "failures" / run
        for pair in PAIRS:
            s, t = pair.split("-")
            p = hyp_path(run, s, t)
            if not p.exists():
                continue
            src, ref, src_top, ref_len = data[pair]
            hyp = read(p)
            assert len(hyp) == len(ref), f"{run} {pair}: {len(hyp)} vs {len(ref)}"
            is_off = off(hyp, t)
            flags = {k: [] for k in KINDS + ["any"]}
            for i, h in enumerate(hyp):
                ht = toks(h, t)
                f = {"empty": not ht,
                     "osc": top_ngram(ht) - src_top[i] >= 2,
                     "off_target": bool(ht) and is_off[i],
                     "copy": bool(ht) and norm(h) == norm(src[i]) and norm(ref[i]) != norm(src[i]),
                     "trunc": 0 < len(ht) / max(1, ref_len[i]) < 0.5}
                for k, v in f.items():
                    if v:
                        flags[k].append(i)
                if any(f.values()):
                    flags["any"].append(i)
            result[run][pair] = {k: 100 * len(v) / len(hyp) for k, v in flags.items()}
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{pair}.json").write_text(json.dumps(flags))
        if result[run]:
            lines = ["pair\t" + "\t".join(KINDS + ["any"])]
            lines += [f"{p}\t" + "\t".join(f"{r[k]:.2f}" for k in KINDS + ["any"]) for p, r in result[run].items()]
            (out / "summary.tsv").write_text("\n".join(lines) + "\n")
    J.mkdir(parents=True, exist_ok=True)
    (J / "failures.json").write_text(json.dumps(result, indent=1))

    for k in KINDS + ["any"]:
        print(f"\n== {k} (% of segments)")
        print(f"{'run':30s}" + "".join(f"{p:>7s}" for p in PAIRS))
        for run, r in result.items():
            if run == "reference" and k != "off_target":
                continue
            print(f"{run:30s}" + "".join(f"{r[p][k]:7.2f}" if p in r else f"{'-':>7s}" for p in PAIRS))


if __name__ == "__main__":
    main()
