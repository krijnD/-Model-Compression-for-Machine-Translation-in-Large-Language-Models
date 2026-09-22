"""BLEU, chrF++ and hallucination rate for one run (CPU, a few seconds; fine on the login node).

BLEU uses the ALMA-R paper's settings (third_party/ALMA/evals/eval_generation.sh):
sacrebleu corpus BLEU, tokenizer "zh" for Chinese targets, otherwise "13a".
Hallucination: candidate at least 2x as long as the reference, in characters.

Usage (from the repo root, venv active): python scripts/score_lexical.py --run ours
"""
import argparse
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO.parent / "outputs"
TESTSET = REPO / "third_party/ALMA/outputs/wmt22_outputs/wmt-testset"
PAIRS = "de-en,cs-en,is-en,zh-en,ru-en,en-de,en-cs,en-is,en-zh,en-ru".split(",")


def hyp_path(run, src, tgt):  # same mapping as hyp_path in scripts/snellius/env.sh
    return {
        "paper": REPO / f"third_party/ALMA/outputs/wmt22_outputs/ALMA-13B-R/{src}{tgt}/test.{src}-{tgt}.{tgt}",
        "ours": OUTPUTS / f"alma-13b-r/wmt22/test-{src}-{tgt}",
        "ours-beam": OUTPUTS / f"alma-13b-r-beam/wmt22/test-{src}-{tgt}",
    }[run]


def read(path):
    return path.read_text(encoding="utf-8").splitlines()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=["paper", "ours", "ours-beam"])
    run = parser.parse_args().run

    scores = {"bleu": {}, "chrf": {}, "halluc": {}}
    for pair in PAIRS:
        src, tgt = pair.split("-")
        hyps = read(hyp_path(run, src, tgt))
        refs = read(TESTSET / f"{src}{tgt}/test.{src}-{tgt}.{tgt}")
        assert len(hyps) == len(refs), f"{pair}: {len(hyps)} translations vs {len(refs)} references"

        scores["bleu"][pair] = BLEU(tokenize="zh" if tgt == "zh" else "13a").corpus_score(hyps, [refs]).score
        scores["chrf"][pair] = CHRF(word_order=2).corpus_score(hyps, [refs]).score
        n_halluc = sum(len(h) >= 2 * len(r) for h, r in zip(hyps, refs))
        scores["halluc"][pair] = 100 * n_halluc / len(hyps)
        n_empty = sum(not h.strip() for h in hyps)
        print(f"{pair}  BLEU {scores['bleu'][pair]:6.2f}  chrF++ {scores['chrf'][pair]:6.2f}  "
              f"halluc {scores['halluc'][pair]:5.2f}% ({n_halluc})  empty {n_empty}")

    for metric, per_pair in scores.items():
        out = OUTPUTS / metric / run
        out.mkdir(parents=True, exist_ok=True)
        lines = [f"pair\t{metric}"] + [f"{p}\t{s:.2f}" for p, s in per_pair.items()]
        (out / "summary.tsv").write_text("\n".join(lines) + "\n")
    print(f"Written to {OUTPUTS}/{{bleu,chrf,halluc}}/{run}/summary.tsv")


if __name__ == "__main__":
    main()
