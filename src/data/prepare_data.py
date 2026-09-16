#!/usr/bin/env python
"""Prepare eval data: FLORES-200 devtest, NTREX-128, WMT'23 overlap pairs.

Scope per AGENTS.md §2: 18 FLORES directions (9 langs x en<->), 18 NTREX
en-centric directions, WMT'23 test pairs overlapping X-ALMA's eval set.

Eval languages and X-ALMA group mapping:
  de, is (G1) · ru (G3) · fr, mg (G4) · cs (G5) · zh, ja (G6) · ar (G8)

Data sources (verified 2026-09-16, all public):
  FLORES-200: haoranxu/FLORES-200  (config "devtest"; NLLB-coded columns)
  NTREX-128 : mteb/NTREX           (1997 rows; NLLB-coded columns)
           (fallback: Zihao-Li/NTREX-128 text/lang flattened)
  WMT'23    : haoranxu/WMT23-Test  (per-direction configs, e.g. "en-de")

Writes data/<benchmark>/<direction>/src.txt / ref.txt (one sentence/line).

Usage:
  python src/data/prepare_data.py [--bench flores,ntrex,wmt23] [--out data]
"""
from __future__ import annotations

import argparse
import os
import sys

from datasets import load_dataset

LANGS = ["de", "is", "fr", "mg", "ru", "cs", "zh", "ja", "ar"]
# FLORES/NLLB language codes used by both FLORES-200 (haoranxu) and NTREX (mteb)
NLLB = {
    "de": "deu_Latn", "is": "isl_Latn", "fr": "fra_Latn", "mg": "plt_Latn",
    "ru": "rus_Cyrl", "cs": "ces_Latn", "zh": "zho_Hans", "ja": "jpn_Jpan",
    "ar": "arb_Arab", "en": "eng_Latn",
}
# mteb/NTREX codes differ for Malagasy (and a few others)
NTREX_CODES = dict(NLLB)
NTREX_CODES["mg"] = "mlg_Latn"


def write_direction(out_root: str, benchmark: str, direction: str,
                    src_lines: list[str], ref_lines: list[str]) -> None:
    d = os.path.join(out_root, benchmark, direction)
    os.makedirs(d, exist_ok=True)
    assert len(src_lines) == len(ref_lines), (
        benchmark, direction, len(src_lines), len(ref_lines))
    with open(os.path.join(d, "src.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(src_lines) + "\n")
    with open(os.path.join(d, "ref.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ref_lines) + "\n")
    return len(src_lines)


def prepare_flores(out_root: str) -> None:
    print("[flores] loading haoranxu/FLORES-200 (per-direction configs, split=test)")
    total = 0
    for lang in LANGS:
        for d in (f"{lang}-en", f"en-{lang}"):
            try:
                ds = load_dataset("haoranxu/FLORES-200", d, split="test")
            except Exception as e:
                print(f"  [flores] WARN: {d} unavailable ({str(e)[:60]})")
                continue
            src_l, tgt_l = d.split("-")
            src_lines = [r[src_l] for r in ds[d]]
            ref_lines = [r[tgt_l] for r in ds[d]]
            total += write_direction(out_root, "flores", d, src_lines, ref_lines)
    print(f"[flores] done -> {out_root}/flores ({total} rows)")


def prepare_ntrex(out_root: str) -> None:
    print("[ntrex] loading mteb/NTREX split=test")
    ds = load_dataset("mteb/NTREX", split="test")
    total = 0
    for lang in LANGS:
        code = NTREX_CODES[lang]
        if code not in ds.column_names:
            print(f"  [ntrex] WARN: {lang} ({code}) not found; skipping")
            continue
        total += write_direction(out_root, "ntrex", f"en-{lang}",
                                 list(ds["eng_Latn"]), list(ds[code]))
        total += write_direction(out_root, "ntrex", f"{lang}-en",
                                 list(ds[code]), list(ds["eng_Latn"]))
    print(f"[ntrex] done -> {out_root}/ntrex ({total} rows, 18 directions)")


def prepare_wmt23(out_root: str) -> None:
    """WMT'23: haoranxu/WMT23-Test per-direction configs; overlap pairs only."""
    pairs = []
    for lang in LANGS:
        if lang == "en":
            continue
        for d in (f"en-{lang}", f"{lang}-en"):
            try:
                ds = load_dataset("haoranxu/WMT23-Test", d, split="test")
            except Exception:
                continue  # direction not present in WMT23
            col = d
            rows = ds[col]
            src_lines = [r["en"] if d.startswith("en-") else r[lang] for r in rows]
            ref_lines = [r[lang] if d.startswith("en-") else r["en"] for r in rows]
            n = write_direction(out_root, "wmt23", d, src_lines, ref_lines)
            pairs.append((d, n))
    print(f"[wmt23] done -> {out_root}/wmt23 ({pairs})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="flores,ntrex,wmt23")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for b in args.bench.split(","):
        b = b.strip()
        if b == "flores":
            prepare_flores(args.out)
        elif b == "ntrex":
            prepare_ntrex(args.out)
        elif b == "wmt23":
            prepare_wmt23(args.out)
        else:
            print(f"unknown bench {b}", file=sys.stderr)
    print("DONE")


if __name__ == "__main__":
    main()