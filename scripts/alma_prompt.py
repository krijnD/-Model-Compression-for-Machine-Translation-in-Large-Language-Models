"""ALMA prompt format, calibration examples and output cleanup, shared by the quantization scripts.

Copied from third_party/ALMA/utils/utils.py (get_prompt, get_key_suffix, clean_outputstring and the
training format of tokenize_function_train_eval_left_pad). We can't import that module here: it pulls
in peft/datasets/deepspeed from the generation venv, and the quantization venv has a newer transformers.
Keep this file in sync with ALMA if the submodule is updated.
"""
import json
import random
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
ALMA_DATA = REPO_DIR / "third_party/ALMA/human_written_data"
WMT22_PAIRS = ["de-en", "cs-en", "is-en", "zh-en", "ru-en", "en-de", "en-cs", "en-is", "en-zh", "en-ru"]

LANG_TABLE = {"en": "English", "de": "German", "cs": "Czech", "is": "Icelandic", "zh": "Chinese", "ru": "Russian"}


def get_prompt(src, tgt, source_text):
    src_name, tgt_name = LANG_TABLE[src], LANG_TABLE[tgt]
    return f"Translate this from {src_name} to {tgt_name}:\n{src_name}: " + source_text + f"\n{tgt_name}:"


def get_key_suffix(tgt):
    return f"\n{LANG_TABLE[tgt]}:"


def clean_outputstring(output, key_word, split_idx=1):
    """Pull the translation out of the decoded prompt + generation (same logic as ALMA)."""
    try:
        out = output.split(key_word)[split_idx].split("\n")
        if out[0].strip() != "":
            return out[0].strip()
        elif out[1].strip() != "":
            return out[1].strip()
        else:
            return out[2].strip()
    except IndexError:
        pass
    try:
        return output.split(key_word)[2].split("\n")[0].strip()
    except IndexError:
        return ""


def pair_dir(src, tgt):
    """ALMA stores both directions in <xx>en/, e.g. deen/ for de-en and en-de."""
    first = src if src != "en" else tgt
    return ALMA_DATA / f"{first}en", first


def load_test_sources(src, tgt):
    """Source sentences of the WMT'22 (WMT'21 for is) test set, in the file order run_llmmt.py uses."""
    d, _ = pair_dir(src, tgt)
    with open(d / f"test.{src}-{tgt}.json", encoding="utf-8") as f:
        return [ex["translation"][src] for ex in json.load(f)]


def load_train_pairs(src, tgt):
    """(source, target) sentences from ALMA's human-written training data (earlier WMT test sets + Flores)."""
    d, first = pair_dir(src, tgt)
    with open(d / f"train.{first}-en.json", encoding="utf-8") as f:
        exs = [json.loads(line)["translation"] for line in f if line.strip()]
    return [(ex[src], ex[tgt]) for ex in exs]


def calibration_examples(tokenizer, n_samples, pairs=WMT22_PAIRS, seed=42, max_length=512):
    """n_samples tokenized examples in ALMA's fine-tuning format (BOS + prompt + target + EOS),
    balanced over the directions and drawn from the train files only, so there's no overlap with the test sets."""
    rng = random.Random(seed)
    per_pair = [n_samples // len(pairs) + (i < n_samples % len(pairs)) for i in range(len(pairs))]
    examples = []
    for pair, n in zip(pairs, per_pair):
        src, tgt = pair.split("-")
        for s, t in rng.sample(load_train_pairs(src, tgt), n):
            ids = tokenizer(get_prompt(src, tgt, s) + t, max_length=max_length - 1, truncation=True).input_ids
            ids.append(tokenizer.eos_token_id)
            examples.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    rng.shuffle(examples)
    return examples
