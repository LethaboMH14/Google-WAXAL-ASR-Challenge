"""
Inspect the Zindi WAXAL files and report exactly what the submission needs.

Run this FIRST, locally, after dropping the five Zindi downloads into data/zindi/:
    Train.csv  Test.csv  Test_phase2.csv  SampleSubmission.csv  Waxal_Challenge_Starter_Code.ipynb

    python local/inspect_data.py

It answers the questions the rest of the pipeline is blocked on:
  - what columns does the submission need, and which IDs does it cover (phase 1, phase 2, or both)?
  - how do we join Test IDs back to WAXAL audio?
  - what does the reference text actually look like (case, punctuation, digits, diacritics)?

That last one drives the whole normalisation policy, and guessing it costs real WER.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZINDI = ROOT / "data" / "zindi"
OUT = ROOT / "data" / "derived"


def load(name: str) -> pd.DataFrame | None:
    """
    Zindi's CSVs are not standard CSV: quotes inside quoted fields are backslash-escaped
    (`\\"`), which the C parser reads as a field terminator. 23 of the 38,199 Train.csv rows
    trip on it and pandas dies on the first one. `escapechar="\\"` is the fix, and every
    script that touches these files needs it or it silently loses those rows.
    """
    path = ZINDI / name
    if not path.exists():
        print(f"  !! missing: {path}")
        return None
    try:
        df = pd.read_csv(path, escapechar="\\")
    except Exception as e:                                      # noqa: BLE001
        print(f"  !! {name}: escapechar parse failed ({type(e).__name__}), "
              f"falling back to skipping bad lines")
        df = pd.read_csv(path, escapechar="\\", on_bad_lines="skip")
    print(f"  {name}: {len(df):,} rows x {len(df.columns)} cols -> {list(df.columns)}")
    return df


def guess_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Find a column by fuzzy name match, so we do not hard-code Zindi's naming."""
    lowered = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    for cand in candidates:
        for low, orig in lowered.items():
            if cand in low:
                return orig
    return None


def describe_text(series: pd.Series, label: str) -> dict:
    """Characterise the reference transcripts. This is what we must match on output."""
    text = series.dropna().astype(str)
    joined = "".join(text.tolist())
    chars = Counter(joined)

    has_upper = any(c.isupper() for c in joined)
    upper_ratio = sum(c.isupper() for c in joined) / max(1, sum(c.isalpha() for c in joined))
    punct = {c: n for c, n in chars.items() if unicodedata.category(c).startswith("P")}
    digits = {c: n for c, n in chars.items() if c.isdigit()}
    # non-ASCII letters = diacritics / special orthography we must keep in the CTC vocab
    non_ascii = {c: n for c, n in chars.items() if c.isalpha() and ord(c) > 127}

    words = re.findall(r"\S+", " ".join(text.tolist()))
    lengths = text.str.split().str.len()

    print(f"\n--- reference text profile: {label} ---")
    print(f"  utterances           : {len(text):,}")
    print(f"  words (total/unique) : {len(words):,} / {len(set(words)):,}")
    print(f"  words per utt        : mean {lengths.mean():.1f}, p50 {lengths.median():.0f}, p95 {lengths.quantile(0.95):.0f}, max {lengths.max()}")
    print(f"  uppercase present    : {has_upper}  (ratio of alpha chars: {upper_ratio:.3f})")
    print(f"  punctuation          : {dict(sorted(punct.items(), key=lambda kv: -kv[1])[:15]) or 'NONE'}")
    print(f"  digits               : {digits or 'NONE'}")
    print(f"  non-ASCII letters    : {dict(sorted(non_ascii.items(), key=lambda kv: -kv[1])[:20]) or 'NONE'}")
    print(f"  distinct chars       : {len(chars)}")
    print("  samples:")
    for s in text.sample(min(5, len(text)), random_state=0):
        print(f"    | {s[:110]}")

    return {
        "n_utterances": int(len(text)),
        "n_words": int(len(words)),
        "vocab_words": int(len(set(words))),
        "has_uppercase": bool(has_upper),
        "uppercase_ratio": round(float(upper_ratio), 4),
        "punctuation": {k: int(v) for k, v in punct.items()},
        "digits": {k: int(v) for k, v in digits.items()},
        "non_ascii_letters": {k: int(v) for k, v in non_ascii.items()},
        "distinct_chars": sorted(chars.keys()),
    }


def main() -> int:
    if not ZINDI.exists():
        print(f"Create {ZINDI} and put the Zindi downloads there first.")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    print("=== files ===")
    train = load("Train.csv")
    test1 = load("Test.csv")
    test2 = load("Test_phase2.csv")
    sample = load("SampleSubmission.csv")

    if train is None or sample is None:
        print("\nNeed at least Train.csv and SampleSubmission.csv to continue.")
        return 1

    report: dict = {}

    # ---------- submission contract ----------
    print("\n=== submission contract ===")
    sub_id = guess_col(sample, "id", "audio_id", "utt_id", "filename")
    sub_txt = guess_col(sample, "transcription", "transcript", "text", "target", "prediction")
    print(f"  id column        : {sub_id}")
    print(f"  target column    : {sub_txt}")
    print(f"  rows             : {len(sample):,}")
    print(f"  first 3 ids      : {sample[sub_id].head(3).tolist() if sub_id else 'n/a'}")

    ids_sub = set(sample[sub_id].astype(str)) if sub_id else set()
    coverage = {}
    for label, df in (("Test.csv (phase 1)", test1), ("Test_phase2.csv (phase 2)", test2)):
        if df is None:
            continue
        c = guess_col(df, "id", "audio_id", "utt_id", "filename")
        ids = set(df[c].astype(str)) if c else set()
        inter = len(ids & ids_sub)
        coverage[label] = {"rows": len(df), "id_col": c, "overlap_with_submission": inter}
        pct = 100 * inter / max(1, len(ids))
        print(f"  {label}: {len(df):,} rows, {inter:,} of its ids in SampleSubmission ({pct:.1f}%)")

    if coverage:
        covered = [k for k, v in coverage.items() if v["overlap_with_submission"] > 0]
        print(f"\n  >> SampleSubmission covers: {', '.join(covered) if covered else 'NEITHER — inspect manually'}")
        if len(covered) == 2:
            print("  >> Submission spans BOTH phases: predict every id in SampleSubmission.")
        elif covered and "phase 2" in covered[0]:
            print("  >> Phase 2 only: the leaderboard is now scoring the unseen set. This is the one that counts.")

    # ---------- train schema ----------
    print("\n=== train schema ===")
    tr_id = guess_col(train, "id", "audio_id", "utt_id")
    tr_txt = guess_col(train, "transcription", "transcript", "text", "target", "sentence")
    tr_lang = guess_col(train, "language", "lang", "locale")
    tr_audio = guess_col(train, "audio", "path", "url", "file")
    print(f"  id={tr_id}  text={tr_txt}  language={tr_lang}  audio={tr_audio}")
    if tr_lang:
        print(f"  language distribution:\n{train[tr_lang].value_counts().to_string()}")
    if test1 is not None and guess_col(test1, "language", "lang"):
        print(f"  NOTE: Test.csv HAS a language column — Test_phase2.csv is expected not to.")
    if test2 is not None:
        print(f"  Test_phase2.csv columns: {list(test2.columns)}")
        if not guess_col(test2, "language", "lang"):
            print("  >> CONFIRMED: no language column in phase 2. Language ID from audio is mandatory.")

    # ---------- the important bit ----------
    if tr_txt:
        report["train_text_overall"] = describe_text(train[tr_txt], "ALL LANGUAGES")
        if tr_lang:
            for lang, grp in train.groupby(tr_lang):
                report[f"train_text_{lang}"] = describe_text(grp[tr_txt], str(lang))

    report["submission"] = {"id_col": sub_id, "text_col": sub_txt, "rows": len(sample)}
    report["coverage"] = coverage
    report["train_schema"] = {"id": tr_id, "text": tr_txt, "language": tr_lang, "audio": tr_audio}

    (OUT / "data_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # The CTC vocab must be derived from the real training text, never assumed.
    if tr_txt:
        chars = sorted(set("".join(train[tr_txt].dropna().astype(str))))
        (OUT / "charset_raw.json").write_text(json.dumps(chars, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  wrote {OUT/'charset_raw.json'} ({len(chars)} raw chars) and {OUT/'data_report.json'}")

    print("\n=== normalisation decision ===")
    if tr_txt:
        prof = report["train_text_overall"]
        if prof["uppercase_ratio"] < 0.005:
            print("  Reference is effectively lowercase -> lowercase the output. Safe.")
        else:
            print(f"  Reference KEEPS case (ratio {prof['uppercase_ratio']:.3f}) -> do NOT lowercase blindly;")
            print("  CTC lowercase output would cost WER on every capitalised word. Consider truecasing,")
            print("  or keeping case in the CTC vocab if the ratio is high enough to learn.")
        if prof["punctuation"]:
            print(f"  Reference CONTAINS punctuation {list(prof['punctuation'])[:8]} -> stripping it costs CER.")
        else:
            print("  Reference has no punctuation -> strip punctuation from output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
