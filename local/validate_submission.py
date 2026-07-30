"""
Validate a submission CSV against SampleSubmission before uploading.

    python local/validate_submission.py path/to/submission.csv

You get 5 submissions a day and 200 total. A rejected file still costs you the round trip,
and a silently malformed one costs you a slot and a misleading score. Run this every time.
"""

from __future__ import annotations

import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

# Windows consoles default to cp1252 and raise UnicodeEncodeError on the characters this
# pipeline works with (ŋ, ᵑ, ’ are all in the real WAXAL charset), killing a run mid-print.
# Force UTF-8 on our own streams instead of relying on PYTHONIOENCODING being set.
# No-op on Linux/Kaggle, where stdout is already UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "zindi" / "SampleSubmission.csv"


def guess_col(df, *cands):
    low = {c.lower().strip(): c for c in df.columns}
    for c in cands:
        if c in low:
            return low[c]
    for c in cands:
        for k, v in low.items():
            if c in k:
                return v
    return None


def main(path: str) -> int:
    if not SAMPLE.exists():
        print(f"missing {SAMPLE}")
        return 1

    # escapechar: Zindi backslash-escapes quotes inside quoted fields, which the C parser
    # otherwise reads as a field terminator (23 ragged rows in Train.csv).
    sample = pd.read_csv(SAMPLE, escapechar="\\")
    sub = pd.read_csv(path, escapechar="\\")
    fails: list[str] = []
    warns: list[str] = []

    # --- structure
    if list(sub.columns) != list(sample.columns):
        fails.append(f"columns {list(sub.columns)} != sample {list(sample.columns)}")
    sid = guess_col(sample, "id", "audio_id", "utt_id", "filename")
    stxt = guess_col(sample, "transcription", "transcript", "text", "target", "prediction")

    if len(sub) != len(sample):
        fails.append(f"row count {len(sub):,} != sample {len(sample):,}")

    if sid in sub.columns:
        want = sample[sid].astype(str)
        got = sub[sid].astype(str)
        if set(want) != set(got):
            fails.append(f"id set mismatch: {len(set(want)-set(got)):,} missing, "
                         f"{len(set(got)-set(want)):,} unexpected")
        elif not want.equals(got):
            warns.append("ids present but in a different order (usually fine; reorder to be safe)")
        if got.duplicated().any():
            fails.append(f"{int(got.duplicated().sum()):,} duplicate ids")

    # --- content
    if stxt in sub.columns:
        text = sub[stxt]
        n_null = int(text.isna().sum())
        s = text.fillna("").astype(str)
        n_blank = int((s.str.strip() == "").sum())
        if n_null:
            fails.append(f"{n_null:,} NaN predictions — write '' instead, NaN can break the parser")
        if n_blank:
            pct = 100 * n_blank / len(s)
            (fails if pct > 5 else warns).append(f"{n_blank:,} blank predictions ({pct:.1f}%)")

        lens = s.str.split().str.len()
        joined = "".join(s.tolist())
        upper = sum(c.isupper() for c in joined) / max(1, sum(c.isalpha() for c in joined))
        punct = Counter(c for c in joined if unicodedata.category(c).startswith("P"))

        print("--- prediction profile ---")
        print(f"  rows            : {len(s):,}")
        print(f"  words per utt   : mean {lens.mean():.1f}, p50 {lens.median():.0f}, max {lens.max()}")
        print(f"  uppercase ratio : {upper:.4f}")
        print(f"  punctuation     : {dict(punct.most_common(10)) or 'none'}")
        print(f"  distinct chars  : {len(set(joined))}")

        if lens.max() > 200:
            warns.append(f"longest prediction is {lens.max()} words — likely a CTC repetition loop")
        if (lens == 0).sum() == 0 and lens.min() > 0:
            pass
        if s.str.contains(r"[\[\]<>]").any():
            warns.append("special tokens like [UNK]/<pad> leaked into the output — strip them")
        if s.str.contains(",").any():
            warns.append("commas present; confirm the CSV quoting survived a round-trip")

        # Compare against the reference distribution — a large gap here is a WER leak.
        train_p = ROOT / "data" / "zindi" / "Train.csv"
        if train_p.exists():
            tr = pd.read_csv(train_p, escapechar="\\")
            ttxt = guess_col(tr, "transcription", "transcript", "text", "target", "sentence")
            if ttxt:
                ref = tr[ttxt].dropna().astype(str)
                ref_join = "".join(ref.tolist())
                ref_upper = sum(c.isupper() for c in ref_join) / max(1, sum(c.isalpha() for c in ref_join))
                ref_len = ref.str.split().str.len().mean()
                print(f"\n--- vs reference (Train.csv) ---")
                print(f"  words per utt   : pred {lens.mean():.1f}  vs  ref {ref_len:.1f}")
                print(f"  uppercase ratio : pred {upper:.4f}  vs  ref {ref_upper:.4f}")
                if abs(upper - ref_upper) > 0.02:
                    warns.append(f"casing mismatch vs reference ({upper:.3f} vs {ref_upper:.3f}) — direct WER cost")
                if ref_len and abs(lens.mean() - ref_len) / ref_len > 0.35:
                    warns.append(f"length mismatch vs reference ({lens.mean():.1f} vs {ref_len:.1f}) — "
                                 "insertions or deletions dominating")
                unseen = set(c for c in set(joined) if c not in set(ref_join))
                if unseen:
                    warns.append(f"chars in output never seen in reference: {sorted(unseen)[:15]}")

    print()
    for w in warns:
        print(f"WARN  {w}")
    for f in fails:
        print(f"FAIL  {f}")
    if not fails:
        print("\nOK — safe to upload." + ("  (review warnings first)" if warns else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
