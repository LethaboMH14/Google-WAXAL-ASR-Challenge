"""
Is punctuation restoration actually worth building? Answer it on CPU, today, before spending GPU.

THE CASE FOR ASKING
-------------------
Every strong open checkpoint for these languages is a character-CTC model whose vocabulary
contains no sentence punctuation — waxal-benchmarking's carry only '&' and '/'. The competition
metric lowercases and then compares verbatim, so every '.' and ',' in a reference that our
hypothesis does not reproduce is a word error AND several character errors. Measured on the dev
set, a PERFECT transcriber that emits no punctuation scores 0.9367 rather than 1.0.

That is a ~0.063 ceiling nobody at the top of the leaderboard appears to have taken: the top
cluster sits at 0.7206-0.7257, which is almost exactly where those checkpoints land once you
dock them the no-punctuation penalty.

WHAT THIS SCRIPT MEASURES
-------------------------
A deliberately cheap restorer — logistic regression over word-identity and context features, no
GPU, no pretrained encoder — trained ONLY on Train.csv rows marked original_split == "train", and
evaluated on the frozen dev set it has never seen. Two evaluations, and the second is the honest
one:

  1. on perfect (reference) words     — the ceiling this class of model reaches
  2. on words corrupted to a realistic WER — what it is actually worth on top of a real ASR

If (2) is small, punctuation is a dead end and we drop it and spend the GPU on the acoustic model
instead. If (2) is large, this script is also the floor: a transformer token-classifier should
beat logistic regression, so the real number is at least this.
"""

from __future__ import annotations

import io
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import score as S  # noqa: E402

PUNCT = "!\"&'(),-./:;?«»“”"
# The three marks worth modelling. Measured density in the dev references, per 1,000 reference
# words: '.' 65.2, ',' 27.6, "'" 16.3, everything else under 1.4 combined. The apostrophe is
# excluded here because it sits INSIDE words (a spelling question, handled by a lexicon) whereas
# '.' and ',' attach after a word, which is what a token classifier can decide.
TRAILING = [".", ","]
SEED = 1337


def strip_punct(text: str) -> str:
    return re.sub(r"\s+", " ", "".join(c for c in text if c not in PUNCT)).strip()


def to_labels(text: str) -> tuple[list[str], list[int]]:
    """A punctuated sentence -> (bare words, label per word). Label 0=none, 1='.', 2=','."""
    words, labels = [], []
    for tok in text.split():
        bare = "".join(c for c in tok if c not in PUNCT)
        if not bare:
            continue
        lab = 0
        for i, mark in enumerate(TRAILING, start=1):
            if tok.rstrip().endswith(mark):
                lab = i
                break
        words.append(bare)
        labels.append(lab)
    return words, labels


def featurise(words: list[str], i: int, lang: str) -> dict[str, int]:
    """Context window around word i. Lowercased because the metric lowercases anyway."""
    n = len(words)
    w = words[i].lower()
    prev = words[i - 1].lower() if i else "<s>"
    nxt = words[i + 1].lower() if i + 1 < n else "</s>"
    nxt2 = words[i + 2].lower() if i + 2 < n else "</s>"
    return {
        f"lang={lang}": 1,
        f"w={w}": 1,
        f"prev={prev}": 1,
        f"next={nxt}": 1,
        f"next2={nxt2}": 1,
        f"w|next={w}|{nxt}": 1,
        f"prev|w={prev}|{w}": 1,
        f"suf3={w[-3:]}": 1,
        f"nsuf3={nxt[-3:]}": 1,
        # Position matters enormously: '.' is overwhelmingly sentence-final, so "how close am I
        # to the end" carries most of the signal for the majority class.
        f"last={i == n - 1}": 1,
        f"from_end={min(n - 1 - i, 5)}": 1,
        f"pos={min(i, 5)}": 1,
        f"len={min(n // 5, 8)}": 1,
    }


def main() -> None:
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.linear_model import SGDClassifier

    df = pd.read_csv(REPO / "data" / "zindi" / "Train.csv",
                     engine="python", on_bad_lines="skip")
    dev = json.load(open(HERE / "devset.json", encoding="utf-8"))
    dev_ids = {it["id"] for it in dev["items"]}

    # Train ONLY on the train split, and belt-and-braces exclude any dev id. The dev set is drawn
    # from the validation split so this cannot overlap, but a restorer that has memorised the
    # sentences it is scored on would report a number that means nothing.
    tr = df[(df["original_split"] == "train") & (~df["id"].isin(dev_ids))]
    tr = tr[tr["transcription"].astype(str).str.strip().str.len() > 1]
    print(f"training rows: {len(tr):,}   dev rows: {len(dev['items']):,}")

    X, y = [], []
    for row in tr.itertuples():
        words, labels = to_labels(str(row.transcription))
        for i in range(len(words)):
            X.append(featurise(words, i, row.language))
            y.append(labels[i])
    print(f"training tokens: {len(y):,}   label mix: "
          f"{ {k: int(v) for k, v in pd.Series(y).value_counts().items()} }")

    hasher = FeatureHasher(n_features=2 ** 20, input_type="dict", alternate_sign=False)
    Xh = hasher.transform(X)
    # class_weight balanced: '.' and ',' are ~9% of tokens between them, and an unweighted model
    # maximises accuracy by predicting "no punctuation" everywhere — which is exactly the baseline
    # we are trying to beat.
    clf = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=12, tol=None,
                        random_state=SEED, class_weight="balanced")
    clf.fit(Xh, np.array(y))

    def restore(words: list[str], lang: str) -> str:
        if not words:
            return ""
        feats = [featurise(words, i, lang) for i in range(len(words))]
        pred = clf.predict(hasher.transform(feats))
        return " ".join(w + (TRAILING[p - 1] if p else "") for w, p in zip(words, pred))

    refs = [it["reference"] for it in dev["items"]]
    langs = [it["language"] for it in dev["items"]]

    print("\n--- 1. on PERFECT words (the ceiling for this model class) ---")
    bare = [strip_punct(r) for r in refs]
    hyp_r = [restore(b.split(), lg) for b, lg in zip(bare, langs)]
    for name, h in (("no punctuation", bare),
                    ("trailing '.' always", [b + "." for b in bare]),
                    ("restored", hyp_r),
                    ("ORACLE (= reference)", refs)):
        print(f"   {name:24s} {S.score(refs, h)}")

    # Token-level quality, so we know WHERE it fails rather than only that it does.
    tp = dict.fromkeys(range(1, len(TRAILING) + 1), 0)
    fp, fn = dict(tp), dict(tp)
    for r, lg in zip(refs, langs):
        w, gold = to_labels(r)
        if not w:
            continue
        pr = clf.predict(hasher.transform([featurise(w, i, lg) for i in range(len(w))]))
        for g, p in zip(gold, pr):
            if g == p and g:
                tp[g] += 1
            else:
                if p:
                    fp[p] += 1
                if g:
                    fn[g] += 1
    print("\n   token-level:")
    for k, mark in enumerate(TRAILING, start=1):
        prec = tp[k] / max(1, tp[k] + fp[k])
        rec = tp[k] / max(1, tp[k] + fn[k])
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        print(f"     {mark!r}: P={prec:.3f} R={rec:.3f} F1={f1:.3f}  (support {tp[k] + fn[k]:,})")

    print("\n--- 2. on CORRUPTED words (what it is worth on a real ASR) ---")
    vocab = sorted({w for b in bare for w in b.split()})
    rng = random.Random(SEED)

    def corrupt(t: str, rate: float) -> str:
        out = []
        for w in t.split():
            p = rng.random()
            if p < rate * 0.7:
                out.append(rng.choice(vocab))
            elif p < rate * 0.9:
                continue
            elif p < rate:
                out += [w, rng.choice(vocab)]
            else:
                out.append(w)
        return " ".join(out) or rng.choice(vocab)

    print(f"   {'sim WER':>8}  {'none':>8}  {'always .':>9}  {'restored':>9}  "
          f"{'oracle':>8}   {'restored-none':>14}")
    for rate in (0.15, 0.25, 0.32, 0.42):
        cor = [corrupt(b, rate) for b in bare]
        a = S.score(refs, cor).multi
        b_ = S.score(refs, [c + "." for c in cor]).multi
        c = S.score(refs, [restore(x.split(), lg) for x, lg in zip(cor, langs)]).multi
        # oracle: re-attach the reference's marks to whichever words survived corruption
        orc = []
        for r, x in zip(refs, cor):
            pm = {strip_punct(w): w for w in r.split() if any(ch in PUNCT for ch in w)}
            orc.append(" ".join(pm.get(w, w) for w in x.split()))
        d = S.score(refs, orc).multi
        print(f"   {rate:8.2f}  {a:8.4f}  {b_:9.4f}  {c:9.4f}  {d:8.4f}   {c - a:+14.4f}")


if __name__ == "__main__":
    main()
