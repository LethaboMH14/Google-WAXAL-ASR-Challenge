"""
STAGE 0 — build the language-model text corpora. CPU ONLY. Run this FIRST, before anything else.

Why this script exists, and why it is arguably the highest-value file in the repo:

  [Benchmarking ASR Models for African Languages, arXiv:2512.10968] measured w2v-bert-2.0 CTC
  with and without a 5-gram KenLM on exactly our three languages. The LM is not a marginal
  gain, it is the result:

    Luganda  w2v-bert 39.75 WER -> 16.30 with LM   (-59% relative)
    Shona    w2v-bert 22.56 WER ->  9.28 with LM   (-59% relative)
    Lingala  w2v-bert 24.19 WER -> 22.74 with LM   ( -6% relative)

  Their LMs were trained on 9.0M (lin), 9.2M (lug), 5.4M (sna) words of external text.
  The Zindi Train.csv holds a few thousand transcripts — call it 50k words. Two orders of
  magnitude short. A 5-gram estimated on 50k words is a lookup table for the training set; it
  will not generalise to unseen test utterances and it is exactly the case where LM fusion
  *hurts* rather than helps (see XLS-R+LM regressing on Lingala and Shona in that same paper).

  So: pull real monolingual text for lin / lug / sna from open corpora, normalise it into the
  acoustic model's exact character space, and hand stage 3 a corpus big enough for a 5-gram to
  mean something.

Rules position: the challenge explicitly permits this — "Participants may supplement the
provided challenge data with other publicly available open-source speech or language datasets.
Any external datasets used must be publicly accessible, legally licensed for research or
development, and disclosed in the final solution documentation." Every source below is public
and openly licensed, and this script writes `lm_sources.json` so the disclosure section writes
itself. Do not add a source here that you cannot point at a licence for.

Where to run: any CPU box. Kaggle CPU session (no GPU quota burned), or the Azure
Standard_D4s_v3. Takes ~20-40 min, mostly download. Output is a few hundred MB of text.

Output lands in persistent `waxal-work/lm/`, which stage 3 reads directly. (Kaggle-era note
removed: we no longer publish this as a mounted Kaggle Dataset.)
"""

import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- config
# Windows consoles default to cp1252 and raise UnicodeEncodeError on the characters this
# pipeline works with (ŋ, ᵑ, ’ are all in the real WAXAL charset), killing a run mid-print.
# Force UTF-8 on our own streams instead of relying on PYTHONIOENCODING being set.
# No-op on Linux/Kaggle, where stdout is already UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


# ---------------------------------------------------------------- 0. where am I running
# Kaggle and Lightning have opposite storage models, and that is the whole reason for this block.
#   Kaggle     read-only Datasets at /kaggle/input + an ephemeral /kaggle/working. Every stage's
#              output must be re-uploaded as a Dataset before the next stage can read it.
#   Lightning  one persistent home. Stage N writes exactly where stage N+1 looks, so the three
#              manual dataset uploads — and the chance of attaching a stale checkpoint — vanish.
# ART(name) hides the difference: it returns wherever the named artefact actually lives.
if Path("/kaggle/working").exists():
    ENV, WORK = "kaggle", Path("/kaggle/working")
    # The Zindi CSVs are 8 MB and committed to the repo, so a kernel that clones the repo already
    # has them — maintaining a `waxal-zindi` Dataset alongside is a second copy to keep in sync
    # and a way to silently work off a stale Train.csv. Prefer an attached Dataset if one is
    # there (that is how the earlier runs were wired), else fall back to the clone.
    _ds = Path("/kaggle/input/waxal-zindi")
    try:
        _repo_zindi = Path(__file__).resolve().parents[1] / "data" / "zindi"
    except NameError:                                     # pasted into a notebook cell
        _repo_zindi = Path.cwd() / "data" / "zindi"
    ZINDI_DIR = Path(os.environ["WAXAL_ZINDI_DIR"]) if os.environ.get("WAXAL_ZINDI_DIR") else (
        _ds if _ds.exists() else _repo_zindi)
    def ART(name: str) -> Path:
        return Path("/kaggle/input") / name               # <- the Dataset you attached
else:
    ENV = "lightning" if Path("/teamspace/studios/this_studio").exists() else "local"
    try:
        REPO = Path(__file__).resolve().parents[1]
    except NameError:                                     # pasted into a notebook cell
        REPO = Path.cwd()
    HOME = Path("/teamspace/studios/this_studio") if ENV == "lightning" else REPO
    WORK = HOME / "waxal-work"                            # persistent across sessions
    WORK.mkdir(parents=True, exist_ok=True)
    ZINDI_DIR = REPO / "data" / "zindi"                   # the csvs are committed to the repo
    def ART(name: str) -> Path:
        return WORK                                       # everything in one persistent tree
print(f"env={ENV}  work={WORK}  zindi={ZINDI_DIR}")

OUT = WORK / "lm_corpus"                                  # stage 3 reads this back via ART()
OUT.mkdir(parents=True, exist_ok=True)

LANGS = ["lin", "sna", "lug"]

# Must match stage 2 / stage 3 exactly. If you change these, the LM stops matching the
# acoustic model's label space and beam search degrades silently.
# Policy is measured, not guessed: the organisers' starter notebook scores with
# jiwer.wer([r.lower()...], [p.lower()...]) — lowercase both sides, punctuation untouched.
LOWERCASE = True
KEEP_PUNCT = set(".,'’-;:!?")

# The in-domain WAXAL transcripts are worth far more per word than Wikipedia prose — they
# carry the register, the disfluencies and the topic distribution of the actual test audio.
# Repetition is a crude but effective stand-in for proper LM interpolation, and it is one line.
INDOMAIN_REPEAT = 10

# Sources, per language. `hf` entries are (repo, config, split, text_column).
# Each is tried independently and failures are non-fatal — HF configs get renamed, and a dead
# source must not take the whole corpus down with it.
# FLEURS is an *audio* dataset. We want one text column out of it, but `streaming=True` still
# pulls the multi-GB wav shards, because that is how the archives are packed — measured on
# Lightning as ~10 minutes of 1% CPU and no output per language. It contributes a few thousand
# read-speech sentences against ~35k we already have, so it is off by default and the corpus is
# built without it. Turn it back on when wall-clock is cheap:
#     WAXAL_INCLUDE_FLEURS=1 python kaggle/00_build_lm_corpus.py
INCLUDE_AUDIO_DATASETS = os.getenv("WAXAL_INCLUDE_FLEURS", "0") == "1"

# Wikipedia + MasakhaNEWS alone left Luganda at 28.5% of the published corpus size and Lingala at
# 51.9% — Luganda being exactly the language where the published LM gain is largest (-59% rel).
# HPLT 2.0 and GlotCC are cleaned, language-identified web crawls that both carry all three of our
# languages and are both **CC0-1.0**, i.e. public domain, which is the cleanest thing we can put in
# the disclosure. HPLT lug_Latn alone is 21,276 documents / 40 MB of Luganda.
#
# Rejected after checking, so nobody re-proposes them:
#   castorini/afriberta-corpus  12 languages, none of ours. No Luganda at all.
#   masakhane/mafand            CC-BY-NC-4.0. Non-commercial, on a $10k-prize entry, for a few
#                               thousand news sentences. Not worth the licence question.
#   statmt/cc100                has lg + ln, but the HF card licence is literally "unknown", and
#                               "must be legally licensed" is a stated rule. CC0 sources cover it.
#   google/fleurs               audio dataset — see audio_heavy below.
#
# `max_words` bounds a source so a web crawl cannot run away with the wall clock. Streaming order
# is deterministic, so a cap stays reproducible — which the rules require ("rerunning your model
# should always place you at the same position"). 12M is above every published corpus size below.
SOURCES: dict[str, list[dict]] = {
    "lin": [
        {"repo": "wikimedia/wikipedia", "config": "20231101.ln", "split": "train", "col": "text",
         "licence": "CC-BY-SA-4.0"},
        {"repo": "masakhane/masakhanews", "config": "lin", "split": "train", "col": "text",
         "licence": "AFL-3.0"},
        {"repo": "HPLT/HPLT2.0_cleaned", "config": "lin_Latn", "split": "train", "col": "text",
         "licence": "CC0-1.0", "max_words": 12_000_000},
        {"repo": "cis-lmu/GlotCC-V1", "config": "lin-Latn", "split": "train", "col": "content",
         "licence": "CC0-1.0", "max_words": 12_000_000},
        {"repo": "google/fleurs", "config": "ln_cd", "split": "train", "col": "transcription",
         "licence": "CC-BY-4.0", "audio_heavy": True},
    ],
    "sna": [
        {"repo": "wikimedia/wikipedia", "config": "20231101.sn", "split": "train", "col": "text",
         "licence": "CC-BY-SA-4.0"},
        {"repo": "masakhane/masakhanews", "config": "sna", "split": "train", "col": "text",
         "licence": "AFL-3.0"},
        {"repo": "HPLT/HPLT2.0_cleaned", "config": "sna_Latn", "split": "train", "col": "text",
         "licence": "CC0-1.0", "max_words": 12_000_000},
        {"repo": "cis-lmu/GlotCC-V1", "config": "sna-Latn", "split": "train", "col": "content",
         "licence": "CC0-1.0", "max_words": 12_000_000},
        {"repo": "google/fleurs", "config": "sn_zw", "split": "train", "col": "transcription",
         "licence": "CC-BY-4.0", "audio_heavy": True},
    ],
    "lug": [
        {"repo": "wikimedia/wikipedia", "config": "20231101.lg", "split": "train", "col": "text",
         "licence": "CC-BY-SA-4.0"},
        {"repo": "masakhane/masakhanews", "config": "lug", "split": "train", "col": "text",
         "licence": "AFL-3.0"},
        {"repo": "HPLT/HPLT2.0_cleaned", "config": "lug_Latn", "split": "train", "col": "text",
         "licence": "CC0-1.0", "max_words": 12_000_000},
        {"repo": "cis-lmu/GlotCC-V1", "config": "lug-Latn", "split": "train", "col": "content",
         "licence": "CC0-1.0", "max_words": 12_000_000},
        {"repo": "google/fleurs", "config": "lg_ug", "split": "train", "col": "transcription",
         "licence": "CC-BY-4.0", "audio_heavy": True},
    ],
}


# ---------------------------------------------------------------- normalisation
# Measured on Train.csv (38,199 rows): 17,063 ASCII U+0027 and ZERO of any curly variant. So the
# stage 2 CTC vocab contains ' and nothing else apostrophe-like — while scraped web text is full
# of U+2019. Without this fold that mismatch costs us twice, silently, in the direction of the
# biggest lever we have:
#   - acceptable() rejects a line once >2% of its characters are outside CHARSET, and one curly
#     apostrophe in a short sentence is already ~1.5-3%. Whole sentences vanish from the corpus.
#   - the lines that DO survive keep a character the acoustic model cannot emit, so pyctcdecode
#     can never match those words and their probability mass is wasted.
# It matters most exactly where we need it most: Luganda's ng' digraph puts an apostrophe inside
# very common words. Folding is free — U+2019 does not occur in the in-domain text at all, so
# nothing in-domain changes.
APOSTROPHES = {"’": "'", "ʼ": "'", "‘": "'", "´": "'", "`": "'"}


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).strip()
    text = text.translate(str.maketrans(APOSTROPHES))
    if LOWERCASE:
        text = text.lower()
    text = "".join(c if (c.isalpha() or c.isspace() or c in KEEP_PUNCT) else " " for c in text)
    return re.sub(r"\s+", " ", text).strip()


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def to_sentences(raw: str) -> list[str]:
    """Wikipedia articles are documents; a 5-gram wants sentences. Split before normalising,
    because normalisation is what destroys the sentence-final punctuation we split on."""
    out = []
    for para in str(raw).split("\n"):
        for sent in SENT_SPLIT.split(para):
            s = normalise(sent)
            if s:
                out.append(s)
    return out


# ---------------------------------------------------------------- the model's character space
# Lines containing characters the acoustic model cannot emit are useless to the decoder and
# actively harmful: they inflate the vocabulary with unreachable words and steal probability
# mass. Derive the allowed charset from the same place stage 2 derives its vocab.
def load_charset() -> set[str]:
    for cand in (ART("waxal-ckpt") / "w2vbert-waxal" / "vocab.json",
                 WORK / "w2vbert-waxal" / "vocab.json"):
        if cand.exists():
            vocab = json.loads(cand.read_text(encoding="utf-8"))
            chars = {c for c in vocab if len(c) == 1 and c != "|"}
            print(f"charset from {cand} ({len(chars)} chars)")
            return chars | {" "}

    # Stage 2 has not run yet — rederive with the SAME rule it uses, so the two agree.
    import pandas as pd
    from collections import Counter as _C
    tr = pd.read_csv(ZINDI_DIR / "Train.csv", escapechar="\\")
    col = next(c for c in tr.columns
               if c.lower() in ("transcription", "transcript", "text", "target", "sentence"))
    counts = _C("".join(normalise(t) for t in tr[col].dropna().astype(str)))
    chars = {c for c, n in counts.items() if n >= 25 and not c.isspace()}
    print(f"charset rederived from Train.csv ({len(chars)} chars) — stage 2 has not run yet")
    return chars | {" "}


CHARSET = load_charset()
print(f"allowed characters: {''.join(sorted(c for c in CHARSET if c.strip()))}")


def acceptable(line: str) -> bool:
    if len(line) < 8 or " " not in line:
        return False
    bad = sum(1 for c in line if c not in CHARSET)
    # A couple of stray characters is a typo; a lot of them means the line is another language,
    # a table of numbers, or markup that survived extraction.
    return bad / len(line) <= 0.02


# ---------------------------------------------------------------- in-domain WAXAL text
def indomain(lang: str) -> list[str]:
    import pandas as pd
    p = ZINDI_DIR / "Train.csv"
    if not p.exists():
        print(f"  !! {p} not found — in-domain text skipped, LM will be weaker")
        return []
    tr = pd.read_csv(p, escapechar="\\")
    tcol = next((c for c in tr.columns
                 if c.lower() in ("transcription", "transcript", "text", "target", "sentence")), None)
    lcol = next((c for c in tr.columns if c.lower() in ("language", "lang", "locale")), None)
    if tcol is None:
        return []
    rows = tr
    if lcol:
        rows = tr[tr[lcol].astype(str).str.lower().str[:3] == lang]
    # HOLD THE DEV SET OUT. Train.csv contains the `original_split == "validation"` rows that ARE
    # the 900-clip dev set, and this text is then repeated INDOMAIN_REPEAT times into the corpus.
    # An LM that has memorised the dev references — at several times their natural weight — makes
    # the dev harness report a score the leaderboard will never reproduce. That mistake cost us a
    # +0.2473 phantom bias (see kaggle/03_decode_and_submit.py, `DEV_IDS`). Unconditional, because
    # 900 sentences out of ~884k in-domain words is below the LM's noise floor, and a flag is a
    # thing you can forget to set.
    icol = next((c for c in tr.columns if c.lower() in ("id", "audio_id", "utt_id")), None)
    dev_path = Path(__file__).resolve().parents[1] / "local" / "harness" / "devset.json"
    if icol and dev_path.exists():
        dev_ids = {it["id"] for it in json.load(open(dev_path, encoding="utf-8"))["items"]}
        before = len(rows)
        rows = rows[~rows[icol].astype(str).isin(dev_ids)]
        print(f"  {lang}: held {before - len(rows):,} dev sentences out of the LM corpus")
    else:
        print(f"  !! {lang}: devset.json not found — the LM may contain dev text, and any dev "
              f"score measured with it will be inflated")
    lines = [normalise(t) for t in rows[tcol].dropna().astype(str)]
    return [l for l in lines if l]


# ---------------------------------------------------------------- build
manifest: dict[str, dict] = {}

for lang in LANGS:
    print(f"\n{'='*60}\n{lang}\n{'='*60}")
    seen: set[str] = set()
    lines: list[str] = []
    used: list[dict] = []

    # in-domain first, repeated
    dom = indomain(lang)
    if dom:
        for _ in range(INDOMAIN_REPEAT):
            lines.extend(dom)
        n_words = sum(len(l.split()) for l in dom)
        print(f"  WAXAL Train.csv     : {len(dom):,} sentences, {n_words:,} words "
              f"(x{INDOMAIN_REPEAT} = {n_words*INDOMAIN_REPEAT:,})")
        used.append({"source": "Zindi Train.csv (challenge data)", "sentences": len(dom),
                     "words": n_words, "repeated": INDOMAIN_REPEAT, "licence": "challenge data"})
        seen.update(dom)

    for src in SOURCES[lang]:
        tag = f"{src['repo']}:{src['config']}"
        if src.get("audio_heavy") and not INCLUDE_AUDIO_DATASETS:
            print(f"  {tag:<42}: skipped (audio dataset — set WAXAL_INCLUDE_FLEURS=1 to use)")
            used.append({"source": tag, "sentences": 0, "words": 0, "skipped": "audio_heavy",
                         "licence": src["licence"]})
            continue
        try:
            from datasets import load_dataset
            # Print BEFORE the work, not after. GlotCC has 1,255 configs and HPLT 191, so just
            # resolving the data files for one of them can take minutes with nothing on stdout —
            # which is indistinguishable from a hang, and cost us a run once already.
            print(f"  {tag:<42}: resolving...", flush=True)
            ds = load_dataset(src["repo"], src["config"], split=src["split"], streaming=True)
            cap = src.get("max_words")
            got, words, capped = 0, 0, False
            for row in ds:
                raw = row.get(src["col"])
                if not raw:
                    continue
                for s in to_sentences(raw):
                    if not acceptable(s) or s in seen:
                        continue
                    seen.add(s)
                    lines.append(s)
                    got += 1
                    words += len(s.split())
                if cap and words >= cap:
                    capped = True
                    break
            print(f"  {tag:<42}: {got:,} sentences, {words:,} words"
                  f"{'  (hit max_words cap)' if capped else ''}")
            used.append({"source": tag, "sentences": got, "words": words,
                         "repeated": 1, "licence": src["licence"], "capped": capped})
        except Exception as e:                                  # noqa: BLE001
            # Non-fatal by design: a renamed config must not cost us the other sources.
            print(f"  {tag:<42}: FAILED — {type(e).__name__}: {str(e)[:140]}")
            used.append({"source": tag, "sentences": 0, "words": 0, "error": str(e)[:200],
                         "licence": src["licence"]})

    out_p = OUT / f"{lang}.txt"
    out_p.write_text("\n".join(lines), encoding="utf-8")
    total_words = sum(len(l.split()) for l in lines)
    vocab = Counter(w for l in lines for w in l.split())

    manifest[lang] = {
        "file": out_p.name,
        "sentences": len(lines),
        "words": total_words,
        "unique_words": len(vocab),
        "sources": used,
    }
    print(f"  -> {out_p.name}: {len(lines):,} lines, {total_words:,} words, "
          f"{len(vocab):,} unique")

    # The reference point from arXiv:2512.10968 for the LMs that produced those WER gains.
    target = {"lin": 9_025_326, "lug": 9_168_243, "sna": 5_366_168}[lang]
    pct = 100 * total_words / target
    verdict = "OK" if pct >= 40 else ("THIN" if pct >= 10 else "TOO SMALL")
    print(f"  vs published LM corpus ({target:,} words): {pct:.1f}%  [{verdict}]")
    if verdict == "TOO SMALL":
        print("  !! At this size the LM may well score worse than greedy. Stage 3 tunes "
              "against greedy and will fall back, so this is safe — but the big win is lost.")

(OUT / "lm_sources.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
print(f"\nwrote {OUT}/lm_sources.json — this is your rules-required external-data disclosure")
print(f"\nNow: nothing to publish — {OUT} is persistent and stage 3 reads it from there.")
