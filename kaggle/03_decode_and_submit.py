"""
STAGE 3 — the biggest single win: KenLM shallow fusion + beam search, then submit.

Where to run: any GPU box with internet, ~2 GPU-hours. REQUIRES stage 0 to have run first.

WAXAL-NET set the published numbers with **CTC greedy decoding**. Beam search against a
word-level 5-gram is how we get past them. This is not a hopeful 15-25%: arXiv:2512.10968
measured w2v-bert-2.0 with and without exactly this setup on exactly these three languages —

    Luganda  39.75 -> 16.30 WER   (-59% relative)
    Shona    22.56 ->  9.28 WER   (-59% relative)
    Lingala  24.19 -> 22.74 WER   ( -6% relative)

— which makes the LM the single largest lever available to us, worth more than the acoustic
model choice. The catch is that those numbers came from 5-9M-word text corpora. A 5-gram built
on the few thousand Zindi train transcripts is far too sparse and can score WORSE than greedy;
the same paper shows XLS-R+LM regressing on Lingala and Shona for that reason. So the corpus
comes from `kaggle/00_build_lm_corpus.py` (read straight out of persistent storage), with the Zindi
transcripts as a fallback that is honestly labelled as the weak path.

alpha/beta are TUNED on the validation split, never guessed, and greedy is kept as a baseline
that the LM has to beat per-language before it is used. That guard is what makes a thin corpus
survivable rather than actively harmful.
"""

import gc
import json
import multiprocessing
import os
import re
import subprocess
import sys
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
    # REPO is defined in BOTH branches, not only the local one — the dev harness in section 3b
    # imports local/harness/score.py relative to it, and on Kaggle that import is the entire point
    # of the run. (It was missing here once: the kernel loaded both models, then died on the very
    # first line of dev mode with NameError, after the GPU was already warm.)
    try:
        REPO = Path(__file__).resolve().parents[1]
    except NameError:                                     # pasted into a notebook cell
        REPO = Path.cwd()
    _repo_zindi = REPO / "data" / "zindi"
    ZINDI_DIR = Path(os.environ["WAXAL_ZINDI_DIR"]) if os.environ.get("WAXAL_ZINDI_DIR") else (
        _ds if _ds.exists() else _repo_zindi)
    # Artefact names are NOT mount names on Kaggle. A kernel's output mounts at
    # /kaggle/input/<kernel-slug>, so stage 2's checkpoint arrives as `waxal-stage2-train` and
    # stage 1's lang_map as `waxal-baseline` — while an uploaded Dataset arrives under whatever
    # it was named. Hard-coding either spelling means stage 3 dies on a missing path after the
    # models have loaded. Resolve by CONTENT instead: look for the mount that actually holds the
    # artefact. WAXAL_<NAME> overrides it outright when a run needs a specific one.
    _MARKER = {
        "waxal-ckpt": "w2vbert-waxal",                    # stage 2: the fine-tuned model dir
        "waxal-lm": "lm_corpus",                          # stage 0: the KenLM text
        "waxal-baseline": "lang_map.json",                # stage 1: the LID routing decisions
        "waxal-router": "router_result.json",             # the router kernel: measured routing
    }

    def ART(name: str) -> Path:
        override = os.environ.get("WAXAL_" + name.replace("-", "_").upper())
        if override:
            return Path(override)
        direct = Path("/kaggle/input") / name
        if direct.exists():
            return direct
        marker = _MARKER.get(name)
        if marker:
            for cand in sorted(Path("/kaggle/input").glob("*")):
                if (cand / marker).exists():
                    return cand
        # Nothing matched. Return the name we were asked for so the caller's own error names it,
        # rather than inventing a path that exists but holds the wrong thing.
        return direct
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

# ---------------------------------------------------------------- datasets version guard
# datasets 4.0 decodes Audio columns via torchcodec, whose prebuilt .so is linked against a
# specific libtorch ABI and which declares no torch dependency on PyPI — so pip installs a wheel
# that may not match the host's torch, and it fails as `undefined symbol: torch_from_blob` from
# inside the dataset iterator, i.e. after models and audio have already downloaded. Three stage-1
# runs died that way. Fail here instead, in the first second, with the fix on screen.
import datasets as _ds

if int(_ds.__version__.split(".")[0]) >= 4:
    raise SystemExit(
        f"\n  datasets {_ds.__version__} is installed; this pipeline needs 3.x."
        "\n  4.0 moved audio decoding to torchcodec (see requirements-gpu.txt)."
        "\n\n      pip install -q 'datasets>=3.6,<4.0'"
        "\n\n  Then re-run this script."
    )


CKPT = ART("waxal-ckpt") / "w2vbert-waxal"                # stage 2 output

# WAXAL_LM_CORPUS_DIR overrides where stage 0's text is read from. ART("waxal-lm") assumes the
# corpus arrives as the waxal-lm kernel's output mounted at /kaggle/input/waxal-lm — and on
# 1 Aug that assumption silently produced nothing: kernel_sources named waxal-lm, Kaggle's own
# stored metadata confirmed it, and /kaggle/input still had no lm_corpus. Both scored
# submissions were decoded without a real LM, partly because of this. A caller that has
# LOCATED the corpus (waxal-lugA rglobs for lug.txt, so it works off a Dataset mount or a
# kernel mount) can now say where it is instead of re-deriving the same broken path.
LM_CORPUS_DIR = (Path(os.environ["WAXAL_LM_CORPUS_DIR"])
                 if os.environ.get("WAXAL_LM_CORPUS_DIR")
                 else ART("waxal-lm") / "lm_corpus")       # stage 0 output
PHASE2_URL = "https://storage.googleapis.com/waxalphase2/audio.zip"

# Preflight. The model does not load until after ~5 GB of phase-2 audio has downloaded and been
# decoded, so a wrong or unattached checkpoint mount otherwise announces itself half an hour into
# a GPU session. Check the inputs first — this costs a stat() call.
#
# Only the w2vbert backend needs a checkpoint; the default MMS backend pulls its weights from the
# Hub. Read the env directly because BACKEND is defined in section 2, and this check has to run
# before the download, not after it.
# Every backend except `w2vbert` pulls its weights from the Hub, so none of them needs the local
# stage-2 checkpoint to exist.
if (os.environ.get("WAXAL_BACKEND", "mms").lower() not in ("mms", "waxalnet", "whisper")
        and not CKPT.exists()):
    raise SystemExit(
        f"\n  no fine-tuned checkpoint at {CKPT}"
        f"\n  mounts present: {[p.name for p in sorted(Path('/kaggle/input').glob('*'))] if Path('/kaggle/input').exists() else 'none'}"
        "\n\n  Attach the stage 2 kernel's output (kernel_sources) or set WAXAL_WAXAL_CKPT"
        "\n  to the directory that CONTAINS w2vbert-waxal.\n")
if not LM_CORPUS_DIR.exists():
    # Survivable, and quietly so — which is the danger. corpus_lines() falls back to building
    # each KenLM from Train.csv alone, so shallow fusion still happens and nothing errors; the
    # LM is just trained on a fraction of the text and gives back a fraction of the win. That
    # reads as a disappointing score rather than as a missing input, so say it here.
    print(f"\n!! no LM corpus at {LM_CORPUS_DIR} — every language will fall back to a "
          f"Train.csv-only LM (marked WEAK below).\n"
          f"!! Attach the waxal-lm kernel output unless you are deliberately measuring that.\n")

LANGS = ["lin", "sna", "lug"]
HF_CONFIGS = {"lin": "lin_asr", "sna": "sna_asr", "lug": "lug_asr"}
LID_MODEL = "facebook/mms-lid-256"
NGRAM_ORDER = 5
BLANK_FILL = "a"                  # what to write when the decoder returns nothing; see §5
MAX_SECONDS = 40
SEED = 1337
# 100 is the pyctcdecode default and the usual sweet spot; past ~250 the gain is noise and the
# decode becomes the slowest thing in the pipeline.
BEAM_WIDTH = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

# Must match stage 2 exactly. The organisers' starter notebook scores with
# jiwer.wer([r.lower()...], [p.lower()...]) — lowercase both sides, punctuation left alone.
LOWERCASE = True
KEEP_PUNCT = set(".,'’-;:!?")


# Train.csv is 17,063 ASCII U+0027 and zero curly variants, so ' is the only apostrophe in the
# stage 2 vocab. Fold every lookalike onto it: on a character metric an unfolded U+2019 is a
# guaranteed wrong character in every word carrying one, and it costs nothing if the scorer
# already normalises. See the longer note in 00_build_lm_corpus.py for what it costs the LM.
APOSTROPHES = {"’": "'", "ʼ": "'", "‘": "'", "´": "'", "`": "'"}


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).strip()
    text = text.translate(str.maketrans(APOSTROPHES))
    if LOWERCASE:
        text = text.lower()
    text = "".join(c if (c.isalpha() or c.isspace() or c in KEEP_PUNCT) else " " for c in text)
    return re.sub(r"\s+", " ", text).strip()


def lang_from_id(utt_id: str) -> str | None:
    """
    Every Phase 1 id is `<iso3>_<number>` (lug_96114, lin_9487, sna_221), so the language is
    free and exact — no LID model, no error rate. Phase 2 promises no metadata and may well
    strip this, which is why LID stays in the pipeline as the fallback rather than being
    deleted. Free accuracy when it is there; graceful when it is not.
    """
    prefix = str(utt_id).split("_", 1)[0].strip().lower()
    return prefix if prefix in LANGS else None


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


# ---------------------------------------------------------------- 1. build KenLM per language
# Built into WORK, not the cwd. On Kaggle that's the same thing; on Lightning WORK is persistent,
# so the cmake+make (several minutes, and on Lightning those minutes are billed against your 15
# free credits) happens once ever instead of once per session.
#
# WAXAL_NO_LM=1 skips this whole section: no apt, no cmake, no corpus, no alpha/beta grid. That is
# ~40 minutes of a GPU session, and for a dev run whose question is "how good is this acoustic
# model on its own?" every one of those minutes buys nothing. Never set it for a submission run —
# shallow fusion is the largest single lever in this pipeline.
NO_LM = os.environ.get("WAXAL_NO_LM", "0") == "1"

KENLM = WORK / "kenlm"
LMPLZ = KENLM / "build" / "bin" / "lmplz"
BUILD_BINARY = KENLM / "build" / "bin" / "build_binary"
if NO_LM:
    print("WAXAL_NO_LM=1 -> skipping KenLM build and alpha/beta tuning; decoding GREEDILY.")
    os.system("pip -q install pyctcdecode")
elif not LMPLZ.exists():
    # Kaggle runs as root and has no sudo; Lightning gives you a normal user with sudo. Try
    # plain apt first and fall back, rather than assuming either.
    apt = ("apt-get -qq install -y build-essential cmake libboost-system-dev libboost-thread-dev "
           "libboost-program-options-dev libboost-test-dev libboost-filesystem-dev libeigen3-dev "
           "zlib1g-dev")
    if os.system(apt) != 0:
        os.system(f"sudo {apt}")
    os.system(f"mkdir -p {KENLM} && wget -q -O - https://kheafield.com/code/kenlm.tar.gz "
              f"| tar xz -C {KENLM} --strip-components=1")
    os.system(f"mkdir -p {KENLM}/build && cd {KENLM}/build && cmake .. -DCMAKE_BUILD_TYPE=Release "
              "> /dev/null && make -j4 lmplz build_binary > /dev/null")
assert NO_LM or LMPLZ.exists(), (
    f"KenLM did not build at {LMPLZ}. Without it there is no shallow fusion and no ~59% WER "
    f"win — stop and fix this rather than falling through to greedy decoding."
)
if not NO_LM:
    os.system("pip -q install pyctcdecode https://github.com/kpu/kenlm/archive/master.zip")

LM_DIR = WORK / "lm"
LM_DIR.mkdir(exist_ok=True)
lm_paths, unigram_sets = {}, {}


# THE DEV IDS ARE HELD OUT OF EVERY LANGUAGE MODEL, ALWAYS — including submission runs.
#
# Found 2026-08-01. `corpus_lines()` read ALL of Train.csv, and Train.csv contains the
# `original_split == "validation"` rows that ARE our 900-clip dev set. So the KenLM had memorised
# the dev references verbatim, and shallow fusion then decoded those same clips with an LM that
# contained their exact sentences. That is why the dev harness reported 0.7392 for a configuration
# the leaderboard scored 0.4919: a +0.2473 "bias" that was leakage, not domain shift. Every
# LM-enabled dev number produced before this fix is inflated and must not be compared against one
# produced after it. (kaggle/kernels/bakeoff ran WAXAL_NO_LM=1 and is unaffected.)
#
# The holdout is unconditional rather than gated on WAXAL_DEV because a flag is a thing you can
# forget to set, and the cost of being wrong is silently believing a number that is 0.25 too high.
# Dropping 900 sentences from a corpus of ~884k in-domain words plus external text is far below
# the noise floor of the LM, so there is no submission-quality reason to keep the footgun.
try:
    _dev_ids_path = REPO / "local" / "harness" / "devset.json"
    _dev_json = json.load(open(_dev_ids_path, encoding="utf-8"))["items"]
    DEV_IDS = {it["id"] for it in _dev_json}
    # A corpus stage 0 already wrote has the dev text baked in and no ids left to filter on, so
    # hold out by CONTENT as well. Matching on the normalised string is exactly the right key:
    # normalise() is what both the corpus writer and the scorer apply, so a line that survives
    # this filter is a line the LM could not have memorised from our dev references.
    DEV_TEXT = {normalise(it["reference"]) for it in _dev_json}
    print(f"LM holdout: excluding {len(DEV_IDS):,} dev ids / {len(DEV_TEXT):,} dev sentences "
          f"from every language model")
except Exception as e:                                    # noqa: BLE001
    DEV_IDS, DEV_TEXT = set(), set()
    print(f"LM holdout: could NOT read devset.json ({type(e).__name__}: {e}) — "
          f"dev scores from this run are NOT trustworthy")


def corpus_lines(lang: str) -> tuple[list[str], str]:
    """External corpus from stage 0 if it is mounted, else the thin Train.csv fallback."""
    ext = LM_CORPUS_DIR / f"{lang}.txt"
    if ext.exists():
        # Fold apostrophes here too. Stage 0 does it at write time, but a corpus built before
        # that fix existed is still a valid mount, and a KenLM carrying a character the acoustic
        # model cannot emit makes those words unreachable in the beam. Cheap insurance: this is
        # a str.translate over text we are reading anyway, not a re-normalisation.
        _fold = str.maketrans(APOSTROPHES)
        lines = [l.translate(_fold)
                 for l in ext.read_text(encoding="utf-8").splitlines() if l.strip()]
        if DEV_TEXT:
            before = len(lines)
            lines = [l for l in lines if normalise(l) not in DEV_TEXT]
            if before != len(lines):
                print(f"    {lang}: held {before - len(lines):,} dev sentences out of the "
                      f"stage-0 corpus (it was built before the holdout existed)")
        if lines:
            return lines, "external (stage 0)"

    train_csv = pd.read_csv(ZINDI_DIR / "Train.csv", escapechar="\\")
    txt_col = guess_col(train_csv, "transcription", "transcript", "text", "target", "sentence")
    lng_col = guess_col(train_csv, "language", "lang", "locale")
    id_col = guess_col(train_csv, "id", "audio_id", "utt_id")
    rows = (train_csv[train_csv[lng_col].astype(str).str.lower().str[:3] == lang]
            if lng_col else train_csv)
    if DEV_IDS and id_col:
        before = len(rows)
        rows = rows[~rows[id_col].astype(str).isin(DEV_IDS)]
        print(f"    {lang}: held {before - len(rows):,} dev sentences out of the LM corpus")
    lines = [normalise(t) for t in rows[txt_col].dropna().astype(str)]
    return [l for l in lines if l], "Train.csv only (WEAK)"


for lang in [] if NO_LM else LANGS:
    lines, provenance = corpus_lines(lang)
    if not lines:
        print(f"  {lang}: no text at all, skipping LM — this language decodes greedily")
        continue

    n_words = sum(len(l.split()) for l in lines)
    print(f"\n  {lang}: {len(lines):,} sentences, {n_words:,} words  [{provenance}]")
    if n_words < 500_000:
        print(f"  {lang}: WARNING — under 500k words. A 5-gram this sparse often loses to "
              f"greedy. Run kaggle/00_build_lm_corpus.py. The sweep below will fall back.")

    txt = LM_DIR / f"{lang}.txt"
    txt.write_text("\n".join(lines), encoding="utf-8")
    arpa = LM_DIR / f"{lang}.arpa"
    # --discount_fallback: Kneser-Ney cannot estimate its discounts on a small corpus and
    # lmplz hard-errors without this. Harmless on a large one, so it stays unconditional.
    # --prune keeps the ARPA from ballooning at 4/5-gram order on a multi-million-word corpus;
    # singleton high-order n-grams are noise anyway.
    prune = "--prune 0 0 0 1 1" if n_words > 2_000_000 else ""
    subprocess.run(
        f"{LMPLZ} -o {NGRAM_ORDER} --text {txt} --arpa {arpa} "
        f"--discount_fallback {prune}",
        shell=True, check=True,
    )
    # Binary format loads in seconds instead of minutes, which matters because the alpha/beta
    # sweep constructs a decoder 12 times per language.
    binary = LM_DIR / f"{lang}.bin"
    subprocess.run(f"{BUILD_BINARY} {arpa} {binary}", shell=True, check=True)
    arpa.unlink(missing_ok=True)

    lm_paths[lang] = str(binary)
    # Cap the unigram list: pyctcdecode uses it only to flag out-of-vocabulary words, and a
    # multi-million-entry set costs memory for no decoding benefit.
    counts = pd.Series([w for l in lines for w in l.split()]).value_counts()
    unigram_sets[lang] = sorted(counts[counts >= 2].index[:400_000])
    print(f"  {lang}: {len(unigram_sets[lang]):,} unigrams -> {binary.name}")


# ---------------------------------------------------------------- 2. model + decoders
# BACKEND picks the acoustic model. It defaults to MMS because the w2v-bert fine-tune FAILED:
# leg 1 collapsed to the CTC blank solution (WER and CER exactly 1.000 at every eval, train loss
# flat at ~24.0 from step 400, grad_norm 167 -> 2). Evidence and post-mortem in
# docs/sbu-lethabo-log.md, 1 Aug. Its CTC head was randomly initialised and WAXAL utterances are
# long (176 chars / 26 words average), which is close to the worst case for bootstrapping a CTC
# alignment — blank is a deep local minimum and it deepens with target length.
#
# mms-1b-all does not have that failure mode available to it: its per-language CTC heads are
# already trained on lin/sna/lug, so we start from a model that transcribes rather than one that
# has to discover an alignment from scratch. The KenLM fusion below — the single biggest lever we
# have — never depended on the fine-tune at all. It needs logits, and MMS produces good ones now.
#
# "w2vbert" is kept working so the failed run stays reproducible for the rules' code review.
# "waxalnet" is the backend that matters. The WAXAL organisers published their own baseline
# fine-tunes at huggingface.co/waxal-benchmarking — one model per language, trained on the WAXAL
# corpus itself, ungated and Apache/MIT. The rules allow this explicitly: "You may use pretrained
# models as long as they are openly available to everyone."
#
# Their published test numbers, projected onto this competition's metric (weighted by each
# language's share of reference words/characters, because jiwer POOLS rather than averages):
#
#     mms-300m-waxal-*     WER 0.317  CER 0.096  ->  multi 0.794
#     whisper-small-waxal-* WER 0.332  CER 0.113  ->  multi 0.777
#     zero-shot mms-1b-all (what we submitted)    ->  multi 0.492  (measured, rank 120)
#
# Those projections assume punctuation is reproduced. It is not: their vocabs carry only '&' and
# '/', no '.' ',' or apostrophe. Docking the measured cost of emitting no punctuation puts
# mms-300m-waxal-* at ~0.731 — which is the leaderboard's entire top cluster (0.7206-0.7257).
# That is almost certainly what the leaders are running.
#
# CORRECTION, 1 Aug — I read the conclusion off that wrongly, and the bakeoff (kernel
# lethabomh14/waxal-bakeoff, 10 candidates, LM disabled so nothing leaks) settled it. I claimed
# "the way past them is punctuation, not a bigger acoustic model". The measurement says the
# opposite: the acoustic model is the dominant lever and punctuation is a rounding error on top
# of it.
#
#     lin  mms-300m 0.6893 -> w2vbert-lin-waxal-aug-ft 0.7788   +0.0895  (acoustic)
#     sna  mms-300m 0.7815 -> waxal-whisper-large-v3   0.8034   +0.0219  (acoustic)
#     lug  mms-300m 0.8163 -> nothing beat it                    ——
#     trailing '.' where it helps:                    +0.004 lin, +0.012 lug, -0.018 sna
#
# So the ordering is: pick the right checkpoint per language first, then add the period only
# where the checkpoint does not already punctuate. See local/harness/ for how this was measured
# and docs/MODEL-CANDIDATES.md for the full table.
BACKEND = os.environ.get("WAXAL_BACKEND", "mms").lower()
ASR_MODEL = os.environ.get("WAXAL_ASR_MODEL", "facebook/mms-1b-all")
MMS_ADAPTER = {"lin": "lin", "sna": "sna", "lug": "lug"}

# One checkpoint per language, not one model with adapters — so these swap as whole models.
WAXALNET = {
    "lin": os.environ.get("WAXAL_LIN", "waxal-benchmarking/mms-300m-waxal-lin"),
    "sna": os.environ.get("WAXAL_SNA", "waxal-benchmarking/mms-300m-waxal-sna"),
    "lug": os.environ.get("WAXAL_LUG", "waxal-benchmarking/mms-300m-waxal-lug"),
}

# PER-LANGUAGE backend. The bakeoff (kaggle/kernels/bakeoff, 2026-08-01) measured every open
# checkpoint on the frozen dev set and no single architecture won all three languages:
#   lin  douyeszn/w2vbert-lin-waxal-aug-ft        0.7788   (CTC)      vs mms-300m 0.6893
#   sna  Mubarak127/waxal-whisper-large-v3-sna    0.8034   (seq2seq)  vs mms-300m 0.7815
#   lug  waxal-benchmarking/mms-300m-waxal-lug    0.8163   (CTC)      control held
# jiwer POOLS errors rather than averaging per-language scores, so the submission is three
# independent decoding problems weighted by their share of reference words. Mixing architectures
# across languages therefore costs nothing and wins ~0.05 overall — but it needs the backend to be
# a property of the LANGUAGE, not of the run. Format: "lin=waxalnet,sna=whisper,lug=waxalnet".
BACKENDS = {lg: BACKEND for lg in LANGS}
for _pair in os.environ.get("WAXAL_BACKENDS", "").split(","):
    if "=" in _pair:
        _lg, _bk = _pair.split("=", 1)
        _lg, _bk = _lg.strip(), _bk.strip().lower()
        if _bk not in ("mms", "waxalnet", "whisper"):
            raise SystemExit(f"WAXAL_BACKENDS: unknown backend {_bk!r} for {_lg!r}")
        BACKENDS[_lg] = _bk
MIXED = len(set(BACKENDS.values())) > 1
# A stable name for the run, fixed here — BACKEND becomes per-language below and by the time the
# CSV is written it holds whichever language decoded last, which is not what the file should be
# named after.
RUN_TAG = os.environ.get("WAXAL_RUN_TAG") or ("mixed" if MIXED else BACKEND)

# Languages that get a trailing '.' appended. This is per-language and measured, not a global
# tidy-up: the metric counts punctuation and 82.4% of dev references end in a full stop, but the
# rate is nothing like uniform (lin 64.6%, sna 95.9%, lug 97.8%) and it only pays when the model
# does not already punctuate. Bakeoff deltas:
#   lin  w2vbert  0.7788 -> 0.7828  (+0.0040)   CTC, no punctuation in vocab   -> ON
#   lug  mms-300m 0.8163 -> 0.8286  (+0.0123)   CTC, no punctuation in vocab   -> ON
#   sna  whisper  0.8034 -> 0.7853  (-0.0181)   BPE, punctuates natively       -> OFF
# Appending to a model that already emits '.' produces '..' and costs a word AND a character, so
# this list must be re-derived whenever a checkpoint changes.
PLUS_PERIOD = {x.strip() for x in os.environ.get("WAXAL_PLUS_PERIOD", "").split(",") if x.strip()}
if PLUS_PERIOD:
    print(f"trailing '.' will be appended for: {sorted(PLUS_PERIOD)}")
if PLUS_PERIOD - set(LANGS):
    raise SystemExit(f"WAXAL_PLUS_PERIOD: unknown language(s) {sorted(PLUS_PERIOD - set(LANGS))}")

# Which Whisper language token to force, per WAXAL language. Format: "lug=sw,sna=sw", or a bare
# code to apply to all ("sw"). Defined HERE, above the loader, because _wn_load runs at import.
#
# Off by default, and the default is still right for a checkpoint that carries its own forced
# decoder ids (see transcribe_whisper). But "carries its own" is an assumption, not a fact:
# cdli/whisper-large-v3_finetuned_ugandan_luganda_waxal_7 ships generation_config language="sw",
# while every KasuleTrevor/cdli-whisper-ml-* checkpoint ships language=null. On the second kind,
# passing nothing does not mean "use the fine-tuned default" — it means Whisper runs its own
# language ID on each clip and forces whatever that returns, independently, 1500 times. That is a
# silent per-clip coin flip in the middle of a decode, so it has to be overridable.
#
# Swahili is the code to reach for: it is the closest language Whisper has a token for at all, and
# it is what CDLI fine-tuned and evaluated Luganda under. Whisper has no Luganda, Lingala or Shona
# token, so for these three there is nothing "correct" to pass — only nearer misses.
WHISPER_LANG: dict[str, str] = {}
for _pair in os.environ.get("WAXAL_WHISPER_LANG", "").split(","):
    _pair = _pair.strip()
    if not _pair:
        continue
    if "=" in _pair:
        _lg, _code = _pair.split("=", 1)
        WHISPER_LANG[_lg.strip()] = _code.strip()
    else:
        WHISPER_LANG.update({lg: _pair for lg in LANGS})
if WHISPER_LANG:
    print(f"whisper language tokens forced: {WHISPER_LANG}")
if set(WHISPER_LANG) - set(LANGS):
    raise SystemExit(f"WAXAL_WHISPER_LANG: unknown language(s) {sorted(set(WHISPER_LANG) - set(LANGS))}")


def finish(text: str, lang: str) -> str:
    """The single point where a decoded hypothesis becomes a submitted cell.

    Dev and submission both call this, which is the whole point: the harness can only predict a
    leaderboard score if it scores the exact string that would be uploaded, punctuation included.
    The endswith guard stops a model that already punctuates from getting '..'.
    """
    t = normalise(text)
    if lang in PLUS_PERIOD and t and not t.endswith((".", "?", "!")):
        t += "."
    return t
if MIXED:
    print(f"MIXED backends: {BACKENDS}")
    if "mms" in BACKENDS.values():
        # mms is one model with swappable adapters; the others are whole-model swaps. Mixing the
        # two would need both resident at once and there is no measured reason to want it.
        raise SystemExit("WAXAL_BACKENDS: 'mms' cannot be mixed with per-language checkpoints")

from pyctcdecode import build_ctcdecoder

if BACKEND in ("waxalnet", "whisper"):
    # ONE loader for both whole-model backends, dispatching on the language's own backend, because
    # the winning lineup is mixed (CTC for lin/lug, seq2seq for sna) and a loader that can only
    # build one architecture cannot express it.
    #
    # AutoModelForCTC, NOT Wav2Vec2ForCTC. The three env vars above accept ANY per-language CTC
    # checkpoint on the Hub, and the good ones are not all the same architecture:
    #   Wav2Vec2ForCTC      mms-300m-waxal-*, keystats/lingala-xlsr-waxal-finetuned
    #   Wav2Vec2BertForCTC  douyeszn/w2vbert-*-waxal-aug, dhasmana/WAXAL-*-w2v-bert-2.0
    # Hardcoding Wav2Vec2ForCTC silently excluded every w2v-bert checkpoint — and w2v-bert is what
    # won Lingala by +0.094 over the organisers' baseline.
    #
    # Whisper is seq2seq: no frame logits, so no pyctcdecode and no KenLM shallow fusion on that
    # language. It earns its slot because its tokenizer is BPE over ordinary text, so a Whisper
    # checkpoint fine-tuned on WAXAL transcripts emits punctuation natively — which is why blindly
    # appending '.' HURTS it (sna 0.8034 -> 0.7853) while it helps every CTC model here.
    from transformers import AutoModelForCTC, AutoProcessor, WhisperForConditionalGeneration

    _wn_cache: dict[str, tuple] = {}

    def _wn_load(lang: str):
        """Load (and cache) one language's checkpoint, dispatching on THAT language's backend."""
        if lang not in _wn_cache:
            repo = WAXALNET.get(lang) or WAXALNET["lug"]
            kind = BACKENDS.get(lang, BACKEND)
            print(f"  loading {lang} [{kind}]: {repo}", flush=True)
            pr = AutoProcessor.from_pretrained(repo)
            if kind == "whisper":
                md = WhisperForConditionalGeneration.from_pretrained(
                    repo, torch_dtype=torch.float16).to(DEVICE).eval()
                # Say out loud which language token this checkpoint will decode under. A null here
                # with no WAXAL_WHISPER_LANG override means per-clip language detection, which
                # looks identical in the log to a checkpoint that knows its own language.
                _cfg_lang = getattr(md.generation_config, "language", None)
                if not _cfg_lang and not WHISPER_LANG.get(lang):
                    print(f"    !! {repo} has generation_config.language=null and no "
                          f"WAXAL_WHISPER_LANG for {lang} — Whisper will language-detect EACH "
                          f"clip independently. Set WAXAL_WHISPER_LANG={lang}=sw to pin it.")
                else:
                    print(f"    whisper language: {WHISPER_LANG.get(lang) or _cfg_lang}"
                          f"{' (forced)' if WHISPER_LANG.get(lang) else ' (from checkpoint)'}")
            else:
                md = AutoModelForCTC.from_pretrained(
                    repo, torch_dtype=torch.float16).to(DEVICE).eval()
            # mms-300m is ~0.6 GB in fp16 and whisper-large-v3 ~3 GB. The decode loop groups by
            # language, so eviction is nearly free; keep two only when everything is small.
            cap = 1 if "whisper" in BACKENDS.values() else 2
            if len(_wn_cache) >= cap:
                for k in [k for k in _wn_cache if k != lang]:
                    del _wn_cache[k]
                torch.cuda.empty_cache()
            _wn_cache[lang] = (pr, md)
        return _wn_cache[lang]

    _first = next((lg for lg in LANGS if lg in WAXALNET), "lug")
    processor, model = _wn_load(_first)
    BACKEND = BACKENDS.get(_first, BACKEND)     # so `labels` below is built for the right kind
    print(f"backends={BACKENDS}  { {k: v.split('/')[-1] for k, v in WAXALNET.items()} }")
elif BACKEND == "mms":
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    processor = AutoProcessor.from_pretrained(ASR_MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL, torch_dtype=torch.float16).to(DEVICE).eval()
    print(f"backend=mms  {ASR_MODEL}")
else:
    from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor

    processor = Wav2Vec2BertProcessor.from_pretrained(CKPT)
    model = Wav2Vec2BertForCTC.from_pretrained(CKPT, torch_dtype=torch.float16).to(DEVICE).eval()
    print(f"backend=w2vbert  {CKPT}")


# Tokens that must never reach a transcript. MMS's vocab carries <s>, </s> and <unk> alongside
# the characters, and pyctcdecode does NOT strip them the way tokenizer.decode() does — it warns
# ("Found entries of length > 1 in alphabet") and then emits them literally. Verified locally on
# the real lug vocab: a decode came back as 'a<s>b?</s>a1j...'. Every one of those is straight
# WER and CER damage on a metric that is half CER.
#
# They cannot simply be mapped to "" because pyctcdecode rejects duplicate labels (only one entry
# may be the blank). So they keep unique placeholder labels here and are masked out of the LOGITS
# in logits_for(), which is the more robust fix anyway: masked columns can never win argmax, so
# the greedy fallback path gets the same protection as the beam path.
_NEVER_EMIT = ("<s>", "</s>", "<unk>", "[UNK]")
_BLANKS = ("<pad>", "[PAD]")
_special_ids: list[int] = []


def _labels_from_tokenizer() -> list[str]:
    """CTC label list in id order, in pyctcdecode's convention: blank is "", delimiter is " "."""
    global _special_ids
    if BACKEND == "whisper":
        # Whisper is seq2seq — there is no CTC alphabet and nothing here applies. Return empty
        # rather than building a 51,866-entry "alphabet" out of a BPE vocab, which pyctcdecode
        # would accept and then use to produce confident nonsense.
        _special_ids = []
        return []
    vocab_dict = processor.tokenizer.get_vocab()
    # MMS's tokenizer holds one vocab per target language. Depending on the transformers version
    # get_vocab() returns either the active language's flat dict or the whole nested mapping, so
    # unwrap the nested case rather than trusting one shape. (Measured on transformers 5.9 it is
    # flat, and the vocabs genuinely differ per language: lin 81, sna 65, lug 79 tokens, with
    # different index order — which is why set_language() has to rebuild this.)
    if vocab_dict and all(isinstance(v, dict) for v in vocab_dict.values()):
        active = getattr(processor.tokenizer, "target_lang", None)
        vocab_dict = vocab_dict.get(active) or next(iter(vocab_dict.values()))
    sorted_vocab = [k for k, _ in sorted(vocab_dict.items(), key=lambda kv: kv[1])]
    _special_ids = [i for i, t in enumerate(sorted_vocab) if t in _NEVER_EMIT]
    out = []
    for i, t in enumerate(sorted_vocab):
        if t in _BLANKS:
            out.append("")                     # w2v-bert uses [PAD]; MMS uses <pad>
        elif t == "|":
            out.append(" ")                    # both use | as the word delimiter
        elif t in _NEVER_EMIT:
            # One control char each: unique (pyctcdecode forbids duplicates), absent from every
            # transcript, and SINGLE-character so the alphabet stays char-type. Multi-char entries
            # make pyctcdecode warn that it cannot tell whether the alphabet is BPE, which is a
            # confusing thing to leave in a log for tokens that can never be emitted anyway.
            out.append(chr(1 + _NEVER_EMIT.index(t)))
        else:
            out.append(t)
    return out


_active_lang: str | None = None      # so we never re-swap an adapter we already hold
labels = _labels_from_tokenizer()


def set_language(lang: str) -> None:
    """Point MMS at `lang`'s CTC adapter and rebuild the label list to match.

    This is the one thing that genuinely differs from the w2v-bert path. w2v-bert had a single
    shared 46-token vocab for all three languages, so `labels` could be built once at import.
    MMS carries a SEPARATE vocabulary per adapter, so a decoder built against the wrong language's
    labels silently maps logit indices onto the wrong characters — it does not raise, it just
    produces convincing-looking rubbish. Hence: rebuild labels on every swap, and build any
    decoder afterwards, never before.
    """
    global _active_lang, labels, processor, model, BACKEND
    if lang == _active_lang:
        return
    kind = BACKENDS.get(lang, BACKEND)
    if kind in ("waxalnet", "whisper"):
        # Whole-model swap, not an adapter swap. Each checkpoint carries its OWN vocab (lin 72
        # tokens, sna 51, lug 38 — measured), so `labels` must be rebuilt from the new processor
        # before any decoder is constructed, for exactly the reason documented below.
        processor, model = _wn_load(lang if lang in WAXALNET else "lug")
        # BACKEND is what every downstream branch keys off — logits_for's feature-name choice, the
        # dev loop's batch size and its seq2seq test, _labels_from_tokenizer's early return. In a
        # mixed lineup it has to track the ACTIVE LANGUAGE rather than the run, and this is the
        # line that makes that true. Set it BEFORE rebuilding labels, which reads it.
        BACKEND = kind
        labels = [] if kind == "whisper" else _labels_from_tokenizer()
        _active_lang = lang
        return
    if kind != "mms":
        return
    adapter = MMS_ADAPTER.get(lang, lang)
    try:
        processor.tokenizer.set_target_lang(adapter)
        model.load_adapter(adapter)
    except Exception as e:                                  # noqa: BLE001
        # LID can name a language mms-1b-all has no adapter for. Falling back to lug is stage 1's
        # rule and the reasoning carries: everything LID confuses these clips with is Bantu, and
        # lug is the nearest of our three. A wrong-but-related adapter still beats no output.
        print(f"  no mms adapter for {lang} ({type(e).__name__}) -> decoding it as lug")
        adapter = "lug"
        processor.tokenizer.set_target_lang(adapter)
        model.load_adapter(adapter)
    labels = _labels_from_tokenizer()
    _active_lang = lang


def make_decoder(lang: str, alpha: float, beta: float):
    set_language(lang)      # `labels` must belong to `lang` before the decoder captures it
    return build_ctcdecoder(
        labels,
        kenlm_model_path=lm_paths.get(lang),
        unigrams=unigram_sets.get(lang),
        alpha=alpha, beta=beta,
    )


WHISPER_BEAMS = int(os.environ.get("WAXAL_WHISPER_BEAMS", "1"))


def transcribe_whisper(wavs: list[np.ndarray]) -> list[str]:
    """Seq2seq decode. Returns text directly — there is no logits/beam stage to hand to KenLM.

    By default this does NOT pass `language=`/`task=`: a checkpoint fine-tuned on one language
    carries its own generation_config with the right forced decoder ids, and overriding them with
    a guess is how you get an English-flavoured transliteration back. WAXAL_WHISPER_LANG opts out
    of that default for checkpoints whose generation_config leaves `language` null — there the
    "default" is per-clip language detection, which is not a default anyone chose.
    """
    kw = {}
    code = WHISPER_LANG.get(_active_lang or "")
    if code:
        kw = {"language": code, "task": "transcribe"}
    inp = processor(wavs, sampling_rate=16000, return_tensors="pt",
                    return_attention_mask=True)
    with torch.inference_mode():
        ids = model.generate(
            inp.input_features.to(DEVICE).half(),
            attention_mask=inp.attention_mask.to(DEVICE),
            num_beams=WHISPER_BEAMS,
            # Whisper's failure mode on out-of-distribution audio is a repetition loop that runs to
            # max_length. Capping it bounds the damage to one clip instead of one batch's runtime.
            max_new_tokens=200,
            repetition_penalty=1.1,
            **kw,
        )
    return processor.batch_decode(ids, skip_special_tokens=True)


def logits_for(wavs: list[np.ndarray]) -> list[np.ndarray]:
    inp = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
    # Ask the PROCESSOR what it produced rather than inferring it from BACKEND. wav2vec2/MMS/XLSR
    # are raw-waveform (input_values) and w2v-bert takes mel features (input_features) — and since
    # the waxalnet slots accept any checkpoint on the Hub, a single run can now legitimately mix
    # the two across languages. Keying off BACKEND was right only while each backend was one
    # architecture.
    feats = inp["input_values"] if "input_values" in inp else inp["input_features"]
    with torch.inference_mode():
        out = model(
            feats.to(DEVICE).half(),
            attention_mask=inp.attention_mask.to(DEVICE),
        ).logits.float().cpu().numpy()
    # Kill <s>, </s> and <unk> before anything reads these logits. See _labels_from_tokenizer:
    # pyctcdecode emits them literally, and even on the greedy path an <unk> is a character we
    # would rather spend on a real guess. -1e9 rather than -inf: log_softmax over an all--inf
    # row is NaN, and a NaN frame poisons the whole beam.
    if _special_ids:
        out[:, :, _special_ids] = -1e9
    lens = inp.attention_mask.sum(-1).cpu().numpy()
    # Trim padding frames before decoding, or the LM scores a tail of silence.
    ratio = out.shape[1] / inp.attention_mask.shape[1]
    return [out[i, : max(1, int(lens[i] * ratio))] for i in range(len(wavs))]


# ---------------------------------------------------------------- 3. tune alpha/beta on validation
from datasets import Audio, load_dataset
import evaluate


# requirements-gpu.txt pins datasets < 4 so Audio columns decode through soundfile, not
# torchcodec — see the note at the bottom of that file. This helper is the single place audio
# becomes 16 kHz mono float32, and it takes either a decoded cell or raw bytes, so the HF path
# and the phase 2 zip path cannot drift apart.
def decode_audio_cell(cell) -> np.ndarray:
    """An undecoded datasets audio cell -> 16 kHz mono float32."""
    import io

    import soundfile as sf

    if isinstance(cell, dict) and cell.get("array") is not None:   # already decoded upstream
        wav = np.asarray(cell["array"], dtype=np.float32)
        sr = int(cell.get("sampling_rate") or 16000)
    else:
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav.astype(np.float32)


wer_metric, cer_metric = evaluate.load("wer"), evaluate.load("cer")
N_TUNE = 150

tuned = {}
for lang, cfg in HF_CONFIGS.items():
    if lang not in lm_paths:
        continue
    # BEFORE logits_for, not just before make_decoder: on MMS the adapter decides what the
    # logits MEAN, so tuning alpha/beta against another language's adapter would tune on noise.
    set_language(lang)
    ds = load_dataset("google/WaxalNLP", cfg, split="validation", streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    wavs, refs = [], []
    n_skipped = 0
    for row in ds:
        # SECOND LEAK, fixed 1 Aug. The dev set is itself drawn from this same validation split
        # (see the dev-audio loader below, which streams split="validation" for the same configs).
        # Tuning alpha/beta on the first 150 rows therefore tuned on clips the dev score is then
        # reported over — the LM weights got fitted to the very utterances used to judge them.
        # This is independent of the KenLM corpus leak: fixing the corpus does not stop the grid
        # search from overfitting the eval sample. Skipping by id costs nothing but a longer
        # stream, because validation is far larger than 900 + 150.
        if str(row["id"]) in DEV_IDS:
            n_skipped += 1
            continue
        w = decode_audio_cell(row["audio"])
        if not (1.0 * 16000 <= len(w) <= MAX_SECONDS * 16000):
            continue
        wavs.append(w)
        refs.append(normalise(row["transcription"]))
        if len(wavs) >= N_TUNE:
            break
    print(f"  {lang}: tuning on {len(wavs)} clips, {n_skipped} dev clips skipped")
    if not wavs:
        print(f"  {lang}: no non-dev tuning clips found — skipping LM tuning for this language")
        continue

    all_logits = []
    for k in range(0, len(wavs), 4):
        all_logits.extend(logits_for(wavs[k:k + 4]))

    greedy = [normalise(processor.tokenizer.decode(l.argmax(-1))) for l in all_logits]
    g_wer = wer_metric.compute(predictions=greedy, references=refs)
    g_cer = cer_metric.compute(predictions=greedy, references=refs)
    best = {"alpha": None, "beta": None, "score": 1 - 0.5 * (g_wer + g_cer), "wer": g_wer, "cer": g_cer}
    print(f"\n[{lang}] greedy: WER {g_wer:.4f} CER {g_cer:.4f} score {best['score']:.4f}")

    # Grid reaches to alpha=1.5: with a genuine multi-million-word LM the optimum sits well
    # above the 0.5 that suits a corpus-of-transcripts LM, and stopping at 0.9 would quietly
    # cap the single biggest win in the pipeline.
    with multiprocessing.get_context("fork").Pool(os.cpu_count()) as pool:
        for alpha in (0.3, 0.5, 0.7, 0.9, 1.2, 1.5):
            for beta in (0.5, 1.5, 3.0):
                dec = make_decoder(lang, alpha, beta)
                hyp = [normalise(t) for t in
                       dec.decode_batch(pool, all_logits, beam_width=BEAM_WIDTH)]
                w = wer_metric.compute(predictions=hyp, references=refs)
                c = cer_metric.compute(predictions=hyp, references=refs)
                s = 1 - 0.5 * (w + c)
                flag = "  <-- best" if s > best["score"] else ""
                print(f"  a={alpha} b={beta}: WER {w:.4f} CER {c:.4f} score {s:.4f}{flag}")
                if s > best["score"]:
                    best = {"alpha": alpha, "beta": beta, "score": s, "wer": w, "cer": c}

    tuned[lang] = best
    print(f"[{lang}] BEST -> {best}")
    if best["alpha"] is None:
        print(f"[{lang}] LM did not beat greedy; this language will decode greedily. "
              f"That usually means the corpus is too thin — check stage 0's word counts.")
    elif best["alpha"] == 1.5:
        print(f"[{lang}] optimum sits at the edge of the grid; widen it if you have a rerun.")
    gc.collect()

json.dump(tuned, open(WORK / "lm_tuning.json", "w"), indent=2)


# ---------------------------------------------------------------- 3b. DEV MODE (WAXAL_DEV=1)
# Simulate a submission and predict its leaderboard score, without spending a submission.
#
# This runs the SAME model, the SAME set_language(), the SAME decoders and the SAME normalise()
# that section 6 uses to build the real file — it only swaps what audio goes in and scores the
# result instead of writing a csv. A harness that re-implements the decode measures a different
# system than the one we upload, which is worse than no harness at all.
#
# The references come from local/harness/devset.json: a seeded sample of Train.csv rows marked
# original_split == "validation". That is the organisers' own held-out split, handed to us
# labelled. It is NOT the phase-1 test labels, which the rules forbid using and which we never
# read. See local/harness/make_devset.py for why it is sampled to the test set's language mix.
if os.environ.get("WAXAL_DEV", "0") == "1":
    import multiprocessing as _mp

    sys.path.insert(0, str(REPO / "local" / "harness"))
    import score as HARNESS                                   # noqa: E402

    dev = json.load(open(REPO / "local" / "harness" / "devset.json", encoding="utf-8"))
    dev_items = dev["items"]

    # WAXAL_DEV_LANGS restricts the run to a subset of languages. The bakeoff needs this: the
    # best checkpoint is chosen PER LANGUAGE (there is no single publisher who is strongest at all
    # three), so a candidate for lin should cost one language's worth of decode, not three.
    # WAXAL_DEV_TAG keeps each candidate's result file separate — without it every run in a sweep
    # overwrites dev_result_waxalnet.json and only the last survives.
    DEV_LANGS = [x for x in LANGS
                 if x in os.environ.get("WAXAL_DEV_LANGS", ",".join(LANGS)).split(",")]
    DEV_TAG = os.environ.get("WAXAL_DEV_TAG", RUN_TAG)

    # WAXAL_MISROUTE=1 decodes every dev clip as the WRONG language while still scoring it against
    # its true reference. It exists because the cost of a routing error was the last number in this
    # pipeline that was estimated rather than measured, and it is the number the whole phase-2
    # projection swings on: our 0.4919 is 0.7453 acoustics plus a router that answered zero Lingala
    # and zero Shona, and how bad that is depends entirely on what a misrouted clip scores.
    # A derangement, not a random draw — every clip moves, and the same way on every run.
    MISROUTE = {"lin": "sna", "sna": "lug", "lug": "lin"}
    _misroute = os.environ.get("WAXAL_MISROUTE") == "1"
    if _misroute and len(DEV_LANGS) < len(LANGS):
        raise SystemExit("WAXAL_MISROUTE needs all three languages loaded — it sends each "
                         "language's clips to another language's model.")
    def route(lg: str) -> str:
        return MISROUTE[lg] if _misroute else lg
    if _misroute:
        print(f"    *** MISROUTE MODE: decoding {MISROUTE}, scoring against TRUE references.\n"
              f"        The multi below is what a 100%-wrong router scores, i.e. the floor the\n"
              f"        real number is interpolated against. It is not a submission candidate.")
    want = {it["id"]: it for it in dev_items if it["language"] in DEV_LANGS}
    print(f"\n=== DEV MODE === {len(want):,} clips, seed {dev['seed']}, mix {dev['actual_mix']}"
          f"\n    languages: {DEV_LANGS}   tag: {DEV_TAG}")

    # Cached to WORK because the dev set is FROZEN — the same 900 clips every run, forever. The
    # streaming pull is minutes; the load is seconds. Comparing two backends in one GPU session
    # should not cost the download twice, and more importantly both backends must see byte-identical
    # audio or the comparison measures the decoder plus the resampler.
    dev_cache = WORK / f"dev_audio_{dev['seed']}_{dev['n']}.npz"
    dev_audio = {}
    if dev_cache.exists():
        with np.load(dev_cache) as z:
            dev_audio = {k: z[k] for k in z.files}
        print(f"dev audio from cache: {len(dev_audio):,} clips ({dev_cache.name})")
    else:
        # Resolve ALL languages, not just DEV_LANGS. The cache name is keyed on (seed, n) only, so
        # a run restricted to one language must not be allowed to write a partial file under that
        # name — the next full run would load it, silently decode a third of the dev set, and
        # report a confident score for a sample that is no longer the frozen 900.
        _all = {it["id"]: it for it in dev_items}
        for lang, cfg in HF_CONFIGS.items():
            need = {i for i, it in _all.items() if it["language"] == lang}
            if not need:
                continue
            ds = load_dataset("google/WaxalNLP", cfg, split="validation", streaming=True)
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
            got = 0
            for row in ds:
                rid = str(row["id"])
                if rid in need and rid not in dev_audio:
                    dev_audio[rid] = decode_audio_cell(row["audio"])
                    got += 1
                    if got % 100 == 0:
                        print(f"  {lang} audio {got:,}/{len(need):,}", flush=True)
                if got >= len(need):
                    break
        np.savez(dev_cache, **dev_audio)
        print(f"resolved {len(dev_audio):,}/{len(_all):,} dev clips -> cached {dev_cache.name}")

    dev_dec = {}
    for lang in DEV_LANGS:
        t = tuned.get(lang)
        # No CTC decoder for whisper — it is seq2seq, there are no frame logits to beam over. Keyed
        # off THIS language's backend, not the run's: in a mixed lineup sna is seq2seq while lin
        # and lug still want their KenLM beam, and this loop runs before any set_language() call.
        dev_dec[lang] = make_decoder(lang, t["alpha"], t["beta"]) \
            if BACKENDS.get(lang, BACKEND) != "whisper" and t and t["alpha"] is not None else None

    dev_pred = {}
    with _mp.get_context("fork").Pool(os.cpu_count()) as pool:
        for lang in DEV_LANGS:
            # Grouped by the language each clip is DECODED as, which is what `lang` means to
            # set_language() below. Identical to the true language unless MISROUTE is on.
            ids = [i for i in want if route(want[i]["language"]) == lang and i in dev_audio]
            if not ids:
                continue
            set_language(lang)
            dec = dev_dec.get(lang)
            mode = "seq2seq" if BACKEND == "whisper" else ("beam+LM" if dec else "greedy")
            print(f"\ndecoding dev/{lang}: {len(ids):,} clips ({mode})")
            # Length-sorted so each batch pads to roughly its own longest clip rather than to the
            # longest in the set. Order does not affect the score — dev_pred is keyed by id.
            ids.sort(key=lambda i: len(dev_audio[i]))
            bi, bl = [], []

            def _flush():
                if not bi:
                    return
                txt = (dec.decode_batch(pool, bl, beam_width=BEAM_WIDTH) if dec
                       else [processor.tokenizer.decode(l.argmax(-1)) for l in bl])
                for i, t_ in zip(bi, txt):
                    dev_pred[i] = finish(t_, lang)
                bi.clear()
                bl.clear()

            step = 8 if BACKEND == "whisper" else 4
            for k in range(0, len(ids), step):
                ch = ids[k:k + step]
                wavs = [dev_audio[i][:16000 * MAX_SECONDS] for i in ch]
                if BACKEND == "whisper":
                    # Text comes straight out of generate(); there is no logits buffer to fill.
                    for i, t_ in zip(ch, transcribe_whisper(wavs)):
                        dev_pred[i] = finish(t_, lang)
                    if (k // step) % 10 == 0:
                        print(f"  {lang} {k:,}/{len(ids):,}", flush=True)
                    continue
                bl.extend(logits_for(wavs))
                bi.extend(ch)
                if len(bi) >= 64:
                    _flush()
            _flush()

    ok = [i for i in want if i in dev_pred]
    refs = [want[i]["reference"] for i in ok]
    hyps = [dev_pred[i] for i in ok]
    lgs = [want[i]["language"] for i in ok]
    res = HARNESS.report(refs, hyps, lgs, title=f"DEV — {DEV_TAG} ({','.join(DEV_LANGS)})",
                         ci=len(DEV_LANGS) == len(LANGS))
    if len(DEV_LANGS) < len(LANGS):
        print("  (partial-language run: 'reweighted to test mix' above is NOT comparable to a "
              "full run — use the per-language multi)")

    # What a trailing full stop is worth. These CTC vocabs mostly emit no sentence punctuation, and
    # '.' alone is 65 marks per 1,000 reference words in the dev references. Measured per language,
    # not globally, because the rate of period-final references is nothing like uniform:
    # lin 64.6%, sna 95.9%, lug 97.8%. A rule that pays for itself on sna and lug can easily lose
    # money on lin, and a single pooled number would hide that.
    res["plus_period"] = HARNESS.score(refs, [h + "." if h.strip() else h for h in hyps]).multi
    print(f"\n  + trailing '.' everywhere -> multi={res['plus_period']:.4f} "
          f"(delta {res['plus_period'] - res['per_language']['overall']['multi']:+.4f})")
    res["plus_period_by_lang"] = {}
    for lg in DEV_LANGS:
        idx = [k for k, x in enumerate(lgs) if x == lg]
        if not idx:
            continue
        r_, h_ = [refs[k] for k in idx], [hyps[k] for k in idx]
        base = HARNESS.score(r_, h_).multi
        dotted = HARNESS.score(r_, [h + "." if h.strip() else h for h in h_]).multi
        res["plus_period_by_lang"][lg] = {"base": base, "plus_period": dotted}
        print(f"      {lg}: {base:.4f} -> {dotted:.4f}  ({dotted - base:+.4f})")

    # ---------------------------------------------------------- routing-aware projection
    # The number printed above is an ORACLE-ROUTING score: every dev clip was decoded as the
    # language it actually is, because dev ids carry it. Phase 2 ids do not, so on the split that
    # decides the prize a model never gets that for free — a router does, and it gets some of them
    # wrong. Reporting the oracle number as "the predicted leaderboard score" is exactly the error
    # that let 0.7453 sit next to an actual 0.4919 for a week without anyone being able to name the
    # difference. So: state the assumption, and price it whenever the inputs to do so exist.
    _oracle = res["per_language"]["overall"]["multi"]
    _rr = ART("waxal-router") / "router_result.json"
    _acc = json.load(open(_rr))["accuracy"] if _rr.exists() else {}
    _floor = float(os.environ.get("WAXAL_MISROUTE_MULTI", "nan"))   # from a WAXAL_MISROUTE=1 run
    res["oracle_routing_multi"] = _oracle
    res["router_accuracy"] = _acc
    print(f"\n  ROUTING: the {_oracle:.4f} above assumes a PERFECT router (dev ids name their")
    print(f"  language; phase-2 ids do not). Phase 2 is worth less than this by the router's "
          f"error rate.")
    if _acc and _floor == _floor:            # nan != nan; both inputs measured
        best = max(_acc, key=_acc.get)
        p = _acc[best]
        proj = p * _oracle + (1 - p) * _floor
        res["projected_phase2_multi"] = proj
        res["projection_inputs"] = {"router": best, "accuracy": p, "misroute_multi": _floor}
        print(f"  router {best} measured {p:.4f} accurate; a misrouted clip measured {_floor:.4f}")
        print(f"  -> projected phase 2: {proj:.4f}")
    else:
        miss = ("router_result.json (run kaggle/kernels/router)" if not _acc else "") + \
               (" and " if not _acc and _floor != _floor else "") + \
               ("WAXAL_MISROUTE_MULTI (run this script once with WAXAL_MISROUTE=1)"
                if _floor != _floor else "")
        print(f"  cannot price it yet — missing {miss}. Until both exist, treat {_oracle:.4f} as")
        print(f"  an upper bound and do NOT quote it as a leaderboard prediction.")

    # RUN_TAG / BACKENDS, not BACKEND — the latter now holds whichever language decoded last, so
    # a mixed run would record a single arbitrary backend as though it produced every language.
    res["backend"] = RUN_TAG
    res["backends"] = BACKENDS
    res["plus_period_applied"] = sorted(PLUS_PERIOD)
    res["languages"] = DEV_LANGS
    res["models"] = {k: v for k, v in WAXALNET.items() if k in DEV_LANGS} \
        if set(BACKENDS.values()) & {"waxalnet", "whisper"} else ASR_MODEL
    res["n_decoded"] = len(ok)
    HARNESS.save(res, WORK / f"dev_result_{DEV_TAG}.json")
    json.dump({i: dev_pred[i] for i in ok}, open(WORK / f"dev_preds_{DEV_TAG}.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"\nwrote {WORK / f'dev_result_{DEV_TAG}.json'}")
    raise SystemExit(0)


# ---------------------------------------------------------------- 4. resolve test audio
# TWO submission templates, and they are disjoint sets with different shapes:
#   SampleSubmission.csv  4,253 rows, ids like `lug_96114`  -> phase 1
#   Test_phase2.csv       1,500 rows, ids like `ID_TBDTM`   -> phase 2, already ID/Target shaped
# Measured 30 Jul: zero id overlap between them. We predict the union and write one file per
# template, so whichever phase is open we have a correctly-shaped file ready and never have
# to guess which one Zindi wants.
TEMPLATES = []
for fname in ("SampleSubmission.csv", "Test_phase2.csv"):
    p = ZINDI_DIR / fname
    if not p.exists():
        print(f"  {fname}: absent, skipping")
        continue
    df = pd.read_csv(p, escapechar="\\")
    tid = guess_col(df, "id", "audio_id", "utt_id", "filename")
    ttxt = guess_col(df, "transcription", "transcript", "text", "target", "prediction")
    if ttxt is None:                      # an id-only file is still a valid target list
        ttxt = "Target"
        df[ttxt] = ""
    TEMPLATES.append((fname, df, tid, ttxt))
    print(f"  {fname}: {len(df):,} rows, id={tid}, target={ttxt}")

assert TEMPLATES, "no submission template found in ZINDI_DIR"
sample, SUB_ID, SUB_TXT = TEMPLATES[0][1], TEMPLATES[0][2], TEMPLATES[0][3]

needed_ids, _seen = [], set()
for _, df, tid, _ in TEMPLATES:
    for i in df[tid].astype(str):
        if i not in _seen:
            _seen.add(i)
            needed_ids.append(i)
needed = set(needed_ids)
print(f"submission contract: {len(needed_ids):,} unique ids across {len(TEMPLATES)} template(s)")

audio_store, known_lang = {}, {}
# Routing source, best first. The router kernel's map wins outright when it is mounted, because it
# is the only routing in this repo whose accuracy was MEASURED against labelled audio; stage 1's is
# an unmeasured open-set argmax from a 256-language model that has never seen this corpus, and it
# is what produced 0.4919. Kept as the fallback, not the default.
_router_map = ART("waxal-router") / "lang_map.json"
_lang_map = _router_map if _router_map.exists() else ART("waxal-baseline") / "lang_map.json"
if _lang_map.exists():
    known_lang = json.load(open(_lang_map))
    print(f"routing map: {_lang_map}  ({len(known_lang):,} ids)"
          + ("  [MEASURED — router kernel]" if _lang_map == _router_map else
             "  [stage 1, open-set, UNMEASURED — mount waxal-router instead if you have it]"))

# Cheapest and most reliable source first: the id prefix.
for i in needed_ids:
    lg = lang_from_id(i)
    if lg:
        known_lang[i] = lg
n_prefix = sum(1 for i in needed_ids if i in known_lang)
print(f"language from id prefix: {n_prefix:,} / {len(needed_ids):,}")

# Phase 2 ids are `ID_` + 5 uniformly-random uppercase letters (verified: letter frequencies
# 0.036-0.043 against a uniform 0.0385, all 26 letters used). There is no language in them and
# nothing to exploit. So on the set that actually decides the prize, LID is not a fallback that
# never fires — it is load-bearing for every single clip, and a LID error means decoding with
# the wrong KenLM, which corrupts the whole utterance rather than costing a few WER points.
#
# CORRECTION, 1 Aug — this block used to say stage 1's open-set routing was the fix and that
# stage 1's phase-2 file was the better one to upload. Both were wrong, and the cost was the whole
# competition so far.
#
# Open-set routing asks "which of 256 languages is this", and mms-lid-256 answered phase 2 with
# luo 42.5% / lug 27.5% / nyn 20% / guz,xog,kin,kam 2.5% each — zero Lingala, zero Shona, on a
# corpus that is 43.9% Lingala and 41.1% Shona (local/diagnose_lid_unconstrained.py, commit
# e9b3885). We shipped that, and it scored 0.4919 with checkpoints that measure 0.7453 on dev when
# the language is known. Solving 0.4919 = p*0.7453 + (1-p)*0.30 puts routing accuracy near 0.43.
#
# The error in the old reasoning was treating "the model can emit Dholuo" as an advantage. The
# REFERENCE transcripts are always lin, sna or lug — that is the whole challenge — so a clip
# decoded perfectly in Dholuo scores against a Lingala reference exactly as badly as noise does.
# Being able to express "this is Dholuo" is worth nothing when no cell may contain Dholuo.
# Closed-set is not a limitation here; it is the correct prior, and it is now enforced below.
#
# What replaces it is a router chosen by measurement rather than by argument: see
# kaggle/kernels/router/, which scores our own corpus-tuned ASR checkpoints and Whisper's language
# head against labelled phase-1 test audio and writes the winner's map. Mount it as waxal-router.
n_lid = len(needed_ids) - n_prefix
if n_lid:
    print(f"*** {n_lid:,} ids carry no language ({100*n_lid/len(needed_ids):.0f}%) -> "
          f"{LID_MODEL} decides their decoder. This stage is still CLOSED-SET over "
          f"{{lin, sna, lug}} while stage 1 is open-set, so do NOT check the mix below against "
          f"~44/41/15 and do NOT assume it beats stage 1 on phase 2 — see the note above.")

# An explicit language column, if a future Test csv ever carries one, overrides the prefix.
for fname in ("Test.csv", "Test_phase2.csv"):
    p = ZINDI_DIR / fname
    if not p.exists():
        continue
    df = pd.read_csv(p, escapechar="\\")
    cid, clang = guess_col(df, "id", "audio_id", "utt_id"), guess_col(df, "language", "lang")
    if clang:
        for i, l in zip(df[cid].astype(str), df[clang].astype(str)):
            known_lang[i] = l.strip().lower()[:3]

zip_path = WORK / "phase2_audio.zip"
if not zip_path.exists():
    os.system(f"wget -q -O {zip_path} {PHASE2_URL}")
if zip_path.exists() and zip_path.stat().st_size > 0:
    import soundfile as sf, librosa
    # The zip is not purely audio — the LID probe crashed on a member soundfile reported as
    # "Format not recognised". This loop happened to survive it because a junk member's stem is
    # not in `needed`, but that is luck, not a guard: one corrupt member whose name DOES match a
    # phase-2 id would take down a submission run at the last step. Catch per member and let the
    # id fall through to the HF fallback below, then to BLANK_FILL. Never abort the whole run for
    # one unreadable clip.
    bad = []
    with zipfile.ZipFile(zip_path) as zf:
        for n in [x for x in zf.namelist() if not x.endswith("/")]:
            stem = Path(n).stem
            if stem not in needed:
                continue
            try:
                with zf.open(n) as fh:
                    w, sr = sf.read(fh, dtype="float32")
            except Exception as e:  # noqa: BLE001
                bad.append(f"{n}: {type(e).__name__}")
                continue
            if w.ndim > 1:
                w = w.mean(axis=1)
            if sr != 16000:
                w = librosa.resample(w, orig_sr=sr, target_sr=16000)
            audio_store[stem] = w.astype(np.float32)
    if bad:
        print(f"  {len(bad):,} zip member(s) would not decode, e.g. {bad[:5]}\n"
              f"  those ids fall through to the HuggingFace fallback below")
print(f"phase 2 clips: {len(audio_store):,}")

missing = needed - set(audio_store)
if missing:
    for lang, cfg in HF_CONFIGS.items():
        ds = load_dataset("google/WaxalNLP", cfg, split="test", streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        # RULES GUARD: never read labels from the test split.
        ds = ds.remove_columns([c for c in ("transcription", "text") if c in ds.column_names])
        for row in ds:
            rid = str(row["id"])
            if rid in missing and rid not in audio_store:
                audio_store[rid] = decode_audio_cell(row["audio"])
                known_lang.setdefault(rid, lang)
print(f"resolved {len(audio_store):,} / {len(needed):,}")


# ---------------------------------------------------------------- 5. language ID
# Every cell of every reference is lin, sna or lug, so a routing decision outside those three is
# always wrong no matter how right it is about the audio. Discard such labels and let the
# closed-set LID below re-decide those clips. Re-measuring beats remapping: a hand-written
# "luo -> lug, guz -> sna" neighbour table would be my guess about Bantu proximity dressed up as a
# rule, and this path already owns a model that answers the question directly.
_off_set = [i for i in needed_ids
            if i in audio_store and known_lang.get(i) is not None and known_lang[i] not in LANGS]
if _off_set:
    print(f"routing map named a language outside {LANGS} on {len(_off_set):,} clip(s) "
          f"({dict(pd.Series([known_lang[i] for i in _off_set]).value_counts().head(8))}) — "
          f"re-deciding those closed-set, since no reference cell can contain them")
    for i in _off_set:
        known_lang.pop(i, None)

unknown = [i for i in needed_ids if i in audio_store and i not in known_lang]

# WAXAL_LANG_MAP overrides the LID model for clips that carry no language of their own.
#
# Why this exists. mms-lid-256 routes phase 2 to 94% Luganda. It does that with the
# attention_mask correctly supplied, so the padding bug noted above is not the cause — the
# model simply calls it that way, and its lug recall of exactly 1.000 on labelled audio is
# the class-bias signature. The public leaderboard refutes the 94% claim outright: if phase 2
# really were 94% Luganda then our submitted file, which decoded 87.8% Luganda, was already
# routed nearly right, and inverting 0.4919 = a*s + (1-a)*f at a = 0.8367 puts its
# perfect-routing CEILING at 0.5685 — below a score six other teams have posted on these same
# clips. A ceiling cannot sit under an observed floor. See scripts/anchor_calibration.py.
#
# The surviving hypothesis is the CTC-confidence router (scripts + kaggle/kernels/router),
# which is architecturally independent of the MMS LID family, has balanced per-class recalls
# of 0.9525/0.9800/0.9650, and calls phase 2 lin 36.9 / sna 15.5 / lug 47.5. Its measured
# agreement with the submitted file, 0.5680, matches the routing accuracy independently
# implied by the leaderboard arithmetic (0.51-0.59). Two separate routes, one number.
#
# Routing and decoding are separable, so a map produced by one set of models is a legitimate
# input to a decode by another. Ids absent from the map fall through to the LID as before.
_map_path = os.environ.get("WAXAL_LANG_MAP", "")
if unknown and _map_path and Path(_map_path).exists():
    _ext = json.loads(Path(_map_path).read_text(encoding="utf-8"))
    _bad = sorted({v for v in _ext.values() if v not in LANGS})
    if _bad:
        raise SystemExit(f"WAXAL_LANG_MAP names {_bad}, outside {LANGS} — those clips would be "
                         f"dropped to BLANK_FILL. Fix the map.")
    _hit = [i for i in unknown if i in _ext]
    for i in _hit:
        known_lang[i] = _ext[i]
    print(f"WAXAL_LANG_MAP {Path(_map_path).name}: routed {len(_hit):,} / {len(unknown):,} "
          f"unlabelled clip(s); {len(unknown) - len(_hit):,} fall through to {LID_MODEL}")
    print("  map mix: " + str(dict(pd.Series([_ext[i] for i in _hit]).value_counts())))
    unknown = [i for i in unknown if i not in _ext]

if unknown:
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
    fe = AutoFeatureExtractor.from_pretrained(LID_MODEL)
    lid = Wav2Vec2ForSequenceClassification.from_pretrained(LID_MODEL).to(DEVICE).eval().half()
    allowed = [i for i, l in lid.config.id2label.items() if l in LANGS]
    with torch.inference_mode():
        for k in range(0, len(unknown), 8):
            chunk = unknown[k:k + 8]
            inp = fe([audio_store[i][:16000 * 30] for i in chunk],
                     sampling_rate=16000, return_tensors="pt", padding=True)
            # attention_mask is NOT optional. mms-lid-256 has feat_extract_norm="layer" and
            # return_attention_mask=True, and the sequence-classification head mean-pools over
            # time — without the mask, zero padding in a variable-length batch is pooled in as
            # signal and the argmax collapses onto one class. Stage 1 shipped a 94%-Luganda
            # phase 2 routing because of this exact omission. Stage 1 also calibrates LID
            # against the phase-1 prefixes; read that number before trusting this block.
            sub_logits = lid(inp.input_values.to(DEVICE).half(),
                             attention_mask=inp.attention_mask.to(DEVICE)
                             ).logits.float()[:, allowed]
            for i, p in zip(chunk, sub_logits.argmax(-1).cpu().numpy()):
                known_lang[i] = lid.config.id2label[allowed[int(p)]]
    del lid; gc.collect(); torch.cuda.empty_cache()
print(pd.Series([known_lang.get(i, "??") for i in needed_ids]).value_counts().to_string())


# ---------------------------------------------------------------- 6. decode + submit
decoders = {}
for lang in LANGS:
    t = tuned.get(lang)
    decoders[lang] = make_decoder(lang, t["alpha"], t["beta"]) if t and t["alpha"] is not None else None

# DECODE_LANGS used to extend past LANGS so that clips stage 1 routed to luo/nyn/xog/kam/kin could
# be decoded with those MMS adapters. That is gone: every reference cell is lin, sna or lug, so
# emitting Dholuo is not "the language it is actually in", it is an unmatchable string. Section 5
# now strips off-set labels and re-decides them closed-set, so this should be a no-op — the check
# stays because a silent regression here is worth a quarter of a point, and BLANK_FILL would hide
# it as a merely-poor score rather than a broken one.
DECODE_LANGS = list(LANGS)
_stray = sorted({l for i in needed_ids if (l := known_lang.get(i)) and l not in LANGS})
if _stray:
    raise SystemExit(f"routing still names {_stray} after the closed-set pass in section 5 — "
                     f"those clips would be dropped to BLANK_FILL. Fix the routing, do not "
                     f"widen DECODE_LANGS: no reference cell can contain them.")

preds = {}
with multiprocessing.get_context("fork").Pool(os.cpu_count()) as pool:
    for lang in DECODE_LANGS:
        ids = [i for i in needed_ids if known_lang.get(i) == lang and i in audio_store]
        if not ids:
            continue
        # Re-arm the adapter for THIS language. `decoders` was built in a loop above, so the
        # model is currently holding whichever language that loop finished on; without this the
        # first language decoded here would run its audio through the last language's adapter.
        set_language(lang)
        dec = decoders.get(lang)      # .get: DECODE_LANGS extends past the keys built above
        if BACKEND == "whisper":
            # set_language() has just swapped BACKEND to this language's kind. Whisper is seq2seq:
            # there are no frame logits, so no CTC beam and no KenLM — drop the decoder rather
            # than hand decode_batch() something it cannot read.
            dec = None
        mode = ("seq2seq" if BACKEND == "whisper" else
                "beam+LM" if dec else "greedy" if lang in LANGS else "greedy (no KenLM)")
        print(f"\ndecoding {lang}: {len(ids):,} clips  ({mode})")
        # Length-sorted so padded batches stay tight; roughly halves GPU time on skewed lengths.
        ids.sort(key=lambda i: len(audio_store[i]))

        # GPU forward stays at 4 (a 40s clip through a 600M model is what sizes T4 VRAM), but
        # logits are buffered so the CPU beam search runs over a batch worth parallelising.
        buf_ids: list[str] = []
        buf_logits: list[np.ndarray] = []

        def flush() -> None:
            if not buf_ids:
                return
            if dec:
                texts = dec.decode_batch(pool, buf_logits, beam_width=BEAM_WIDTH)
            else:
                texts = [processor.tokenizer.decode(l.argmax(-1)) for l in buf_logits]
            for i, t in zip(buf_ids, texts):
                preds[i] = finish(t, lang)
            buf_ids.clear()
            buf_logits.clear()

        step = 8 if BACKEND == "whisper" else 4
        for k in range(0, len(ids), step):
            chunk = ids[k:k + step]
            wavs = [audio_store[i][:16000 * MAX_SECONDS] for i in chunk]
            if BACKEND == "whisper":
                for i, t in zip(chunk, transcribe_whisper(wavs)):
                    preds[i] = finish(t, lang)
                if (k // step) % 25 == 0:
                    print(f"  {k}/{len(ids)}", flush=True)
                continue
            buf_logits.extend(logits_for(wavs))
            buf_ids.extend(chunk)
            if len(buf_ids) >= 64:
                flush()
            if k % 400 == 0:
                print(f"  {k}/{len(ids)}")
        flush()

# One file per template, each keeping that template's own row order and column names.
# Upload the one matching the phase that is currently open.
SUFFIX = {"SampleSubmission.csv": "phase1", "Test_phase2.csv": "phase2"}
for fname, df, tid, ttxt in TEMPLATES:
    sub = df.copy()
    # Count the misses BEFORE filling, or the diagnostic below reports zero and we lose the one
    # signal that says the audio for those ids never arrived.
    mapped = sub[tid].astype(str).map(preds).fillna("")
    blank = int((mapped.str.strip() == "").sum())
    # Fill with a single character rather than leaving the cell empty. Metric-neutral — an empty
    # hypothesis against an N-word reference is N deletions, and "a" is N-1 deletions plus one
    # substitution, the same N errors — but it means no cell reads back as NaN in a parser that
    # treats an empty field as missing. Same BLANK_FILL as stage 1, deliberately.
    sub[ttxt] = mapped.replace(r"^\s*$", BLANK_FILL, regex=True)
    # Name the file after the lineup, via RUN_TAG frozen at import. NOT off BACKEND: that is now
    # per-language and mutates on every set_language(), so by the time this line runs it holds
    # whichever language happened to decode last — a mixed run would be named after an accident.
    out = WORK / f"submission_03_{RUN_TAG}_lm_{SUFFIX.get(fname, Path(fname).stem)}.csv"
    sub.to_csv(out, index=False)

    langs = pd.Series([known_lang.get(i, "??") for i in sub[tid].astype(str)]).value_counts()
    print(f"\nwrote {out}")
    print(f"  rows={len(sub):,}  blank={blank:,} ({100*blank/len(sub):.1f}%)")
    print(f"  language mix: {langs.to_dict()}")
    if blank:
        # On phase 2 this almost always means the audio zip did not contain that id.
        print(f"  WARNING: {blank:,} ids got no prediction. They carry BLANK_FILL and score as a "
              f"total miss on those rows — the fill is cosmetic, it does not recover anything.")
    print(sub.head(5).to_string())
