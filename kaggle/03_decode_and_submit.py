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
LM_CORPUS_DIR = ART("waxal-lm") / "lm_corpus"             # stage 0 output
PHASE2_URL = "https://storage.googleapis.com/waxalphase2/audio.zip"

# Preflight. The model does not load until after ~5 GB of phase-2 audio has downloaded and been
# decoded, so a wrong or unattached checkpoint mount otherwise announces itself half an hour into
# a GPU session. Check the inputs first — this costs a stat() call.
#
# Only the w2vbert backend needs a checkpoint; the default MMS backend pulls its weights from the
# Hub. Read the env directly because BACKEND is defined in section 2, and this check has to run
# before the download, not after it.
if (os.environ.get("WAXAL_BACKEND", "mms").lower() not in ("mms", "waxalnet")
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
        if lines:
            return lines, "external (stage 0)"

    train_csv = pd.read_csv(ZINDI_DIR / "Train.csv", escapechar="\\")
    txt_col = guess_col(train_csv, "transcription", "transcript", "text", "target", "sentence")
    lng_col = guess_col(train_csv, "language", "lang", "locale")
    rows = (train_csv[train_csv[lng_col].astype(str).str.lower().str[:3] == lang]
            if lng_col else train_csv)
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
# That is almost certainly what the leaders are running, and it says the way past them is
# punctuation, not a bigger acoustic model. See local/harness/ for how all of this was measured.
BACKEND = os.environ.get("WAXAL_BACKEND", "mms").lower()
ASR_MODEL = os.environ.get("WAXAL_ASR_MODEL", "facebook/mms-1b-all")
MMS_ADAPTER = {"lin": "lin", "sna": "sna", "lug": "lug"}

# One checkpoint per language, not one model with adapters — so these swap as whole models.
WAXALNET = {
    "lin": os.environ.get("WAXAL_LIN", "waxal-benchmarking/mms-300m-waxal-lin"),
    "sna": os.environ.get("WAXAL_SNA", "waxal-benchmarking/mms-300m-waxal-sna"),
    "lug": os.environ.get("WAXAL_LUG", "waxal-benchmarking/mms-300m-waxal-lug"),
}

from pyctcdecode import build_ctcdecoder

if BACKEND == "waxalnet":
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    _wn_cache: dict[str, tuple] = {}

    def _wn_load(lang: str):
        """Load (and cache) one language's WAXALNet checkpoint.

        Cached rather than reloaded because the decode loop groups by language but the phase-2
        LID pass can interleave them; three mms-300m in fp16 is ~1.9 GB, which a T4 holds
        comfortably next to the beam search.
        """
        if lang not in _wn_cache:
            repo = WAXALNET.get(lang) or WAXALNET["lug"]
            print(f"  loading {lang}: {repo}", flush=True)
            pr = AutoProcessor.from_pretrained(repo)
            md = Wav2Vec2ForCTC.from_pretrained(repo, torch_dtype=torch.float16).to(DEVICE).eval()
            _wn_cache[lang] = (pr, md)
        return _wn_cache[lang]

    processor, model = _wn_load("lin")
    print(f"backend=waxalnet  { {k: v.split('/')[-1] for k, v in WAXALNET.items()} }")
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
    global _active_lang, labels, processor, model
    if lang == _active_lang:
        return
    if BACKEND == "waxalnet":
        # Whole-model swap, not an adapter swap. Each WAXALNet checkpoint carries its OWN vocab
        # (lin 72 tokens, sna 51, lug 38 — measured), so `labels` must be rebuilt from the new
        # processor before any decoder is constructed, for exactly the reason documented below.
        processor, model = _wn_load(lang if lang in WAXALNET else "lug")
        labels = _labels_from_tokenizer()
        _active_lang = lang
        return
    if BACKEND != "mms":
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


def logits_for(wavs: list[np.ndarray]) -> list[np.ndarray]:
    inp = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
    # MMS and WAXALNet are raw-waveform wav2vec2 (input_values); w2v-bert takes mel features.
    feats = inp.input_values if BACKEND in ("mms", "waxalnet") else inp.input_features
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
    for row in ds:
        w = decode_audio_cell(row["audio"])
        if not (1.0 * 16000 <= len(w) <= MAX_SECONDS * 16000):
            continue
        wavs.append(w)
        refs.append(normalise(row["transcription"]))
        if len(wavs) >= N_TUNE:
            break

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
    want = {it["id"]: it for it in dev_items}
    print(f"\n=== DEV MODE === {len(want):,} clips, seed {dev['seed']}, mix {dev['actual_mix']}")

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
        for lang, cfg in HF_CONFIGS.items():
            need = {i for i, it in want.items() if it["language"] == lang}
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
        print(f"resolved {len(dev_audio):,}/{len(want):,} dev clips -> cached {dev_cache.name}")

    dev_dec = {}
    for lang in LANGS:
        t = tuned.get(lang)
        dev_dec[lang] = make_decoder(lang, t["alpha"], t["beta"]) \
            if t and t["alpha"] is not None else None

    dev_pred = {}
    with _mp.get_context("fork").Pool(os.cpu_count()) as pool:
        for lang in LANGS:
            ids = [i for i in want if want[i]["language"] == lang and i in dev_audio]
            if not ids:
                continue
            set_language(lang)
            dec = dev_dec.get(lang)
            print(f"\ndecoding dev/{lang}: {len(ids):,} clips ({'beam+LM' if dec else 'greedy'})")
            ids.sort(key=lambda i: len(dev_audio[i]))
            bi, bl = [], []

            def _flush():
                if not bi:
                    return
                txt = (dec.decode_batch(pool, bl, beam_width=BEAM_WIDTH) if dec
                       else [processor.tokenizer.decode(l.argmax(-1)) for l in bl])
                for i, t_ in zip(bi, txt):
                    dev_pred[i] = normalise(t_)
                bi.clear()
                bl.clear()

            for k in range(0, len(ids), 4):
                ch = ids[k:k + 4]
                bl.extend(logits_for([dev_audio[i][:16000 * MAX_SECONDS] for i in ch]))
                bi.extend(ch)
                if len(bi) >= 64:
                    _flush()
            _flush()

    ok = [i for i in want if i in dev_pred]
    refs = [want[i]["reference"] for i in ok]
    hyps = [dev_pred[i] for i in ok]
    lgs = [want[i]["language"] for i in ok]
    res = HARNESS.report(refs, hyps, lgs, title=f"DEV — backend={BACKEND}")

    # The single most valuable extra number here: what a trailing full stop is worth. These CTC
    # vocabs emit no sentence punctuation at all, and on the dev references '.' alone accounts for
    # 65 marks per 1,000 reference words. Measured, not assumed — if it does not help, we see that.
    res["plus_period"] = HARNESS.score(refs, [h + "." if h.strip() else h for h in hyps]).multi
    print(f"  + trailing '.' on every hypothesis -> multi={res['plus_period']:.4f} "
          f"(delta {res['plus_period'] - res['per_language']['overall']['multi']:+.4f})")

    res["backend"] = BACKEND
    res["models"] = WAXALNET if BACKEND == "waxalnet" else ASR_MODEL
    res["n_decoded"] = len(ok)
    HARNESS.save(res, WORK / f"dev_result_{BACKEND}.json")
    json.dump({i: dev_pred[i] for i in ok}, open(WORK / f"dev_preds_{BACKEND}.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"\nwrote {WORK / f'dev_result_{BACKEND}.json'}")
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
_lang_map = ART("waxal-baseline") / "lang_map.json"       # stage 1 wrote it; reuse, don't re-LID
if _lang_map.exists():
    known_lang = json.load(open(_lang_map))

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
# KNOWN GAP — this stage still routes CLOSED-SET, and stage 1 no longer does. Unconstrained,
# mms-lid-256 calls the phase-2 clips luo/nyn/lug/kin/kam/xog (commit e9b3885,
# local/diagnose_lid_unconstrained.py). This stage cannot simply copy stage 1's open-set fix,
# because what it decodes with — a w2v-bert fine-tuned on lin/sna/lug transcripts, plus one KenLM
# per those three languages — has no way to emit Dholuo or Runyankole at all. Fixing it properly
# means splitting the stage: fine-tuned model + KenLM for phase 1 and the lug slice of phase 2,
# MMS adapters for the rest. Until that lands, stage 1's phase-2 file is the better one to upload
# and this stage's phase-2 output should be treated as phase-1-quality only.
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
    with zipfile.ZipFile(zip_path) as zf:
        for n in [x for x in zf.namelist() if not x.endswith("/")]:
            stem = Path(n).stem
            if stem not in needed:
                continue
            with zf.open(n) as fh:
                w, sr = sf.read(fh, dtype="float32")
            if w.ndim > 1:
                w = w.mean(axis=1)
            if sr != 16000:
                w = librosa.resample(w, orig_sr=sr, target_sr=16000)
            audio_store[stem] = w.astype(np.float32)
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
unknown = [i for i in needed_ids if i in audio_store and i not in known_lang]
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

# Which languages do we actually have to decode? LANGS is the set we hold KenLMs for, but stage 1
# routes OPEN-SET: on phase 2 its lang_map.json can name luo/nyn/xog/kam/kin, none of which are in
# LANGS. Iterating LANGS alone silently skips those clips and hands them to BLANK_FILL — a
# guaranteed total miss on every one, on the split that decides the prize.
#
# This is the "KNOWN GAP" flagged further up, and dropping w2v-bert is what closes it. That note
# was right that a model fine-tuned on lin/sna/lug transcripts can never emit Dholuo; mms-1b-all
# ships adapters for those languages, so each clip decodes in the language it is actually in.
# There is no KenLM for them, so they decode greedily — worse than lin/sna/lug, vastly better
# than a blank row.
_routed = [l for l in dict.fromkeys(known_lang.get(i) for i in needed_ids) if l]
DECODE_LANGS = LANGS + [l for l in _routed if l not in LANGS]
if len(DECODE_LANGS) > len(LANGS):
    print(f"\nopen-set routing sent clips to {DECODE_LANGS[len(LANGS):]} — decoding each with "
          f"its own MMS adapter, greedily (no KenLM for those).")

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
        print(f"\ndecoding {lang}: {len(ids):,} clips  "
              f"({'beam+LM' if dec else 'greedy' if lang in LANGS else 'greedy (no KenLM)'})")
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
                preds[i] = normalise(t)
            buf_ids.clear()
            buf_logits.clear()

        for k in range(0, len(ids), 4):
            chunk = ids[k:k + 4]
            buf_logits.extend(logits_for([audio_store[i][:16000 * MAX_SECONDS] for i in chunk]))
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
    # Name the file after the backend that produced it. Hardcoding "w2vbert" here was fine when
    # there was one backend; with two it is how the wrong CSV gets uploaded to Zindi.
    out = WORK / f"submission_03_{BACKEND}_lm_{SUFFIX.get(fname, Path(fname).stem)}.csv"
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
