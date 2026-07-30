"""
STAGE 3 — the biggest single win: KenLM shallow fusion + beam search, then submit.

Run on Kaggle GPU, Internet ON. ~2 GPU-hours. REQUIRES stage 0 to have run first.

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
comes from `kaggle/00_build_lm_corpus.py` (mounted as the `waxal-lm` Dataset), with the Zindi
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
    ZINDI_DIR = Path("/kaggle/input/waxal-zindi")         # <- the Dataset you uploaded
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

CKPT = ART("waxal-ckpt") / "w2vbert-waxal"                # stage 2 output
LM_CORPUS_DIR = ART("waxal-lm") / "lm_corpus"             # stage 0 output
PHASE2_URL = "https://storage.googleapis.com/waxalphase2/audio.zip"

LANGS = ["lin", "sna", "lug"]
HF_CONFIGS = {"lin": "lin_asr", "sna": "sna_asr", "lug": "lug_asr"}
LID_MODEL = "facebook/mms-lid-256"
NGRAM_ORDER = 5
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


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).strip()
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
KENLM = WORK / "kenlm"
LMPLZ = KENLM / "build" / "bin" / "lmplz"
BUILD_BINARY = KENLM / "build" / "bin" / "build_binary"
if not LMPLZ.exists():
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
assert LMPLZ.exists(), (
    f"KenLM did not build at {LMPLZ}. Without it there is no shallow fusion and no ~59% WER "
    f"win — stop and fix this rather than falling through to greedy decoding."
)
os.system("pip -q install pyctcdecode https://github.com/kpu/kenlm/archive/master.zip")

LM_DIR = WORK / "lm"
LM_DIR.mkdir(exist_ok=True)
lm_paths, unigram_sets = {}, {}


def corpus_lines(lang: str) -> tuple[list[str], str]:
    """External corpus from stage 0 if it is mounted, else the thin Train.csv fallback."""
    ext = LM_CORPUS_DIR / f"{lang}.txt"
    if ext.exists():
        lines = [l for l in ext.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            return lines, "external (stage 0)"

    train_csv = pd.read_csv(ZINDI_DIR / "Train.csv", escapechar="\\")
    txt_col = guess_col(train_csv, "transcription", "transcript", "text", "target", "sentence")
    lng_col = guess_col(train_csv, "language", "lang", "locale")
    rows = (train_csv[train_csv[lng_col].astype(str).str.lower().str[:3] == lang]
            if lng_col else train_csv)
    lines = [normalise(t) for t in rows[txt_col].dropna().astype(str)]
    return [l for l in lines if l], "Train.csv only (WEAK)"


for lang in LANGS:
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
from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor
from pyctcdecode import build_ctcdecoder

processor = Wav2Vec2BertProcessor.from_pretrained(CKPT)
model = Wav2Vec2BertForCTC.from_pretrained(CKPT, torch_dtype=torch.float16).to(DEVICE).eval()

vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab = [k for k, _ in sorted(vocab_dict.items(), key=lambda kv: kv[1])]
# pyctcdecode's label convention: blank is "", the word delimiter is a literal space.
labels = [("" if t == "[PAD]" else " " if t == "|" else t) for t in sorted_vocab]


def make_decoder(lang: str, alpha: float, beta: float):
    return build_ctcdecoder(
        labels,
        kenlm_model_path=lm_paths.get(lang),
        unigrams=unigram_sets.get(lang),
        alpha=alpha, beta=beta,
    )


def logits_for(wavs: list[np.ndarray]) -> list[np.ndarray]:
    inp = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.inference_mode():
        out = model(
            inp.input_features.to(DEVICE).half(),
            attention_mask=inp.attention_mask.to(DEVICE),
        ).logits.float().cpu().numpy()
    lens = inp.attention_mask.sum(-1).cpu().numpy()
    # Trim padding frames before decoding, or the LM scores a tail of silence.
    ratio = out.shape[1] / inp.attention_mask.shape[1]
    return [out[i, : max(1, int(lens[i] * ratio))] for i in range(len(wavs))]


# ---------------------------------------------------------------- 3. tune alpha/beta on validation
from datasets import Audio, load_dataset
import evaluate


# datasets >= 4 decodes Audio columns through torchcodec, which pins against specific torch
# builds and needs FFmpeg present. On a machine we don't control that is a dependency we can
# lose a run to — it raised ImportError on Lightning with torch 2.8. Ask datasets for raw bytes
# (Audio(decode=False)) and decode with soundfile, which we already depend on.
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
    ds = load_dataset("google/WaxalNLP", cfg, split="validation", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))       # see decode_audio_cell()
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
_lang_map = ART("waxal-ckpt") / "lang_map.json"           # stage 1 wrote it; reuse, don't re-LID
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
n_lid = len(needed_ids) - n_prefix
if n_lid:
    print(f"*** {n_lid:,} ids carry no language ({100*n_lid/len(needed_ids):.0f}%) -> "
          f"{LID_MODEL} decides their decoder. Check the distribution it produces below "
          f"against the ~44/41/15 lin/sna/lug split of the corpus; a wildly different "
          f"split means LID is misfiring and the submission is not trustworthy.")

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
        # Cast before remove_columns: the latter is a map on streaming datasets and freezes the
        # decoding formatter, which makes Audio(decode=False) a no-op.
        ds = ds.cast_column("audio", Audio(decode=False))   # see decode_audio_cell()
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
            sub_logits = lid(inp.input_values.to(DEVICE).half()).logits.float()[:, allowed]
            for i, p in zip(chunk, sub_logits.argmax(-1).cpu().numpy()):
                known_lang[i] = lid.config.id2label[allowed[int(p)]]
    del lid; gc.collect(); torch.cuda.empty_cache()
print(pd.Series([known_lang.get(i, "??") for i in needed_ids]).value_counts().to_string())


# ---------------------------------------------------------------- 6. decode + submit
decoders = {}
for lang in LANGS:
    t = tuned.get(lang)
    decoders[lang] = make_decoder(lang, t["alpha"], t["beta"]) if t and t["alpha"] is not None else None

preds = {}
with multiprocessing.get_context("fork").Pool(os.cpu_count()) as pool:
    for lang in LANGS:
        ids = [i for i in needed_ids if known_lang.get(i) == lang and i in audio_store]
        if not ids:
            continue
        dec = decoders[lang]
        print(f"\ndecoding {lang}: {len(ids):,} clips  "
              f"({'beam+LM' if dec else 'greedy (LM lost the sweep)'})")
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
    sub[ttxt] = sub[tid].astype(str).map(preds).fillna("")
    out = WORK / f"submission_03_w2vbert_lm_{SUFFIX.get(fname, Path(fname).stem)}.csv"
    sub.to_csv(out, index=False)

    blank = int((sub[ttxt].str.strip() == "").sum())
    langs = pd.Series([known_lang.get(i, "??") for i in sub[tid].astype(str)]).value_counts()
    print(f"\nwrote {out}")
    print(f"  rows={len(sub):,}  blank={blank:,} ({100*blank/len(sub):.1f}%)")
    print(f"  language mix: {langs.to_dict()}")
    if blank:
        # On phase 2 this almost always means the audio zip did not contain that id.
        print(f"  WARNING: {blank:,} ids got no prediction and will score as pure deletions.")
    print(sub.head(5).to_string())
