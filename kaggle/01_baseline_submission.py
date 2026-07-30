"""
STAGE 1 — get a real score on the board today, with zero training.

Run on Kaggle with GPU (T4 x2 or P100), Internet ON. ~1.5 GPU-hours.

Strategy: facebook/mms-1b-all already ships CTC adapters for lin / sna / lug. On this exact
corpus the WAXAL-NET paper measured it zero-shot at 44.7 / 36.9 / 32.1 WER — worse than a
fine-tune, but a genuine mid-pack score and, more importantly, it exercises the entire
pipeline (audio resolution -> language ID -> decode -> submission format) before we spend
16 GPU-hours training. Never let the first test of your submission plumbing be the run that
matters.

Upload data/zindi/*.csv as a Kaggle Dataset and point ZINDI_DIR at it.

Rules note: the HF `test` split carries ground-truth `transcription`. We load audio from it
and DROP that column on sight. Using it would be an explicit disqualification.
"""

import gc
import json
import os
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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

PHASE2_URL = "https://storage.googleapis.com/waxalphase2/audio.zip"

LANGS = ["lin", "sna", "lug"]                           # Lingala, Shona, Luganda
HF_CONFIGS = {"lin": "lin_asr", "sna": "sna_asr", "lug": "lug_asr"}
# WAXAL uses ISO 639-3; MMS adapter names happen to match for these three.
MMS_ADAPTER = {"lin": "lin", "sna": "sna", "lug": "lug"}
# mms-lid-256 label space is also ISO 639-3.
LID_MODEL = "facebook/mms-lid-256"
ASR_MODEL = "facebook/mms-1b-all"

SEED = 1337
BATCH_AUDIO_SECONDS = 60          # chunk long clips; MMS is a CTC model, no length limit but VRAM is
MAX_CLIP_SECONDS = 40

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEVICE}  torch={torch.__version__}")


# ---------------------------------------------------------------- helpers
def guess_col(df, *candidates):
    lowered = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    for cand in candidates:
        for low, orig in lowered.items():
            if cand in low:
                return orig
    return None


# Measured, not guessed. The organisers' starter notebook scores with
# jiwer.wer([r.lower()...], [p.lower()...]): lowercase both sides, punctuation untouched.
LOWERCASE = True
KEEP_PUNCT = set(".,'’-;:!?")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).strip()
    if LOWERCASE:
        text = text.lower()
    text = "".join(c if (c.isalpha() or c.isspace() or c in KEEP_PUNCT) else " " for c in text)
    return re.sub(r"\s+", " ", text).strip()


def lang_from_id(utt_id: str) -> str | None:
    """Phase 1 ids are `<iso3>_<number>` (lug_96114), so language is exact and free. Phase 2
    may strip it, so LID below stays as the fallback."""
    prefix = str(utt_id).split("_", 1)[0].strip().lower()
    return prefix if prefix in LANGS else None


# ---------------------------------------------------------------- 1. what do we need to predict
# Two submission templates, disjoint sets with different id conventions:
#   SampleSubmission.csv  4,253 rows, `lug_96114`  -> phase 1, language readable from the id
#   Test_phase2.csv       1,500 rows, `ID_TBDTM`   -> phase 2, no language anywhere
# Measured 30 Jul: zero id overlap. Predict the union, write one file per template, so whichever
# phase is open we have a correctly-shaped file and never have to guess which one Zindi wants.
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

known_lang = {}   # id -> iso3
for i in needed_ids:                      # id prefix: exact, free, no model
    lg = lang_from_id(i)
    if lg:
        known_lang[i] = lg
print(f"language from id prefix: {len(known_lang):,} / {len(needed):,}")

# If a Test csv ever ships an explicit language column it beats the prefix. Phase 1's does not,
# and Phase 2's is `ID,Target` only — this loop is insurance, not the mechanism.
for fname in ("Test.csv", "Test_phase2.csv"):
    p = ZINDI_DIR / fname
    if not p.exists():
        continue
    df = pd.read_csv(p, escapechar="\\")
    cid = guess_col(df, "id", "audio_id", "utt_id", "filename")
    clang = guess_col(df, "language", "lang", "locale")
    print(f"{fname}: {len(df):,} rows, id={cid}, language={clang or 'NONE'}")
    if clang:
        for i, l in zip(df[cid].astype(str), df[clang].astype(str)):
            known_lang[i] = l.strip().lower()[:3]

# Phase 2 ids are `ID_` + 5 uniformly-random uppercase letters (letter frequencies 0.036-0.043
# against a uniform 0.0385, all 26 used). No language, nothing to exploit. So on the set that
# actually decides the prize, LID is not a fallback that never fires — it decides every clip.
n_lid = len(needed) - len(known_lang)
print(f"language known for {len(known_lang):,} / {len(needed):,} ids"
      f"   ({n_lid:,} -> {LID_MODEL})")


# ---------------------------------------------------------------- audio decoding
# datasets >= 4 decodes Audio columns through torchcodec, which pins against specific torch
# builds and needs FFmpeg present. On a machine we don't control that is a dependency we can
# lose a run to — it raised ImportError on Lightning with torch 2.8. We already decode the
# phase 2 zip with soundfile, so ask datasets for raw bytes (Audio(decode=False)) and use the
# same decoder for both paths. One less moving part, and identical resampling either way.
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


# ---------------------------------------------------------------- 2. resolve audio
audio_store: dict[str, np.ndarray] = {}

# --- 2a. Phase 2 audio: a plain zip of waveforms, no metadata at all.
zip_path = WORK / "phase2_audio.zip"
if not zip_path.exists():
    print("downloading phase 2 audio ...")
    os.system(f"wget -q -O {zip_path} {PHASE2_URL}")
if zip_path.exists() and zip_path.stat().st_size > 0:
    import soundfile as sf
    import librosa

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        print(f"phase 2 zip: {len(names):,} entries")
        for n in names:
            stem = Path(n).stem
            if stem not in needed:
                continue
            with zf.open(n) as fh:
                wav, sr = sf.read(fh, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            audio_store[stem] = wav.astype(np.float32)
    print(f"loaded {len(audio_store):,} phase-2 clips")

# --- 2b. Phase 1 audio: lives in the HF `test` split. Audio only.
still_missing = needed - set(audio_store)
if still_missing:
    from datasets import Audio, load_dataset

    print(f"{len(still_missing):,} ids unresolved -> pulling audio from HF test splits")
    for lang, cfg in HF_CONFIGS.items():
        ds = load_dataset("google/WaxalNLP", cfg, split="test", streaming=True)
        # === RULES GUARD: the labels in this split may never be read. ===
        ds = ds.remove_columns([c for c in ("transcription", "text") if c in ds.column_names])
        ds = ds.cast_column("audio", Audio(decode=False))      # see decode_audio_cell()
        hits = 0
        for row in ds:
            rid = str(row["id"])
            if rid in still_missing and rid not in audio_store:
                audio_store[rid] = decode_audio_cell(row["audio"])
                known_lang.setdefault(rid, lang)
                hits += 1
        print(f"  {cfg}: matched {hits:,}")
        gc.collect()

missing = needed - set(audio_store)
print(f"\nresolved {len(audio_store):,} / {len(needed):,} clips  (missing {len(missing):,})")
if missing:
    print("  WARNING: unresolved ids will be submitted blank. Sample:", list(missing)[:5])


# ---------------------------------------------------------------- 3. language ID where unknown
unknown = [i for i in needed_ids if i in audio_store and i not in known_lang]
if unknown:
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

    print(f"\nrunning LID on {len(unknown):,} clips ({LID_MODEL})")
    fe = AutoFeatureExtractor.from_pretrained(LID_MODEL)
    lid = Wav2Vec2ForSequenceClassification.from_pretrained(LID_MODEL).to(DEVICE).eval().half()
    id2label = lid.config.id2label
    # Restrict to the three competition languages: an unconstrained argmax over 256 languages
    # will happily emit `swh` or `nya` for a Bantu clip and route it to the wrong decoder.
    allowed_idx = [i for i, l in id2label.items() if l in LANGS]
    assert allowed_idx, f"none of {LANGS} in LID label space"
    print(f"  constraining argmax to {[id2label[i] for i in allowed_idx]}")

    with torch.inference_mode():
        for k in range(0, len(unknown), 8):
            chunk = unknown[k:k + 8]
            wavs = [audio_store[i][:16000 * 30] for i in chunk]
            inp = fe(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
            logits = lid(inp.input_values.to(DEVICE).half()).logits.float()
            sub = logits[:, allowed_idx]
            picks = sub.argmax(-1).cpu().numpy()
            for i, p in zip(chunk, picks):
                known_lang[i] = id2label[allowed_idx[int(p)]]
            if k % 400 == 0:
                print(f"  lid {k}/{len(unknown)}")
    del lid
    gc.collect()
    torch.cuda.empty_cache()

dist = pd.Series([known_lang.get(i, "??") for i in needed_ids]).value_counts()
print(f"\nlanguage distribution over submission ids:\n{dist.to_string()}")


# ---------------------------------------------------------------- 4. transcribe, one adapter at a time
from transformers import AutoProcessor, Wav2Vec2ForCTC

processor = AutoProcessor.from_pretrained(ASR_MODEL)
model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL, torch_dtype=torch.float16).to(DEVICE).eval()

preds: dict[str, str] = {}
for lang in LANGS:
    ids = [i for i in needed_ids if known_lang.get(i) == lang and i in audio_store]
    if not ids:
        continue
    print(f"\n--- {lang}: {len(ids):,} clips ---")
    processor.tokenizer.set_target_lang(MMS_ADAPTER[lang])
    model.load_adapter(MMS_ADAPTER[lang])
    model.to(DEVICE).eval()

    # Sort by length so padded batches stay tight — roughly 2x throughput on skewed lengths.
    ids.sort(key=lambda i: len(audio_store[i]))
    B = 4
    with torch.inference_mode():
        for k in range(0, len(ids), B):
            chunk = ids[k:k + B]
            wavs = [audio_store[i][:16000 * MAX_CLIP_SECONDS] for i in chunk]
            inp = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
            logits = model(
                inp.input_values.to(DEVICE).half(),
                attention_mask=inp.attention_mask.to(DEVICE) if "attention_mask" in inp else None,
            ).logits
            for i, row in zip(chunk, logits.argmax(-1).cpu()):
                preds[i] = normalise(processor.decode(row))
            if k % (B * 50) == 0:
                print(f"  {k}/{len(ids)}")
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- 5. write submission
# One file per template, each keeping that template's own row order and column names.
# Upload the one matching the phase that is currently open.
SUFFIX = {"SampleSubmission.csv": "phase1", "Test_phase2.csv": "phase2"}
for fname, df, tid, ttxt in TEMPLATES:
    sub = df.copy()
    sub[ttxt] = sub[tid].astype(str).map(preds).fillna("")
    out = WORK / f"submission_01_mms_zeroshot_{SUFFIX.get(fname, Path(fname).stem)}.csv"
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

# Keep the language decisions — stage 3 reuses them instead of re-running LID.
json.dump(known_lang, open(WORK / "lang_map.json", "w"))
print("wrote lang_map.json")
