"""
STAGE 1 — get a real score on the board today, with zero training.

Where to run: any GPU box with internet. We run it on a Lightning Studio T4 (~1.5 GPU-hours);
Kaggle T4 x2 / P100 also works. The ART() block below makes both resolve their own storage.

Strategy: facebook/mms-1b-all already ships CTC adapters for lin / sna / lug. On this exact
corpus the WAXAL-NET paper measured it zero-shot at 44.7 / 36.9 / 32.1 WER — worse than a
fine-tune, but a genuine mid-pack score and, more importantly, it exercises the entire
pipeline (audio resolution -> language ID -> decode -> submission format) before we spend
16 GPU-hours training. Never let the first test of your submission plumbing be the run that
matters.

The Zindi CSVs come from data/zindi/ in this repo. On Lightning that path is read directly;
on Kaggle, upload them as a Dataset and point ZINDI_DIR at it.

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


PHASE2_URL = "https://storage.googleapis.com/waxalphase2/audio.zip"

LANGS = ["lin", "sna", "lug"]                           # Lingala, Shona, Luganda
HF_CONFIGS = {"lin": "lin_asr", "sna": "sna_asr", "lug": "lug_asr"}
# WAXAL uses ISO 639-3; MMS adapter names happen to match for these three.
MMS_ADAPTER = {"lin": "lin", "sna": "sna", "lug": "lug"}
# mms-lid-256 label space is also ISO 639-3.
LID_MODEL = "facebook/mms-lid-256"
ASR_MODEL = "facebook/mms-1b-all"

SEED = 1337
# What to write when CTC returns nothing. Metric-neutral (see the write-out below); this exists
# purely so no cell in the CSV reads back as NaN.
BLANK_FILL = "a"
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


# Train.csv uses ASCII ' throughout; MMS emits the curly U+2019 and the modifier U+02BC. On a
# character metric each of those is a guaranteed wrong character in every word carrying one, and
# 512 phase-1 rows carry one. Folding costs nothing if the scorer already normalises, and saves
# those characters if it doesn't.
APOSTROPHES = {"’": "'", "ʼ": "'", "‘": "'", "´": "'", "`": "'"}


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).strip()
    text = text.translate(str.maketrans(APOSTROPHES))
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
# requirements-gpu.txt pins datasets < 4 so Audio columns decode through soundfile, not
# torchcodec — the note at the bottom of that file explains why, and the guard at the top of this
# one enforces it. This helper is the single place audio becomes 16 kHz mono float32, and it
# takes either a decoded cell or raw bytes, so the HF path and the phase 2 zip path cannot
# drift apart.
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
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        # === RULES GUARD: the labels in this split may never be read. ===
        ds = ds.remove_columns([c for c in ("transcription", "text") if c in ds.column_names])
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
# ---------------------------------------------------------------- open-set routing
# Constraining LID's argmax to the three challenge languages is right for phase 1, whose ids
# carry lin_/sna_/lug_ prefixes and never reach LID at all. It is wrong for phase 2. Run the same
# model unconstrained over 40 sampled phase-2 clips (local/diagnose_lid_unconstrained.py) and it
# returns luo 42.5%, lug 27.5%, nyn 20%, guz/xog/kin/kam 2.5% each — zero Lingala, zero Shona, at
# confidences of 0.98-1.00. Phase 1 is ~44% lin / ~41% sna, so drawing zero of both in 40 clips is
# not sampling noise.
#
# A three-class argmax cannot express "this is Dholuo". It can only return the nearest of three,
# and for Ugandan Bantu that is always Luganda — which is precisely the 1403/1500 lug routing we
# saw, and why the forced-Luganda transcripts read `hukendera hu luguudo` (hu- where Luganda takes
# ku-) and `ni ndeeba` (Runyankole). The mask fix was still necessary; it just wasn't this.
#
# mms-1b-all ships 2,396 adapters, luo/nyn/xog/kam/kin among them, so a clip can be decoded in the
# language it is actually in. Anything LID names that has no adapter falls back to the closed set.
# Set WAXAL_CLOSED_SET=1 to get the old behaviour back for an A/B.
OPEN_SET = os.environ.get("WAXAL_CLOSED_SET") != "1"
try:
    from huggingface_hub import list_repo_files

    HAS_ADAPTER = {f.split(".")[1] for f in list_repo_files(ASR_MODEL) if f.startswith("adapter.")}
except Exception as e:                       # noqa: BLE001 - offline is survivable, silence isn't
    print(f"could not list {ASR_MODEL} adapters ({type(e).__name__}: {e}); closed set only")
    HAS_ADAPTER, OPEN_SET = set(LANGS), False
print(f"routing: {'open set' if OPEN_SET else 'closed set'}; "
      f"{len(HAS_ADAPTER):,} adapters available on {ASR_MODEL}")

unknown = [i for i in needed_ids if i in audio_store and i not in known_lang]
if unknown:
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

    print(f"\nrunning LID on {len(unknown):,} clips ({LID_MODEL})")
    fe = AutoFeatureExtractor.from_pretrained(LID_MODEL)
    lid = Wav2Vec2ForSequenceClassification.from_pretrained(LID_MODEL).to(DEVICE).eval().half()
    id2label = lid.config.id2label
    allowed_idx = [i for i, l in id2label.items() if l in LANGS]
    assert allowed_idx, f"none of {LANGS} in LID label space"
    # The argmax runs over whichever label subset we can actually decode. Open set = every LID
    # language mms-1b-all has an adapter for; closed set = the three challenge languages.
    route_idx = ([i for i, l in id2label.items() if l in HAS_ADAPTER] if OPEN_SET
                 else allowed_idx)
    assert route_idx, "no LID label has a matching MMS adapter"
    print(f"  argmax over {len(route_idx):,} language(s)"
          f"{'' if OPEN_SET else f' {[id2label[i] for i in route_idx]}'}")

    def lid_predict(ids: list[str], bs: int = 8, label: str = "lid") -> dict[str, str]:
        """Batched LID. Both the calibration below and the real routing call THIS function —
        if they diverged, the calibration would be measuring something we don't ship."""
        out: dict[str, str] = {}
        with torch.inference_mode():
            for k in range(0, len(ids), bs):
                chunk = ids[k:k + bs]
                wavs = [audio_store[i][:16000 * 30] for i in chunk]
                inp = fe(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
                # attention_mask is NOT optional here. mms-lid-256 has feat_extract_norm="layer"
                # and return_attention_mask=True, and Wav2Vec2ForSequenceClassification mean-pools
                # over time — so without the mask every zero-padded frame in a variable-length
                # batch is averaged in as if it were audio. That drags each clip's pooled vector
                # toward the same direction and collapses the argmax onto one class. It is exactly
                # what produced the 94%-Luganda phase 2 routing on the first run.
                logits = lid(inp.input_values.to(DEVICE).half(),
                             attention_mask=inp.attention_mask.to(DEVICE)).logits.float()
                picks = logits[:, route_idx].argmax(-1).cpu().numpy()
                for i, p in zip(chunk, picks):
                    out[i] = id2label[route_idx[int(p)]]
                if k % 400 == 0:
                    print(f"  {label} {k}/{len(ids)}", flush=True)
        return out

    # --- calibration: measure LID against ids whose language we already know ------------------
    # Phase 1 ids carry the language in the prefix, so we have thousands of free labels for the
    # model that decides all 1,500 phase 2 clips on its own. A wrong call there doesn't cost one
    # word — it sends the whole utterance to the wrong adapter and the wrong KenLM in stage 3.
    # This is cheap, and it is the difference between knowing and hoping.
    import random as _random

    _rng = _random.Random(SEED)
    calib: list[str] = []
    for _lang in LANGS:
        pool = [i for i in needed_ids if known_lang.get(i) == _lang and i in audio_store]
        calib += _rng.sample(pool, min(100, len(pool)))
    if calib:
        print(f"\ncalibrating LID on {len(calib)} phase-1 clips with known language")
        got = lid_predict(calib, label="calib")
        truth = [known_lang[i] for i in calib]
        pred = [got[i] for i in calib]
        acc = sum(t == p for t, p in zip(truth, pred)) / len(calib)
        # In open-set mode this is a much harder test than the 97.3% we measured against three
        # classes: a known-Luganda clip now has to come back as `lug` out of every language
        # mms-1b-all can decode, not merely beat lin and sna. That is the right test, because it
        # is the one phase 2 actually sits. A large drop here means open-set routing is scattering
        # clips across near neighbours and needs a confidence floor before we ship it.
        print(f"  LID accuracy ({'open' if OPEN_SET else 'closed'} set): {acc:.1%}")
        print("  confusion (rows = true, cols = predicted):")
        print(pd.crosstab(pd.Series(truth, name="true"),
                          pd.Series(pred, name="pred")).to_string())
        if acc < 0.85:
            print("  !! LID is the weakest link in the phase 2 path. Every point of LID error is\n"
                  "     a whole utterance decoded in the wrong language. Do not ship phase 2 on\n"
                  "     this without looking at the confusion matrix above.")

    print(f"\nrouting {len(unknown):,} phase-2 clips")
    known_lang.update(lid_predict(unknown))
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
# Decode whatever languages routing actually assigned, not a hardcoded three. Each adapter swap
# costs a load, so process a language's clips together; sorted() keeps the order deterministic.
route_langs = sorted(set(known_lang.values()))
missing_adapter = [l for l in route_langs if MMS_ADAPTER.get(l, l) not in HAS_ADAPTER]
if missing_adapter:
    # Nothing sensible to decode these with, so hand them to Luganda: every language LID confused
    # them with is Bantu, and Luganda is the one of our three with an actual adapter nearby. They
    # will score badly. Naming them here beats discovering it in the leaderboard.
    n = sum(1 for i in known_lang if known_lang[i] in missing_adapter)
    print(f"\nno mms-1b-all adapter for {missing_adapter} ({n} clips) -> falling back to lug")
    known_lang = {i: ("lug" if l in missing_adapter else l) for i, l in known_lang.items()}
    route_langs = sorted(set(known_lang.values()))

for lang in route_langs:
    ids = [i for i in needed_ids if known_lang.get(i) == lang and i in audio_store]
    if not ids:
        continue
    adapter = MMS_ADAPTER.get(lang, lang)
    print(f"\n--- {lang}: {len(ids):,} clips (adapter {adapter}) ---")
    processor.tokenizer.set_target_lang(adapter)
    model.load_adapter(adapter)
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
    # An empty Target is neutral for the metric — an empty hypothesis against a 26-word reference
    # is 26 deletions, and a one-word hypothesis is 25 deletions plus a substitution, same total.
    # It is NOT neutral for the parser: pandas reads an empty field back as float NaN, and a NaN
    # handed to jiwer is a type error rather than a bad score. So write a real token instead. We
    # only get 5 submissions a day; none of them should die on a dtype.
    mapped = sub[tid].astype(str).map(preds).fillna("")
    empty = int((mapped.str.strip() == "").sum())      # count BEFORE filling, or we lose the signal
    sub[ttxt] = mapped.replace(r"^\s*$", BLANK_FILL, regex=True)
    out = WORK / f"submission_01_mms_zeroshot_{SUFFIX.get(fname, Path(fname).stem)}.csv"
    sub.to_csv(out, index=False)

    langs = pd.Series([known_lang.get(i, "??") for i in sub[tid].astype(str)]).value_counts()
    print(f"\nwrote {out}")
    print(f"  rows={len(sub):,}  empty-before-fill={empty:,} ({100*empty/len(sub):.1f}%)")
    print(f"  language mix: {langs.to_dict()}")
    if empty:
        # On phase 2 this almost always means the audio zip did not contain that id.
        print(f"  WARNING: {empty:,} ids got no prediction; written as {BLANK_FILL!r}, which scores the same as a deletion but parses.")
    print(sub.head(5).to_string())

# Keep the language decisions — stage 3 reuses them instead of re-running LID.
json.dump(known_lang, open(WORK / "lang_map.json", "w"))
print("wrote lang_map.json")
