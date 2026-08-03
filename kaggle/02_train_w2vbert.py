"""
STAGE 2 — the actual model. One multilingual w2v-bert-2.0 CTC head over lin + sna + lug.

Where to run: a GPU box with internet, and the card matters here more than anywhere else.
BATCH/GRAD_ACCUM below size themselves to the VRAM they find. A 16 GB T4 measures ~24 s/step
(~17 h for 2,500 steps); an 80 GB card drops gradient checkpointing and runs several times
faster. The run is resumable either way — see the resume block — so a session that stops does
not cost you the steps it completed.

Why this shape:
  * ONE model, not three. Phase 2 ships no language metadata, so a per-language model has no
    way to route. All three languages are Bantu in Latin script, so a shared character vocab
    is natural and the languages reinforce each other in the low-resource regime.
  * w2v-bert-2.0 (580M, pretrained on 4.5M hours / 143 languages) beats MMS-300M on Bantu in
    arXiv:2512.10968, and MMS-300M is what WAXAL-NET used to set the published numbers.
  * Streaming from HF: the three train splits are ~25 GB of audio, which is more than we want
    to land on any of these hosts' disks. Streaming + step-based training is also the right fit
    for a time-boxed run.
  * SpecAugment is ON. Phase 2 is explicitly a generalisation test to unseen speakers, so we
    trade a little train fit for robustness. Do not turn this off to make the loss curve pretty.

Rules guard: trains on `train`, early-stops on `validation`. The `test` split is never opened.
"""

import json
import os
import random
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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
    # and a way to silently train a vocab off a stale Train.csv. Prefer an attached Dataset if
    # one is there (that is how the earlier runs were wired), else fall back to the clone.
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


OUTDIR = WORK / "w2vbert-waxal"
# On Kaggle this is last session's output re-uploaded as a Dataset. On Lightning it resolves to
# WORK, i.e. OUTDIR itself — so a session killed at hour 6 resumes simply by re-running this
# script, with no upload step and nothing to remember.
RESUME_DIR = ART("waxal-ckpt")

BASE = "facebook/w2v-bert-2.0"
HF_CONFIGS = {"lin": "lin_asr", "sna": "sna_asr", "lug": "lug_asr"}
# Sampling weights across languages. Measured Train.csv counts are lin 16,244 / sna 15,836 /
# lug 6,119 utterances, i.e. natural proportions of 0.43 / 0.41 / 0.16. The weights below
# leave Lingala at its natural share (it is the hardest language at 42.6 WER, so it needs
# every example it has) and move sampling mass from Shona to Luganda, which is starved at 16%
# and would otherwise be the language the shared model quietly gives up on.
LANG_WEIGHTS = {"lin": 0.42, "sna": 0.34, "lug": 0.24}

# WAXAL_LANGS restricts the sampler to a subset and renormalises what remains, so WAXAL_LANGS=sna
# trains a Shona-only head without disturbing the multilingual defaults above.
#
# The stated case for one shared model is that phase 2 ships no language metadata. That does not
# bind any more: we route phase 2 with okwija, and its map agrees with EVERY scored submission on
# 891-892 of 892 clips (verified 3 Aug by recovering each file's implied routing from its own
# output text). A per-language head therefore has an accurate router to sit behind. Corrected
# phase 2 is ~50% Shona, and douyeszn's per-language Shona card reports 0.7945 on a
# speaker-disjoint split, which is a published number to aim at rather than a guess.
_want = [x.strip() for x in os.environ.get("WAXAL_LANGS", "").split(",") if x.strip()]
if _want:
    _unknown = [x for x in _want if x not in LANG_WEIGHTS]
    if _unknown:
        raise SystemExit(f"WAXAL_LANGS={_want} names unknown languages {_unknown}")
    _tot = sum(LANG_WEIGHTS[x] for x in _want)
    LANG_WEIGHTS = {x: LANG_WEIGHTS[x] / _tot for x in _want}
    print(f"WAXAL_LANGS -> {_want} only, renormalised weights {LANG_WEIGHTS}")

MAX_SECONDS = 20.0          # clips run 3-67s; >20s blows T4 VRAM and adds little
MIN_SECONDS = 1.0

# --- CTC frame rate: the one config choice this dataset actually forces ---
# w2v-bert-2.0 ships with add_adapter=True, and essentially every w2v-bert fine-tuning
# tutorial keeps it. Those tutorials target CommonVoice-style utterances of ~5 words. WAXAL
# is not that: measured over Train.csv, transcripts average **176 characters** (26 words),
# p95 305, max 650.
#
# The adapter is a stride-2 conv after the encoder, so it halves the frame rate from 20ms to
# 40ms. Do the arithmetic on a typical ~12s clip:
#
#   with adapter    12s / 0.040 = 300 frames for ~169 chars -> 1.8 frames per character
#   without adapter 12s / 0.020 = 600 frames for ~169 chars -> 3.6 frames per character
#
# CTC needs at least one frame per label plus a blank between any repeated pair, and Bantu
# orthography here is full of doubled letters (ekkubo, ssukuma, ennyaanya, amaato). At 1.8
# frames/char a large fraction of the training set is either unlearnable or right at the edge,
# which shows up as inf losses, dropped examples and a model that truncates long utterances.
# Disabling the adapter costs a little memory in the CTC head only — the encoder is untouched —
# and buys back the headroom. This is measured, not stylistic.
ADD_ADAPTER = False
SAMPLES_PER_FRAME = 640 if ADD_ADAPTER else 320
# HF Trainer counts OPTIMIZER steps, not forward passes. Measured: 24.4 s/step on a single T4 at
# BATCH=4/ACCUM=8. On 2xT4 each GPU does 4 accumulation passes instead of 8, so ~13-16 s/step
# with DDP sync — call it 9.7 h for 2,500 steps.
#
# That does not fit a Kaggle GPU session, which is capped at 9 hours, and a committed run that
# hits the wall does not reliably save its outputs. WAXAL-NET converged at 2000-3500 steps, so
# 2,000 is inside the converged range rather than a compromise: it gives up the top of a plateau,
# not the climb. Set WAXAL_MAX_STEPS=2000 on Kaggle T4x2 and it lands in ~7.8 h with headroom.
#
# The default stays 2500 because that is the right number anywhere the session isn't capped.
MAX_STEPS = int(os.environ.get("WAXAL_MAX_STEPS", 2500))

# --- batch size follows the card, effective batch does not ---------------------------------
# BATCH=4 + GRAD_ACCUM=8 exists because a 16 GB T4 cannot hold more of a 581M model on 20 s
# audio. On an 80 GB card that leaves most of the GPU idle and costs 24.4 s/step, which is how
# a 2,500-step run becomes 17 hours. Scale the micro-batch with the memory actually present and
# divide the accumulation by the same factor, so the EFFECTIVE batch stays 32 — the WAXAL-NET
# recipe — and the training maths is unchanged. This is a throughput decision, not a recipe one.
#
# Gradient checkpointing is the other T4 concession: it recomputes activations in the backward
# pass to save memory, at roughly 30-40% of the step time. With headroom it is pure loss.
#
# Override with WAXAL_BATCH / WAXAL_ACCUM if a card OOMs; keep their product at 32 per GPU.
_VRAM_GB = (torch.cuda.get_device_properties(0).total_memory / 1e9
            if torch.cuda.is_available() else 0)
if _VRAM_GB >= 60:            # H100 / A100 80GB
    BATCH, _ACCUM_BASE, GRAD_CKPT = 8, 4, False
elif _VRAM_GB >= 38:          # A100 40GB / L40S
    BATCH, _ACCUM_BASE, GRAD_CKPT = 8, 4, True
elif _VRAM_GB >= 22:          # L4 / A10G 24GB
    BATCH, _ACCUM_BASE, GRAD_CKPT = 4, 8, True
else:                         # T4 16GB and below — the original settings
    BATCH, _ACCUM_BASE, GRAD_CKPT = 4, 8, True
BATCH = int(os.environ.get("WAXAL_BATCH", BATCH))
# Effective batch must land near 32 (the WAXAL-NET recipe). Trainer multiplies by device count,
# so halve the accumulation on multi-GPU. _ACCUM_BASE already tracks BATCH above.
GRAD_ACCUM = int(os.environ.get(
    "WAXAL_ACCUM", max(1, _ACCUM_BASE // (2 if torch.cuda.device_count() > 1 else 1))))
LR = 5e-5                   # 1e-4 is the paper's value for MMS-300M; w2v-bert prefers lower
WARMUP = 200
# How often we eval AND checkpoint. 500 was chosen when a step was a step; it is really a bet on
# how much work you are willing to lose. That bet has now been settled empirically: a T4 run died
# at step 96, the first checkpoint was 404 steps away, and every one of those steps was thrown
# away — `w2vbert-waxal/` held tokenizer files and nothing else.
#
# The right number is a wall-clock one, so derive it from the measured step time rather than
# fixing it: aim to checkpoint about every 20 minutes, clamped to [100, 500] steps. On a T4 at
# ~24 s/step that is 100 (~40 min, the floor — checkpointing a 581M model plus optimizer state
# costs real minutes and doing it every 10 would eat the run). On an 80 GB card at ~3 s/step it
# is 400. Free Studios stop every 4 hours, so the exposure has to stay well under that.
#
# Override with WAXAL_EVAL_EVERY if you know better. save_total_limit=2 below caps the disk cost
# at two checkpoints regardless of how often we write them.
_SEC_PER_STEP = {8: 3.0, 4: 12.0}.get(BATCH, 24.0) * (1.6 if GRAD_CKPT else 1.0)
EVAL_EVERY = int(os.environ.get(
    "WAXAL_EVAL_EVERY", min(500, max(100, round(20 * 60 / _SEC_PER_STEP / 50) * 50))))
SEED = 1337

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTDIR.mkdir(parents=True, exist_ok=True)

# --- mid-run resume -------------------------------------------------------------------------
# init_from below only finds a COMPLETED run — trainer.save_model() writes OUTDIR itself, and
# that happens after step 2500. A run killed at step 1400 leaves nothing there; all it leaves is
# OUTDIR/checkpoint-1000/. Without the lookup here, re-running after an interruption silently
# restarted from facebook/w2v-bert-2.0 and threw away every step. That matters because free
# Lightning Studios stop every 4 hours and this run is ~16h on a T4: uninterrupted is the case
# that never happens.
#
# Resuming restores optimizer state, LR schedule and step count from the checkpoint.
from transformers.trainer_utils import get_last_checkpoint

RESUME_CKPT = get_last_checkpoint(OUTDIR) if any(OUTDIR.glob("checkpoint-*")) else None
RESUMED_STEP = int(Path(RESUME_CKPT).name.split("-")[1]) if RESUME_CKPT else 0
# The training stream is an IterableDataset, so Trainer cannot seek into it — its batch-skip
# would re-download and discard every audio file already consumed, which on a 16h run costs
# roughly as much as the training it is trying to recover. We set ignore_data_skip=True and
# reshuffle instead, offsetting the stream seed by the step we resumed at so the second leg
# doesn't retrain on the exact prefix the first leg already saw.
#
# Reproducibility (the rules require a re-run to land in the same leaderboard position): an
# uninterrupted run from an empty OUTDIR has RESUMED_STEP = 0, so STREAM_SEED == SEED and the
# result is bit-identical to before this change. Only interrupted runs diverge, and only in
# data order — which is unavoidable, since the interruption point isn't reproducible either.
STREAM_SEED = SEED + RESUMED_STEP
if RESUME_CKPT:
    print(f"resuming from {RESUME_CKPT} (step {RESUMED_STEP} of {MAX_STEPS}); "
          f"stream seed {STREAM_SEED}")
else:
    print("no checkpoint in OUTDIR — starting from step 0")

BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
print(f"cuda={torch.cuda.is_available()} bf16={BF16} n_gpu={torch.cuda.device_count()}")
print(f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'} "
      f"vram={_VRAM_GB:.0f}GB -> batch={BATCH} accum={GRAD_ACCUM} "
      f"(effective {BATCH * GRAD_ACCUM * max(1, torch.cuda.device_count())}) "
      f"grad_ckpt={GRAD_CKPT} ckpt_every={EVAL_EVERY} steps "
      f"(~{EVAL_EVERY * _SEC_PER_STEP / 60:.0f} min of work at risk)")


# ---------------------------------------------------------------- text normalisation
# MEASURED, not guessed. Two independent sources fix this policy:
#
#   1. The organisers' own starter notebook (Waxal_Challenge_Starter_Code.ipynb, section 8)
#      evaluates with:  jiwer.wer([r.lower() ...], [p.lower() ...])
#      It lowercases BOTH sides and does NOT touch punctuation. So lowercasing is free, and
#      punctuation is scored.
#   2. local/inspect_data.py over the real Train.csv: 67,456 full stops and 28,620 commas
#      across 993,916 words. Dropping punctuation would put a substitution on ~9.7% of all
#      tokens — an own goal of roughly 10 absolute WER before the model does anything.
#
# Hence: lowercase yes, strip punctuation NO. The apostrophe in particular is orthographic in
# Luganda (w'ekkubo, ng'atambulira) — 16,337 occurrences in 6,119 utterances — and removing it
# would corrupt the words themselves, not merely their punctuation.
#
# CORRECTION, 3 Aug — point 1 above is WRONG about the live grader, and it is measured wrong.
# Lethabo's submissions 15 and 16 are the SAME TEXT differing only in capitalisation (892/892 rows
# identical ignoring case; byte counts match) and they scored 0.745030 and 0.745734. If the grader
# lowercased both sides those numbers would be identical. They differ by +0.000703, so casing is
# scored and the starter notebook does not match what actually grades submissions.
#
# The effect is small but it is free, and it points the other way for training: a model that never
# emits a capital can never collect it. douyeszn's Shona card likewise keeps "case + punctuation"
# in the vocab. Default stays True so the multilingual runs already measured are reproducible;
# set WAXAL_LOWERCASE=0 to train a cased head.
LOWERCASE = os.environ.get("WAXAL_LOWERCASE", "1") != "0"

# Punctuation the model is expected to emit. Everything measured above ~300 occurrences and
# genuinely predictable from prosody or orthography.
KEEP_PUNCT = set(".,'’-;:!?")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).strip()
    if LOWERCASE:
        text = text.lower()
    # Keep letters, spaces and the punctuation above. Digits (≈500 chars in the whole corpus),
    # brackets, quotes and stray symbols are dropped: too rare to learn, pure vocab bloat.
    text = "".join(c if (c.isalpha() or c.isspace() or c in KEEP_PUNCT) else " " for c in text)
    return re.sub(r"\s+", " ", text).strip()


def fold_rare(text: str, allowed: set[str]) -> str:
    """
    Map characters too rare to earn a CTC label onto their unaccented base (é -> e), and drop
    what still doesn't fit. Without this, a handful of one-off characters (ĝ, þ, œ) each get a
    vocab slot they will never learn, and every one of them is a permanent [UNK] in the labels.
    """
    out = []
    for c in text:
        if c in allowed:
            out.append(c)
            continue
        base = "".join(ch for ch in unicodedata.normalize("NFD", c)
                       if not unicodedata.combining(ch))
        out.append(base if base and all(b in allowed for b in base) else " ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


# ---------------------------------------------------------------- vocab from Zindi Train.csv
# Using the CSV instead of streaming HF text means we see every training transcript without
# pulling a single byte of audio.
import pandas as pd
from collections import Counter

# Zindi's CSVs backslash-escape quotes inside quoted fields (`\"`), which is not standard CSV.
# 23 of the 38,199 rows have it and pandas dies on the first one without escapechar.
train_csv = pd.read_csv(ZINDI_DIR / "Train.csv", escapechar="\\")
TXT_COL = next(c for c in train_csv.columns
               if c.lower() in ("transcription", "transcript", "text", "target", "sentence"))
raw = [normalise(t) for t in train_csv[TXT_COL].dropna().astype(str)]
raw = [c for c in raw if c]
print(f"train transcripts: {len(raw):,}")

# A character earns a CTC label by being frequent enough to actually learn. The tail below
# this threshold is accents on loanwords and OCR-ish noise; fold_rare maps it onto base
# letters instead of handing each one an untrainable slot.
MIN_CHAR_COUNT = 25
counts = Counter("".join(raw))
allowed = {c for c, n in counts.items() if n >= MIN_CHAR_COUNT and not c.isspace()}
dropped = {c: n for c, n in counts.items() if c not in allowed and not c.isspace()}
print(f"chars kept {len(allowed)}, folded/dropped {len(dropped)}: {dropped}")

corpus = [fold_rare(c, allowed) for c in raw]
corpus = [c for c in corpus if c]

chars = sorted(allowed)
vocab = {c: i for i, c in enumerate(chars)}
vocab["|"] = len(vocab)          # word delimiter
vocab["[UNK]"] = len(vocab)
vocab["[PAD]"] = len(vocab)
(OUTDIR / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
print(f"vocab size {len(vocab)}: {''.join(chars)}")

# Stash the LM corpus and the exact charset, so stages 0 and 3 normalise identically without
# having to re-derive anything. A mismatch here silently degrades beam search.
(OUTDIR / "lm_corpus.txt").write_text("\n".join(corpus), encoding="utf-8")
(OUTDIR / "charset.json").write_text(
    json.dumps({"allowed": sorted(allowed), "lowercase": LOWERCASE,
                "keep_punct": sorted(KEEP_PUNCT), "min_char_count": MIN_CHAR_COUNT},
               ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- processor
from transformers import (
    SeamlessM4TFeatureExtractor,
    Wav2Vec2BertForCTC,
    Wav2Vec2BertProcessor,
    Wav2Vec2CTCTokenizer,
)

tokenizer = Wav2Vec2CTCTokenizer(
    str(OUTDIR / "vocab.json"),
    unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|",
)
feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(BASE)
processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor.save_pretrained(OUTDIR)


# ---------------------------------------------------------------- streaming data
from datasets import Audio, interleave_datasets, load_dataset


# requirements-gpu.txt pins datasets < 4 so Audio columns decode through soundfile, not
# torchcodec — see the note at the bottom of that file. This helper is the single place audio
# becomes 16 kHz mono float32, and it takes either a decoded cell or raw bytes, so every caller
# resamples identically.
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


def build(split: str):
    # Train reshuffles on resume; validation must NOT. eval_ds is .take(400) off this stream,
    # so a different seed would hand each leg of the run a different 400 clips, and
    # metric_for_best_model would then be comparing scores measured on different data.
    seed = STREAM_SEED if split == "train" else SEED
    parts, probs = [], []
    for lang, cfg in HF_CONFIGS.items():
        ds = load_dataset("google/WaxalNLP", cfg, split=split, streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        if split == "train":
            ds = ds.shuffle(seed=seed, buffer_size=1500)
        parts.append(ds)
        probs.append(LANG_WEIGHTS[lang])
    total = sum(probs)
    return interleave_datasets(parts, probabilities=[p / total for p in probs],
                               seed=seed, stopping_strategy="all_exhausted")


def prepare(batch: dict) -> dict:
    wav = decode_audio_cell(batch["audio"])
    feats = feature_extractor(wav, sampling_rate=16000)
    batch["input_features"] = feats.input_features[0]
    batch["labels"] = tokenizer(fold_rare(normalise(batch["transcription"]), allowed)).input_ids
    batch["n_samples"] = len(wav)
    return batch


def keep(batch: dict) -> bool:
    n = batch["n_samples"]
    if not (MIN_SECONDS * 16000 <= n <= MAX_SECONDS * 16000):
        return False
    # CTC cannot emit more labels than it has output frames, and it needs slack on top for the
    # blanks that separate repeated characters. SAMPLES_PER_FRAME is 320 because we disable the
    # adapter (see ADD_ADAPTER); the 1.5x margin keeps marginal examples out rather than
    # feeding the model alignments it cannot represent.
    return 0 < len(batch["labels"]) * 1.5 < n / SAMPLES_PER_FRAME


cols_to_drop = ["audio", "transcription", "speaker_id", "language", "gender", "id"]
train_ds = build("train").map(prepare, remove_columns=cols_to_drop).filter(keep)
eval_ds = (build("validation").map(prepare, remove_columns=cols_to_drop)
           .filter(keep).take(400))


@dataclass
class Collator:
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
        )
        batch["labels"] = labels_batch.input_ids.masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        return batch


# ---------------------------------------------------------------- metrics
import evaluate

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_metrics(pred) -> dict:
    logits = pred.predictions
    ids = np.argmax(logits, axis=-1)
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    hyp = processor.batch_decode(ids)
    ref = processor.batch_decode(label_ids, group_tokens=False)
    pairs = [(h, r) for h, r in zip(hyp, ref) if r.strip()]
    if not pairs:
        return {"wer": 1.0, "cer": 1.0, "score": 0.0}
    hyp, ref = zip(*pairs)
    wer = wer_metric.compute(predictions=list(hyp), references=list(ref))
    cer = cer_metric.compute(predictions=list(hyp), references=list(ref))
    # Mirrors the leaderboard: higher is better, so early stopping optimises the real target.
    return {"wer": wer, "cer": cer, "score": 1.0 - 0.5 * (min(wer, 1.0) + min(cer, 1.0))}


# ---------------------------------------------------------------- model
def _has_weights(d: Path) -> bool:
    """Directory existence is NOT the test. On Lightning ART() returns WORK, so
    RESUME_DIR/'w2vbert-waxal' IS OUTDIR — which OUTDIR.mkdir() creates on line 144 and which
    processor.save_pretrained() then fills with tokenizer files. The old `.exists()` check was
    therefore true on a first run, and from_pretrained() would go looking for weights that
    aren't there. Check for the weights themselves."""
    return (d / "config.json").exists() and any(
        (d / f).exists()
        for f in ("model.safetensors", "pytorch_model.bin",
                  "model.safetensors.index.json", "pytorch_model.bin.index.json")
    )


if RESUME_CKPT:                     # mid-run checkpoint wins: it is the newest state we have
    init_from = RESUME_CKPT
elif _has_weights(RESUME_DIR / "w2vbert-waxal"):
    init_from = str(RESUME_DIR / "w2vbert-waxal")   # a previous run that reached the end
else:
    init_from = BASE
print(f"initialising from: {init_from}")

model = Wav2Vec2BertForCTC.from_pretrained(
    init_from,
    attention_dropout=0.05,
    hidden_dropout=0.05,
    feat_proj_dropout=0.0,
    # SpecAugment — the generalisation insurance for Phase 2's unseen speakers.
    mask_time_prob=0.05,
    mask_time_length=10,
    mask_feature_prob=0.05,
    mask_feature_length=64,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    ctc_zero_infinity=True,        # survive the odd pathological alignment instead of NaN-ing
    add_adapter=ADD_ADAPTER,       # False — see the frame-rate note in the config block
    pad_token_id=tokenizer.pad_token_id,
    vocab_size=len(tokenizer),
    ignore_mismatched_sizes=(init_from == BASE),
)
if GRAD_CKPT:
    model.gradient_checkpointing_enable()
print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir=str(OUTDIR),
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=GRAD_CKPT,
    learning_rate=LR,
    warmup_steps=WARMUP,
    lr_scheduler_type="linear",
    bf16=BF16,
    fp16=not BF16,
    eval_strategy="steps",
    eval_steps=EVAL_EVERY,
    save_strategy="steps",
    save_steps=EVAL_EVERY,
    # A checkpoint here is ~7 GB: 2.3 GB of fp32 weights plus AdamW's two fp32 moments. Keeping
    # two of them alongside the final model is ~16 GB, and /kaggle/working is capped at 20 GB —
    # close enough that a run could die on disk at hour 8 rather than on anything to do with
    # training. Lightning's home is 387 GB, so keep the safety margin of a second checkpoint there.
    save_total_limit=1 if ENV == "kaggle" else 2,
    load_best_model_at_end=True,
    metric_for_best_model="score",
    greater_is_better=True,
    logging_steps=50,
    dataloader_num_workers=2,
    report_to=[],
    seed=SEED,
    remove_unused_columns=False,
    ignore_data_skip=True,          # see the resume block near the top
    # NOT passed, because transformers 5 removed both and both were no-ops for us anyway:
    #   group_by_length=False  — already the default, and length grouping cannot work on a
    #                            streaming IterableDataset regardless
    #   save_safetensors=True  — v5 saves safetensors unconditionally
    # Omitting them keeps this file working on 4.44+ as well: the 4.x defaults are identical.
)


# ---------------------------------------------------------------- wall-clock stop
# The binding constraint on a hosted GPU is TIME, not steps — Kaggle kills a GPU session at 9 h
# and a run killed mid-step does not reliably save anything. Sizing MAX_STEPS from an estimated
# step rate is how you find that out at hour nine: the first Kaggle attempt was set to 1,500
# steps on an assumed ~15 s/step, measured 32.7 s/step, and was quietly on course for 13.6 h.
#
# So stop on the clock instead of on a guess. The callback ends training at the budget, saves,
# and lets the normal save_model path run; the next leg resumes from that checkpoint with
# optimizer state and LR schedule intact. MAX_STEPS then only has to be the right END of the
# schedule rather than a number that also has to fit in a session.
#
# Unset means no limit, which is the local and Lightning behaviour.
MAX_HOURS = float(os.environ.get("WAXAL_MAX_HOURS", "0"))

from transformers import TrainerCallback  # noqa: E402


class StopAfterHours(TrainerCallback):
    def __init__(self, hours: float):
        self.deadline = time.monotonic() + hours * 3600
        self.hours = hours

    def on_step_end(self, args, state, control, **kw):
        if time.monotonic() >= self.deadline:
            print(f"\nwall-clock budget of {self.hours} h reached at step {state.global_step} — "
                  f"stopping and saving. Resume with the same command; it continues from here.",
                  flush=True)
            control.should_training_stop = True
            control.should_save = True
            control.should_evaluate = True
        return control


callbacks = [StopAfterHours(MAX_HOURS)] if MAX_HOURS > 0 else []
if MAX_HOURS > 0:
    print(f"wall-clock budget: {MAX_HOURS} h (training stops and saves at that point)")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=Collator(processor=processor),
    compute_metrics=compute_metrics,
    callbacks=callbacks,
)

trainer.train(resume_from_checkpoint=RESUME_CKPT)
trainer.save_model(str(OUTDIR))
processor.save_pretrained(str(OUTDIR))
print(f"\nsaved to {OUTDIR}")
print(f"Now: nothing to publish — {OUTDIR} is persistent. A second leg resumes from it and "
      f"stage 3 loads it directly.")
