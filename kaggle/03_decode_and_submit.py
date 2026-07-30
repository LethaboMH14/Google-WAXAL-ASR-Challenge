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
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ZINDI_DIR = Path("/kaggle/input/waxal-zindi")
CKPT = Path("/kaggle/input/waxal-ckpt/w2vbert-waxal")     # stage 2 output, uploaded as a Dataset
LM_CORPUS_DIR = Path("/kaggle/input/waxal-lm/lm_corpus")  # stage 0 output, uploaded as a Dataset
WORK = Path("/kaggle/working")
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
os.system(
    "apt-get -qq install -y build-essential cmake libboost-system-dev libboost-thread-dev "
    "libboost-program-options-dev libboost-test-dev libboost-filesystem-dev libeigen3-dev zlib1g-dev"
)
if not Path("kenlm/build/bin/lmplz").exists():
    os.system("wget -q -O - https://kheafield.com/code/kenlm.tar.gz | tar xz")
    os.system("mkdir -p kenlm/build && cd kenlm/build && cmake .. -DCMAKE_BUILD_TYPE=Release "
              "> /dev/null && make -j4 lmplz build_binary > /dev/null")
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
        f"kenlm/build/bin/lmplz -o {NGRAM_ORDER} --text {txt} --arpa {arpa} "
        f"--discount_fallback {prune}",
        shell=True, check=True,
    )
    # Binary format loads in seconds instead of minutes, which matters because the alpha/beta
    # sweep constructs a decoder 12 times per language.
    binary = LM_DIR / f"{lang}.bin"
    subprocess.run(f"kenlm/build/bin/build_binary {arpa} {binary}", shell=True, check=True)
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

wer_metric, cer_metric = evaluate.load("wer"), evaluate.load("cer")
N_TUNE = 150

tuned = {}
for lang, cfg in HF_CONFIGS.items():
    if lang not in lm_paths:
        continue
    ds = load_dataset("google/WaxalNLP", cfg, split="validation", streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    wavs, refs = [], []
    for row in ds:
        w = np.asarray(row["audio"]["array"], dtype=np.float32)
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
sample = pd.read_csv(ZINDI_DIR / "SampleSubmission.csv", escapechar="\\")
SUB_ID = guess_col(sample, "id", "audio_id", "utt_id", "filename")
SUB_TXT = guess_col(sample, "transcription", "transcript", "text", "target", "prediction")
needed_ids = sample[SUB_ID].astype(str).tolist()
needed = set(needed_ids)
print(f"submission contract: {len(needed_ids):,} rows, id={SUB_ID}, target={SUB_TXT}")

audio_store, known_lang = {}, {}
if Path("/kaggle/input/waxal-ckpt/lang_map.json").exists():
    known_lang = json.load(open("/kaggle/input/waxal-ckpt/lang_map.json"))

# Cheapest and most reliable source first: the id prefix.
for i in needed_ids:
    lg = lang_from_id(i)
    if lg:
        known_lang[i] = lg
print(f"language from id prefix: {sum(1 for i in needed_ids if i in known_lang):,}"
      f" / {len(needed_ids):,}")

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
        # RULES GUARD: never read labels from the test split.
        ds = ds.remove_columns([c for c in ("transcription", "text") if c in ds.column_names])
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        for row in ds:
            rid = str(row["id"])
            if rid in missing and rid not in audio_store:
                audio_store[rid] = np.asarray(row["audio"]["array"], dtype=np.float32)
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

sub = sample.copy()
sub[SUB_TXT] = sub[SUB_ID].astype(str).map(preds).fillna("")
out = WORK / "submission_03_w2vbert_lm.csv"
sub.to_csv(out, index=False)
blank = int((sub[SUB_TXT].str.strip() == "").sum())
print(f"\nwrote {out}: rows={len(sub):,} blank={blank:,}")
print(sub.head(10).to_string())
