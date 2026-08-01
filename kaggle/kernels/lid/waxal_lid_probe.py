"""Kaggle GPU kernel: LID PROBE — decide, with measurements, how phase 2 gets routed.

WHY THIS IS NOT OPTIONAL
------------------------
Phase 1 ids carry the language (`lug_96114`); phase 2 ids do not (`ID_TBDTM`, verified: `ID_` plus
five uniformly-random uppercase letters, all 26 used, letter frequencies 0.036-0.043 against a
uniform 0.0385). Phase 2 is the split that sets the final ranking. So on the set that decides the
prize, language ID runs on every single clip, and a LID error is not worth a few WER points — it
sends the whole utterance to the wrong acoustic model and the wrong KenLM, which costs essentially
the entire clip. At 1,500 clips, each 1% of LID error is worth roughly 0.007 of final score.

WHAT IS MEASURED, AND ON WHAT
-----------------------------
Labelled audio, for accuracy. The phase-1 TEST clips are the right calibration set: their ids give
the language for free, and they are test-domain audio rather than train-split audio, so they are
the closest labelled proxy we have for phase 2. Reading an id prefix is not reading a label — the
rules forbid using the published phase-1 ground-truth TRANSCRIPTS, and nothing here touches them
(the transcription columns are dropped from the stream on arrival, below).

Unlabelled audio, for agreement. All 1,500 phase-2 clips are then routed by every method, and we
report pairwise agreement and the resulting language mix. Agreement is not accuracy, but on a set
with no labels it is the only honest signal available, and disagreement is a hard lower bound on
the error of at least one method.

THE CANDIDATES
--------------
1. `facebook/mms-lid-256`, closed set over {lin, sna, lug} — what stage 3 ships today.
2. `facebook/mms-lid-256`, open set over every language mms-1b-all has an adapter for, then
   mapped back — what stage 1 ships today. Stage 1 and stage 3 currently disagree, which is
   itself a bug: two stages of one pipeline should not route differently.
3. `Okwija/waxal-lid-lin-sna-lug` — purpose-built for exactly these three languages.

A prior run (commit e9b3885) found that unconstrained mms-lid-256 calls phase-2 clips luo/nyn/lug/
kin/kam/xog — neighbouring Bantu languages, not noise. That is why open-set-then-map exists at all,
and why the closed-set number needs checking rather than assuming: forcing an argmax over three
classes when the model wants to say "Runyankole" can still land on the right one.
"""

import io
import json
import os
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")
WORKING = Path("/kaggle/working")
PHASE2_URL = "https://storage.googleapis.com/waxalphase2/audio.zip"

LANGS = ["lin", "sna", "lug"]
HF_CONFIGS = {"lin": "lin_asr", "sna": "sna_asr", "lug": "lug_asr"}
LID_MODEL = "facebook/mms-lid-256"
ASR_MODEL = "facebook/mms-1b-all"
OKWIJA = "Okwija/waxal-lid-lin-sna-lug"
N_PER_LANG = int(os.environ.get("WAXAL_LID_N", "400"))
SEED = 1337


def sh(cmd, check=True, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=check, **kw)


if not REPO.exists():
    sh(["git", "clone", "--depth", "1", REPO_URL, str(REPO)])
sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements-gpu.txt")])

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification  # noqa: E402

if not torch.cuda.is_available():
    raise SystemExit("no CUDA — set the kernel accelerator to GPU T4 before running")
DEVICE = "cuda"
print(f"gpu: {torch.cuda.get_device_name(0)}")

# ------------------------------------------------------------------ labelled audio (phase 1 test)
# Streaming reads in file order and stops early, so this costs N_PER_LANG clips per language rather
# than the whole 4,253-clip split. Taking a prefix rather than a random sample is a real caveat: it
# is only sound if file order is uncorrelated with difficulty. It buys ~25 minutes of streaming and
# the alternative (materialise the split, then sample) does not fit the session budget.
labelled: dict[str, np.ndarray] = {}
truth: dict[str, str] = {}
for lang, cfg in HF_CONFIGS.items():
    ds = load_dataset("google/WaxalNLP", cfg, split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    # RULES GUARD: the phase-1 ground-truth transcripts are public and using them is an explicit
    # disqualification. Drop the columns on arrival so no later line can reach them by accident.
    ds = ds.remove_columns([c for c in ("transcription", "text") if c in ds.column_names])
    n = 0
    for row in ds:
        rid = str(row["id"])
        w = np.asarray(row["audio"]["array"], dtype=np.float32)
        if w.ndim > 1:
            w = w.mean(axis=1)
        labelled[rid] = w
        truth[rid] = lang
        n += 1
        if n >= N_PER_LANG:
            break
    print(f"  {lang}: {n} labelled clips", flush=True)
print(f"labelled set: {len(labelled):,} clips")

# ------------------------------------------------------------------ unlabelled audio (phase 2)
phase2: dict[str, np.ndarray] = {}
zp = WORKING / "phase2_audio.zip"
if not zp.exists():
    os.system(f"wget -q -O {zp} {PHASE2_URL}")
if zp.exists() and zp.stat().st_size > 0:
    import librosa
    import soundfile as sf

    # v1 died here on a member soundfile could not decode ("Format not recognised"). The zip is
    # not purely audio. Two guards, because a router that silently drops clips is worse than one
    # that crashes: filter by extension first, then catch and COUNT whatever still fails, so the
    # tally prints what was lost instead of hiding it.
    AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}
    skipped, failed = [], []
    with zipfile.ZipFile(zp) as zf:
        for nm in [x for x in zf.namelist() if not x.endswith("/")]:
            if nm.startswith("__MACOSX") or Path(nm).suffix.lower() not in AUDIO_EXT:
                skipped.append(nm)
                continue
            try:
                with zf.open(nm) as fh:
                    w, sr = sf.read(io.BytesIO(fh.read()), dtype="float32")
            except Exception as e:  # noqa: BLE001
                failed.append(f"{nm}: {type(e).__name__}")
                continue
            if w.ndim > 1:
                w = w.mean(axis=1)
            if sr != 16000:
                w = librosa.resample(w, orig_sr=sr, target_sr=16000)
            phase2[Path(nm).stem] = w.astype(np.float32)
    if skipped:
        print(f"  skipped {len(skipped):,} non-audio member(s), e.g. {skipped[:5]}")
    if failed:
        print(f"  FAILED to decode {len(failed):,} member(s), e.g. {failed[:5]}")
print(f"phase 2: {len(phase2):,} clips")
if phase2 and len(phase2) < 1500:
    print(f"  WARNING: expected 1,500 phase-2 clips, loaded {len(phase2):,}. The agreement figures "
          f"below cover only what loaded, not the full private split.")

# ------------------------------------------------------------------ the three routers
try:
    from huggingface_hub import list_repo_files

    HAS_ADAPTER = {f.split(".")[1] for f in list_repo_files(ASR_MODEL) if f.startswith("adapter.")}
except Exception as e:  # noqa: BLE001
    print(f"could not list {ASR_MODEL} adapters ({type(e).__name__}: {e})")
    HAS_ADAPTER = set(LANGS)
print(f"{len(HAS_ADAPTER):,} MMS adapters available")


def run_router(name: str, model_id: str, restrict: set[str] | None, clips: dict[str, np.ndarray]):
    """Argmax over `restrict` (None = the model's whole label space). Returns id -> label."""
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    md = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).to(DEVICE).eval().half()
    id2label = md.config.id2label
    idx = ([i for i, l in id2label.items() if l in restrict] if restrict
           else list(id2label.keys()))
    if not idx:
        raise SystemExit(f"{name}: none of {restrict} in the label space of {model_id}")
    ids = list(clips)
    out: dict[str, str] = {}
    with torch.inference_mode():
        for k in range(0, len(ids), 8):
            ch = ids[k:k + 8]
            inp = fe([clips[i][:16000 * 30] for i in ch], sampling_rate=16000,
                     return_tensors="pt", padding=True)
            # attention_mask is NOT optional. These models have feat_extract_norm="layer" and the
            # sequence-classification head mean-pools over time, so without the mask the zero
            # padding in a variable-length batch is averaged in as signal and the argmax collapses
            # onto a single class. That is exactly what produced a 94%-Luganda phase-2 routing on
            # the first run of stage 1.
            logits = md(inp.input_values.to(DEVICE).half(),
                        attention_mask=inp.attention_mask.to(DEVICE)).logits.float()
            for i, p in zip(ch, logits[:, idx].argmax(-1).cpu().numpy()):
                out[i] = id2label[idx[int(p)]]
            if k % 400 == 0:
                print(f"  {name} {k}/{len(ids)}", flush=True)
    del md
    torch.cuda.empty_cache()
    return out


ROUTERS = [
    ("mms-closed", LID_MODEL, set(LANGS)),
    ("mms-open", LID_MODEL, HAS_ADAPTER),
    ("okwija", OKWIJA, None),
]

results: dict[str, dict] = {}
for name, model_id, restrict in ROUTERS:
    print(f"\n{'=' * 78}\n=== {name}  {model_id}\n{'=' * 78}", flush=True)
    try:
        lab = run_router(name, model_id, restrict, labelled)
        ph2 = run_router(name, model_id, restrict, phase2) if phase2 else {}
    except Exception as e:  # noqa: BLE001 - one dead router must not cost us the other two
        print(f"  {name} FAILED: {type(e).__name__}: {e}")
        continue
    results[name] = {"labelled": lab, "phase2": ph2}

    # Accuracy. An open-set router may answer `nyn`; that is only an error if it would route the
    # clip to the wrong DECODER, so score the mapped-back label, and report the raw one separately.
    correct = sum(1 for i, t in truth.items() if lab.get(i) == t)
    acc = correct / max(1, len(truth))
    off = Counter(lab[i] for i in truth if lab.get(i) not in LANGS)
    print(f"\n  accuracy (exact, over {len(truth):,}): {acc:.4f}  ({correct}/{len(truth)})")
    if off:
        print(f"  answered outside {{lin,sna,lug}} on {sum(off.values()):,} clips: "
              f"{dict(off.most_common(8))}")
    for lg in LANGS:
        pool = [i for i, t in truth.items() if t == lg]
        hit = sum(1 for i in pool if lab.get(i) == lg)
        conf = Counter(lab.get(i) for i in pool if lab.get(i) != lg)
        print(f"    {lg}: recall {hit / max(1, len(pool)):.4f} ({hit}/{len(pool)})"
              f"   confused with {dict(conf.most_common(4)) if conf else '-'}")
    results[name]["accuracy"] = acc
    if ph2:
        mix = Counter(ph2.values())
        tot = sum(mix.values())
        print(f"  phase-2 mix: " + "  ".join(
            f"{k} {v / tot:.1%}" for k, v in mix.most_common(8)))
        results[name]["phase2_mix"] = {k: v / tot for k, v in mix.items()}

# ------------------------------------------------------------------ agreement on phase 2
print(f"\n{'=' * 78}\n=== PHASE-2 AGREEMENT (no labels exist here — this is the only signal)\n"
      f"{'=' * 78}")
names = [n for n in results if results[n].get("phase2")]
for a in range(len(names)):
    for b in range(a + 1, len(names)):
        x, y = results[names[a]]["phase2"], results[names[b]]["phase2"]
        common = set(x) & set(y)
        agree = sum(1 for i in common if x[i] == y[i])
        print(f"  {names[a]:11} vs {names[b]:11}: {agree / max(1, len(common)):.4f} "
              f"({agree}/{len(common)})")

print(f"\n{'-' * 78}\n  VERDICT\n{'-' * 78}")
if results:
    best = max(results, key=lambda n: results[n].get("accuracy", 0.0))
    ba = results[best].get("accuracy", 0.0)
    print(f"  best router on labelled test-domain audio: {best}  acc={ba:.4f}")
    print(f"  implied cost on 1,500 phase-2 clips at that error rate: "
          f"~{(1 - ba) * 0.75:.4f} of final score if a misroute scores ~0")
    print("  phase-1 true mix is lin 43.9% / sna 41.1% / lug 15.0%; a phase-2 mix far from that")
    print("  is evidence of routing bias, not evidence of a different corpus.")
    print("\n  CAVEAT: accuracy here is measured on phase-1 test audio. Phase 2 is described as")
    print("  new unseen recordings, so treat this as an upper bound, not a guarantee.")

json.dump({n: {k: v for k, v in r.items() if k != "labelled"} for n, r in results.items()},
          open(WORKING / "lid_probe.json", "w"), indent=1)
for n, r in results.items():
    json.dump(r.get("phase2", {}), open(WORKING / f"lid_phase2_{n}.json", "w"), indent=1)
print(f"\nwrote {WORKING / 'lid_probe.json'} and per-router phase-2 routings")
