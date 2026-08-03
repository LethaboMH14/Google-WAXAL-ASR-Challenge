"""
Kaggle GPU kernel: cross-check the phase-2 routing with a SECOND, independent router.

WHY
---
okwija routes the corrected phase-2 set as lin 446 / sna 445 / lug 1 — i.e. 0.1% Luganda.
Phase 1's mix is known ground truth, because its ids carry the language (`lug_96114`), and it is
lin 1,866 / sna 1,749 / lug 638 = 15.0% Luganda. A drop from 15% to 0.1% is not a small drift,
and 446/445 is suspiciously close to a perfect two-way split — the shape a 3-class classifier
makes when it collapses onto two classes.

If phase 2 really does contain ~15% Luganda (~130 clips) and we are decoding all of them with the
lin or sna model, that is a large uncorrected error, and it would plausibly explain why both of
our real submissions (0.7065, 0.7131) sit ~0.04 below Lethabo's 0.7450.

So: run `mms-closed` (facebook/mms-lid-256 restricted to the three languages) over the same audio
and see whether it agrees. mms-lid-256 is architecturally independent of okwija, and on labelled
phase-1 clips it measured 0.9792 vs okwija's 0.9917 — close enough that a *large* disagreement
here is evidence about the audio, not about one model being generally weak.

okwija is re-run in the same session rather than read from the committed map. Same reason the
bakeoff re-runs its incumbents: a control that shares a session and an audio cache with the
challenger is the only kind that rules out "the two numbers came from different inputs."

Costs one model download more than the last router run. No decoding, so still minutes not hours.
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

PHASE2_URL = "https://storage.googleapis.com/waxalphase2/newaudios.zip"
OKWIJA = "Okwija/waxal-lid-lin-sna-lug"
MMS_LID = "facebook/mms-lid-256"
LANGS = ["lin", "sna", "lug"]

# Phase 1's mix, from ids that encode the language. This is ground truth, not an estimate, and it
# is the yardstick the phase-2 numbers below get read against.
PHASE1_MIX = {"lin": 1866, "sna": 1749, "lug": 638}


def sh(cmd, check=True, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=check, **kw)


if not REPO.exists():
    sh(["git", "clone", "--depth", "1", REPO_URL, str(REPO)])
head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
print(f"repo at commit {head}")

sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements-gpu.txt")])

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if DEVICE != "cuda":
    raise SystemExit("no CUDA — set the kernel accelerator to GPU T4 before running")

# ------------------------------------------------------------------ audio (one cache, both routers)
import soundfile as sf  # noqa: E402

phase2: dict[str, "np.ndarray"] = {}
zp = WORKING / "phase2_audio_new.zip"
if not zp.exists():
    os.system(f"wget -q -O {zp} {PHASE2_URL}")
if not (zp.exists() and zp.stat().st_size > 0):
    raise SystemExit(f"phase-2 audio did not download from {PHASE2_URL}")
print(f"archive: {zp.stat().st_size / 1e6:,.1f} MB")

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}
failed = []
with zipfile.ZipFile(zp) as zf:
    for nm in [x for x in zf.namelist() if not x.endswith("/")]:
        if nm.startswith("__MACOSX") or Path(nm).suffix.lower() not in AUDIO_EXT:
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
if failed:
    print(f"  FAILED to decode {len(failed):,}: {failed[:5]}")
print(f"loaded {len(phase2):,} clips")

ids = sorted(phase2)


def run_router(name: str, model_id: str, restrict: set[str] | None) -> dict[str, str]:
    """Argmax over `restrict` (None = the model's whole label space). Returns id -> label."""
    print(f"\n{'=' * 78}\n=== {name}  {model_id}\n{'=' * 78}", flush=True)
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    md = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).to(DEVICE).eval().half()
    id2label = md.config.id2label
    idx = ([i for i, l in id2label.items() if l in restrict] if restrict
           else list(id2label.keys()))
    if not idx:
        raise SystemExit(f"{name}: none of {restrict} in the label space of {model_id}")
    print(f"  label space in use: {[id2label[i] for i in idx]}")
    out: dict[str, str] = {}
    with torch.inference_mode():
        for k in range(0, len(ids), 8):
            ch = ids[k:k + 8]
            inp = fe([phase2[i][:16000 * 30] for i in ch], sampling_rate=16000,
                     return_tensors="pt", padding=True)
            # attention_mask is NOT optional — these models mean-pool over time, so without it the
            # zero padding in a variable-length batch is averaged in as signal and the argmax
            # collapses onto one class. That artefact produced a bogus 94%-Luganda routing once
            # already; a run diagnosing a suspected class collapse must not reintroduce it.
            logits = md(inp.input_values.to(DEVICE).half(),
                        attention_mask=inp.attention_mask.to(DEVICE)).logits.float()
            for i, p in zip(ch, logits[:, idx].argmax(-1).cpu().numpy()):
                out[i] = id2label[idx[int(p)]]
            if k % 400 == 0:
                print(f"  {name} {k}/{len(ids)}", flush=True)
    del md
    torch.cuda.empty_cache()
    return out


okwija = run_router("okwija", OKWIJA, None)
mmsclosed = run_router("mms-closed", MMS_LID, set(LANGS))

json.dump(okwija, open(WORKING / "lang_map_okwija_phase2.json", "w"), indent=1)
json.dump(mmsclosed, open(WORKING / "lang_map_mmsclosed_phase2.json", "w"), indent=1)

# ------------------------------------------------------------------ the comparison
n = len(ids)
p1_total = sum(PHASE1_MIX.values())
print(f"\n{'=' * 78}\nPHASE-2 MIX, TWO INDEPENDENT ROUTERS ({n:,} clips)\n{'=' * 78}")
print(f"{'lang':<6}{'phase-1 (truth)':>18}{'okwija':>18}{'mms-closed':>18}")
for lg in LANGS:
    p1 = PHASE1_MIX[lg] / p1_total
    a = sum(1 for v in okwija.values() if v == lg)
    b = sum(1 for v in mmsclosed.values() if v == lg)
    print(f"{lg:<6}{p1:>17.1%}{a:>10,} {a / n:>6.1%}{b:>10,} {b / n:>6.1%}")

agree = sum(1 for i in ids if okwija[i] == mmsclosed[i])
print(f"\nrouters agree on {agree:,}/{n:,} clips ({agree / n:.1%})")

print("\nconfusion (rows = okwija, cols = mms-closed):")
print(f"{'':<8}" + "".join(f"{lg:>10}" for lg in LANGS))
for a in LANGS:
    row = Counter(mmsclosed[i] for i in ids if okwija[i] == a)
    print(f"{a:<8}" + "".join(f"{row.get(b, 0):>10,}" for b in LANGS))

print(f"""
{'=' * 78}
HOW TO READ THIS
{'=' * 78}
If mms-closed also says ~0% lug, the near-total absence of Luganda is a property of the corrected
test set and okwija is fine — the 0.04 gap to Lethabo's 0.7450 is somewhere else entirely.

If mms-closed says ~15% lug (matching phase 1), okwija is collapsing lug onto lin/sna on this
audio, every one of those clips is being decoded by the wrong acoustic model, and that is the
first thing to fix — worth far more than the LM/period dials.

Either way this is a routing question answered without spending a submission.""")

print("\n--- /kaggle/working ---", flush=True)
for p in sorted(WORKING.rglob("*")):
    if p.is_file() and p.suffix != ".zip":
        print(f"{p.stat().st_size / 1e6:10.2f} MB  {p.relative_to(WORKING)}")
