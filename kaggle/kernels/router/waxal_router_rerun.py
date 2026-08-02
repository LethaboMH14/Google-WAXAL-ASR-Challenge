"""
Kaggle GPU kernel: regenerate the phase-2 routing map against the CORRECTED test audio.

WHY THIS EXISTS
---------------
On 2026-08-02 the organisers withdrew the phase-2 test set they had shipped a week earlier
(Zindi discussion #34268) and published a replacement. The id spaces are disjoint — 1,500 clips
with five-character ids became 892 with six — so every routing map we hold matches zero clips and
is void. They are parked in data/routing/withdrawn-phase2-2026-08-02/.

WHY OKWIJA ONLY
---------------
The router bakeoff does not need repeating. It was scored on LABELLED phase-1 clips, which the
swap did not touch: okwija 0.9917, mms-closed 0.9792, mms-open 0.9700, asr-conf 0.9658. The
winner is still the winner. What is void is only the phase-2 *map* okwija produced, so this runs
the one model on the one new input and writes the one file. Running the full probe again would
cost three extra model loads to re-derive numbers we already trust.

WHAT TO WATCH IN THE OUTPUT
---------------------------
The language mix. The withdrawn set came out ~95% Luganda, and a great deal of our reasoning
leaned on that. It was measured on audio that no longer exists. Treat the mix this run prints as
the first real observation of the corrected set, and do not assume it matches.
"""

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")
WORKING = Path("/kaggle/working")

# The corrected archive. The old audio.zip is a hard 404 now, so a stale URL fails loudly.
PHASE2_URL = "https://storage.googleapis.com/waxalphase2/newaudios.zip"
OKWIJA = "Okwija/waxal-lid-lin-sna-lug"
LANGS = ["lin", "sna", "lug"]


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
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if DEVICE != "cuda":
    raise SystemExit("no CUDA — set the kernel accelerator to GPU T4 before running")

# ------------------------------------------------------------------ the ids we must cover
# Read the expected id list from the repo rather than trusting the zip alone: the submission is
# scored against these, so a clip in the zip that is not here (or vice versa) is a problem worth
# seeing now instead of at submission time.
expected = pd.read_csv(REPO / "data" / "zindi" / "Test_phase2.csv", escapechar="\\")
expected_ids = set(expected["ID"].astype(str))
print(f"Test_phase2.csv: {len(expected_ids):,} ids, e.g. {sorted(expected_ids)[:3]}")

# ------------------------------------------------------------------ audio
phase2: dict[str, "np.ndarray"] = {}
zp = WORKING / "phase2_audio_new.zip"
if not zp.exists():
    os.system(f"wget -q -O {zp} {PHASE2_URL}")
if not (zp.exists() and zp.stat().st_size > 0):
    raise SystemExit(f"phase-2 audio did not download from {PHASE2_URL}")
print(f"archive: {zp.stat().st_size / 1e6:,.1f} MB")

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
print(f"loaded {len(phase2):,} clips")

# Reconcile against the scored id list rather than against a hardcoded count — the count is
# exactly the thing that changed under us this week.
missing = expected_ids - set(phase2)
extra = set(phase2) - expected_ids
if missing:
    raise SystemExit(f"{len(missing):,} scored id(s) have no audio, e.g. {sorted(missing)[:5]}. "
                     f"Routing them is impossible; check the archive before going further.")
if extra:
    print(f"  note: {len(extra):,} clip(s) in the archive are not in Test_phase2.csv "
          f"(e.g. {sorted(extra)[:5]}) — routed anyway, harmless, just not scored.")

# ------------------------------------------------------------------ okwija
fe = AutoFeatureExtractor.from_pretrained(OKWIJA)
md = Wav2Vec2ForSequenceClassification.from_pretrained(OKWIJA).to(DEVICE).eval().half()
id2label = md.config.id2label
idx = list(id2label.keys())
print(f"okwija label space: {[id2label[i] for i in idx]}")

ids = sorted(phase2)
lang_map: dict[str, str] = {}
with torch.inference_mode():
    for k in range(0, len(ids), 8):
        ch = ids[k:k + 8]
        inp = fe([phase2[i][:16000 * 30] for i in ch], sampling_rate=16000,
                 return_tensors="pt", padding=True)
        # attention_mask is NOT optional — same reason as in waxal_lid_probe.py. These models
        # mean-pool over time, so without the mask the zero padding in a variable-length batch is
        # averaged in as signal and the argmax collapses onto one class. That artefact is what
        # produced a bogus 94%-Luganda routing the first time stage 1 ran.
        logits = md(inp.input_values.to(DEVICE).half(),
                    attention_mask=inp.attention_mask.to(DEVICE)).logits.float()
        for i, p in zip(ch, logits.argmax(-1).cpu().numpy()):
            lang_map[i] = id2label[int(p)]
        if k % 200 == 0:
            print(f"  okwija {k}/{len(ids)}", flush=True)

bad = sorted({v for v in lang_map.values() if v not in LANGS})
if bad:
    raise SystemExit(f"okwija emitted {bad}, outside {LANGS} — stage 3 would refuse this map.")

out = WORKING / "lang_map_okwija_phase2.json"
json.dump(lang_map, open(out, "w"), indent=1)

mix = pd.Series(list(lang_map.values())).value_counts()
print(f"\n{'=' * 78}\nCORRECTED phase-2 routing — {len(lang_map):,} clips\n{'=' * 78}")
for lang, n in mix.items():
    print(f"  {lang}: {n:,}  ({n / len(lang_map):.1%})")
print(f"\nwrote {out}")
print("\nWithdrawn set was lug 1,430 / sna 57 / lin 13 (95.3 / 3.8 / 0.9) across 1,500 clips.")
print("Compare deliberately: that mix is not evidence about this one.")

print("\n--- /kaggle/working ---", flush=True)
for p in sorted(WORKING.rglob("*")):
    if p.is_file() and p.suffix != ".zip":
        print(f"{p.stat().st_size / 1e6:10.2f} MB  {p.relative_to(WORKING)}")
