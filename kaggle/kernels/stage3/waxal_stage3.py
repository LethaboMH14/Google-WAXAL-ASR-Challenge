"""
Kaggle GPU kernel: STAGE 3 — KenLM shallow fusion on top of MMS. The biggest lever we have left.

Stage 2 was meant to feed this kernel a fine-tuned w2v-bert. It collapsed to the CTC blank
solution instead (WER and CER exactly 1.000 at every eval — see docs/sbu-lethabo-log.md, 1 Aug),
so `kaggle/03_decode_and_submit.py` now defaults to BACKEND=mms and this kernel does NOT mount a
stage 2 checkpoint. That is a demotion in the acoustic model and it is worth being clear-eyed
about, but the fusion itself is untouched: the published numbers for this technique on these
languages are lug 39.75 -> 16.30 WER and sna 22.56 -> 9.28, and none of that came from the
fine-tune. It comes from the 5-gram, and the 5-gram is built from the stage 0 corpus mounted below.

What this kernel needs mounted (kernel_sources):
  - lethabomh14/waxal-lm        stage 0's lm_corpus/ — the external text the KenLMs are built from.
                                Without it every language silently falls back to a Train.csv-only
                                LM, which is a fraction of the win and reads as a bad score rather
                                than a missing input.
  - lethabomh14/waxal-baseline  stage 1's lang_map.json — the open-set LID routing decisions.
                                Reusing them means we do not re-run LID, and it is the routing we
                                already calibrated at 97.7% (293/300) against the phase-1 prefixes.

Runs on ONE T4. Internet ON: KenLM is compiled from source, mms-1b-all and its adapters download
from the Hub, phase-2 audio comes from Google storage and phase-1 audio streams from HF.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")           # outside /kaggle/working, which is the 20 GB output volume
WORKING = Path("/kaggle/working")


def sh(cmd, check=True, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=check, **kw)


# ------------------------------------------------------------------ 1. code
if not REPO.exists():
    sh(["git", "clone", "--depth", "1", REPO_URL, str(REPO)])
head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
print(f"repo at commit {head}")

sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements-gpu.txt")])

import torch  # noqa: E402  — after pip, so this is the version we will actually decode on

print(f"\ntorch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"n_gpu={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("no CUDA — set the kernel accelerator to GPU T4 x2 before running")

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name}  {p.total_memory / 1e9:.0f} GB  sm_{p.major}{p.minor}")

# Same guard as stages 1 and 2: `cuda=True` is not the question, whether this torch has kernels
# for THIS card is. A P100 is sm_60 and torch 2.10+cu128 ships no Pascal kernels.
_cc = torch.cuda.get_device_capability(0)
_sm = f"sm_{_cc[0]}{_cc[1]}"
if _sm not in torch.cuda.get_arch_list():
    raise SystemExit(
        f"this card is {torch.cuda.get_device_name(0)} ({_sm}) and the installed "
        f"torch {torch.__version__} was built for {torch.cuda.get_arch_list()} — it has no "
        f"kernels for this device.\nSet machine_shape to 'NvidiaTeslaT4' (T4 is sm_75). Note "
        f"the capital N: 'nvidiaTeslaT4' is accepted silently and gives you a P100.")

# ------------------------------------------------------------------ 2. show what is mounted
# The script resolves its mounts by CONTENT rather than by name (artefact names are not mount
# names), and prints its own warnings. Print the raw listing anyway: when a score comes back
# disappointing, "was the LM corpus actually attached?" is the first question, and it should be
# answerable from this log without re-running anything.
print("\n--- /kaggle/input ---")
inp = Path("/kaggle/input")
if inp.exists():
    for d in sorted(inp.glob("*")):
        kids = sorted(c.name for c in d.glob("*"))[:8]
        print(f"  {d.name}: {kids}")
else:
    print("  nothing mounted — the LM will fall back to Train.csv and LID will be re-run")

# ------------------------------------------------------------------ 3. run
env = dict(os.environ)
env.setdefault("PYTHONUNBUFFERED", "1")

# One GPU: the decode is a sequential loop over clips, so the second card sits idle either way.
# Pinning makes the run reproducible rather than dependent on which device torch picks.
env["CUDA_VISIBLE_DEVICES"] = "0"

# MMS, explicitly, rather than relying on the script's default. This is the line to change if the
# acoustic model is ever swapped again, and being explicit means the log records which one ran.
env.setdefault("WAXAL_BACKEND", "mms")

sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")], env=env)

# ------------------------------------------------------------------ 4. validate before download
# Same reasoning as stage 1: a malformed CSV should be known here, not after it has been uploaded
# to Zindi and burned one of five daily submissions. check=False so a validation failure never
# deletes the CSV — a wrong-shaped file we can inspect beats no file at all.
for name in ("submission_03_mms_lm_phase1.csv", "submission_03_mms_lm_phase2.csv"):
    csv = WORKING / name
    if csv.exists():
        sh([sys.executable, str(REPO / "local" / "validate_submission.py"), str(csv)], check=False)
    else:
        print(f"\n!! {name} was not written — check the TEMPLATES section of the log above")

# ------------------------------------------------------------------ 5. clean the output volume
# Everything left in /kaggle/working becomes this kernel's output and counts against 20 GB. The
# phase-2 zip is a re-downloadable input and the LM working files are intermediates; the .arpa is
# already deleted by the script, but the plain-text corpora it writes are hundreds of MB. Keep the
# compiled .bin (small, and the thing that would be slow to rebuild) and the CSVs.
zip_path = WORKING / "phase2_audio.zip"
if zip_path.exists():
    mb = zip_path.stat().st_size / 1e6
    zip_path.unlink()
    print(f"\nremoved phase2_audio.zip ({mb:,.0f} MB) from the output — re-downloadable input")

for txt in sorted((WORKING / "lm").glob("*.txt")) if (WORKING / "lm").exists() else []:
    mb = txt.stat().st_size / 1e6
    txt.unlink()
    print(f"removed lm/{txt.name} ({mb:,.0f} MB) — rebuilt from the stage 0 mount on demand")

print("\n--- /kaggle/working ---", flush=True)
for p in sorted(WORKING.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 1e6:10.1f} MB  {p.relative_to(WORKING)}")
