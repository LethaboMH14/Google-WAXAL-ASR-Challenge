"""
Kaggle GPU kernel: STAGE 1 — the zero-shot MMS submission.

This is the kernel that ends the state of having NOTHING to upload. Stage 2's fine-tune is the
better model, but it is hours away and its checkpoint still has to be decoded; stage 1 needs no
training at all and writes BOTH `submission_01_mms_zeroshot_phase1.csv` and `..._phase2.csv` in
one pass. Until it finishes, a leaderboard entry is not something we could make even if we
wanted to, and 5 submissions/day means a day not spent submitting is capacity we cannot bank.

It is also the first end-to-end test of the submission plumbing — audio resolution, LID routing,
CTC decode, template-shaped write-out. Finding a formatting bug here costs an hour. Finding it
in the run that carries the fine-tuned model costs the run.

Runs on ONE T4 (see the CUDA_VISIBLE_DEVICES note below, same reasoning as stage 2). Internet
must be ON: the phase-2 audio zip comes from Google storage, the phase-1 audio streams from the
HF test split, and both MMS checkpoints download at start.

Rules note, restated here because this is the file that produces the artefact we upload: the HF
`test` split carries ground-truth `transcription`, and the script drops that column on sight
(section 2b of 01_baseline_submission.py). Reading it would be an explicit disqualification.
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
    # mms-1b-all is 2 B parameters over ~5,700 clips. On CPU this does not finish inside a session.
    raise SystemExit("no CUDA — set the kernel accelerator to GPU T4 x2 before running")

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name}  {p.total_memory / 1e9:.0f} GB  sm_{p.major}{p.minor}")

# Same guard as stage 2, same reason: a P100 is sm_60 and the Kaggle image's torch 2.10+cu128
# carries no Pascal kernels, so `cuda=True` is not the question — whether this torch has kernels
# for THIS card is. Fail in the first second rather than after the model downloads.
_cc = torch.cuda.get_device_capability(0)
_sm = f"sm_{_cc[0]}{_cc[1]}"
if _sm not in torch.cuda.get_arch_list():
    raise SystemExit(
        f"this card is {torch.cuda.get_device_name(0)} ({_sm}) and the installed "
        f"torch {torch.__version__} was built for {torch.cuda.get_arch_list()} — it has no "
        f"kernels for this device.\nSet machine_shape to 'NvidiaTeslaT4' (T4 is sm_75). Note "
        f"the capital N: 'nvidiaTeslaT4' is accepted silently and gives you a P100.")

# ------------------------------------------------------------------ 2. run
env = dict(os.environ)
env.setdefault("PYTHONUNBUFFERED", "1")

# One GPU. Nothing in stage 1 is data-parallel — it is a sequential loop over clips — so the
# second card would sit idle regardless; pinning it makes the run reproducible rather than
# dependent on which device torch happens to pick.
env["CUDA_VISIBLE_DEVICES"] = "0"

sh([sys.executable, str(REPO / "kaggle" / "01_baseline_submission.py")], env=env)

# ------------------------------------------------------------------ 3. validate before download
# The validator is the same one we would run locally, and running it HERE means a malformed CSV
# is known before the file is downloaded rather than after it has been uploaded to Zindi and
# burned one of five daily submissions. check=False: a validation failure must not delete the
# CSVs, because a wrong-shaped file we can inspect beats no file at all.
for name in ("submission_01_mms_zeroshot_phase1.csv", "submission_01_mms_zeroshot_phase2.csv"):
    csv = WORKING / name
    if csv.exists():
        sh([sys.executable, str(REPO / "local" / "validate_submission.py"), str(csv)], check=False)
    else:
        print(f"\n!! {name} was not written — check the TEMPLATES section of the log above")

# ------------------------------------------------------------------ 4. clean the output volume
# The phase-2 audio zip lands in /kaggle/working, and everything in /kaggle/working becomes this
# kernel's output — mounted read-only into whatever runs next and counted against the 20 GB cap.
# It is a re-downloadable input, not a result. Drop it so the output is just the CSVs.
zip_path = WORKING / "phase2_audio.zip"
if zip_path.exists():
    mb = zip_path.stat().st_size / 1e6
    zip_path.unlink()
    print(f"\nremoved phase2_audio.zip ({mb:,.0f} MB) from the output — re-downloadable input")

print("\n--- /kaggle/working ---", flush=True)
for p in sorted(WORKING.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 1e6:10.1f} MB  {p.relative_to(WORKING)}")
