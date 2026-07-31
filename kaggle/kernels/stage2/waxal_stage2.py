"""
Kaggle kernel wrapper for STAGE 2. This file is the entire notebook: it clones the public repo,
installs what the Kaggle image is missing, and runs `kaggle/02_train_w2vbert.py` unmodified.

Why a wrapper instead of pasting the training script in here: the script is the reviewed artefact
and it lives in git. A copy pasted into a Kaggle notebook is a fork that drifts, and the challenge
rules can ask for our code at any point in the top 10 — one source of truth is worth the extra
40 lines. `git rev-parse HEAD` is printed below so any run can be tied to an exact commit.

The Zindi CSVs come from the clone (they are committed, 8 MB), so this kernel needs no attached
Dataset. Internet must be ON: for the clone, for pip, and for streaming audio from HF.

Leg 2 resume: add this kernel to its own `kernel_sources`, and Kaggle mounts the previous run's
output read-only at /kaggle/input/<slug>. Section 3b below copies the checkpoint from there into
/kaggle/working, because the training script resumes from `get_last_checkpoint(OUTDIR)` and OUTDIR
is /kaggle/working/w2vbert-waxal — a read-only mount is not something Trainer can continue into.
The copy is what buys back the optimizer state, the LR schedule and the step count; initialising
from the weights alone would restart the schedule and waste the warmup.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
# Clone OUTSIDE /kaggle/working. Everything in /kaggle/working becomes the kernel's output, and
# the output has a 20 GB cap that a 7 GB checkpoint plus the final model already eats into.
REPO = Path("/kaggle/repo")


def sh(cmd, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


# ------------------------------------------------------------------ 1. code
if not REPO.exists():
    sh(["git", "clone", "--depth", "1", REPO_URL, str(REPO)])
head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
print(f"repo at commit {head}")

# ------------------------------------------------------------------ 2. deps
# torch and transformers ship with the Kaggle image; datasets does NOT ship at <4.0, and the
# pin matters — see the long note at the bottom of requirements-gpu.txt. Install the whole file
# rather than guessing which pieces are already present; pip is a no-op for the satisfied ones.
sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements-gpu.txt")])

import torch  # noqa: E402  — imported after pip so we see the version we will actually train on

print(f"\ntorch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"n_gpu={torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu{i}: {p.name}  {p.total_memory / 1e9:.0f} GB")
else:
    # Do not burn a committed run producing nothing. A CPU stage 2 is ~200x too slow to finish.
    raise SystemExit("no CUDA — set the kernel accelerator to GPU T4 x2 before running")

# ------------------------------------------------------------------ 3. secrets (optional)
# An HF token only lifts anonymous rate limits; the datasets are public and the run works without
# one. Add it as a Kaggle Secret named HF_TOKEN if streaming starts getting throttled.
try:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets")
except Exception as e:                                    # noqa: BLE001 — any failure is fine
    print(f"no HF_TOKEN secret ({type(e).__name__}); streaming anonymously, which is supported")

# ------------------------------------------------------------------ 3b. resume from leg 1
# /kaggle/input is read-only, so a mounted previous output cannot be trained into directly.
# Copy the newest checkpoint across. Trainer's save_total_limit then rotates it out once it has
# written a newer one, which is what keeps this inside the 20 GB /kaggle/working cap.
import shutil  # noqa: E402

WORKING = Path("/kaggle/working")
prev = sorted(Path("/kaggle/input").glob("*/w2vbert-waxal"))
if prev and not any((WORKING / "w2vbert-waxal").glob("checkpoint-*")):
    src = prev[-1]
    ckpts = sorted(src.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    dst = WORKING / "w2vbert-waxal"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():                               # tokenizer/vocab files are small
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    if ckpts:
        print(f"resuming: copying {ckpts[-1]} -> {dst} (this takes a few minutes)", flush=True)
        shutil.copytree(ckpts[-1], dst / ckpts[-1].name, dirs_exist_ok=True)
    else:
        print(f"mounted {src} has no checkpoint-* — leg 1 must have only written tokenizer files")
else:
    print("no previous kernel output mounted; this is leg 1")

# ------------------------------------------------------------------ 4. run
# 1,500 steps, not 2,500. A Kaggle GPU session is capped at 9 h and a committed run that hits the
# wall does not reliably save its outputs — so the first leg is sized to FINISH, at ~15 s/step on
# 2xT4 under DataParallel (~6.3 h) plus ~30 min of model/audio download. Leg 2 resumes from the
# checkpoint and takes it to 2,500. Losing a finished 1,500-step model to a timeout at step 2,400
# is the failure this number is chosen to avoid.
env = dict(os.environ)
env.setdefault("WAXAL_MAX_STEPS", "1500")
env.setdefault("PYTHONUNBUFFERED", "1")

sh([sys.executable, str(REPO / "kaggle" / "02_train_w2vbert.py")], env=env)

# ------------------------------------------------------------------ 5. what came out
print("\n--- /kaggle/working ---", flush=True)
for p in sorted(Path("/kaggle/working").rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 1e6:10.1f} MB  {p.relative_to('/kaggle/working')}")
