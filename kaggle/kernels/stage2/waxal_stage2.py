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
if not torch.cuda.is_available():
    # Do not burn a committed run producing nothing. A CPU stage 2 is ~200x too slow to finish.
    raise SystemExit("no CUDA — set the kernel accelerator to GPU T4 x2 before running")

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name}  {p.total_memory / 1e9:.0f} GB  sm_{p.major}{p.minor}")

# `torch.cuda.is_available()` is NOT enough, and version 1 of this kernel is the proof: it got a
# P100 (sm_60), reported cuda=True, downloaded the model, entered the training loop and only then
# died with `CUDA error: no kernel image is available for execution on the device` — six minutes
# of a committed run for an answer available in the first second. The Kaggle image ships torch
# 2.10+cu128, whose wheels no longer carry Pascal kernels, so a P100 cannot run this at all
# regardless of our code. Check the arch list instead of the flag.
_cc = torch.cuda.get_device_capability(0)
_sm = f"sm_{_cc[0]}{_cc[1]}"
if _sm not in torch.cuda.get_arch_list():
    raise SystemExit(
        f"this card is {torch.cuda.get_device_name(0)} ({_sm}) and the installed "
        f"torch {torch.__version__} was built for {torch.cuda.get_arch_list()} — it has no "
        f"kernels for this device.\nSet machine_shape to 'NvidiaTeslaT4' (T4 is sm_75). Note "
        f"the capital N: 'nvidiaTeslaT4' is accepted silently and gives you a P100.")

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
env = dict(os.environ)
env.setdefault("PYTHONUNBUFFERED", "1")

# ONE GPU, deliberately, even though Kaggle hands us two. Launched with plain `python`, HF Trainer
# wraps the model in DataParallel, which re-replicates all 581M parameters onto the second card on
# every accumulation pass. Measured on version 2 of this kernel: 32.7 s/step on 2xT4 against a
# 24.4 s/step measurement on ONE T4 doing the same 32 samples. The second GPU was costing us 34%.
#
# The fix that would actually use both cards is DDP via torchrun, which replicates once and
# all-reduces gradients instead. Not done here because the training set is a streaming
# IterableDataset: under DDP each rank must get a DIFFERENT shard, and transformers 5.9's Trainer
# has no split_dataset_by_node call — the sharding would be happening inside accelerate's
# dispatcher, unverified, where getting it wrong silently trains on every sample twice and halves
# the effective batch without erroring. That is a bad thing to discover from a leaderboard score
# two days before close. Single-GPU is 34% faster than what we measured and has no such question.
env["CUDA_VISIBLE_DEVICES"] = "0"

# Stop on the clock, not on a step count (see the StopAfterHours note in the training script).
# 7.5 h against Kaggle's 9 h cap leaves room for setup, the closing eval, and the final save.
env.setdefault("WAXAL_MAX_HOURS", "7.5")

# !! DO NOT PUSH LEG 2. THIS KERNEL IS RETIRED. !!
#
# Leg 1 ran its full 7.5 h and collapsed to the CTC blank solution: WER and CER exactly 1.000 at
# all four evals (steps 250/500/750/998), train loss flat at ~24.0 from step 400 while grad_norm
# fell 167 -> 2. Every hypothesis decodes to the empty string. See docs/sbu-lethabo-log.md,
# 1 Aug, for the evidence and the two artefact traps (the saved model is step 250, not 998).
#
# The annealing arithmetic below is CORRECT AND IRRELEVANT. A perfectly annealed LR schedule on a
# model that emits only blanks still emits only blanks; resuming buys 7.5 h of flat line. We moved
# to facebook/mms-1b-all, whose CTC heads for lin/sna/lug are pretrained rather than randomly
# initialised, which makes this failure mode structurally impossible. Kept below unedited because
# the reasoning is sound for any future run that actually trains.
#
# MAX_STEPS is now only the END of the LR schedule, not a promise about this session.
#
# LEG 2 MUST NOT INHERIT THIS 2,500. Measured on leg 1 from the interval between logging points:
# 26.0 s/step steady state (150 steps in 3,899 s), so a 7.5 h leg is ~1,039 steps and two legs
# reach ~2,077. The scheduler is `linear` with warmup 200, i.e. LR decays to zero AT max_steps —
# so finishing at 2,077 against a 2,500 schedule stops with LR still at
#   5e-5 * (2500-2077)/(2500-200) = 9.2e-6, 18% of peak, never annealed.
# A CTC fine-tune that ends mid-decay is worse than the same compute landed at zero.
#
# So when pushing leg 2: read the actual step number of leg 1's final checkpoint and set
#   WAXAL_MAX_STEPS = <that step> + floor(7.2 * 3600 / 26)     # ~997, leaving setup headroom
# so the decay reaches zero exactly as the wall-clock budget runs out. Do NOT guess it in
# advance — if leg 1 stops early the schedule is wrong in the other direction, and a schedule
# rebuilt SHORTER than the steps already taken makes the LR jump at the seam.
#
# ~2,080 total sits at the low end of the 2,000-3,500 band WAXAL-NET converged in. A third leg
# would buy more, at 7.5 h against a 30 GPU-hour week that also has to carry stage 1 and stage 3.
env.setdefault("WAXAL_MAX_STEPS", "2500")

# Checkpoint every 250 steps (~100 min), not the 100 the script derives. Each checkpoint is ~7 GB;
# at 24 s/step, every 100 steps means writing 7 GB every 40 minutes, and that I/O comes straight
# out of the training budget. The session is committed and not expected to be interrupted, so the
# thing being insured against here is only a crash.
env.setdefault("WAXAL_EVAL_EVERY", "250")

sh([sys.executable, str(REPO / "kaggle" / "02_train_w2vbert.py")], env=env)

# ------------------------------------------------------------------ 5. what came out
print("\n--- /kaggle/working ---", flush=True)
for p in sorted(Path("/kaggle/working").rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 1e6:10.1f} MB  {p.relative_to('/kaggle/working')}")
