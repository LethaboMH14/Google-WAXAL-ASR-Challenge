"""Kaggle GPU kernel: train our own Shona w2v-bert-2.0 CTC head.

WHY TRAIN RATHER THAN SWAP ANOTHER CHECKPOINT
---------------------------------------------
Five checkpoint swaps, five losses: whisper-sna 0.6894, badrex+misterkissi 0.7139, and douyeszn
0.7350 (Lethabo's own run of it). His unidentified 15/16 lineup holds at 0.745734 and nothing on
the Hub has beaten it. Meanwhile every post-processing lever is now measured and small — casing
+0.0007, comma restoration +0.0015 at threshold 0.9, KenLM lands ~0.72 — so stacking them tops out
near 0.750 against a leader at 0.757995. The remaining gap is acoustic, which means a model.

THE SHAPE, AND WHY IT DIFFERS FROM THE MULTILINGUAL DEFAULT
-----------------------------------------------------------
02_train_w2vbert.py trains one shared head over lin+sna+lug because phase 2 ships no language
metadata. That argument no longer binds: we route phase 2 with okwija, and its map agrees with
EVERY scored submission — ours and his — on 891-892 of 892 clips, verified by recovering each
file's implied routing from its own output text. A per-language head has an accurate router to sit
behind.

So WAXAL_LANGS=sna. Corrected phase 2 is ~50% Shona (446 lin / 445 sna / 1 lug), so Shona is where
half the score lives, and douyeszn's per-language Shona card publishes 0.7945 on a speaker-disjoint
split — a real number to aim at instead of a guess.

WAXAL_LOWERCASE=0 — CASED, on evidence that arrived today
----------------------------------------------------------
02_train_w2vbert.py lowercases, citing the organisers' starter notebook, which lowercases both
sides before scoring. That is wrong about the live grader. Lethabo's submissions 15 and 16 are the
same text differing ONLY in capitalisation (892/892 rows identical ignoring case) and scored
0.745030 vs 0.745734. Identical text under a lowercasing grader cannot produce different scores.
Casing is worth +0.000703, and a model that never emits a capital can never collect it. douyeszn's
card keeps case in the vocab too.

RULES AND OVERFITTING
---------------------
Trains on the `train` split and early-stops on `validation`; the `test` split is never opened —
the guard is in 02_train_w2vbert.py itself. SpecAugment stays on precisely because phase 2 is a
generalisation test to unseen speakers, which is the same reason douyeszn used noise + speed
augmentation. Base model facebook/w2v-bert-2.0 is openly available, which is the rule's test.

TIME
----
The script's own measurement: ~24 s/step on one T4, ~13-16 s/step on T4 x2. WAXAL_MAX_STEPS=2000
lands around 7.8 h on T4 x2, inside Kaggle's 12 h ceiling. The run is checkpointed and resumable —
re-running this kernel picks up from the last checkpoint in /kaggle/working rather than restarting,
so a session that dies at hour 6 does not cost those 6 hours.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")
WORKING = Path("/kaggle/working")


def sh(cmd, check=True, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=check, **kw)


if not REPO.exists():
    sh(["git", "clone", "--depth", "1", REPO_URL, str(REPO)])
print("repo at", subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip())

sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements-gpu.txt")])
sh([sys.executable, "-m", "pip", "install", "-q", "jiwer"])

import torch  # noqa: E402

print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  n={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("no CUDA — machine_shape must be NvidiaTeslaT4")
_cap = torch.cuda.get_device_capability()
print(f"gpu: {torch.cuda.get_device_name(0)}  sm_{_cap[0]}{_cap[1]}")
if f"sm_{_cap[0]}{_cap[1]}" not in torch.cuda.get_arch_list():
    raise SystemExit(
        f"torch has no kernels for sm_{_cap[0]}{_cap[1]} — a P100 (sm_60) cannot run this. Set "
        f"machine_shape to NvidiaTeslaT4 in kernel-metadata.json.")

# Resume state lives in /kaggle/working, which persists across versions of the same kernel, so a
# re-run continues rather than restarting. Report it up front: silently restarting from step 0
# after a 6-hour session would be the expensive failure here.
ckpts = sorted(WORKING.glob("w2vbert-waxal/checkpoint-*"))
print(f"\nexisting checkpoints: {[p.name for p in ckpts] or 'none — starting fresh'}")

env = dict(os.environ)
env["PYTHONUNBUFFERED"] = "1"
env["WAXAL_LANGS"] = "sna"          # per-language head; okwija routes phase 2 at 99.9%
env["WAXAL_LOWERCASE"] = "0"        # casing is scored: +0.000703, measured on his 15 vs 16
env["WAXAL_MAX_STEPS"] = os.environ.get("WAXAL_MAX_STEPS", "2000")
env["WAXAL_ZINDI_DIR"] = str(REPO / "data" / "zindi")

print(f"\nWAXAL_LANGS={env['WAXAL_LANGS']}  WAXAL_LOWERCASE={env['WAXAL_LOWERCASE']}  "
      f"WAXAL_MAX_STEPS={env['WAXAL_MAX_STEPS']}")
sh([sys.executable, str(REPO / "kaggle" / "02_train_w2vbert.py")], env=env)

print("\n--- /kaggle/working ---", flush=True)
for p in sorted(WORKING.rglob("*")):
    if p.is_file() and p.stat().st_size > 1_000_000:
        print(f"{p.stat().st_size / 1e6:10.1f} MB  {p.relative_to(WORKING)}")
