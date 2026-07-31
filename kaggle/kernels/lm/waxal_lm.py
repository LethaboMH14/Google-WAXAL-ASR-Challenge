"""
Kaggle CPU kernel: STAGE 0 (the LM corpora) followed by the open-set LID calibration.

CPU on purpose. Both jobs are download-bound, neither needs a GPU, and Kaggle meters CPU
sessions separately from the 30 GPU-hours/week — so this runs alongside stage 2's training
kernel without taking a minute from it.

Why these two together: they are the two things we can learn today that we currently cannot.
Stage 0 produces the text behind KenLM shallow fusion, which arXiv:2512.10968 measures at ~59%
relative WER reduction on Luganda and Shona — the biggest single lever in the pipeline. The
calibration produces the open-set LID accuracy number Sbu asked for, which decides whether the
phase-2 submission is trustworthy at all. Stage 0 runs first and the calibration is allowed to
fail, so a problem in the diagnostic cannot cost us the corpus.

The kernel slug is `waxal-lm` deliberately: Kaggle mounts a kernel's output at
/kaggle/input/<slug>, and stage 3 reads its corpus from ART("waxal-lm") / "lm_corpus". Matching
the slug to the artefact name is what makes stage 3 need no edits to find this.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")           # outside /kaggle/working, which is the 20 GB output volume


def sh(cmd, check=True, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=check, **kw)


if not REPO.exists():
    sh(["git", "clone", "--depth", "1", REPO_URL, str(REPO)])
head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
print(f"repo at commit {head}")

sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements-gpu.txt")])

env = dict(os.environ)
env.setdefault("PYTHONUNBUFFERED", "1")

# ---------------------------------------------------------------- stage 0: the corpora
sh([sys.executable, str(REPO / "kaggle" / "00_build_lm_corpus.py")], env=env)

print("\n" + "=" * 78)
print("stage 0 done. Corpus is safe in /kaggle/working/lm_corpus regardless of what follows.")
print("=" * 78, flush=True)

# ---------------------------------------------------------------- the LID calibration
# check=False: this is a diagnostic. If it dies, the corpus above is still the kernel's output
# and still the thing we came for.
r = sh([sys.executable, str(REPO / "local" / "calibrate_lid_openset.py"), "100"],
       check=False, env=env)
if r.returncode:
    print(f"\ncalibration exited {r.returncode} — corpus is unaffected; rerun the diagnostic "
          f"on its own rather than rebuilding the corpus.")

print("\n--- /kaggle/working ---")
for p in sorted(Path("/kaggle/working").rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 1e6:10.1f} MB  {p.relative_to('/kaggle/working')}")
