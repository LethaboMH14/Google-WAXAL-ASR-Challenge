"""
Kaggle GPU kernel: the lineup submission WITH KenLM shallow fusion, self-contained.

Same bakeoff-winning checkpoints and okwija routing as run_selfcontained.py, but this one leaves
WAXAL_NO_LM unset so 03_decode_and_submit.py builds KenLM, runs its 18-combo alpha/beta grid per
language, and decodes with shallow fusion — the paper's ~59% relative WER win, and the largest
single lever in this pipeline per README.md.

WHY THIS IS RUNNING AGAIN (2026-08-02, second time)
----------------------------------------------------
The first LM run's DEV pass said no-LM beats with-LM by 0.0159 on the corrected phase-2 mix — a
small delta. We then submitted the no-LM+period config for real and it scored 0.7065 against a
DEV estimate of 0.7899: an 8.3-point miss. An error bar that size makes every DEV comparison this
session produced, including "LM hurts," unconfirmed — the delta was inside the noise. This run
exists to get a real, non-DEV number for LM fusion, not another DEV opinion about it.
WAXAL_PLUS_PERIOD is set to match the config that scored 0.7065, so this is a single-variable
test (LM on vs LM off) against a real, already-measured baseline, not a two-variable guess.

The LM corpus is built INLINE, in this same session, by running 00_build_lm_corpus.py first and
pointing WAXAL_LM_CORPUS_DIR at its output directly. This is deliberate, not a style choice: the
normal path (ART("waxal-lm") expecting stage 0's output mounted via a kernel_sources attach) has
already silently failed once — 03_decode_and_submit.py's own comments say so ("on 1 Aug that
assumption silently produced nothing... both scored submissions were decoded without a real LM").
Building the corpus in-process removes the cross-kernel mount as a failure mode entirely, same
fix as the routing map in run_selfcontained.py.

Costs more GPU time than the no-LM lineup run: corpus build + KenLM compile are CPU-only work
that still burns wall-clock inside a GPU-billed session, and the alpha/beta sweep is 18 decode
passes per language before the "real" decode even starts. Expect this to run considerably longer
than the ~48 min no-LM baseline.
"""

import json
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
head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
print(f"repo at commit {head}")

sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements-gpu.txt")])
sh([sys.executable, "-m", "pip", "install", "-q", "jiwer"])

# ------------------------------------------------------------------ stage 0: LM corpus, inline
print("\n=== STAGE 0: building LM corpus in this session ===")
sh([sys.executable, str(REPO / "kaggle" / "00_build_lm_corpus.py")])
LM_CORPUS_DIR = WORKING / "lm_corpus"
if not LM_CORPUS_DIR.exists():
    raise SystemExit(f"stage 0 did not write {LM_CORPUS_DIR} — check the log above")
print(f"LM corpus ready at {LM_CORPUS_DIR}:")
for p in sorted(LM_CORPUS_DIR.glob("*.txt")):
    print(f"  {p.name}: {p.stat().st_size / 1e6:.1f} MB")

import torch  # noqa: E402 — after pip, so this is the version we actually decode on

print(f"\ntorch {torch.__version__}  cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("no CUDA — set the kernel accelerator to GPU T4 x2 before running")

LANG_MAP = REPO / "data" / "routing" / "lang_map_okwija_phase2.json"
routing = json.load(open(LANG_MAP))
counts = {}
for lang in routing.values():
    counts[lang] = counts.get(lang, 0) + 1
print(f"phase-2 routing map (okwija): {counts}  n={len(routing)}")

env = dict(os.environ)
env["PYTHONUNBUFFERED"] = "1"
env["CUDA_VISIBLE_DEVICES"] = "0"
# WAXAL_NO_LM intentionally NOT set — this is the whole point of this variant.
env["WAXAL_LM_CORPUS_DIR"] = str(LM_CORPUS_DIR)
env["WAXAL_BACKEND"] = "waxalnet"
env["WAXAL_BACKENDS"] = "lin=waxalnet,sna=waxalnet,lug=waxalnet"
env["WAXAL_LIN"] = "douyeszn/w2vbert-lin-waxal-aug-ft"
env["WAXAL_SNA"] = "waxal-benchmarking/mms-300m-waxal-sna"
env["WAXAL_LUG"] = "waxal-benchmarking/mms-300m-waxal-lug"
env["WAXAL_PLUS_PERIOD"] = "lin,sna,lug"  # matches the config that actually scored 0.7065
env["WAXAL_RUN_TAG"] = "lineup-lm"
env["WAXAL_LANG_MAP"] = str(LANG_MAP)

dev_env = dict(env, WAXAL_DEV="1")
print("\n=== RUN 1/2: DEV (with shallow fusion + alpha/beta tuning) ===")
sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")], env=dev_env)

dev_path = WORKING / "dev_result_lineup-lm.json"
if dev_path.exists():
    print("DEV RESULT:", json.load(open(dev_path)))
else:
    print("DEV RESULT: (no dev_result_lineup-lm.json written — check the run above)")

tuning_path = WORKING / "lm_tuning.json"
if tuning_path.exists():
    print("ALPHA/BETA TUNING:", json.load(open(tuning_path)))

print("\n=== RUN 2/2: SUBMISSION ===")
sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")], env=env)

for name in ("submission_03_lineup-lm_lm_phase1.csv", "submission_03_lineup-lm_lm_phase2.csv"):
    csv = WORKING / name
    if csv.exists():
        sh([sys.executable, str(REPO / "local" / "validate_submission.py"), str(csv)], check=False)
    else:
        print(f"\n!! {name} was not written")

print("\n--- /kaggle/working ---", flush=True)
for p in sorted(WORKING.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 1e6:10.1f} MB  {p.relative_to(WORKING)}")
