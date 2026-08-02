"""
Kaggle GPU kernel: the lineup submission, without the private kernel_sources dependency.

Same run as waxal_lineup.py — bakeoff-winning checkpoints, no KenLM, DEV pass then a real
submission pass — but waxal_lineup.py's kernel-metadata.json mounts the phase-2 routing map from
Lethabo's private `waxal-router` kernel via kernel_sources, which is not attachable across Kaggle
accounts (private kernels return 0 results when searched from another account). This script reads
the same routing decision from the repo directly: data/routing/lang_map_okwija_phase2.json is
already committed, so no kernel mount is needed.

Meant to be pushed directly via `kaggle kernels push` rather than typed into the notebook editor —
browser-automated typing into Kaggle's Monaco-in-iframe editor proved unreliable (reproducible even
on a fresh notebook: the cell shows focus but keystrokes don't land), so this runs as a plain
script kernel instead.

WAXAL_PLUS_PERIOD="lin,sna,lug" and WAXAL_NO_LM="1" are both deliberate, not defaults. Measured
2026-08-02 across four combinations (LM on/off x period on/off) reweighted to the CORRECTED
phase-2 mix (lin 50.0% / sna 49.9% / lug 0.1% — see that day's log entries for how this differs
from the withdrawn set): no-LM beats with-LM in both period conditions (0.7899 vs 0.7754 with
period on; 0.7800 vs 0.7642 with period off), and period-on beats period-off in both LM
conditions. no-LM + period is the best of the four at 0.7899. Do not "fix" either of these
without a fresh measurement — an earlier version of this file had the opposite belief about
period, from misreading a diagnostic field that double-counted it.
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

import torch  # noqa: E402 — after pip, so this is the version we actually decode on

print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
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
env["WAXAL_NO_LM"] = "1"
env["WAXAL_BACKEND"] = "waxalnet"
env["WAXAL_BACKENDS"] = "lin=waxalnet,sna=waxalnet,lug=waxalnet"
env["WAXAL_LIN"] = "douyeszn/w2vbert-lin-waxal-aug-ft"
env["WAXAL_SNA"] = "waxal-benchmarking/mms-300m-waxal-sna"
env["WAXAL_LUG"] = "waxal-benchmarking/mms-300m-waxal-lug"
env["WAXAL_PLUS_PERIOD"] = "lin,sna,lug"
env["WAXAL_RUN_TAG"] = "lineup"
env["WAXAL_LANG_MAP"] = str(LANG_MAP)

dev_env = dict(env, WAXAL_DEV="1")
print("\n=== RUN 1/2: DEV ===")
sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")], env=dev_env)

dev_path = WORKING / "dev_result_lineup.json"
if dev_path.exists():
    print("DEV RESULT:", json.load(open(dev_path)))
else:
    print("DEV RESULT: (no dev_result_lineup.json written — check the run above)")

print("\n=== RUN 2/2: SUBMISSION ===")
sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")], env=env)

for name in ("submission_03_lineup_lm_phase1.csv", "submission_03_lineup_lm_phase2.csv"):
    csv = WORKING / name
    if csv.exists():
        sh([sys.executable, str(REPO / "local" / "validate_submission.py"), str(csv)], check=False)
    else:
        print(f"\n!! {name} was not written")

print("\n--- /kaggle/working ---", flush=True)
for p in sorted(WORKING.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 1e6:10.1f} MB  {p.relative_to(WORKING)}")
