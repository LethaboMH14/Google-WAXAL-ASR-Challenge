"""Kaggle GPU kernel: same bakeoff, for Lingala — the other half of the scored set.

WHY RUN THIS BEFORE SPENDING A SUBMISSION
-----------------------------------------
The Shona bakeoff just returned a result worth generalising from: two checkpoints that have NEVER
seen WAXAL (badrex +0.0351, manassehzw +0.0312) both beat the WAXAL-fine-tuned mms-300m incumbent
on WAXAL's own validation clips. Encoder quality beat domain match, and it did so twice
independently, which says the incumbent's edge was partly the dev set being familiar to it.

Shona is ~50% of corrected phase 2, so +0.035 there is worth roughly +0.018 overall — enough to
beat our own 0.7131 but not the team's 0.7457. Lingala is the other ~50%. If the same effect exists
there, the two together clear 0.7457; separately, neither does. So this runs first and the phase-2
submission carries both winners at once.

THE ASYMMETRY WITH SHONA, WHICH IS WHY THIS MAY FIND NOTHING
------------------------------------------------------------
Shona was on mms-300m — the family that lost Lingala by 0.0895 — so it had obvious headroom.
Lingala is ALREADY on douyeszn/w2vbert-lin-waxal-aug-ft, which is both w2v-bert-2.0 AND WAXAL
fine-tuned: the exact combination that won. There may simply be nothing better available, and a
flat result here is a real answer, not a failed run.

Challengers are the open Lingala field, biggest-first. BrainTheos is the interesting one: built on
mms-1b rather than mms-300m, so it tests scale where Shona tested encoder family.

DEV-only, Lingala-only, one session, one audio cache. No phase-2 decode, no submission spent.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")
WORKING = Path("/kaggle/working")

# (tag, repo, backend). The control runs FIRST so a wrecked session shows up before the challengers
# rather than after, and so the incumbent's number is re-measured here rather than quoted from a
# different run.
CANDIDATES = [
    ("control-w2vb",  "douyeszn/w2vbert-lin-waxal-aug-ft",                 "waxalnet"),
    ("mms300-ctrl",   "waxal-benchmarking/mms-300m-waxal-lin",             "waxalnet"),
    ("braintheos",    "BrainTheos/wav2vec2-large-mms-1b-all-lingala-ojpl", "waxalnet"),
    ("keystats",      "keystats/lingala-xlsr-waxal-finetuned",             "waxalnet"),
    ("exyms-robust",  "exyms/lingala-asr-xlsr-robust",                     "waxalnet"),
    ("misterkissi",   "misterkissi/w2v2-lg-xls-r-300m-lingala",            "waxalnet"),
]


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

print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("no CUDA — set the notebook accelerator to GPU T4 x2 before running")
_cap = torch.cuda.get_device_capability()
print(f"gpu: {torch.cuda.get_device_name(0)}  sm_{_cap[0]}{_cap[1]}")
if f"sm_{_cap[0]}{_cap[1]}" not in torch.cuda.get_arch_list():
    raise SystemExit(
        f"this torch has no kernels for sm_{_cap[0]}{_cap[1]} — set the accelerator to GPU T4 x2 "
        f"in the UI. Pushing via the API silently resets it to P100.")

# Optional token: only the gated candidate needs it, and its absence must not kill the open ones.
try:
    from kaggle_secrets import UserSecretsClient

    tok = (UserSecretsClient().get_secret("HF_TOKEN") or "").strip()
    if tok:
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
        print("HF_TOKEN present")
except Exception as e:  # noqa: BLE001
    print(f"no HF_TOKEN ({type(e).__name__}) — gated candidates will be skipped")

# Access is checked by fetching a real FILE. model_info() reads metadata, which HF serves for gated
# repos to anyone, and trusting it already cost this project a nine-minute run that died on a 403.
from huggingface_hub import hf_hub_download  # noqa: E402

runnable = []
for tag, repo, backend in CANDIDATES:
    try:
        hf_hub_download(repo, "config.json", token=os.environ.get("HF_TOKEN") or None)
        runnable.append((tag, repo, backend))
        print(f"  OK    {tag:<16} {repo}")
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP  {tag:<16} {repo}  ({type(e).__name__})")
if not runnable:
    raise SystemExit("no candidate is reachable — nothing to measure")

results = {}
for tag, repo, backend in runnable:
    print(f"\n{'=' * 78}\n=== {tag}  {repo}  [{backend}]\n{'=' * 78}", flush=True)
    env = dict(os.environ)
    env.update(
        PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0", WAXAL_NO_LM="1", WAXAL_DEV="1",
        WAXAL_DEV_LANGS="lin", WAXAL_BACKEND=backend,
        WAXAL_BACKENDS=f"lin={backend}", WAXAL_LIN=repo,
        # The trailing period is per-architecture: CTC vocabs here contain no '.' while 95.9% of
        # Shona references end in one, but Whisper emits punctuation natively and appending another
        # gives '..', which measured -0.0181. Set to match the architecture, not a global default.
        WAXAL_PLUS_PERIOD="" if backend == "whisper" else "lin",
        WAXAL_RUN_TAG=f"linbake-{tag}",
    )
    r = sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")], env=env, check=False)
    if r.returncode != 0:
        print(f"!! {tag} exited {r.returncode} — recorded as failed, continuing")
        results[tag] = None
        continue
    p = WORKING / f"dev_result_linbake-{tag}.json"
    if p.exists():
        d = json.load(open(p))
        results[tag] = (d.get("per_language") or {}).get("lin", {}).get("multi")
    else:
        results[tag] = None

print(f"\n{'=' * 78}\nLINGALA BAKEOFF — dev multi on the same 395 clips\n{'=' * 78}")
base = results.get("control-w2vb")
print(f"{'candidate':<18}{'multi':>9}{'vs control':>13}   model")
for tag, repo, _ in runnable:
    v = results.get(tag)
    if v is None:
        print(f"{tag:<18}{'FAILED':>9}{'':>13}   {repo}")
        continue
    delta = f"{v - base:+.4f}" if base else "n/a"
    print(f"{tag:<18}{v:>9.4f}{delta:>13}   {repo}")

good = {k: v for k, v in results.items() if v is not None}
if good:
    win = max(good, key=good.get)
    print(f"\nWINNER: {win}  {good[win]:.4f}")
    if base and good[win] > base:
        print(f"  beats the mms-300m incumbent by {good[win] - base:+.4f} on the dev Lingala clips.")
        print("  Next step is a phase-2 run with this checkpoint, single-variable against the\n"
              "  0.7065 no-LM submission — NOT a leaderboard claim from this number.")
    else:
        print("  nothing beats the incumbent; keep douyeszn/w2vbert-lin-waxal-aug-ft and spend no submission here.")
json.dump(results, open(WORKING / "lin_bakeoff.json", "w"), indent=1)
