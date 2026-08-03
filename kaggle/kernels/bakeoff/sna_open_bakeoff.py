"""Kaggle GPU kernel: bake off every UNGATED Shona checkpoint against the incumbent.

WHY SHONA, AND WHY ONLY SHONA
-----------------------------
Corrected phase 2 routes lin 446 / sna 445 / lug 1. Shona is half the scored set, Luganda is one
clip. Lingala already sits on douyeszn/w2vbert-lin-waxal-aug-ft (0.7788) which beat its mms-300m
control by +0.0895. Shona is still on waxal-benchmarking/mms-300m-waxal-sna (0.7815) — the same
family that lost Lingala by that margin. So the single largest measurable lever we can still reach
is the Shona checkpoint, and this run prices every open option for it at once.

WHY A BAKEOFF INSTEAD OF PICKING ONE
------------------------------------
No ungated candidate has both properties that made the Lingala winner win. That model is BOTH
w2v-bert-2.0 AND fine-tuned on WAXAL. Among open Shona checkpoints you get one or the other:

    badrex/w2v-bert-2.0-shona-asr        w2v-bert-2.0   badrex/shona-speech    out-of-domain
    manassehzw/sna-w2v-bert-2.0-asr      w2v-bert-2.0   own annotated set      out-of-domain
    noirlab/whisper-large-v3-shona-asr   whisper-lg-v3  FLEURS                 out-of-domain
    imaginashaun/whisper-tiny-shona-waxal whisper-tiny  google/WaxalNLP        IN-domain, tiny

Encoder quality and domain match pull in opposite directions here and we have no measurement that
settles which wins on this audio, so guessing would just be the whisper-sna mistake again. Every
candidate decodes the SAME 370 dev Shona clips in ONE session off ONE audio cache, which is the
only way a comparison rules out "the two numbers came from different inputs".

DEV-ONLY AND SHONA-ONLY, so this is cheap: no phase-2 decode, no submission written, no submission
spent. WAXAL_DEV_LANGS=sna skips lin and lug entirely.

READING THE NUMBERS
-------------------
This harness has been wrong by up to 0.08 in ABSOLUTE terms on the corrected set, so no number
below is a leaderboard prediction. It is reliable for exactly what it is used for here: an A/B of
one language against a control measured the same way, in the same session, on the same clips —
which is how docs/MODEL-CANDIDATES.md's bakeoff table was built in the first place.

PROVENANCE — all four terminate at a public base model (facebook/w2v-bert-2.0, openai/whisper-*),
all are ungated, all carry a license, and three of the four were created BEFORE the challenge data
dropped on 2026-06-30, so they cannot have been fitted to the public phase-1 test key. That is a
stronger position than the Mubarak127 checkpoint the team rejected on 1 Aug, whose declared base
model was itself.
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
    ("control-mms",   "waxal-benchmarking/mms-300m-waxal-sna",  "waxalnet"),
    ("badrex-w2vb",   "badrex/w2v-bert-2.0-shona-asr",          "waxalnet"),
    ("manassehzw",    "manassehzw/sna-w2v-bert-2.0-asr",        "waxalnet"),
    ("noirlab-whlg",  "noirlab/whisper-large-v3-shona-asr",     "whisper"),
    ("imaginashaun",  "imaginashaun/whisper-tiny-shona-waxal",  "whisper"),
    # Gated "auto" — instant IF the terms were accepted by the account holding HF_TOKEN. Included
    # because it is the only candidate that is both w2v-bert AND WAXAL-trained, i.e. the exact
    # combination that won Lingala. Skipped without failing the run if access is not there.
    ("douyeszn-gated", "douyeszn/w2vbert-sna-waxal-aug",        "waxalnet"),
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
        WAXAL_DEV_LANGS="sna", WAXAL_BACKEND=backend,
        WAXAL_BACKENDS=f"sna={backend}", WAXAL_SNA=repo,
        # The trailing period is per-architecture: CTC vocabs here contain no '.' while 95.9% of
        # Shona references end in one, but Whisper emits punctuation natively and appending another
        # gives '..', which measured -0.0181. Set to match the architecture, not a global default.
        WAXAL_PLUS_PERIOD="" if backend == "whisper" else "sna",
        WAXAL_RUN_TAG=f"snabake-{tag}",
    )
    r = sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")], env=env, check=False)
    if r.returncode != 0:
        print(f"!! {tag} exited {r.returncode} — recorded as failed, continuing")
        results[tag] = None
        continue
    p = WORKING / f"dev_result_snabake-{tag}.json"
    if p.exists():
        d = json.load(open(p))
        results[tag] = (d.get("per_language") or {}).get("sna", {}).get("multi")
    else:
        results[tag] = None

print(f"\n{'=' * 78}\nSHONA BAKEOFF — dev multi on the same 370 clips\n{'=' * 78}")
base = results.get("control-mms")
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
        print(f"  beats the mms-300m incumbent by {good[win] - base:+.4f} on the dev Shona clips.")
        print("  Next step is a phase-2 run with this checkpoint, single-variable against the\n"
              "  0.7065 no-LM submission — NOT a leaderboard claim from this number.")
    else:
        print("  nothing beats the incumbent; keep mms-300m-waxal-sna and spend no submission here.")
json.dump(results, open(WORKING / "sna_bakeoff.json", "w"), indent=1)
