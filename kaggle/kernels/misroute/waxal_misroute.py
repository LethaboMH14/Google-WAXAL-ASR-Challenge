"""waxal-misroute — measure what a routing error actually costs.

WHY THIS RUN EXISTS

Every phase-2 projection we have swings on one number that has never been
measured. The 1,500 phase-2 ids carry no language, so a router picks each clip's
decoder, and the file's score is

    observed = a * s + (1 - a) * f

        a  fraction of clips the router sends to the right language
        s  score on a correctly-routed clip
        f  score on a MISROUTED clip

`a` we measure (router accuracy on labelled audio, agreement on phase 2).
`s` we measure (the dev run).
`f` we have only ever guessed at, with 0.30 as a placeholder.

That guess is not harmless. Feeding the corrected phase-2 routing through the
projection gives 0.75 at f=0.15 and 0.61 at f=0.35 — the difference between
clearing the leaders and finishing well behind them, decided entirely by a
number nobody has run. Principle 3 in CLAUDE.md says calibrated numbers only.
This run retires the placeholder.

HOW

WAXAL_MISROUTE=1 sends every dev clip to a DIFFERENT language's model, on a
fixed derangement (lin->sna, sna->lug, lug->lin), and scores the output against
the clip's TRUE reference. Every clip moves, and the same way on every run, so
the result is a measurement rather than a sample. The resulting `multi` is what
a 100%-wrong router scores: the floor the real number is interpolated against.

The output is NOT a submission candidate. It is deliberately, completely wrong
transcription. It exists to price being wrong.

Models are the shipped lineup, because f depends on the decoder: a model with a
larger vocabulary emits more plausible-looking wrong words, and that changes the
floor. Measuring f with different checkpoints than we ship would just move the
guess somewhere less visible.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")
WORKING = Path("/kaggle/working")

# Identical to the lineup. If these drift apart the measured floor stops applying
# to the file we actually upload.
LINEUP = {
    "lin": ("waxalnet", "douyeszn/w2vbert-lin-waxal-aug-ft"),
    "sna": ("waxalnet", "waxal-benchmarking/mms-300m-waxal-sna"),
    "lug": ("waxalnet", "waxal-benchmarking/mms-300m-waxal-lug"),
}
PLUS_PERIOD = "lin,sna,lug"
OBSERVED = 0.491944347                 # our one real leaderboard row, submitted 30 Jul
TOP = 0.725666538                      # rank 1 at the time of writing
A_SUB = 0.5680                         # submitted file's agreement with the CTC router
A_CAND = 0.9658                        # CTC router accuracy on labelled audio


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

import torch  # noqa: E402  — after pip, so this is the version we actually decode on

print(f"\ntorch {torch.__version__}  cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("no CUDA — set the kernel accelerator to GPU T4 before running")
_cc = torch.cuda.get_device_capability(0)
_sm = f"sm_{_cc[0]}{_cc[1]}"
if _sm not in torch.cuda.get_arch_list():
    raise SystemExit(
        f"this card is {torch.cuda.get_device_name(0)} ({_sm}) and torch {torch.__version__} "
        f"was built for {torch.cuda.get_arch_list()}. Set machine_shape to 'NvidiaTeslaT4'.")
print(f"gpu: {torch.cuda.get_device_name(0)} {_sm}")

env = dict(os.environ)
env.update(
    PYTHONUNBUFFERED="1",
    CUDA_VISIBLE_DEVICES="0",
    WAXAL_NO_LM="1",
    WAXAL_BACKEND="waxalnet",
    WAXAL_BACKENDS=",".join(f"{lg}={bk}" for lg, (bk, _) in LINEUP.items()),
    WAXAL_LIN=LINEUP["lin"][1],
    WAXAL_SNA=LINEUP["sna"][1],
    WAXAL_LUG=LINEUP["lug"][1],
    WAXAL_PLUS_PERIOD=PLUS_PERIOD,
    WAXAL_DEV="1",
    WAXAL_MISROUTE="1",                # the whole point of this kernel
    WAXAL_RUN_TAG="misroute",
    WAXAL_DEV_TAG="misroute",
)

print(f"\n{'=' * 78}\n=== MISROUTE RUN — 900 dev clips, each through the WRONG model\n{'=' * 78}",
      flush=True)
sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")], env=env, check=False)

res_path = WORKING / "dev_result_misroute.json"
if not res_path.exists():
    print("\n  NO RESULT — read the traceback above. f remains unmeasured and every phase-2\n"
          "  projection keeps its 0.15-0.35 band.")
    raise SystemExit(1)

res = json.load(open(res_path))
f = res["per_language"]["overall"]["multi"]
o = res["per_language"]["overall"]

print(f"\n{'=' * 78}\n=== MEASURED MISROUTE FLOOR\n{'=' * 78}")
print(f"\n  f = {f:.4f}    (WER {o['wer']:.4f}, CER {o['cer']:.4f}, n={res['n_decoded']})")
for lg in ("lin", "sna", "lug"):
    if lg in res["per_language"]:
        s = res["per_language"][lg]
        print(f"      {lg} -> wrong model: multi={s['multi']:.4f} "
              f"WER={s['wer']:.4f} CER={s['cer']:.4f}")

print("\n  Why f is not near zero: the wrong model still hears the same phonemes and")
print("  writes them in a related Bantu orthography, so character overlap survives")
print("  even when almost no whole word does. CER carries the floor, WER does not.")

print(f"\n{'=' * 78}\n=== WHAT THIS DOES TO THE PROJECTION\n{'=' * 78}")
s_base = (OBSERVED - (1 - A_SUB) * f) / A_SUB
print(f"\n  Inverting our leaderboard row at the measured floor:")
print(f"    {OBSERVED:.6f} = {A_SUB:.4f}*s + {1 - A_SUB:.4f}*{f:.4f}   ->   s = {s_base:.4f}")
print(f"\n  s is the score our phase-2 decode reaches when the routing is right.")
print(f"  Re-routed by the CTC router, the same decode projects:\n")
print("      router acc on phase 2 | projected")
print("      " + "-" * 34)
for a in (0.80, 0.85, 0.90, A_CAND):
    print(f"      {a:>21.4f} | {a * s_base + (1 - a) * f:.4f}")
print(f"\n  leaders: {TOP:.4f}")
print("\n  The router's accuracy on phase 2 is now the ONLY unmeasured term left,")
print("  and it cannot be measured without phase-2 labels. It is bounded below by")
print("  agreement with an independent router and above by its labelled-audio")
print("  accuracy, so the row span above is the honest remaining uncertainty.")

json.dump({"misroute_multi": f, "per_language": res["per_language"],
           "implied_s": s_base, "a_sub": A_SUB, "observed": OBSERVED},
          open(WORKING / "misroute_floor.json", "w"), indent=2)
print(f"\n  wrote {WORKING / 'misroute_floor.json'}")
print(f"\n  Feed this forward as WAXAL_MISROUTE_MULTI={f:.4f} so the lineup prices its own")
print("  routing error instead of reporting the oracle number.")
