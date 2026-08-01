"""
Kaggle GPU kernel: DEV EVAL — simulate a submission and predict its leaderboard score.

WHY THIS EXISTS
---------------
Zindi allows 5 submissions a day and reports one number hours later, on a public split that is
only ~30% of the test set. Every "this change should help" claim in this repo was, until now,
untested until it had already cost a submission. This kernel closes that loop: it decodes a frozen
900-clip dev set through the SAME code path that writes the real submission, and scores it with
the competition's exact metric (local/harness/score.py).

It runs TWO backends in one session, and the pairing is the point:

  A. mms      zero-shot facebook/mms-1b-all. We have submitted exactly this and it scored
              0.491944347 on the public leaderboard. It is the CALIBRATION ANCHOR — if the dev
              score does not land near 0.492, the harness does not predict the leaderboard and
              nothing it says afterwards can be trusted.

  B. waxalnet waxal-benchmarking/mms-300m-waxal-{lin,sna,lug}: the organisers' own baseline
              fine-tunes, trained on the WAXAL corpus, ungated, openly available to everyone
              (which is what the rules require of a pretrained model). Their published numbers
              project to multi ~0.79 with punctuation, ~0.73 without — and their vocabs contain
              no sentence punctuation, which is very likely why the leaderboard's top cluster
              sits at 0.7206-0.7257.

Both run with WAXAL_NO_LM=1: no KenLM build, no alpha/beta grid. That is deliberate. This session
asks one question — how good is the acoustic model on its own — and skipping the LM saves ~40
minutes of GPU. Shallow fusion is the next experiment, not this one, and it is measured the same
way before it is ever submitted.

Both also share one audio cache, so they see byte-identical input and the comparison is of the
models rather than of two resampling passes.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")            # outside /kaggle/working, which is the 20 GB output volume
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
sh([sys.executable, "-m", "pip", "install", "-q", "jiwer"])

import torch  # noqa: E402  — after pip, so this is the version we actually decode on

print(f"\ntorch {torch.__version__}  cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("no CUDA — set the kernel accelerator to GPU T4 before running")

# Same guard as every other stage: `cuda=True` is not the question, whether this torch has
# kernels for THIS card is. A P100 is sm_60 and torch 2.10+cu128 ships no Pascal kernels.
_cc = torch.cuda.get_device_capability(0)
_sm = f"sm_{_cc[0]}{_cc[1]}"
if _sm not in torch.cuda.get_arch_list():
    raise SystemExit(
        f"this card is {torch.cuda.get_device_name(0)} ({_sm}) and torch {torch.__version__} "
        f"was built for {torch.cuda.get_arch_list()}. Set machine_shape to 'NvidiaTeslaT4' "
        f"(capital N — 'nvidiaTeslaT4' is accepted silently and gives you a P100).")
print(f"gpu: {torch.cuda.get_device_name(0)} {_sm}")

# ------------------------------------------------------------------ 2. run both backends
base = dict(os.environ)
base.update(PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0",
            WAXAL_DEV="1", WAXAL_NO_LM="1")

RUNS = [
    ("mms", "zero-shot mms-1b-all — the calibration anchor (leaderboard says 0.491944)"),
    ("waxalnet", "waxal-benchmarking mms-300m-waxal-* — the organisers' own fine-tunes"),
]
for backend, why in RUNS:
    print(f"\n{'=' * 78}\n=== {backend}: {why}\n{'=' * 78}", flush=True)
    env = dict(base, WAXAL_BACKEND=backend)
    # check=False: if one backend dies we still want the other's number and the traceback, rather
    # than an empty session. A missing dev_result_*.json below is the signal that it failed.
    sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")], env=env, check=False)

# ------------------------------------------------------------------ 3. verdict
print(f"\n{'=' * 78}\n=== VERDICT\n{'=' * 78}")
LEADERBOARD_MMS = 0.491944347          # our one real observation, submitted 30 Jul
TOP = 0.725666538                      # KanYi2026, rank 1 at the time of writing

results = {}
for backend, _ in RUNS:
    p = WORKING / f"dev_result_{backend}.json"
    if not p.exists():
        print(f"  {backend}: NO RESULT — that run failed, read its traceback above")
        continue
    results[backend] = json.load(open(p))

for backend, r in results.items():
    o = r["per_language"]["overall"]
    print(f"\n  {backend}: dev multi={o['multi']:.4f}  WER={o['wer']:.4f}  CER={o['cer']:.4f}"
          f"   (test-mix reweighted {r['test_mix_multi']:.4f}, n={r['n_decoded']})")
    for lg in ("lin", "sna", "lug"):
        if lg in r["per_language"]:
            s = r["per_language"][lg]
            print(f"      {lg}: multi={s['multi']:.4f} WER={s['wer']:.4f} CER={s['cer']:.4f}")
    print(f"      + trailing '.': {r['plus_period']:.4f}")

if "mms" in results:
    dev = results["mms"]["test_mix_multi"]
    gap = dev - LEADERBOARD_MMS
    print(f"\n  CALIBRATION: dev predicts {dev:.4f}, leaderboard gave {LEADERBOARD_MMS:.4f}, "
          f"bias {gap:+.4f}")
    print("    The dev set is validation audio and the leaderboard is test audio, so a small bias\n"
          "    is expected and fine — what matters is that it is SMALL and CONSTANT, because then\n"
          "    it subtracts out of every A/B comparison we make from here on.")
    if abs(gap) > 0.08:
        print("    *** bias is large. Do not trust dev deltas as absolute leaderboard predictions\n"
              "    *** until this is understood — use them only to RANK configurations.")

if "waxalnet" in results and "mms" in results:
    a, b = results["mms"]["test_mix_multi"], results["waxalnet"]["test_mix_multi"]
    print(f"\n  waxalnet - mms = {b - a:+.4f} on dev")
    print(f"  projected leaderboard if the bias holds: {LEADERBOARD_MMS + (b - a):.4f}  "
          f"(rank-1 is {TOP:.4f})")

print("\n--- /kaggle/working ---", flush=True)
for p in sorted(WORKING.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 1e6:10.1f} MB  {p.relative_to(WORKING)}")
