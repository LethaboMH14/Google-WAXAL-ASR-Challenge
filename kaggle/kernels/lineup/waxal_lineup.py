"""
Kaggle GPU kernel: THE LINEUP — measure the bakeoff winner, then write the submission from it.

WHY THIS EXISTS
---------------
The bakeoff (kernel lethabomh14/waxal-bakeoff v1) scored ten checkpoints one language at a time
and produced a per-language winner. But a per-language table is not a submission: jiwer POOLS
errors across the whole corpus rather than averaging per-language scores, so a word-weighted
estimate of three separate runs is an approximation, not the number. This kernel produces the
number, and then produces the CSV that the number describes.

Two runs, in this order, deliberately:

  1. DEV      — all three languages, one process, the frozen 900-clip dev set. This is the
                predicted leaderboard score for the exact configuration in run 2.
  2. SUBMIT   — the same configuration against SampleSubmission.csv and Test_phase2.csv.

Same env in both apart from WAXAL_DEV. That is the whole discipline: we never upload a CSV whose
configuration has not first been scored on dev. Uploading is still the human's job — this kernel
writes files, it does not talk to Zindi.

THE CONFIGURATION, AND WHY EACH PIECE
-------------------------------------
    lin  douyeszn/w2vbert-lin-waxal-aug-ft            0.7788   (mms-300m control: 0.6893)
    sna  waxal-benchmarking/mms-300m-waxal-sna        0.7815   (see PROVENANCE below)
    lug  waxal-benchmarking/mms-300m-waxal-lug        0.8163   (nothing in the bakeoff beat it)

PROVENANCE, 1 Aug — why sna is NOT the higher-scoring checkpoint
----------------------------------------------------------------
Mubarak127/waxal-whisper-large-v3-sna_asr measured 0.8034 on dev, 0.0219 better than the official
benchmark model, and it is dropped anyway. Its card is unedited Trainer boilerplate: "fine-tuned
version of Mubarak127/waxal-whisper-large-v3-sna_asr on an unknown dataset", "Training and
evaluation data: More information needed", empty results table. Its declared base model is itself,
so the chain never terminates at a public checkpoint. Created 2026-07-01, the day after the
challenge data dropped.

Three challenge rules bear on that, and it fails all three as an evidentiary matter:
  - using the (public) Phase-1 test ground truth in a submission is a disqualifying breach;
  - external data must be publicly accessible, legally licensed, and DISCLOSED in the final
    solution documentation;
  - top-10 finishers hand over code within 48 hours.
We cannot disclose what we cannot establish. Nothing here says the author did anything wrong — the
point is we have no way to show they didn't, and at code review the burden is ours.

The swap costs 0.459*0 + 0.366*0.0219 + 0.175*0 = 0.0080 of final score. Correct price for a
lineup we can account for end to end:
  - douyeszn: card states "WAXAL train split only", speaker-disjoint validation, augmentation for
    Phase-2 robustness, full training curve published. Exemplary, and its honest speaker-disjoint
    number (0.7550 on the Zindi scale) is 0.0278 below what our dev split says — evidence our own
    dev shares speakers with train and is therefore optimistic.
  - waxal-benchmarking/*: the benchmark suite's own models, dataset tag waxal-benchmarking/waxal,
    arXiv:2606.02375, published April 2026, before the challenge opened.

All three are CTC now, so the mixed-architecture path is unused — WAXAL_BACKENDS is kept anyway so
the seq2seq branch stays exercised and the swap is one env var to reverse.

Trailing '.' now on ALL THREE, which is a consequence of the sna swap above, not a separate idea.
It was worth +0.0040 on lin and +0.0123 on lug but -0.0181 on sna, because Whisper punctuates
natively and appending gave '..' — one word error AND one character error on a cell that was
otherwise correct. mms-300m-waxal-sna is CTC and its vocab is 51 symbols with no '.' in it (nor
has lug's 38, nor douyeszn's 63), so it cannot produce the character at all, while 95.1% of Shona
references end in one. The reason to exclude sna was Whisper-specific and left with Whisper.
Expect roughly what lug got; the dev run re-measures all three, so the number is checked, not
assumed.

NO KENLM (WAXAL_NO_LM=1), and this is not laziness. Every alpha/beta pair we have was tuned on
leaked data: corpus_lines() read all of Train.csv, which contains the original_split=="validation"
rows that ARE the dev set, so the LM had memorised the dev references and shallow fusion decoded
those same clips against it. That is what made dev predict 0.7392 for a config that scored 0.4919.
The holdout is fixed (commit db22db2) but the tuning has not been redone, so the honest thing is to
submit without fusion and re-sweep alpha/beta as its own experiment.

SEED: the rules require reruns to reproduce. Everything here is greedy/beam-1 argmax decoding with
no sampling; the only stochastic choice in the pipeline is the dev-set draw, frozen in
local/harness/devset.json at seed 1337.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")            # outside /kaggle/working, which is the 20 GB output volume
WORKING = Path("/kaggle/working")

LINEUP = {
    "lin": ("waxalnet", "douyeszn/w2vbert-lin-waxal-aug-ft", 0.7788),
    "sna": ("waxalnet", "waxal-benchmarking/mms-300m-waxal-sna", 0.7815),
    "lug": ("waxalnet", "waxal-benchmarking/mms-300m-waxal-lug", 0.8163),
}
PLUS_PERIOD = "lin,sna,lug"            # measured per language; see the header
WORD_SHARE = {"lin": 0.459, "sna": 0.366, "lug": 0.175}   # share of dev reference WORDS
LEADERBOARD_MMS = 0.491944347          # our one real observation, submitted 30 Jul
TOP = 0.725666538                      # KanYi2026, rank 1 at the time of writing


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

# `cuda=True` is not the question; whether this torch has kernels for THIS card is. A P100 is
# sm_60 and torch 2.10+cu128 ships no Pascal kernels.
_cc = torch.cuda.get_device_capability(0)
_sm = f"sm_{_cc[0]}{_cc[1]}"
if _sm not in torch.cuda.get_arch_list():
    raise SystemExit(
        f"this card is {torch.cuda.get_device_name(0)} ({_sm}) and torch {torch.__version__} "
        f"was built for {torch.cuda.get_arch_list()}. Set machine_shape to 'NvidiaTeslaT4' "
        f"(capital N — 'nvidiaTeslaT4' is accepted silently and gives you a P100).")
print(f"gpu: {torch.cuda.get_device_name(0)} {_sm}")

# ------------------------------------------------------------------ 2. the shared configuration
env = dict(os.environ)
env.update(
    PYTHONUNBUFFERED="1",
    CUDA_VISIBLE_DEVICES="0",
    WAXAL_NO_LM="1",
    WAXAL_BACKEND="waxalnet",          # the majority kind; WAXAL_BACKENDS overrides per language
    WAXAL_BACKENDS=",".join(f"{lg}={bk}" for lg, (bk, _, _) in LINEUP.items()),
    WAXAL_LIN=LINEUP["lin"][1],
    WAXAL_SNA=LINEUP["sna"][1],
    WAXAL_LUG=LINEUP["lug"][1],
    WAXAL_PLUS_PERIOD=PLUS_PERIOD,
    WAXAL_RUN_TAG="lineup",
)

# ---- phase-2 routing -------------------------------------------------------
# The 1,500 phase-2 ids carry no language, so something has to choose their decoder. Version 2
# of this kernel let mms-lid-256 choose and it returned 94% Luganda. That is refuted by our own
# leaderboard row: at that routing our submitted file was already ~right, and its perfect-routing
# ceiling works out to 0.5685 — under a score six teams have posted on the same clips. A ceiling
# cannot sit below an observed floor. Full arithmetic in scripts/anchor_calibration.py.
#
# So we route phase 2 with the CTC-confidence map from the waxal-router kernel instead:
# architecturally independent of the MMS LID family, balanced recalls 0.9525/0.9800/0.9650, and
# its agreement with our submitted file (0.5680) matches the routing accuracy the leaderboard
# arithmetic implies independently (0.51-0.59).
#
# Hard-fail rather than fall through. A silent fall-through would quietly reproduce the 94%
# Luganda file we already know scores ~0.49, and it would do it after an hour of GPU time.
LANG_MAP = Path("/kaggle/input/waxal-router/lang_map_asr-conf_z_neg_entropy.json")
if not LANG_MAP.exists():
    found = sorted(p.name for p in Path("/kaggle/input").glob("*/lang_map*.json"))
    raise SystemExit(
        f"{LANG_MAP} is missing — add waxal-router to kernel_sources and re-push.\n"
        f"  lang_map files visible under /kaggle/input: {found or 'none'}\n"
        f"  Refusing to fall through to mms-lid-256: that path produces the 94%-Luganda\n"
        f"  routing this kernel exists to replace, and it would cost an hour to find out.")
env["WAXAL_LANG_MAP"] = str(LANG_MAP)

_m = json.load(open(LANG_MAP))
_mix = {k: sum(v == k for v in _m.values()) for k in ("lin", "sna", "lug")}
print(f"phase-2 routing: {LANG_MAP.name}  n={len(_m):,}  "
      + "  ".join(f"{k}={v:,} ({v / len(_m):.1%})" for k, v in _mix.items()))

print(f"\n{'=' * 78}\n=== CONFIGURATION\n{'=' * 78}")
for lg, (bk, mid, solo) in LINEUP.items():
    dot = "yes" if lg in PLUS_PERIOD.split(",") else "no"
    print(f"  {lg}  {bk:9s}  {mid:48s}  +period={dot:3s}  bakeoff solo={solo:.4f}")
est = sum(WORD_SHARE[lg] * (solo + (0.004 if lg == "lin" else 0.012 if lg == "lug" else 0.0))
          for lg, (_, _, solo) in LINEUP.items())
print(f"\n  word-weighted estimate from the bakeoff: {est:.4f}")
print(f"  the DEV run below replaces that estimate with a pooled measurement — trust that one")

# ------------------------------------------------------------------ 3. run 1: dev
print(f"\n{'=' * 78}\n=== RUN 1/2: DEV — 900 frozen clips, all three languages\n{'=' * 78}",
      flush=True)
# check=False: if dev dies we still want to see the traceback AND attempt the submission run,
# rather than losing the whole session. The missing dev_result_lineup.json is the signal.
sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")],
   env=dict(env, WAXAL_DEV="1"), check=False)

dev_path = WORKING / "dev_result_lineup.json"
dev = json.load(open(dev_path)) if dev_path.exists() else None
if dev is None:
    print("\n  DEV RUN PRODUCED NO RESULT — read its traceback above. The submission run below\n"
          "  will still write CSVs, but they will be UNMEASURED. Do not upload them on that basis.")
else:
    o = dev["per_language"]["overall"]
    print(f"\n  DEV: multi={o['multi']:.4f}  WER={o['wer']:.4f}  CER={o['cer']:.4f}"
          f"   (test-mix reweighted {dev['test_mix_multi']:.4f}, n={dev['n_decoded']})")
    for lg in ("lin", "sna", "lug"):
        if lg in dev["per_language"]:
            s, solo = dev["per_language"][lg], LINEUP[lg][2]
            print(f"      {lg}: multi={s['multi']:.4f} WER={s['wer']:.4f} CER={s['cer']:.4f}"
                  f"   (bakeoff solo {solo:.4f}, delta {s['multi'] - solo:+.4f})")
    print(f"\n  vs our submitted 0.4919: {dev['test_mix_multi'] - LEADERBOARD_MMS:+.4f}")
    print(f"  vs rank-1 {TOP:.4f}:       {dev['test_mix_multi'] - TOP:+.4f}")
    print("\n  CAVEAT worth stating plainly: dev is validation audio, the leaderboard is test\n"
          "  audio, and the only LM-free calibration point we have is indirect (our harness puts\n"
          "  the organisers' mms-300m set at 0.7453 while the leaderboard top cluster sits at\n"
          "  0.7206-0.7257, implying a bias near +0.02). That inference assumes the leaders run\n"
          "  those checkpoints. It is not confirmed. Treat the dev number as a good RANKING\n"
          "  signal and a rough absolute one.")

# ------------------------------------------------------------------ 4. run 2: submission
print(f"\n{'=' * 78}\n=== RUN 2/2: SUBMISSION — phase 1 + phase 2 CSVs\n{'=' * 78}", flush=True)
sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")], env=env, check=False)

# ------------------------------------------------------------------ 5. what to upload
print(f"\n{'=' * 78}\n=== VERDICT\n{'=' * 78}")
csvs = sorted(WORKING.glob("submission_03_lineup_lm_*.csv"))
if not csvs:
    print("  NO SUBMISSION CSV WAS WRITTEN — read the run-2 traceback above.")
for p in csvs:
    import pandas as pd

    df = pd.read_csv(p)
    txt = df.columns[-1]
    blank = int((df[txt].astype(str).str.strip().isin(["", "nan", "a"])).sum())
    print(f"\n  {p.name}")
    print(f"    rows={len(df):,}  suspiciously-empty={blank:,}")
    print(f"    mean chars/row={df[txt].astype(str).str.len().mean():.1f}")
    print(f"    ends with '.': {100 * df[txt].astype(str).str.rstrip().str.endswith('.').mean():.1f}%")

print("\n  Upload the PHASE the competition currently has open. Phase 2 (1,500 rows, ID_XXXXX ids)\n"
      "  is the split that sets the final ranking and carries no language metadata, so its\n"
      "  language routing came from the LID model rather than from an id prefix.")
print("\n  Zindi allows 5 submissions/day, 200 total, and 2 must be selected for the private\n"
      "  leaderboard before the 09 Aug close.")

# Drop the bulky intermediates. /kaggle/working IS the output volume, and the dev audio cache
# alone is ~1.1 GB — pulling this kernel's output to fetch a 400 KB CSV should not mean pulling
# that. Both are reconstructible: the npz from the frozen dev ids, the zip from PHASE2_URL.
freed = 0
for junk in list(WORKING.glob("dev_audio_*.npz")) + list(WORKING.glob("phase2_audio.zip")):
    freed += junk.stat().st_size
    junk.unlink()
if freed:
    print(f"\n  cleaned {freed / 1e6:,.0f} MB of regenerable cache from the output volume")

print("\n--- /kaggle/working ---", flush=True)
for p in sorted(WORKING.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 1e6:10.1f} MB  {p.relative_to(WORKING)}")
