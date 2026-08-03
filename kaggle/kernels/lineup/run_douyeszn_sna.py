"""Kaggle GPU kernel: douyeszn's compliant Shona checkpoint — the config the evidence points at.

WHAT THE SUBMISSION TRAIL SAYS
------------------------------
Lethabo's filenames and scores, read off Zindi:

    submission_14_af51v2_linsna_phase2v2.csv   0.700815
    submission_15_linsna_routed_phase2v2.csv   0.745030
    submission_16_linsna_capfirst.csv          0.745734   <- team best
    submission_17_ctc_douyeszn_phase2v2.csv    0.734984

Our LCJutFUw ran douyeszn for lin and mms-300m for sna and scored 0.706477. His 17 reads as
douyeszn for BOTH languages and scored 0.734984. Same lin checkpoint, +0.028, and the only thing
left to differ is the Shona side — douyeszn/w2vbert-sna-waxal-aug, which we have never been able to
run because it is gated.

Three hypotheses were killed cheaply before landing here, all without spending a submission:
  - routing: his implied routing and ours agree on 892/892 clips (recovered by classifying each
    output's text against unigram LMs built from Train.csv; the classifier scores 100% on held-in
    Train rows, so it is trustworthy);
  - repetition: his file repeats n-grams MORE than ours (0.595% vs 0.336%), so our duplication is
    not the defect;
  - truncation: the longest phase-2 clip is 35.2s against our MAX_SECONDS=40, so nothing is cut.

WHY THIS CHECKPOINT IS DIFFERENT FROM THE LAST FOUR WE TRIED
------------------------------------------------------------
Its card answers, by construction, every failure mode this project has hit:

  - "case + punctuation kept in the vocab" — punctuation emitted by the acoustic model, conditioned
    on audio. docs/MODEL-CANDIDATES.md priced full punctuation at +0.029 over a bare trailing period
    and our own XLM-RoBERTa restorer measured -0.0050, because a text-only model cannot tell which
    words the ASR got wrong and so punctuates garbage. A model with punctuation in its vocab does
    not have that problem.
  - "speaker-disjoint validation split" — the author's own note says it "predicts fresh-audio
    (Phase-2) performance rather than the inflated numbers a leaking split gives". That is exactly
    the failure our dev harness has: our dev clips come from WaxalNLP validation, and dev has now
    mispredicted the leaderboard five times running.
  - "noise + speed augmentation for out-of-domain (Phase-2) robustness" — built for the domain
    shift that has eaten every gain we measured on dev.
  - Reported honest validation: combined 0.5*WER + 0.5*CER = 0.2055, i.e. 0.7945.

COMPLIANCE — the standard that killed the Mubarak127 checkpoint
---------------------------------------------------------------
"Training data: WAXAL Shona train split only. The Phase-1 test split is never read (no test audio or
transcriptions used at any point)." Base model facebook/w2v-bert-2.0, apache-2.0. That is the exact
objection commit 96419be raised against Mubarak127, answered in the card.

PLUS_PERIOD EXCLUDES sna, DELIBERATELY
--------------------------------------
This model already emits punctuation. Appending a trailing period would double-punctuate exactly as
it did on the whisper checkpoint, where it measured -0.0181. lin keeps the period because its
CTC vocab has no '.' at all.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")
WORKING = Path("/kaggle/working")

SNA = "douyeszn/w2vbert-sna-waxal-aug"          # gated auto — the whole point of this run
# lin: prefer douyeszn's lin (what his 17 used) but it flipped to gated "manual" on 3 Aug and 401s.
# misterkissi is the best ungated lin we measured, so it is the fallback rather than a blocker.
LIN_PREFERRED = "douyeszn/w2vbert-lin-waxal-aug-ft"
LIN_FALLBACK = "misterkissi/w2v2-lg-xls-r-300m-lingala"
LUG = "waxal-benchmarking/mms-300m-waxal-lug"


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
    raise SystemExit("no CUDA — accelerator must be GPU T4 (machine_shape NvidiaTeslaT4)")
_cap = torch.cuda.get_device_capability()
print(f"gpu: {torch.cuda.get_device_name(0)}  sm_{_cap[0]}{_cap[1]}")
if f"sm_{_cap[0]}{_cap[1]}" not in torch.cuda.get_arch_list():
    raise SystemExit(f"torch has no kernels for sm_{_cap[0]}{_cap[1]}; set machine_shape to "
                     f"NvidiaTeslaT4 in kernel-metadata.json")

# ---------------------------------------------------------------- preflight: fetch a real FILE
# All three are ungated today, but douyeszn was ungated this morning and 401s this afternoon, so
# "it was open when I checked" is not a guarantee that survives to decode time. Fetching config.json
# exercises the same permission on the same path the loader uses; model_info() would not, because HF
# serves metadata for gated repos to anyone — that mistake already cost a nine-minute run.
from huggingface_hub import hf_hub_download  # noqa: E402

for lg, repo in LINEUP.items():
    try:
        hf_hub_download(repo, "config.json")
        print(f"  OK  {lg}: {repo}")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"  FAIL {lg}: {repo} — {type(e).__name__}\n"
            f"  This repo is not downloadable anonymously. If it has just been gated, either "
            f"request access or swap in the next-best ungated candidate from the bakeoff before "
            f"burning an hour of GPU.")

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
env["WAXAL_LIN"] = LINEUP["lin"]
env["WAXAL_SNA"] = LINEUP["sna"]
env["WAXAL_LUG"] = LINEUP["lug"]
# All three are CTC and no CTC vocab here contains '.', while 82.4% of references end in one.
# The exclusion that applied to sna was Whisper-specific and left with Whisper.
env["WAXAL_PLUS_PERIOD"] = "lin,lug"  # NOT sna: punctuation is already in its vocab
env["WAXAL_RUN_TAG"] = "douyesna"
env["WAXAL_LANG_MAP"] = str(LANG_MAP)

print("\n=== RUN 1/2: DEV ===")
sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")], env=dict(env, WAXAL_DEV="1"))

dev_path = WORKING / "dev_result_douyesna.json"
if dev_path.exists():
    res = json.load(open(dev_path))
    per = res.get("per_language") or {}
    print("\nDEV per-language multi:")
    for lg in ("lin", "sna", "lug"):
        v = (per.get(lg) or {}).get("multi")
        if v is not None:
            print(f"   {lg}: {v:.4f}")
    print(f"   overall: {(per.get('overall') or {}).get('multi')}")
    print("\n  This harness has missed the real score by up to 0.08 in absolute terms on the\n"
          "  corrected set. Treat these as a smoke test that the right checkpoints loaded, not as\n"
          "  a leaderboard prediction. The only trustworthy comparison is the submitted score\n"
          "  against 0.7065 (no-LM) and 0.7131 (with-LM).")
else:
    print("DEV RESULT: missing — check the run above")

print("\n=== RUN 2/2: SUBMISSION ===")
sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")], env=env)

for name in ("submission_03_douyesna_lm_phase1.csv", "submission_03_douyesna_lm_phase2.csv"):
    csv = WORKING / name
    if csv.exists():
        sh([sys.executable, str(REPO / "local" / "validate_submission.py"), str(csv)], check=False)
    else:
        print(f"\n!! {name} was not written")

print("\n--- /kaggle/working ---", flush=True)
for p in sorted(WORKING.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 1e6:10.1f} MB  {p.relative_to(WORKING)}")
