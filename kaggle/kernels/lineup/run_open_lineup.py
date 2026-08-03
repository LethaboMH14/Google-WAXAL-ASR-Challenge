"""Kaggle GPU kernel: the all-ungated lineup — both bakeoff winners, phase-2 submission.

THE LINEUP
----------
    lin  misterkissi/w2v2-lg-xls-r-300m-lingala   CTC   ungated  cc-by-nc-sa-4.0   dev 0.8482
    sna  badrex/w2v-bert-2.0-shona-asr            CTC   ungated  cc-by-4.0         dev 0.8331
    lug  waxal-benchmarking/mms-300m-waxal-lug    CTC   ungated  cc-by-nc-4.0      1 clip

Both challengers won their language's bakeoff against an in-session control, measured on the same
dev clips in the same session. Shona: badrex 0.8331 vs the mms-300m-waxal-sna incumbent's 0.7980,
+0.0351. Lingala: misterkissi 0.8482 against an in-session mms-300m-waxal-lin control of 0.6924,
+0.156, where the previous incumbent douyeszn historically ran +0.0895 over that same control.

NEITHER WINNER HAS EVER SEEN WAXAL
----------------------------------
badrex trained on badrex/shona-speech (Nov 2025) and misterkissi on its own Lingala data (Jul 2025),
both predating the challenge data by many months. That matters twice over. It rules out the dev set
being inflated by memorisation — our dev clips come from WaxalNLP's validation split, so a
WAXAL-fine-tuned checkpoint could score well by having seen them, and these two cannot. And it means
neither can have been fitted to the public phase-1 test key, which is a disqualifying breach.

They beat WAXAL-fine-tuned models on WAXAL's own validation audio anyway. Read that as the encoder
mattering more than the domain match, which is the same thing Lingala already showed when w2v-bert
beat mms-300m by +0.0895.

WHY UNGATED IS NOW A REQUIREMENT, NOT A PREFERENCE
--------------------------------------------------
douyeszn/w2vbert-lin-waxal-aug-ft — the lin checkpoint every previous submission used — flipped to
gated "manual" on 3 Aug, between two runs hours apart, and now returns 401. The rule is "You may use
pretrained models as long as they are OPENLY AVAILABLE TO EVERYONE". docs/MODEL-CANDIDATES.md
already called auto-gating "a grey area" against that wording; manual gating, where an owner decides
case by case who may download, is harder to defend still. Every model here is ungated, so the
question does not arise.

On the licences: NonCommercial is not disqualifying. The rule about licences applies to external
DATASETS ("legally licensed for research or development"), the model rule tests only open
availability, and the organisers' own benchmark models are themselves cc-by-nc-4.0. All three
repos and their licences go in the final solution documentation, which the rules require.

NO KENLM in this run. It measured +0.0066 on a real submission and it would apply here (all three
are CTC), but this run already changes two checkpoints at once against the 0.7065 no-LM baseline.
Stacking a third change would make a disappointing result unattributable. If this lands, LM is the
next thing to add.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")
WORKING = Path("/kaggle/working")

LINEUP = {
    "lin": "misterkissi/w2v2-lg-xls-r-300m-lingala",
    "sna": "badrex/w2v-bert-2.0-shona-asr",
    "lug": "waxal-benchmarking/mms-300m-waxal-lug",
}


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
env["WAXAL_PLUS_PERIOD"] = "lin,sna,lug"
env["WAXAL_RUN_TAG"] = "openlineup"
env["WAXAL_LANG_MAP"] = str(LANG_MAP)

print("\n=== RUN 1/2: DEV ===")
sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")], env=dict(env, WAXAL_DEV="1"))

dev_path = WORKING / "dev_result_openlineup.json"
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

for name in ("submission_03_openlineup_lm_phase1.csv", "submission_03_openlineup_lm_phase2.csv"):
    csv = WORKING / name
    if csv.exists():
        sh([sys.executable, str(REPO / "local" / "validate_submission.py"), str(csv)], check=False)
    else:
        print(f"\n!! {name} was not written")

print("\n--- /kaggle/working ---", flush=True)
for p in sorted(WORKING.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 1e6:10.1f} MB  {p.relative_to(WORKING)}")
