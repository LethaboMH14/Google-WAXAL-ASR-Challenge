"""Kaggle GPU kernel: swap Shona onto the w2v-bert encoder that never got tested.

THE ONE CHANGE
--------------
    lin  douyeszn/w2vbert-lin-waxal-aug-ft   waxalnet (CTC)   0.7788   unchanged
    sna  douyeszn/w2vbert-sna-waxal-aug      waxalnet (CTC)   UNTESTED  <-- this
    lug  waxal-benchmarking/mms-300m-waxal-lug waxalnet (CTC) 0.8286   unchanged

WHY THIS ONE IS WORTH AN HOUR OF GPU
------------------------------------
On Lingala the w2v-bert-2.0 encoder is worth an enormous amount over the official benchmark
model: douyeszn's lin checkpoint measured 0.7788 against mms-300m-waxal-lin's 0.6893, a lift of
+0.0895. Shona is still decoded by mms-300m-waxal-sna (0.7815) because the same publisher's
w2v-bert Shona checkpoint showed as gated in docs/MODEL-CANDIDATES.md's access table and was never
benchmarked. It is gated "auto" (instant click-through), not "manual", so it was one click away the
whole time. The corrected phase-2 set is ~50% Shona, so if that encoder is worth even a fraction
on sna what it is worth on lin, this is the largest untested lever we have.

It is a prediction, not a measurement — the bakeoff never scored this checkpoint. The DEV pass
below prices it against the 0.7815 incumbent before the submission CSV is written, so we find out
in this run rather than on the leaderboard.

PROVENANCE, checked before running (the standard that killed the Mubarak127 sna checkpoint)
-------------------------------------------------------------------------------------------
That checkpoint was dropped on 1 Aug because its declared base model was ITSELF, so the chain
never terminated at a public checkpoint, and it appeared the day after the challenge data dropped.
This one declares base_model facebook/w2v-bert-2.0 — a public, pre-existing, unrelated-to-WAXAL
encoder — is apache-2.0, was created 2026-07-28, and comes from the same publisher as the lin
checkpoint the team already accepted and shipped. Same evidentiary test, opposite answer.

PLUS_PERIOD IS lin,sna,lug HERE — sna included, unlike the whisper run
---------------------------------------------------------------------
Dropping the period on sna was Whisper-specific: Whisper emits punctuation natively so appending
another gives '..'. This is a CTC model, and CTC vocabs for these languages contain no '.' at all
while 95.9% of Shona references end in one. The reason to exclude sna left with Whisper.

NO KENLM in this run, deliberately. LM fusion measured +0.0066 on a real submission and it does
apply here (all three are CTC now), but bundling it would make this a two-variable change against
a one-variable baseline. This run is single-variable against LCJutFUw (0.7065): same routing, same
lin, same lug, same period setting, only sna's checkpoint differs. If sna moves, we stack the LM on
top in a follow-up rather than guess which half did the work.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")
WORKING = Path("/kaggle/working")
SNA_MODEL = "douyeszn/w2vbert-sna-waxal-aug"


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
        f"this torch has no kernels for sm_{_cap[0]}{_cap[1]} — the notebook is on the wrong "
        f"accelerator. Set it to GPU T4 x2 in the UI (pushing via the API silently resets it).")

# ---------------------------------------------------------------- gated-model auth, checked FIRST
# The sna checkpoint is gated. Auth has to be proven before the hour of decoding, not after: a
# 401 discovered at the sna stage would waste the whole run and every lin clip already decoded.
from kaggle_secrets import UserSecretsClient  # noqa: E402

try:
    token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:  # noqa: BLE001
    raise SystemExit(
        f"could not read the HF_TOKEN secret ({type(e).__name__}: {e}).\n"
        f"  Add it in the notebook: Add-ons -> Secrets -> Add secret, label exactly HF_TOKEN,\n"
        f"  value = a Hugging Face access token with read permission, then attach it to this\n"
        f"  notebook. {SNA_MODEL} is gated, so it cannot be downloaded anonymously.")
if not token or not token.strip():
    raise SystemExit("HF_TOKEN secret is present but empty")
os.environ["HF_TOKEN"] = token.strip()
os.environ["HUGGING_FACE_HUB_TOKEN"] = token.strip()

from huggingface_hub import model_info  # noqa: E402

try:
    mi = model_info(SNA_MODEL, token=token.strip())
    print(f"gated-model access OK: {mi.id}  (base_model="
          f"{(mi.cardData or {}).get('base_model')}, license={(mi.cardData or {}).get('license')})")
except Exception as e:  # noqa: BLE001
    raise SystemExit(
        f"cannot access {SNA_MODEL}: {type(e).__name__}: {e}\n"
        f"  Accept the terms at https://huggingface.co/{SNA_MODEL} with the SAME account the\n"
        f"  HF_TOKEN belongs to. Gating is per-account, so accepting as one user does not grant\n"
        f"  a token issued by another.")

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
env["WAXAL_SNA"] = SNA_MODEL
env["WAXAL_LUG"] = "waxal-benchmarking/mms-300m-waxal-lug"
env["WAXAL_PLUS_PERIOD"] = "lin,sna,lug"   # all CTC, no '.' in any vocab — see the header
env["WAXAL_RUN_TAG"] = "w2vbertsna"
env["WAXAL_LANG_MAP"] = str(LANG_MAP)

print("\n=== RUN 1/2: DEV — prices the new sna checkpoint against mms-300m's 0.7815 ===")
sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")],
   env=dict(env, WAXAL_DEV="1"))

dev_path = WORKING / "dev_result_w2vbertsna.json"
if dev_path.exists():
    res = json.load(open(dev_path))
    print("DEV RESULT:", res)
    sna = (res.get("per_language") or {}).get("sna", {}).get("multi")
    if sna is not None:
        print(f"\n  sna: {sna:.4f} vs mms-300m incumbent 0.7815  -> {sna - 0.7815:+.4f}")
        print("  (DEV has been off by up to 0.08 in absolute terms on the corrected set, so read\n"
              "   this as a per-language A/B against a number measured the same way, not as a\n"
              "   leaderboard prediction.)")
else:
    print("DEV RESULT: (no dev_result_w2vbertsna.json written — check the run above)")

print("\n=== RUN 2/2: SUBMISSION ===")
sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")], env=env)

for name in ("submission_03_w2vbertsna_lm_phase1.csv", "submission_03_w2vbertsna_lm_phase2.csv"):
    csv = WORKING / name
    if csv.exists():
        sh([sys.executable, str(REPO / "local" / "validate_submission.py"), str(csv)], check=False)
    else:
        print(f"\n!! {name} was not written")

print("\n--- /kaggle/working ---", flush=True)
for p in sorted(WORKING.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 1e6:10.1f} MB  {p.relative_to(WORKING)}")
