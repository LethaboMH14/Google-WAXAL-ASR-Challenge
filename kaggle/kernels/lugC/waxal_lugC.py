"""waxal-lugC — Sunbird's SALT Whisper for Luganda. The best available model; needs an HF token.

WHY THIS IS THE ONE TO WANT

Sunbird/asr-whisper-large-v3-salt, MIT licensed, is whisper-large-v3 adapted to Luganda + ten
other Ugandan languages. Three things make it the best-matched checkpoint found so far:

  * TRAINED FOR PHONE AUDIO. Its card states training used added random noise, random
    downsampling to 8 kHz to simulate phone speech, and street noise sampled from urban Uganda.
    Phase-2 WAXAL audio is crowd-collected phone recordings. Every other candidate is read
    studio/Common-Voice speech.
  * IT PUNCTUATES. Sunbird's own MMS SALT card points at this model for "punctuation and
    capitalisation". Measured on Train.csv, a perfect but punctuation-free Luganda transcriber
    caps at 0.9378 (0.8933 without apostrophes) — see waxal_lugD.py's header for the full
    breakdown. A CTC model with no punctuation in its vocab cannot ever claim that.
  * SUNBIRD SAY SO. Their asr-mms-salt card: the Whisper model "supports additional Ugandan
    languages, has better accuracy". asr-mms-salt is also cc-by-nc-4.0; this one is MIT, which is
    the cleaner answer if this solution is ever code-reviewed for the top 10.

THE LANGUAGE TOKEN — READ THIS BEFORE CHANGING IT

generation_config.language is null, so left alone Whisper language-detects each of 1,403 clips
independently. This checkpoint has the stock 100-language table (vocab 51,866) and NO Luganda
entry; SALT instead REPURPOSES existing slots. From Sunbird's card, cross-checked against this
checkpoint's own lang_to_id:

    lug 50355 = <|ba|>      nyn 50354 = <|ha|>      teo 50353 = <|ln|>
    ach 50357 = <|su|>      lgg 50356 = <|jw|>      xog 50352 = <|haw|>
    swa 50318 = <|sw|>      eng 50259 = <|en|>

So Luganda here is language="ba" — Bashkir's slot. Not "sw", which is what the CDLI checkpoints
in waxal-lugD use, and not "lg", which this model does not have. The two conventions look
interchangeable and are not; passing sw here would decode Luganda as Swahili.

THE GATE, AND WHY THIS KERNEL CAN STILL FAIL

The repo is gated=auto. Lethabo accepted it on 1 Aug, so it downloads from any machine holding
his HF token. A Kaggle kernel does not hold one — the lugB log shows "You are sending
unauthenticated requests to the HF Hub" — and a gated repo 401s for an anonymous client no
matter who accepted the licence. So this kernel needs HF_TOKEN as a Kaggle Secret:

    Kaggle notebook editor -> Add-ons -> Secrets -> Add secret
        label: HF_TOKEN
        value: an HF access token with READ scope
    then tick it for this notebook.

The token is never printed, never written to disk by this script, and never committed — this
repo is PUBLIC. If the secret is missing this kernel exits in ~1 minute with that instruction
rather than burning a GPU hour to fail on the model download.

waxal-lugD is the no-token version of this experiment: KasuleTrevor's cdli-whisper-ml-* is
derived from this same Sunbird lineage and is ungated MIT. Run lugD if this stays blocked.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")
WORKING = Path("/kaggle/working")

LUG_MODEL = "Sunbird/asr-whisper-large-v3-salt"       # gated=auto, MIT
LIN_MODEL = "waxal-benchmarking/mms-300m-waxal-lin"   # 8 clips, along for the ride
SNA_MODEL = "waxal-benchmarking/mms-300m-waxal-sna"   # 89 clips, ditto
LUG_WHISPER_LANG = "ba"                               # SALT's Luganda slot, token 50355
RUN_TAG = "lugC"
OBSERVED = 0.491944347                                # sub_01, the anchor for every run

# ---------------------------------------------------------------- the token, before anything else
_tok = None
try:
    from kaggle_secrets import UserSecretsClient

    _tok = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as exc:                                          # noqa: BLE001
    print(f"could not read the HF_TOKEN secret: {type(exc).__name__}: {exc}")
if not _tok:
    raise SystemExit(
        f"\n{LUG_MODEL} is gated and this kernel has no HF token.\n"
        "  Kaggle notebook editor -> Add-ons -> Secrets -> Add secret\n"
        "      label HF_TOKEN, value = an HF access token with READ scope\n"
        "  then tick it for this notebook and re-run.\n"
        "  Accepting the licence on huggingface.co is NOT enough on its own: a Kaggle kernel is\n"
        "  an anonymous HF client and gated repos 401 for anonymous clients.\n"
        "  No token? Run waxal-lugD instead — same experiment, ungated MIT checkpoint.")
# Both names, because huggingface_hub has read HF_TOKEN and HUGGING_FACE_HUB_TOKEN at different
# versions and the kernel image's version is not pinned by us.
os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = _tok
print("HF token loaded from Kaggle Secrets (value not logged)")


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

import torch  # noqa: E402

print(f"\ntorch {torch.__version__}  cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("no CUDA — set the kernel accelerator to GPU T4 before running")

# Probe the GATED model first: this is the step that 401s, and it should cost one minute, not one
# GPU hour. lugB v1 learned that the expensive way.
from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402

print(f"\nprobing {LUG_MODEL} ...", flush=True)
try:
    _pr = WhisperProcessor.from_pretrained(LUG_MODEL)
    _md = WhisperForConditionalGeneration.from_pretrained(LUG_MODEL, torch_dtype=torch.float16)
except Exception as exc:                                          # noqa: BLE001
    raise SystemExit(
        f"could not load {LUG_MODEL}: {type(exc).__name__}: {exc}\n"
        "  A 401/403 here means the HF_TOKEN secret is present but the account behind it has not\n"
        "  accepted this model's licence at https://huggingface.co/" + LUG_MODEL) from exc

print(f"  ok — {type(_md).__name__}, vocab {_md.config.vocab_size:,}")
_l2i = getattr(_md.generation_config, "lang_to_id", None) or {}
_id = _l2i.get(f"<|{LUG_WHISPER_LANG}|>")
print(f"  generation_config.language = {getattr(_md.generation_config, 'language', None)}")
print(f"  forcing lug -> <|{LUG_WHISPER_LANG}|> = token {_id}   (SALT's Luganda slot is 50355)")
if _id != 50355:
    raise SystemExit(
        f"<|{LUG_WHISPER_LANG}|> maps to {_id}, not 50355. Either this checkpoint's token table "
        "changed or the SALT mapping in this file's header is stale — re-derive it from the "
        "model card before decoding 1,403 clips under the wrong language.")
del _md, _pr
torch.cuda.empty_cache()

# ---------------------------------------------------------------- guard: the corpus must be here
# lin and sna still decode through KenLM; lug does not (seq2seq has no frame logits, and stage 3
# now skips LM tuning for a whisper language rather than crashing inside logits_for).
_hits = sorted(Path("/kaggle/input").rglob("lug.txt"))
if not _hits:
    tree = sorted(str(p) for p in Path("/kaggle/input").glob("*/*"))[:60]
    raise SystemExit(
        "no lug.txt anywhere under /kaggle/input — the LM corpus did not mount.\n"
        f"  what IS mounted: {tree or 'NOTHING — /kaggle/input is empty'}\n"
        "  Attach lethabomh14/waxal-lm-corpus as a dataset_source and re-push.")
CORPUS = _hits[0].parent
print(f"\nfound LM corpus at {CORPUS}")
print(f"  files: {sorted(p.name for p in CORPUS.glob('*.txt'))}")

# ---------------------------------------------------------------- guard: routing must match lugA
LANG_MAP = REPO / "data" / "routing" / "lang_map_mmsclosed_phase2.json"
if not LANG_MAP.exists():
    raise SystemExit(f"{LANG_MAP} missing from the repo clone — commit it before running.")
_m = json.loads(LANG_MAP.read_text(encoding="utf-8"))
_mix = {lg: sum(1 for v in _m.values() if v == lg) for lg in ("lin", "sna", "lug")}
print(f"routing (replayed from sub_01): {len(_m):,} clips {_mix}")
assert _mix["lug"] == 1403 and _mix["sna"] == 89 and _mix["lin"] == 8, (
    f"routing map is not sub_01's ({_mix}) — the runs stop being comparable, fix the map")

env = dict(os.environ)
env.update(
    PYTHONUNBUFFERED="1",
    CUDA_VISIBLE_DEVICES="0",
    WAXAL_BACKEND="waxalnet",                       # the default for the two ride-along languages
    WAXAL_BACKENDS="lug=whisper,lin=waxalnet,sna=waxalnet",
    WAXAL_LUG=LUG_MODEL,
    WAXAL_LIN=LIN_MODEL,
    WAXAL_SNA=SNA_MODEL,
    WAXAL_WHISPER_LANG=f"lug={LUG_WHISPER_LANG}",   # else 1,403 independent language guesses
    WAXAL_WHISPER_BEAMS="5",
    WAXAL_LANG_MAP=str(LANG_MAP),
    WAXAL_LM_CORPUS_DIR=str(CORPUS),
    WAXAL_RUN_TAG=RUN_TAG,
    # WAXAL_NO_LM unset: lin/sna keep the same KenLM as lugA/lugB. lug skips it by backend.
    # WAXAL_PLUS_PERIOD unset: this model punctuates natively; appending would produce '..'.
)
env.pop("WAXAL_NO_LM", None)
env.pop("WAXAL_PLUS_PERIOD", None)

print(f"\n{'=' * 78}\n=== {RUN_TAG}: lug={LUG_MODEL} [whisper/{LUG_WHISPER_LANG}], sub_01 routing"
      f"\n{'=' * 78}", flush=True)
sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")], env=env, check=False)

print(f"\n{'=' * 78}\n=== WHAT TO READ OFF THIS RUN\n{'=' * 78}")
subs = sorted(WORKING.glob("*.csv"))
print(f"\n  submission files written: {[p.name for p in subs] or 'NONE'}")
for p in subs:
    import csv

    with p.open(encoding="utf-8") as fh:
        rows = list(csv.reader(fh))[1:]
    txt = [r[-1] for r in rows if r]
    n_pun = sum(1 for t in txt if t.strip()[-1:] in ".?!")
    n_apo = sum(t.count("'") + t.count("’") for t in txt)
    blank = sum(1 for t in txt if not t.strip())
    print(f"    {p.name}: {len(txt)} rows, {n_pun} end in .?! , {n_apo} apostrophes, {blank} blank")
    print("      ^ apostrophes should be the STRAIGHT ' — WAXAL references use ' (17,030 of them "
          "in Train.csv) and a curly ’ scores as a wrong character AND a wrong word.")
print(f"\n  Compare against lugD (same experiment, ungated model), lugA (LM lever) and "
      f"{OBSERVED:.6f} (sub_01).")
