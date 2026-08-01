"""waxal-lugB — same LM, same routing, DIFFERENT Luganda acoustic model.

WHY THIS RUN EXISTS

lugA holds the acoustic model fixed and adds the language model. This run holds the language
model and the routing fixed and changes the Luganda acoustic model. Read together they separate
the two levers; read alone, neither would.

    lugA   mms-1b-all zero-shot                + 5-gram KenLM + sub_01 routing
    lugB   dmusingu w2v-bert-2.0 Luganda CV7   + 5-gram KenLM + sub_01 routing   <-- this file

Phase 2 is ~93.5% Luganda (sub_04's CER 0.3932 sits on the measured all-wrong floor of 0.3944,
which is what a file routed to the wrong model looks like), so the Luganda slot is essentially
the whole score. lin (8 clips) and sna (89) ride along on the WAXAL checkpoints purely so the
routing stays byte-identical to lugA's; they cannot move the number by more than ~0.03.

WHY A COMMON-VOICE MODEL RATHER THAN ANOTHER WAXAL FINE-TUNE

The leaderboard already priced the WAXAL fine-tunes on phase-2 audio and the answer was bad.
Backing per-clip quality out of the two scored files:

    zero-shot mms-1b-all      ~0.51 on phase-2 Luganda
    mms-300m-waxal-lug        ~0.38 on phase-2 Luganda   (dev said 0.8286)

A model fine-tuned on WAXAL scores *worse* on phase 2 than the generic checkpoint it was built
from, while scoring far better on WAXAL-domain dev. That is domain shift, not noise: the dev
0.8286 was measured greedily with the dev ids held out of the LM, so there is no leakage left
to explain it away. Phase-2 audio is not WAXAL audio, and fitting harder to WAXAL moves the
wrong way. So the candidate has to be trained on Luganda from OUTSIDE this corpus.

dmusingu/w2v-bert-2.0-luganda-CV-train-validation-7.0 is that: Wav2Vec2BertForCTC fine-tuned on
Common Voice 7 Luganda, 19.33% WER on the CV test set — a different recording domain, different
speakers, different prompts. Ungated on the Hub, which is what the rules require ("You may use
pretrained models as long as they are openly available to everyone").

    (The first draft of this file used Sunbird/asr-mms-salt. It is gated=auto: the Kaggle run
    died on GatedRepoError 401 three minutes in. Accepting that licence is the account owner's
    click to make, not this script's, so the model is out until someone accepts it by hand.)

w2v-bert takes mel features (input_features) where MMS takes a raw waveform (input_values).
Stage 3 asks the processor which it produced rather than inferring it from the backend name, so
this checkpoint drops into the waxalnet slot without a code change — and its logits are still
CTC frame logits, so KenLM shallow fusion applies exactly as in lugA.

WHAT WOULD FALSIFY IT

If lugB lands below lugA, out-of-domain Common Voice Luganda is no closer to phase-2 audio than
MMS's generic multilingual pretraining, and the search should move to Ugandan-corpus models
(Sunbird SALT, once ungated) rather than more read-speech corpora. Cheap to find out: one GPU
hour and one of 197 remaining submissions.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")
WORKING = Path("/kaggle/working")

LUG_MODEL = "dmusingu/w2v-bert-2.0-luganda-CV-train-validation-7.0"   # the variable under test
LIN_MODEL = "waxal-benchmarking/mms-300m-waxal-lin"                   # 8 clips, along for the ride
SNA_MODEL = "waxal-benchmarking/mms-300m-waxal-sna"                   # 89 clips, ditto
RUN_TAG = "lugB"
OBSERVED = 0.491944347                   # sub_01


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

# Load the checkpoint BEFORE an hour of audio downloads, not after. The previous lugB spent its
# whole session discovering a 401 that this three-minute probe surfaces immediately.
#
# AutoModelForCTC, not Wav2Vec2ForCTC: this checkpoint is Wav2Vec2BertForCTC and the hardcoded
# class raises on it. That is the same mistake stage 3 already fixed in its waxalnet loader; the
# probe has to load the model the same way the real run will or it is not a probe.
from transformers import AutoModelForCTC, AutoProcessor  # noqa: E402

print(f"\nprobing {LUG_MODEL} ...", flush=True)
_pr = AutoProcessor.from_pretrained(LUG_MODEL)
_md = AutoModelForCTC.from_pretrained(LUG_MODEL, torch_dtype=torch.float16)
_vocab = _pr.tokenizer.get_vocab()
print(f"  ok — {type(_md).__name__}, vocab {len(_vocab):,} tokens")
print(f"  feature extractor: {type(_pr.feature_extractor).__name__}")
# The tokenizer's alphabet is what pyctcdecode builds its labels from, and a KenLM over Luganda
# text is useless against a vocab that cannot spell it. Print it so a mismatch is visible in the
# log rather than showing up as a mysteriously bad score.
print(f"  alphabet: {''.join(sorted(t for t in _vocab if len(t) == 1))}")
del _md, _pr
torch.cuda.empty_cache()

# ---------------------------------------------------------------- guard: the corpus must be here
# Same search as lugA, for the same reason: the lineup run failed silently on a missing corpus and
# cost an hour producing a greedy file we then submitted. Find it at any depth, under either
# mechanism (an attached Dataset or a kernel_sources mount), and hand the located path to stage 3
# rather than letting stage 3 re-derive an assumption that has already been wrong once.
_hits = sorted(Path("/kaggle/input").rglob("lug.txt"))
if not _hits:
    tree = sorted(str(p) for p in Path("/kaggle/input").glob("*/*"))[:60]
    raise SystemExit(
        "no lug.txt anywhere under /kaggle/input — the LM corpus did not mount.\n"
        f"  what IS mounted: {tree or 'NOTHING — /kaggle/input is empty'}\n"
        "  Attach lethabomh14/waxal-lm-corpus as a dataset_source and re-push.\n"
        "  Refusing to fall through to the Train.csv-only LM: lugA and lugB have to share a "
        "language model or the acoustic comparison between them means nothing.")
LUG_TXT = _hits[0]
CORPUS = LUG_TXT.parent
print(f"\nfound LM corpus at {CORPUS}")
print(f"  files: {sorted(p.name for p in CORPUS.glob('*.txt'))}")
_words = sum(len(line.split()) for line in LUG_TXT.read_text(encoding="utf-8").splitlines())
print(f"\nLM corpus: lug.txt {_words:,} words")
if _words < 2_000_000:
    raise SystemExit(f"lug.txt has only {_words:,} words — expected ~8.2M. Wrong/truncated mount.")

# ---------------------------------------------------------------- guard: routing must match lugA
LANG_MAP = REPO / "data" / "routing" / "lang_map_mmsclosed_phase2.json"
if not LANG_MAP.exists():
    raise SystemExit(f"{LANG_MAP} missing from the repo clone — commit it before running.")
_m = json.loads(LANG_MAP.read_text(encoding="utf-8"))
_mix = {lg: sum(1 for v in _m.values() if v == lg) for lg in ("lin", "sna", "lug")}
print(f"routing (replayed from sub_01): {len(_m):,} clips {_mix}")
assert _mix["lug"] == 1403 and _mix["sna"] == 89 and _mix["lin"] == 8, (
    f"routing map is not sub_01's ({_mix}) — lugA/lugB stop being comparable, fix the map")

env = dict(os.environ)
env.update(
    PYTHONUNBUFFERED="1",
    CUDA_VISIBLE_DEVICES="0",
    WAXAL_BACKEND="waxalnet",          # whole-model swaps, not MMS adapter swaps
    WAXAL_LUG=LUG_MODEL,
    WAXAL_LIN=LIN_MODEL,
    WAXAL_SNA=SNA_MODEL,
    WAXAL_LANG_MAP=str(LANG_MAP),
    WAXAL_LM_CORPUS_DIR=str(CORPUS),   # the corpus we FOUND, not the one stage 3 would assume
    WAXAL_RUN_TAG=RUN_TAG,
    # WAXAL_NO_LM deliberately unset — lugB must decode with the same LM as lugA.
    # WAXAL_PLUS_PERIOD deliberately unset — the +0.0123 trailing-'.' win was measured on the
    # WAXAL dev set with a different checkpoint, and dev has stopped predicting phase 2. Adding it
    # here would be a second uncontrolled variable on top of the one this run exists to measure.
)
env.pop("WAXAL_NO_LM", None)
env.pop("WAXAL_PLUS_PERIOD", None)

print(f"\n{'=' * 78}\n=== {RUN_TAG}: lug={LUG_MODEL} + 5-gram KenLM, sub_01 routing\n{'=' * 78}",
      flush=True)
sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")], env=env, check=False)

print(f"\n{'=' * 78}\n=== WHAT TO READ OFF THIS RUN\n{'=' * 78}")
tune = WORKING / "lm_tuning.json"
if tune.exists():
    print(f"\n  tuned alpha/beta: {tune.read_text(encoding='utf-8')}")
else:
    print("\n  no lm_tuning.json — the LM section did not complete; read the traceback above.")
subs = sorted(WORKING.glob("*.csv"))
print(f"\n  submission files written: {[p.name for p in subs] or 'NONE'}")
print("\n  Compare against lugA (same LM, same routing, different acoustics) to price the")
print(f"  Luganda acoustic model, and against {OBSERVED:.6f} (sub_01) for the total move.")
