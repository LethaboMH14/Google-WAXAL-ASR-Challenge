"""waxal-lugB — same LM, same routing, DIFFERENT Luganda acoustic model.

WHY THIS RUN EXISTS

lugA holds the acoustic model fixed and adds the language model. This run holds the language
model and the routing fixed and changes the acoustic model. Read together they separate the
two levers; read alone, neither would.

    lugA   mms-1b-all zero-shot   + 5-gram KenLM + sub_01 routing
    lugB   Sunbird/asr-mms-salt   + 5-gram KenLM + sub_01 routing     <-- this file

WHY SUNBIRD RATHER THAN ANOTHER WAXAL FINE-TUNE

The leaderboard already priced the WAXAL fine-tunes on phase-2 audio and the answer was bad.
Backing per-clip quality out of the two scored files:

    zero-shot mms-1b-all      ~0.51 on phase-2 Luganda
    mms-300m-waxal-lug        ~0.38 on phase-2 Luganda   (dev said 0.8286)

A model fine-tuned on WAXAL scores *worse* on phase 2 than the generic checkpoint it was built
from, while scoring far better on WAXAL-domain dev. That is domain shift, not noise: the dev
0.8286 was measured greedily with the dev ids held out of the LM, so there is no leakage left
to explain it away. Phase-2 audio is not WAXAL audio, and fitting harder to WAXAL moves the
wrong way.

So the candidate has to be a model trained on Luganda from OUTSIDE this corpus.
Sunbird/asr-mms-salt is that: adapter-based Wav2Vec2ForCTC over the SALT dataset — real-world
Ugandan speech across lug/ach/lgg/nyn/teo, monolingual and English code-switched. Openly
available on the Hub, which is what the rules require ("You may use pretrained models as long
as they are openly available to everyone"), and it loads through the same
`model.load_adapter('lug')` path the mms backend already implements, so nothing in stage 3
needs to change.

WHAT WOULD FALSIFY IT

If lugB lands below lugA, SALT's domain is no closer to phase 2 than MMS's generic pretraining
and the search should move to Common-Voice-trained Luganda checkpoints
(dmusingu/w2v-bert-2.0-luganda-CV7, 19.33 WER on CV) rather than more Ugandan-corpus models.
Cheap to find out: this is one GPU hour and one of 197 remaining submissions.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")
WORKING = Path("/kaggle/working")

ASR_MODEL = "Sunbird/asr-mms-salt"
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

# The adapter has to exist before an hour of decoding is spent finding out it does not. Stage 3
# catches a missing adapter and silently falls back to lug on mms-1b-all, which for THIS run
# would mean quietly re-running lugA under lugB's name.
from transformers import AutoProcessor, Wav2Vec2ForCTC  # noqa: E402

print(f"\nprobing {ASR_MODEL} for a lug adapter ...", flush=True)
_pr = AutoProcessor.from_pretrained(ASR_MODEL)
_md = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL, torch_dtype=torch.float16)
_pr.tokenizer.set_target_lang("lug")
_md.load_adapter("lug")
print(f"  ok — lug adapter loaded, vocab {len(_pr.tokenizer.get_vocab()):,} tokens")
del _md, _pr
torch.cuda.empty_cache()

CORPUS = Path("/kaggle/input/waxal-lm/lm_corpus")
LUG_TXT = CORPUS / "lug.txt"
if not LUG_TXT.exists():
    found = sorted(str(p) for p in Path("/kaggle/input").glob("*/lm_corpus/*.txt"))
    raise SystemExit(
        f"{LUG_TXT} is missing — add lethabomh14/waxal-lm to kernel_sources and re-push.\n"
        f"  corpus files visible under /kaggle/input: {found or 'none'}")
_words = sum(len(line.split()) for line in LUG_TXT.read_text(encoding="utf-8").splitlines())
print(f"\nLM corpus: lug.txt {_words:,} words")
if _words < 2_000_000:
    raise SystemExit(f"lug.txt has only {_words:,} words — expected ~8.2M. Wrong/truncated mount.")

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
    WAXAL_BACKEND="mms",
    WAXAL_ASR_MODEL=ASR_MODEL,
    WAXAL_LANG_MAP=str(LANG_MAP),
    WAXAL_RUN_TAG=RUN_TAG,
)
env.pop("WAXAL_NO_LM", None)

print(f"\n{'=' * 78}\n=== {RUN_TAG}: {ASR_MODEL} + 5-gram KenLM, sub_01 routing\n{'=' * 78}",
      flush=True)
sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")], env=env, check=False)

print(f"\n{'=' * 78}\n=== WHAT TO READ OFF THIS RUN\n{'=' * 78}")
tune = WORKING / "lm_tuning.json"
if tune.exists():
    print(f"\n  tuned alpha/beta: {tune.read_text(encoding='utf-8')}")
subs = sorted(WORKING.glob("*.csv"))
print(f"\n  submission files written: {[p.name for p in subs] or 'NONE'}")
print(f"\n  Compare against lugA (same LM, same routing, different acoustics) to price the")
print(f"  acoustic model, and against {OBSERVED:.6f} (sub_01) for the total move.")
