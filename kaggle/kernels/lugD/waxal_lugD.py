"""waxal-lugD — Ugandan-domain Whisper for Luganda. Ungated, MIT, no HF token needed.

WHY THIS RUN EXISTS

lugA/lugB hold routing fixed and vary (LM) and (Luganda acoustics) inside the CTC family. This
run leaves the CTC family entirely. Two independent reasons, both measured, not assumed:

1. PUNCTUATION IS WORTH REAL POINTS AND CTC CANNOT SPELL IT.
   Measured on Train.csv (38,048 rows), a *perfect* transcriber that emits no punctuation does
   not score 1.0:

       lug   perfect words, no punctuation, keeps apostrophe   WER 0.1085 CER 0.0159 -> 0.9378
       lug   perfect words, no punctuation, no apostrophe      WER 0.1846 CER 0.0288 -> 0.8933

   Broken down for Luganda, the cost of each class of mark on its own:

       apostrophes missing only        -> 0.9478   (-5.22 pts)   ng', k', ky', ez' are everywhere
       sentence-final .?! missing only -> 0.9629   (-3.71 pts)   83.7% of rows end in one
       commas missing only             -> 0.9762   (-2.38 pts)

   Those are ceilings, and they scale down with our accuracy — on a 0.76-WER transcriber the
   recoverable WER share is roughly (1 - WER) of the ideal, while the CER share lands nearly in
   full. So this is worth single-digit tenths of a point today, not six points. It grows as the
   acoustics improve, which is the point: it compounds with reason 2 rather than competing.

   sub_01 (our 0.4919) emits 1,715 apostrophes and exactly one full stop across 1,500 rows. The
   apostrophes are already right — and WAXAL uses the straight ' , which is what MMS's lug vocab
   carries. The sentence-final mark is simply absent.

2. THE ACOUSTIC DOMAIN.
   Phase 2 is ~93.5% Luganda and our Luganda is the whole score. The WAXAL fine-tunes went
   BACKWARDS on phase 2 (~0.38/clip vs zero-shot MMS ~0.51), so more WAXAL fitting is not the
   answer; the answer is Luganda from a different corpus, recorded the way phase-2 audio was.

KasuleTrevor/cdli-whisper-ml-ugeng-lug-swa-sunbird-full-a40-lr5e5 is whisper-large-v3 continued
from the Sunbird SALT lineage on Ugandan Luganda + Ugandan English + Swahili. Its own card
reports WER 0.3797 / CER 0.2041 on its test set (implied ~0.708 on this metric). MIT licensed,
gated=False — no token, no licence click, "openly available to everyone" as the rules require.

WHY NOT Sunbird/asr-whisper-large-v3-salt DIRECTLY

That is the better model (MIT; Sunbird's own card says it beats their MMS SALT model; trained
with added street noise and random 8 kHz downsampling to simulate phone speech, which is what
this corpus is). It is gated=auto. Lethabo has now accepted it, so it downloads fine from a
machine holding his HF token — but a Kaggle kernel runs UNAUTHENTICATED and a gated repo 401s
there regardless of who accepted what. waxal-lugC runs it via a Kaggle Secret; this file is the
version that needs nothing from anybody, and lugC's cdli checkpoint is derived from that same
Sunbird lineage anyway.

WHISPER LANGUAGE TOKEN

This checkpoint ships generation_config.language = null, which means Whisper language-detects
EVERY clip independently — on 1,403 Luganda clips that is 1,403 chances to guess English. It has
no Luganda token (vocab 51,866, the stock 100-language set), and CDLI fine-tuned and evaluated
Luganda under Swahili, so lug=sw is the checkpoint's own convention, not our guess.

    (Sunbird's own SALT checkpoints instead REPURPOSE stock token slots: lug is token 50355,
    which is Bashkir's slot, so there the setting is lug=ba. Recorded here because the two look
    interchangeable and are not.)

NO TRAILING '.' HERE

WAXAL_PLUS_PERIOD is deliberately unset. Whisper's tokenizer is BPE over ordinary text and this
checkpoint punctuates natively; appending '.' to a model that already emitted one produces '..'
and costs a word AND a character. The bakeoff measured exactly that on the seq2seq slot
(sna 0.8034 -> 0.7853). Reason 1 above is claimed by the MODEL here, not by string surgery.

WHAT WOULD FALSIFY IT

If lugD lands at or below lugA, then Ugandan-domain Whisper is no closer to phase-2 audio than
zero-shot MMS, and the remaining explanation for the gap to ~0.75 is neither the LM (lugA) nor
the acoustic corpus (lugB, lugD) — at which point the routing map itself goes back on trial.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")
WORKING = Path("/kaggle/working")

LUG_MODEL = "KasuleTrevor/cdli-whisper-ml-ugeng-lug-swa-sunbird-full-a40-lr5e5"  # under test
LIN_MODEL = "waxal-benchmarking/mms-300m-waxal-lin"       # 8 clips, along for the ride
SNA_MODEL = "waxal-benchmarking/mms-300m-waxal-sna"       # 89 clips, ditto
LUG_WHISPER_LANG = "sw"                                   # CDLI's own convention, see header
RUN_TAG = "lugD"
OBSERVED = 0.491944347                                    # sub_01, the anchor for every run


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

# Probe BEFORE an hour of audio downloads. lugB v1 spent a whole session discovering a 401 that a
# three-minute probe surfaces immediately.
from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402

print(f"\nprobing {LUG_MODEL} ...", flush=True)
_pr = WhisperProcessor.from_pretrained(LUG_MODEL)
_md = WhisperForConditionalGeneration.from_pretrained(LUG_MODEL, torch_dtype=torch.float16)
print(f"  ok — {type(_md).__name__}, vocab {_md.config.vocab_size:,}")
_l2i = getattr(_md.generation_config, "lang_to_id", None) or {}
_tok = _l2i.get(f"<|{LUG_WHISPER_LANG}|>")
print(f"  generation_config.language = {getattr(_md.generation_config, 'language', None)}")
print(f"  forcing lug -> <|{LUG_WHISPER_LANG}|> = token {_tok}")
if _tok is None:
    raise SystemExit(
        f"<|{LUG_WHISPER_LANG}|> is not in this checkpoint's lang_to_id ({len(_l2i)} entries) — "
        f"generate(language={LUG_WHISPER_LANG!r}) would raise. Pick a code this model knows.")
del _md, _pr
torch.cuda.empty_cache()

# ---------------------------------------------------------------- guard: the corpus must be here
# lin and sna still decode through KenLM; lug does not (seq2seq has no frame logits to beam over,
# and stage 3 now skips LM tuning for a whisper language rather than crashing in logits_for).
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
    f"routing map is not sub_01's ({_mix}) — lugA/B/D stop being comparable, fix the map")

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
    # WAXAL_NO_LM unset: lin/sna still get the same KenLM as lugA/lugB. lug skips it by backend.
    # WAXAL_PLUS_PERIOD unset: this model punctuates natively — see header.
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
    print("      ^ if 'end in .?!' is ~0 this model did NOT punctuate and PLUS_PERIOD should go "
          "back on for lug; if it is high, the punctuation ceiling above is being claimed.")
print(f"\n  Compare against lugA (LM lever), lugB (CTC acoustic lever) and {OBSERVED:.6f} (sub_01).")
