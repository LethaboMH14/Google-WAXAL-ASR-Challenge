"""waxal-lugA — the controlled A/B: change ONLY the language model.

WHY THIS RUN EXISTS

Two files have been scored on the leaderboard:

    sub_01  zero-shot mms-1b-all, routed by mms-lid-256   WER 0.7601  CER 0.2560  ->  0.4919
    sub_04  WAXAL fine-tunes, routed by the CTC router    WER 0.9346  CER 0.3932  ->  0.3361

sub_04's CER (0.3932) is the CER of a file where EVERY clip went to the wrong model
(measured floor: 0.3944). That refutes the CTC router's 47.5%-Luganda read of phase 2 and
confirms mms-lid-256's 93.5%. Phase 2 is overwhelmingly Luganda.

It also killed the previous plan. sub_04 changed the routing AND the acoustic models at the
same time, so when it fell 0.156 there was no way to attribute the loss. This run does not
repeat that. Against sub_01 it changes exactly one thing:

    routing        IDENTICAL to sub_01 (the mms-lid-256 map, replayed from the repo)
    acoustic model IDENTICAL to sub_01 (facebook/mms-1b-all, zero-shot)
    language model GREEDY  ->  5-gram KenLM shallow fusion        <-- the only delta

Whatever this run scores, the difference from 0.4919 is the LM and nothing else.

WHY THE LM IS THE RIGHT SINGLE VARIABLE

Neither scored submission used one. sub_01 and sub_04 both ran WAXAL_NO_LM=1, and the lineup
kernel additionally never attached waxal-lm, so its corpus was not even mounted — the log says
`no LM corpus at /kaggle/input/waxal-lm/lm_corpus`. Two independent reasons the largest lever
in the pipeline has never been pulled.

The corpus has existed the whole time: waxal-lm's output carries lug.txt at 8,197,542 words,
the same order of magnitude (9.2M) as the LMs in arXiv:2512.10968, which measures on Luganda:

    w2v-bert CTC   39.75 WER  ->  16.30 with a 5-gram      (-59% relative)

Our WER is 0.7601 against a CER of 0.2560. That spread — characters roughly right, words
wrong — is the exact failure mode shallow fusion repairs.

WHAT WOULD FALSIFY THE PLAN

If this scores at or below 0.4919, the LM is not the lever on phase-2 audio and the remaining
gap is acoustic; go to lugB's model class and stop spending runs on decoding. Anything above
~0.55 says fusion works here and the next run stacks a better Luganda acoustic model on top.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")
WORKING = Path("/kaggle/working")

ASR_MODEL = "facebook/mms-1b-all"        # identical to sub_01. Do not "improve" this here.
RUN_TAG = "lugA"
OBSERVED = 0.491944347                   # sub_01, the anchor this run is measured against


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

# ---------------------------------------------------------------- guard: the corpus must be here
# The lineup run failed silently on exactly this and cost an hour producing a greedy file we
# then submitted. A missing mount is not allowed to degrade quietly into the thing this run
# exists to replace.
#
# v1 of this kernel hard-coded /kaggle/input/waxal-lm/lm_corpus and died with "visible: none"
# even though kernel_sources named waxal-lm and Kaggle's stored metadata confirmed it. Rather
# than keep guessing at the mount layout, FIND the corpus: any depth, either mechanism
# (kernel_sources mount or an attached Dataset). Stage 3 is then pointed at whatever we found
# via WAXAL_LM_CORPUS_DIR instead of inheriting the same hard-coded assumption.
_hits = sorted(Path("/kaggle/input").rglob("lug.txt"))
if not _hits:
    tree = sorted(str(p) for p in Path("/kaggle/input").glob("*/*"))[:60]
    raise SystemExit(
        "no lug.txt anywhere under /kaggle/input — the LM corpus did not mount.\n"
        f"  what IS mounted: {tree or 'NOTHING — /kaggle/input is empty'}\n"
        "  Attach lethabomh14/waxal-lm-corpus as a dataset_source (or waxal-lm as a\n"
        "  kernel_source) and re-push.\n"
        "  Refusing to fall through to the Train.csv-only LM: that is a 177k-word corpus for a\n"
        "  5-gram, which the paper measures as WORSE than greedy, and this run's entire purpose\n"
        "  is to measure real shallow fusion.")
LUG_TXT = _hits[0]
CORPUS = LUG_TXT.parent
print(f"\nfound LM corpus at {CORPUS}")
print(f"  files: {sorted(p.name for p in CORPUS.glob('*.txt'))}")
_words = sum(len(line.split()) for line in LUG_TXT.read_text(encoding="utf-8").splitlines())
print(f"\nLM corpus: lug.txt {_words:,} words")
if _words < 2_000_000:
    raise SystemExit(f"lug.txt has only {_words:,} words — expected ~8.2M. Wrong/truncated "
                     f"mount; a 5-gram on this is a lookup table, not a language model.")

# ---------------------------------------------------------------- guard: routing must replay sub_01
LANG_MAP = REPO / "data" / "routing" / "lang_map_mmsclosed_phase2.json"
if not LANG_MAP.exists():
    raise SystemExit(f"{LANG_MAP} missing from the repo clone — commit it before running. "
                     f"Without it this run reroutes and stops being a controlled A/B.")
_m = json.loads(LANG_MAP.read_text(encoding="utf-8"))
_mix = {lg: sum(1 for v in _m.values() if v == lg) for lg in ("lin", "sna", "lug")}
print(f"routing (replayed from sub_01): {len(_m):,} clips {_mix}")
assert _mix["lug"] == 1403 and _mix["sna"] == 89 and _mix["lin"] == 8, (
    f"routing map is not sub_01's ({_mix}) — the A/B is broken, fix the map before running")

env = dict(os.environ)
env.update(
    PYTHONUNBUFFERED="1",
    CUDA_VISIBLE_DEVICES="0",
    WAXAL_BACKEND="mms",
    WAXAL_ASR_MODEL=ASR_MODEL,
    WAXAL_LANG_MAP=str(LANG_MAP),
    WAXAL_LM_CORPUS_DIR=str(CORPUS),   # the corpus we FOUND, not the one stage 3 would assume
    WAXAL_RUN_TAG=RUN_TAG,
    # WAXAL_NO_LM is deliberately NOT set. This is the whole experiment.
)
env.pop("WAXAL_NO_LM", None)

print(f"\n{'=' * 78}\n=== {RUN_TAG}: {ASR_MODEL} + 5-gram KenLM, sub_01 routing\n{'=' * 78}",
      flush=True)
sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")], env=env, check=False)

print(f"\n{'=' * 78}\n=== WHAT TO READ OFF THIS RUN\n{'=' * 78}")
tune = WORKING / "lm_tuning.json"
if tune.exists():
    t = json.loads(tune.read_text(encoding="utf-8"))
    print(f"\n  tuned alpha/beta: {json.dumps(t, indent=2)}")
    lug = t.get("lug") or {}
    if lug.get("alpha") is None:
        print("\n  !! lug fell back to GREEDY — the sweep found no alpha/beta that beat it.\n"
              "     That is a real result: it means fusion does not help this acoustic model on\n"
              "     this text, and the gap to the leaders is acoustic. Go to lugB.")
else:
    print("\n  no lm_tuning.json — the LM section did not complete; read the traceback above.")

subs = sorted(WORKING.glob("*.csv"))
print(f"\n  submission files written: {[p.name for p in subs] or 'NONE'}")
print(f"\n  Anchor to beat: {OBSERVED:.6f} (sub_01, same models, same routing, greedy).")
print("  The delta from that number is the language model, isolated.")
