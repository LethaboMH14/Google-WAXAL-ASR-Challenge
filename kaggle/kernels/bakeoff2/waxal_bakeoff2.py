"""
Kaggle GPU kernel: BAKEOFF ROUND 2 — the candidates round 1 never saw.

WHY A SECOND ROUND
------------------
Round 1 shortlisted from docs/MODEL-CANDIDATES.md, which was built by hand. A systematic sweep of
the Hub afterwards (198 distinct ASR checkpoints matching lingala/shona/luganda/waxal, each access
tested by an actual config.json download rather than by model_info) turned up several strong,
OPEN, untried models. Two facts make this worth GPU time:

  1. **lin carries 45.9% of the metric's reference words and is our weakest language** (0.7788).
     A point of lin is worth 2.6x the same point on lug. Round 1 had exactly one non-control lin
     challenger. This round has four.
  2. Round 1's winner for sna is a whisper-large-v3 fine-tune (0.8034, the best single number we
     have anywhere). `noirlab` publishes the same architecture for BOTH lin and sna, and neither
     was tested.

INCUMBENTS ARE RE-RUN, NOT ASSUMED
----------------------------------
Each language's round-1 winner runs again in this session rather than being quoted from round 1.
The decode path changed since then (per-language backends, the finish() helper), and a control
that shares a session with its challengers is the only kind that rules out drift. If an incumbent
reproduces its round-1 number, that is also a free reproducibility check the rules ask for.

WHAT IS NOT HERE
----------------
The `sulaimank` punct-v2 set — a complete, punctuation-aware w2v-bert lineup for exactly our three
languages — is still 403. It is the single biggest untested lever and it needs a human to accept
the terms on each model page. `sulaimank/xlsr-luganda-waxal` IS open and is included, which is
some evidence the publisher's gating is per-repo rather than blanket.

COST CONTROL
------------
Each run decodes ONE language via WAXAL_DEV_LANGS, off one shared frozen audio cache, so every
candidate sees byte-identical input. LM off (WAXAL_NO_LM=1) — the alpha/beta we had was tuned
against a leaked KenLM and is void; fusion gets re-measured on top of whatever wins here.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")
WORKING = Path("/kaggle/working")


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
_cc = torch.cuda.get_device_capability(0)
_sm = f"sm_{_cc[0]}{_cc[1]}"
if _sm not in torch.cuda.get_arch_list():
    raise SystemExit(
        f"this card is {torch.cuda.get_device_name(0)} ({_sm}) and torch {torch.__version__} was "
        f"built for {torch.cuda.get_arch_list()}. Set machine_shape to 'NvidiaTeslaT4'.")
print(f"gpu: {torch.cuda.get_device_name(0)} {_sm}")

# (tag, language, backend, repo, note). Incumbent first in each group: its number is on screen
# before any challenger runs, and it warms the audio cache for the rest of the group.
CANDIDATES = [
    # ---- lin: 45.9% of dev reference words, our weakest language, so the deepest bench ----
    ("lin-incumbent", "lin", "waxalnet", "douyeszn/w2vbert-lin-waxal-aug-ft",
     "ROUND-1 WINNER 0.7788 — re-run as this session's control"),
    ("lin-noirlab",   "lin", "whisper",  "noirlab/whisper-large-v3-lingala-asr",
     "whisper-large-v3 — same architecture that won sna in round 1, punctuation native"),
    ("lin-braintheos", "lin", "waxalnet", "BrainTheos/wav2vec2-large-mms-1b-all-lingala-ojpl",
     "mms-1b-all fine-tuned on Lingala — 1,193 downloads, the most-used open lin ASR"),
    ("lin-wsmall",    "lin", "whisper",  "waxal-benchmarking/whisper-small-waxal-lin",
     "organisers' own whisper baseline — punctuation native, trained on the WAXAL corpus itself"),

    # ---- sna: 36.6% of words, 44.5% of characters ----
    ("sna-incumbent", "sna", "whisper",  "Mubarak127/waxal-whisper-large-v3-sna_asr",
     "ROUND-1 WINNER 0.8034 — the best single number we have"),
    ("sna-noirlab",   "sna", "whisper",  "noirlab/whisper-large-v3-shona-asr",
     "same size and architecture as the incumbent, different publisher"),
    ("sna-badrex",    "sna", "waxalnet", "badrex/w2v-bert-2.0-shona-asr",
     "w2v-bert-2.0 — the architecture that won lin in round 1"),

    # ---- lug: 17.5% of words, and round 1 found nothing that beat the control ----
    ("lug-incumbent", "lug", "waxalnet", "waxal-benchmarking/mms-300m-waxal-lug",
     "ROUND-1 WINNER 0.8163 — no challenger beat it"),
    ("lug-sulaimank", "lug", "waxalnet", "sulaimank/xlsr-luganda-waxal",
     "1,182 downloads and OPEN, unlike the rest of this publisher's WAXAL set"),
    ("lug-allandclive", "lug", "whisper", "allandclive/whisper-medium-luganda",
     "5,173 downloads — most-used Luganda ASR on the Hub, though not WAXAL-specific"),
]

ENVKEY = {"lin": "WAXAL_LIN", "sna": "WAXAL_SNA", "lug": "WAXAL_LUG"}

for tag, lang, backend, repo, note in CANDIDATES:
    print(f"\n{'=' * 78}\n=== {tag}  [{lang}]  {repo}\n===   {note}\n{'=' * 78}", flush=True)
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0",
               WAXAL_DEV="1", WAXAL_NO_LM="1",
               WAXAL_BACKEND=backend, WAXAL_DEV_LANGS=lang, WAXAL_DEV_TAG=tag,
               # All three slots point at the same repo: only `lang` is decoded, and this way a
               # typo in ENVKEY cannot silently leave a slot on the default and decode the control
               # a second time under the challenger's tag.
               WAXAL_LIN=repo, WAXAL_SNA=repo, WAXAL_LUG=repo)
    env[ENVKEY[lang]] = repo
    # check=False: one dead candidate (gated repo, OOM, missing vocab.json) must not cost us the
    # other nine. A missing dev_result_<tag>.json in the summary is how a failure reports itself.
    sh([sys.executable, str(REPO / "kaggle" / "03_decode_and_submit.py")], env=env, check=False)

# ------------------------------------------------------------------ summary
print(f"\n{'=' * 78}\n=== BAKEOFF RESULTS\n{'=' * 78}")
# Share of dev reference WORDS, measured — this is the weight each language actually carries in a
# pooled jiwer score, and it is not the share of rows.
WORD_SHARE = {"lin": 0.459, "sna": 0.366, "lug": 0.175}
best: dict[str, tuple] = {}

for tag, lang, backend, repo, note in CANDIDATES:
    p = WORKING / f"dev_result_{tag}.json"
    if not p.exists():
        print(f"  {tag:14} [{lang}] FAILED — no result file (see its traceback above)")
        continue
    r = json.load(open(p))
    s = r["per_language"].get(lang) or r["per_language"]["overall"]
    pp = r.get("plus_period_by_lang", {}).get(lang, {})
    dotted = pp.get("plus_period", s["multi"])
    print(f"  {tag:14} [{lang}] multi={s['multi']:.4f}  WER={s['wer']:.4f}  CER={s['cer']:.4f}"
          f"   +period={dotted:.4f}   {repo.split('/')[-1][:38]}")
    score = max(s["multi"], dotted)
    if lang not in best or score > best[lang][0]:
        best[lang] = (score, tag, repo, s["multi"], dotted)

print(f"\n{'-' * 78}\n  BEST PER LANGUAGE (using +period where it helps)\n{'-' * 78}")
total = 0.0
for lang in ("lin", "sna", "lug"):
    if lang not in best:
        print(f"  {lang}: NO RESULT")
        continue
    sc, tag, repo, raw, dotted = best[lang]
    total += WORD_SHARE[lang] * sc
    print(f"  {lang} ({WORD_SHARE[lang]:.1%} of words): {sc:.4f}  <- {tag}  {repo}")
    print(f"        raw={raw:.4f}  +period={dotted:.4f}")

print(f"\n  word-weighted estimate of the combined submission: {total / sum(WORD_SHARE.values()):.4f}")
print("  rank-1 on the public leaderboard is 0.7257; our current submission is 0.4919.")
print("\n  NOTE: this weighting is an approximation. The real metric pools errors across the whole")
print("  corpus rather than averaging per-language scores, so treat it as a ranking aid. The")
print("  number to trust is a full three-language dev run of the winning combination — kernel")
print("  waxal-lineup, not this one.")

print("\n--- /kaggle/working ---", flush=True)
for p in sorted(WORKING.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 1e6:10.1f} MB  {p.relative_to(WORKING)}")
