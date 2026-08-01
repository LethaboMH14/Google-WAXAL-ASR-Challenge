"""
Kaggle GPU kernel: BAKEOFF — measure every open WAXAL checkpoint on our frozen dev set and pick
the best one PER LANGUAGE.

WHY PER LANGUAGE
----------------
There is no single publisher who is strongest at all three competition languages, and the metric
does not care about elegance — jiwer pools, so the submission is just three independent decoding
problems weighted by their share of reference words (lin 45.9%, sna 36.6%, lug 17.5%). Mixing
architectures across languages costs nothing and is very likely to win, so the only question worth
asking is "which checkpoint is best at THIS language", three times.

WHAT IS BEING TESTED (see docs/MODEL-CANDIDATES.md for how the shortlist was built)
-----------------------------------------------------------------------------------
The organisers' own `waxal-benchmarking/mms-300m-waxal-*` are the control in every group — they
are, on the evidence of download counts and the leaderboard's 0.7206-0.7257 cluster, what the
leaders are running. Anything that does not beat the control is not interesting.

The challengers are chosen for one property the control lacks: **punctuation in the vocabulary**.
All three control vocabs contain zero sentence punctuation (verified by downloading vocab.json),
and the metric counts punctuation, so a perfect no-punctuation transcriber caps at 0.9367. That
cap is worth ~0.063 and nobody above us appears to have taken it.

Whisper checkpoints are included for the same reason: their tokenizer is BPE over ordinary text,
so a Whisper model fine-tuned on WAXAL transcripts emits punctuation natively. For Shona that is
the only open route to punctuation we found.

COST CONTROL
------------
Each run decodes ONE language (~135-395 clips), not all 900, via WAXAL_DEV_LANGS. All runs share
one frozen audio cache written on the first run, so every candidate sees byte-identical input and
the comparison measures the model rather than two resampling passes. LM is off (WAXAL_NO_LM=1):
this session asks how good the acoustic models are on their own. Shallow fusion is measured
separately, on top of whichever checkpoint wins here.
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

# (tag, language, backend, repo, note). Controls first in each group so the cache is warm and the
# baseline number is on screen before the challengers run.
CANDIDATES = [
    # ---- lin: 45.9% of dev reference words, the heaviest language ----
    ("lin-control",  "lin", "waxalnet", "waxal-benchmarking/mms-300m-waxal-lin",
     "organisers' baseline — no punctuation in vocab"),
    ("lin-xlsr",     "lin", "waxalnet", "keystats/lingala-xlsr-waxal-finetuned",
     "XLSR, FULL punctuation in vocab"),
    ("lin-w2vbert",  "lin", "waxalnet", "douyeszn/w2vbert-lin-waxal-aug-ft",
     "w2v-bert-2.0, apostrophe only"),

    # ---- sna: 36.6% of words, 44.5% of characters — CER-heaviest ----
    ("sna-control",  "sna", "waxalnet", "waxal-benchmarking/mms-300m-waxal-sna",
     "organisers' baseline — no punctuation in vocab"),
    ("sna-whisper",  "sna", "whisper",  "Mubarak127/waxal-whisper-large-v3-sna_asr",
     "whisper-large-v3, punctuation native to the tokenizer"),
    ("sna-drewmens", "sna", "waxalnet", "DrewMens/mms-waxal-shona",
     "mms fine-tune"),

    # ---- lug: 17.5% of words, the lightest — but the control is already strong here ----
    ("lug-control",  "lug", "waxalnet", "waxal-benchmarking/mms-300m-waxal-lug",
     "organisers' baseline — best published numbers of the three"),
    ("lug-w2vbert",  "lug", "waxalnet", "douyeszn/w2vbert-lug-waxal-aug",
     "w2v-bert-2.0, FULL punctuation in vocab"),
    ("lug-dhasmana", "lug", "waxalnet", "dhasmana/WAXAL-lug-ful-w2v-bert-2.0",
     "w2v-bert-2.0, '.' in vocab"),
    ("lug-whisper",  "lug", "whisper",
     "cdli/whisper-large-v3_finetuned_ugandan_luganda_waxal_7_standard_speech_v1.0",
     "whisper-large-v3, punctuation native"),
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
print("  number to trust is a full three-language dev run of the winning combination — which is")
print("  the next kernel, not this one.")

print("\n--- /kaggle/working ---", flush=True)
for p in sorted(WORKING.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 1e6:10.1f} MB  {p.relative_to(WORKING)}")
