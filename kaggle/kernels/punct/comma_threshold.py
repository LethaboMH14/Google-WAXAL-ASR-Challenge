"""Sweep the restorer's confidence threshold — the question the first probe left unanswered.

WHAT WAS MEASURED, AND WHAT WAS NOT
-----------------------------------
waxal-punct-restorer trained an XLM-RoBERTa token classifier and scored it at ARGMAX: -0.0050
against "always append one period" at sim WER 0.32. I read that as "punctuation restoration does
not pay" and closed the question. That conclusion was too broad. Argmax is threshold 0.34-ish for a
3-way head — it marks a token whenever the mark is merely the most likely of three options. The
documented failure mode is PRECISION: "a false mark corrupts a word that was otherwise correct",
and precision is exactly what a threshold buys. Sweeping it was never tried.

WHY IT IS WORTH TRYING AGAIN
-----------------------------
Every phase-2 system on our team is short of commas by roughly the same amount:

    file                          '.'/utt   ','/utt
    s16  (0.745734, team best)      2.03      0.13
    s18  (0.740833)                 2.00      0.08
    s17  (0.734984)                 1.50      0.05
    Train.csv reference             1.77      0.75

~0.62 missing commas per utterance over 892 utterances is ~550 word errors, ~2.3% WER, worth on the
order of +0.012 — the same size as the gap between our 0.745734 and the leaderboard leader. It is
the largest single defect that is identical across every system we have, ours and his alike.

HOW THIS IS SCORED
------------------
The identical corrupted-reference probe, at the error rates that bracket a real submission, but
sweeping the probability a comma or period must clear before it is written. Threshold 0.0 reproduces
the argmax number already measured, so the sweep contains its own control: if nothing beats
"always .", the answer really is no and this closes the question for good rather than on a
technicality.

Reuses the trained model from the waxal-punct-restorer kernel output rather than retraining, so this
is minutes rather than half an hour.
"""

import csv
import json
import random
import re
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")
WORKING = Path("/kaggle/working")
MODEL_DIR = Path("/kaggle/input/waxal-punct-restorer/punct_restorer")


def sh(cmd, check=True, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=check, **kw)


if not REPO.exists():
    sh(["git", "clone", "--depth", "1", REPO_URL, str(REPO)])
sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements-gpu.txt")])
sh([sys.executable, "-m", "pip", "install", "-q", "jiwer"])

import torch  # noqa: E402
from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__}  device={DEVICE}")
if DEVICE == "cuda":
    _c = torch.cuda.get_device_capability()
    print(f"gpu: {torch.cuda.get_device_name(0)} sm_{_c[0]}{_c[1]}")

if not MODEL_DIR.exists():
    raise SystemExit(
        f"{MODEL_DIR} missing — add the waxal-punct-restorer kernel to this notebook's inputs "
        f"(Add Input -> Notebook Output), or its trained model is not mounted.")

HARNESS = REPO / "local" / "harness"
sys.path.insert(0, str(HARNESS))
import score as S  # noqa: E402
from punct_probe import PUNCT, TRAILING, strip_punct  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR)).to(DEVICE).eval()


@torch.inference_mode()
def restore(words, thr):
    """Attach a mark only when its probability clears `thr`. thr=0 reproduces argmax."""
    if not words:
        return ""
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=256,
              return_tensors="pt")
    logits = model(input_ids=enc["input_ids"].to(DEVICE),
                   attention_mask=enc["attention_mask"].to(DEVICE)).logits[0]
    prob = torch.softmax(logits.float(), -1).cpu()
    first, prev = {}, None
    for pos, wid in enumerate(enc.word_ids()):
        if wid is not None and wid != prev:
            first[wid] = prob[pos]
        prev = wid
    out = []
    for i, w in enumerate(words):
        p = first.get(i)
        mark = ""
        if p is not None:
            k = int(p[1:].argmax()) + 1          # best of {., ,}
            if float(p[k]) >= thr:
                mark = TRAILING[k - 1]
        out.append(w + mark)
    return " ".join(out)


dev = json.load(open(HARNESS / "devset.json", encoding="utf-8"))
refs = [it["reference"] for it in dev["items"]]
bare = [strip_punct(r) for r in refs]
vocab = sorted({w for b in bare for w in b.split()})
rng = random.Random(1337)


def corrupt(t, rate):
    out = []
    for w in t.split():
        p = rng.random()
        if p < rate * 0.7:
            out.append(rng.choice(vocab))
        elif p < rate * 0.9:
            continue
        elif p < rate:
            out += [w, rng.choice(vocab)]
        else:
            out.append(w)
    return " ".join(out) or rng.choice(vocab)


print("\n" + "=" * 92)
print("THRESHOLD SWEEP — restored vs 'always .' at each simulated error rate")
print("=" * 92)
best_overall = {}
for rate in (0.32, 0.38, 0.42):
    cor = [corrupt(b, rate) for b in bare]
    always = S.score(refs, [c + "." for c in cor]).multi
    orc = []
    for r, x in zip(refs, cor):
        pm = {strip_punct(w): w for w in r.split() if any(ch in PUNCT for ch in w)}
        orc.append(" ".join(pm.get(w, w) for w in x.split()))
    oracle = S.score(refs, orc).multi
    print(f"\nsim WER {rate}:   always. {always:.4f}   oracle {oracle:.4f}")
    print(f"   {'thr':>6}{'restored':>11}{'vs always.':>13}{'commas/utt':>13}")
    best = (None, -9)
    for thr in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        hyp = [restore(x.split(), thr) for x in cor]
        m = S.score(refs, hyp).multi
        cpu = sum(h.count(",") for h in hyp) / len(hyp)
        flag = "  <-- beats always." if m > always else ""
        print(f"   {thr:>6.2f}{m:>11.4f}{m - always:>+13.4f}{cpu:>13.2f}{flag}")
        if m > best[1]:
            best = (thr, m)
    best_overall[rate] = (best[0], best[1] - always)

print("\n" + "=" * 92)
for rate, (thr, delta) in best_overall.items():
    print(f"  sim WER {rate}: best threshold {thr}  ->  {delta:+.4f} vs always.")
win = [r for r, (t, d) in best_overall.items() if d > 0]
print("\nVERDICT:", "APPLY at the winning threshold" if win else
      "DO NOT APPLY — no threshold beats 'always .'; the restoration question is now closed")

# ---- apply to the team-best file only if the sweep actually won at a realistic rate
if win:
    thr = best_overall[max(win)][0]
    src = REPO / "data" / "submissions" / "team_best_0.7457_kWVXKLW3.csv"
    if src.exists():
        with open(src, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        head, body = rows[0], rows[1:]
        out = []
        for r in body:
            if len(r) < 2:
                continue
            # strip only . and , — never the apostrophe, which sits inside words in all three
            # languages (g'ennyanja, ak'ekikobe) and is in the lin vocab
            base = re.sub(r"\s+", " ", re.sub(r"[.,]", "", r[1])).strip()
            out.append([r[0], restore(base.split(), thr)])
        dst = WORKING / "submission_teambest_commas.csv"
        with open(dst, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(out)
        print(f"\nwrote {dst.name} at threshold {thr}")
        sh([sys.executable, str(REPO / "local" / "validate_submission.py"), str(dst)], check=False)
