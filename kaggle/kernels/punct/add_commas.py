"""Train the restorer, measure ADDITIVE comma insertion, and apply it to the team-best file.

WHAT CHANGED FROM THE LAST RUN
------------------------------
The sweep found a real, replicated gain: at threshold ~0.9, restoration beats "always append one
period" by +0.0020 / +0.0012 / +0.0018 at sim WER 0.32 / 0.38 / 0.42. But the apply step then
STRIPPED all punctuation from the team-best file and rebuilt it, and at that threshold the model is
too conservative to replace what it removed: output had 1707 periods and 21 commas against the
original's 1811 and ~116. It degraded a 0.745734 file. The probe measured "restore onto bare text";
the operation we actually want is "add to already-good text", and those are not the same thing.

So this run measures the operation it performs. Baseline is the file as it stands (trailing period
already present, which is what `always.` models). Treatment is that same text with high-confidence
commas ADDED at word boundaries that currently have no mark. Nothing is ever removed, no existing
punctuation is touched, and a word that already carries a mark is skipped entirely — so the change
is strictly additive and cannot undo anything that is already scoring.

WHY COMMAS SPECIFICALLY
-----------------------
Every phase-2 system on the team is short of them by the same margin: 0.13 commas/utt on the
0.745734 file, 0.08 and 0.05 on the other two, against 0.75 in Train.csv. jiwer keeps punctuation
welded to its word after tokenising, so a missing comma is a whole word error while CER barely
registers it. That is ~550 word errors on a file we already own.

Expect small. The sweep says the honest number is around +0.0015, not the +0.012 the raw deficit
suggests, because precision collapses long before recall gets anywhere near 0.75 commas/utt.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")
WORKING = Path("/kaggle/working")

BASE_MODEL = "xlm-roberta-base"   # ungated, and the architecture sulaimank's gated restorer uses
EPOCHS = 3
MAX_LEN = 256
BATCH = 16
LR = 3e-5
SEED = 1337


def sh(cmd, check=True, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=check, **kw)


if not REPO.exists():
    sh(["git", "clone", "--depth", "1", REPO_URL, str(REPO)])
print("repo at", subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip())

# Match the environment the lineup kernels already run in. Installing transformers/accelerate
# bare pulls whatever resolves against the preinstalled torch, and requirements-gpu.txt is the
# pinned set those runs proved out.
sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements-gpu.txt")])
sh([sys.executable, "-m", "pip", "install", "-q", "jiwer"])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: E402

HARNESS = REPO / "local" / "harness"
sys.path.insert(0, str(HARNESS))
import score as S  # noqa: E402

sys.path.insert(0, str(REPO / "local" / "harness"))
from punct_probe import PUNCT, TRAILING, strip_punct, to_labels  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if DEVICE != "cuda":
    raise SystemExit("no CUDA — set the kernel accelerator to GPU T4 x2 before running")

# Fail here, loudly, rather than 100 steps into training. Version 1 of this kernel died with
# "CUDA error: no kernel image is available for execution on the device": Kaggle handed it a GPU
# whose compute capability the installed torch has no compiled kernels for, which is a property
# of the accelerator the notebook is set to, not of this code. torch 2.10 ships sm_75 and up, so
# a P100 (sm_60) fails while a T4 (sm_75) works.
_cap = torch.cuda.get_device_capability()
print(f"gpu: {torch.cuda.get_device_name(0)}  sm_{_cap[0]}{_cap[1]}  "
      f"torch arch list: {torch.cuda.get_arch_list()}")
if f"sm_{_cap[0]}{_cap[1]}" not in torch.cuda.get_arch_list():
    raise SystemExit(
        f"this torch has no kernels for sm_{_cap[0]}{_cap[1]} — switch the notebook accelerator "
        f"to GPU T4 x2 (sm_75) and re-run; nothing below would work on this device")
torch.manual_seed(SEED)

# ---------------------------------------------------------------- data, with the leak guard
df = pd.read_csv(REPO / "data" / "zindi" / "Train.csv", engine="python", on_bad_lines="skip")
dev = json.load(open(HARNESS / "devset.json", encoding="utf-8"))
dev_ids = {it["id"] for it in dev["items"]}

tr = df[(df["original_split"] == "train") & (~df["id"].isin(dev_ids))]
tr = tr[tr["transcription"].astype(str).str.strip().str.len() > 1]
print(f"train rows {len(tr):,}   dev rows {len(dev['items']):,}   "
      f"overlap {len(set(tr['id']) & dev_ids)} (must be 0)")
assert not (set(tr["id"]) & dev_ids), "dev ids leaked into the restorer's training set"

tok = AutoTokenizer.from_pretrained(BASE_MODEL)


class PunctData(Dataset):
    """Word-level labels (0 none, 1 '.', 2 ',') aligned to each word's FIRST subword.

    Only the first subword carries the label; continuation pieces get -100 so the loss ignores
    them. Predicting on a continuation piece would let the model put a comma inside a word.
    """

    def __init__(self, texts):
        self.items = []
        for t in texts:
            words, labels = to_labels(str(t))
            if 1 <= len(words) <= MAX_LEN - 2:
                self.items.append((words, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        words, labels = self.items[i]
        enc = tok(words, is_split_into_words=True, truncation=True,
                  max_length=MAX_LEN, padding="max_length")
        out, prev = [], None
        for wid in enc.word_ids():
            if wid is None or wid == prev:
                out.append(-100)
            else:
                out.append(labels[wid])
            prev = wid
        return {"input_ids": torch.tensor(enc["input_ids"]),
                "attention_mask": torch.tensor(enc["attention_mask"]),
                "labels": torch.tensor(out)}


ds = PunctData(tr["transcription"].tolist())
print(f"usable training sentences: {len(ds):,}")
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)

model = AutoModelForTokenClassification.from_pretrained(BASE_MODEL, num_labels=3).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
steps = EPOCHS * len(dl)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
scaler = torch.amp.GradScaler("cuda")

# '.' and ',' are together ~9% of tokens. Unweighted, the model maximises accuracy by predicting
# "none" everywhere, which is precisely the baseline we are trying to beat. Weights are mild
# rather than fully balanced: over-weighting buys recall at the cost of precision, and precision
# is what this metric pays for.
w = torch.tensor([1.0, 3.0, 4.0], device=DEVICE)
lossf = torch.nn.CrossEntropyLoss(weight=w, ignore_index=-100)

print(f"\n=== TRAIN  {EPOCHS} epochs x {len(dl):,} steps ===")
model.train()
step = 0
for ep in range(EPOCHS):
    run = 0.0
    for b in dl:
        b = {k: v.to(DEVICE, non_blocking=True) for k, v in b.items()}
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            loss = lossf(logits.view(-1, 3), b["labels"].view(-1))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run += loss.item()
        step += 1
        if step % 200 == 0:
            print(f"  ep{ep} step {step}/{steps}  loss {run / 200:.4f}", flush=True)
            run = 0.0

model.eval()

# ---------------------------------------------------------------- additive-only restoration
@torch.inference_mode()
def comma_probs(words):
    """P(comma) for each word position. Index 2 is the comma class (0 none, 1 '.', 2 ',')."""
    if not words:
        return []
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=MAX_LEN,
              return_tensors="pt")
    lg = model(input_ids=enc["input_ids"].to(DEVICE),
               attention_mask=enc["attention_mask"].to(DEVICE)).logits[0]
    prob = torch.softmax(lg.float(), -1).cpu()
    first, prev = {}, None
    for pos, wid in enumerate(enc.word_ids()):
        if wid is not None and wid != prev:
            first[wid] = float(prob[pos][2])
        prev = wid
    return [first.get(i, 0.0) for i in range(len(words))]


def add_commas(text, thr):
    """Add ',' only where the word carries no mark already. Never removes, never rewrites."""
    words = text.split()
    if not words:
        return text
    ps = comma_probs(words)
    out = []
    for w, p in zip(words, ps):
        # skip anything already punctuated: it is either right, or a period we must not clobber
        if p >= thr and not re.search(r"[^\w']$", w):
            out.append(w + ",")
        else:
            out.append(w)
    return " ".join(out)


import random  # noqa: E402

refs = [it["reference"] for it in dev["items"]]
bare = [strip_punct(r) for r in refs]
vocab = sorted({w for b in bare for w in b.split()})
rng = random.Random(SEED)


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
print("ADDITIVE COMMA INSERTION — baseline is 'text + trailing period', i.e. what we already ship")
print("=" * 92)
best = {}
for rate in (0.32, 0.38, 0.42):
    cor = [corrupt(b, rate) for b in bare]
    shipped = [c + "." for c in cor]                       # the file as it stands today
    base = S.score(refs, shipped).multi
    print(f"\nsim WER {rate}:   baseline(as-shipped) {base:.4f}")
    print(f"   {'thr':>6}{'+commas':>11}{'delta':>10}{'commas/utt':>13}")
    bt = (None, -9.0)
    for thr in (0.70, 0.80, 0.85, 0.90, 0.95):
        hyp = [add_commas(x, thr) for x in shipped]
        m = S.score(refs, hyp).multi
        cpu = sum(h.count(",") for h in hyp) / len(hyp)
        print(f"   {thr:>6.2f}{m:>11.4f}{m - base:>+10.4f}{cpu:>13.2f}"
              f"{'  <-- gain' if m > base else ''}")
        if m > bt[1]:
            bt = (thr, m)
    best[rate] = (bt[0], bt[1] - base)

print("\n" + "=" * 92)
for rate, (thr, d) in best.items():
    print(f"  sim WER {rate}: best thr {thr}  ->  {d:+.4f}")
win = [r for r, (t, d) in best.items() if d > 0]
print("\nVERDICT:", "APPLY" if len(win) >= 2 else "DO NOT APPLY - additive insertion does not pay either")

if len(win) >= 2:
    import csv  # noqa: E402

    thr = best[0.38][0]          # tune at the rate closest to a real submission
    src = REPO / "data" / "submissions" / "team_best_0.7457_kWVXKLW3.csv"
    rows = list(csv.reader(open(src, encoding="utf-8")))
    out, changed = [], 0
    for r in rows[1:]:
        if len(r) < 2:
            continue
        new = add_commas(r[1], thr)
        changed += (new != r[1])
        out.append([r[0], new])
    dst = WORKING / "submission_teambest_pluscommas.csv"
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(rows[0])
        w.writerows(out)
    print(f"\nwrote {dst.name} at thr {thr}; {changed}/{len(out)} rows changed")
    sh([sys.executable, str(REPO / "local" / "validate_submission.py"), str(dst)], check=False)
