"""Kaggle GPU kernel: train a real sequence-model punctuation restorer, and price it honestly.

WHY THIS AND NOT ANOTHER CHECKPOINT SWAP
----------------------------------------
docs/MODEL-CANDIDATES.md already priced this lever and then left it on the table:

    sim WER   none     always '.'   cheap restorer   oracle (all marks)
    0.32      0.6329   0.6412       0.6373           0.6706

At our real error rate the full punctuation oracle is worth +0.029 over the trailing period we
currently ship — the same order as the whole gap between our best submission (0.7131) and the
team's best (0.7457). The cheap restorer captured none of it and the doc says exactly why: a
logistic head on hashed features had period F1 0.714 but could not beat "always append one period"
on PRECISION, and precision is what this metric rewards, because a false mark corrupts a word that
was otherwise correct. Its own conclusion was that a good restorer "has to be a real sequence
model, not features-and-a-linear-head". This is that model.

Three facts make it worth the GPU time rather than a guess:
  - the references are punctuated and we have 38,199 of them in Train.csv (lin 16k / sna 16k /
    lug 6k), so this is supervised in-domain data we already own;
  - neither our best file nor the team's best emits commas (0.21 and 0.13 per utterance against
    0.75 in the references), so the headroom is real and unclaimed on both;
  - jiwer keeps punctuation welded to its word after lowercasing, so a missing comma is a whole
    word error and CER barely registers it — which is why this gap can be large on the scorer
    while the transcripts look fine to read.

MEASURED BEFORE IT IS SHIPPED
-----------------------------
The comparison below is the identical four-way probe punct_probe.py runs (none / always '.' /
restored / oracle) on the same held-out dev references at the same simulated error rates, so the
numbers land directly next to the table above rather than in a new frame that flatters them. If
this model does not beat the "always '.'" column it does not get applied, and the run has still
paid for itself by closing the question.

LEAK GUARD
----------
Trains ONLY on original_split == "train", and additionally excludes every dev id by hand. The dev
set is drawn from the validation split so the intersection should already be empty; this repo has
shipped two leaks (the KenLM corpus, then alpha/beta tuned on the dev clips) and both were found
after they had produced numbers we believed, so the belt-and-braces check stays.
"""

import json
import os
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


@torch.inference_mode()
def restore(words, _lang=None):
    """Attach '.' / ',' to the words a token classifier marks. Batched over one sentence."""
    if not words:
        return ""
    enc = tok(words, is_split_into_words=True, truncation=True,
              max_length=MAX_LEN, return_tensors="pt")
    ids = enc["input_ids"].to(DEVICE)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        lg = model(input_ids=ids, attention_mask=enc["attention_mask"].to(DEVICE)).logits[0]
    pred = lg.argmax(-1).cpu().tolist()
    first, prev = {}, None
    for pos, wid in enumerate(enc.word_ids()):
        if wid is not None and wid != prev:
            first[wid] = pred[pos]
        prev = wid
    return " ".join(w + (TRAILING[first.get(i, 0) - 1] if first.get(i, 0) else "")
                    for i, w in enumerate(words))


# ---------------------------------------------------------------- the identical four-way probe
refs = [it["reference"] for it in dev["items"]]
langs = [it["language"] for it in dev["items"]]
bare = [strip_punct(r) for r in refs]

print("\n--- 1. on PERFECT words (ceiling for this model class) ---")
print(f"   none {S.score(refs, bare).multi:.4f}   "
      f"always. {S.score(refs, [b + '.' for b in bare]).multi:.4f}   "
      f"restored {S.score(refs, [restore(b.split()) for b in bare]).multi:.4f}")

import random  # noqa: E402

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


print("\n--- 2. on CORRUPTED words (what it is worth on a real ASR) ---")
print(f"   {'sim WER':>8}  {'none':>8}  {'always .':>9}  {'restored':>9}  {'oracle':>8}"
      f"  {'restored - always.':>19}")
verdict = {}
for rate in (0.15, 0.32, 0.42):
    cor = [corrupt(b, rate) for b in bare]
    a = S.score(refs, cor).multi
    b_ = S.score(refs, [c + "." for c in cor]).multi
    c = S.score(refs, [restore(x.split()) for x in cor]).multi
    orc = []
    for r, x in zip(refs, cor):
        pm = {strip_punct(w): w for w in r.split() if any(ch in PUNCT for ch in w)}
        orc.append(" ".join(pm.get(w, w) for w in x.split()))
    d = S.score(refs, orc).multi
    verdict[rate] = c - b_
    print(f"   {rate:>8.2f}  {a:>8.4f}  {b_:>9.4f}  {c:>9.4f}  {d:>8.4f}  {c - b_:>+19.4f}")

gain = verdict[0.32]
print(f"\nVERDICT at the realistic rate (0.32): restored - always. = {gain:+.4f}")
print("APPLY" if gain > 0 else "DO NOT APPLY — 'always .' is still better; ship nothing from this run")

model.save_pretrained(WORKING / "punct_restorer")
tok.save_pretrained(WORKING / "punct_restorer")

# ---------------------------------------------------------------- apply to the candidate files
import csv  # noqa: E402
import re  # noqa: E402


def strip_marks(text: str) -> str:
    """Remove only the marks this model predicts — NOT the apostrophe.

    punct_probe's strip_punct also drops ' and -, which is harmless inside the probe because both
    sides of every comparison get the same treatment. It is not harmless here: the apostrophe sits
    INSIDE words in all three languages (g'ennyanja, ak'ekikobe, k'ennyaanya) and the lin
    checkpoint has it in vocab, so stripping it would rewrite words the ASR got right.
    """
    return re.sub(r"\s+", " ", re.sub(r"[.,]", "", text)).strip()


SUBS = REPO / "data" / "submissions"
if gain > 0 and SUBS.exists():
    for src in sorted(SUBS.glob("*.csv")):
        with open(src, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        head, body = rows[0], rows[1:]
        out = [[r[0], restore(strip_marks(r[1]).split())] for r in body if len(r) >= 2]
        dst = WORKING / f"restored_{src.name}"
        with open(dst, "w", newline="", encoding="utf-8") as f:
            wri = csv.writer(f)
            wri.writerow(head)
            wri.writerows(out)
        print(f"  wrote {dst.name}  ({len(out)} rows)")
        sh([sys.executable, str(REPO / "local" / "validate_submission.py"), str(dst)], check=False)
else:
    print("\nnot applying to submissions (either the probe said no, or data/submissions is empty)")

print("\n--- /kaggle/working ---")
for p in sorted(WORKING.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 1e6:9.2f} MB  {p.relative_to(WORKING)}")
