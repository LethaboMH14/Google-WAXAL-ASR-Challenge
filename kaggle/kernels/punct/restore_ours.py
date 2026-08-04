"""Restore full punctuation on OUR pipeline's output, where the deficit is large.

WHY OUR FILE AND NOT THE TEAM-BEST ONE
--------------------------------------
Applied to an already-punctuated file this is worth +0.0001 — measured, submitted, confirmed. Our
own output is a completely different case:

    file                       '.'/utt   ','/utt
    ours (open lineup)            1.00      0.00
    Train.csv reference           1.77      0.75

Our decode emits no punctuation at all; the single period per row is appended by the pipeline. So
we are missing ~687 internal periods and ~669 commas across 892 rows — about 5.8% of all tokens,
and under jiwer every one of those is a whole word error while CER barely registers it.

This is exactly the scenario the probe measured. Baseline "always append one period" IS our file,
and restoring onto it beat that baseline by +0.0020 / +0.0012 / +0.0017 at sim WER 0.32 / 0.38 /
0.42, threshold ~0.9. The earlier disappointment came from applying it to a file that already had
good punctuation; here the headroom is real.

Threshold 0.9 is carried over rather than re-swept: precision is the failure mode, the sweep found
0.9 optimal at all three error rates, and re-deriving it would only re-measure what is already
known.
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

# ---------------------------------------------------------------- apply to our own output
@torch.inference_mode()
def restore(words, thr=0.90):
    """Attach '.' or ',' where the model clears `thr`. Words already carrying a mark are skipped."""
    if not words:
        return ""
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=MAX_LEN,
              return_tensors="pt")
    lg = model(input_ids=enc["input_ids"].to(DEVICE),
               attention_mask=enc["attention_mask"].to(DEVICE)).logits[0]
    prob = torch.softmax(lg.float(), -1).cpu()
    first, prev = {}, None
    for pos, wid in enumerate(enc.word_ids()):
        if wid is not None and wid != prev:
            first[wid] = prob[pos]
        prev = wid
    out = []
    for i, w in enumerate(words):
        p = first.get(i)
        if p is not None and not re.search(r"[^\w']$", w):
            k = int(p[1:].argmax()) + 1
            if float(p[k]) >= thr:
                w = w + TRAILING[k - 1]
        out.append(w)
    return " ".join(out)


def capfirst(t):
    t = re.sub(r"^(\s*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)
    return re.sub(r"([.?!]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)


import csv  # noqa: E402

# ADDITIVE ONLY. The previous run stripped the pipeline's trailing period and let the model
# re-decide every mark; at threshold 0.9 it is too conservative to put them back and 51 terminal
# periods were lost (892 -> 841), while 82.4% of references end in one. Never strip again: keep
# every mark the file already has and only add marks to words that carry none.
TARGETS = ["ours_kenlm_capfirst.csv", "best_dedup_commas.csv"]

for name in TARGETS:
    src = REPO / "data" / "submissions" / name
    if not src.exists():
        print(f"!! {name} missing")
        continue
    rows = list(csv.reader(open(src, encoding="utf-8")))
    out = []
    for r in rows[1:]:
        if len(r) < 2:
            continue
        words = r[1].split()
        marked = restore(words).split()      # restore() skips already-punctuated words
        if len(marked) != len(words):        # never let a length change through
            marked = words
        # keep the final word exactly as it was: its terminal punctuation is already correct
        if marked:
            marked[-1] = words[-1]
        out.append([r[0], capfirst(" ".join(marked))])

    dst = WORKING / f"restored_{name}"
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(rows[0])
        w.writerows(out)
    t = [x[1] for x in out]
    n = len(t)
    b = [r[1] for r in rows[1:] if len(r) >= 2]
    print(f"\n{name} -> restored_{name}")
    print(f"  '.'/utt {sum(x.count('.') for x in b)/n:.2f} -> {sum(x.count('.') for x in t)/n:.2f}   "
          f"','/utt {sum(x.count(',') for x in b)/n:.2f} -> {sum(x.count(',') for x in t)/n:.2f}   "
          f"words {sum(len(x.split()) for x in b)} -> {sum(len(x.split()) for x in t)}")
    sh([sys.executable, str(REPO / "local" / "validate_submission.py"), str(dst)], check=False)
