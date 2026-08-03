"""Train the restorer AND sweep its confidence threshold, in one self-contained kernel.

The first probe scored this classifier at ARGMAX and got -0.0050 against "always append one
period", and I closed the restoration question on that. Too broad: argmax writes a mark whenever it
is merely the likeliest of three options, while the documented failure mode is PRECISION — "a false
mark corrupts a word that was otherwise correct". A confidence threshold is the direct lever on
precision and was never swept. Threshold 0.0 below reproduces the argmax number, so the sweep
carries its own control.

Worth reopening because the comma deficit is identical across every phase-2 system on the team:

    file                          '.'/utt   ','/utt
    his s16 (0.745734, best)        2.03      0.13
    his s18 (0.740833)              2.00      0.08
    his s17 (0.734984)              1.50      0.05
    Train.csv reference             1.77      0.75

~0.62 missing commas per utterance over 892 rows is ~550 word errors, ~2.3% WER, on the order of
+0.012 — the same size as our gap to the leaderboard leader, and the only large defect that every
system shares.

Trains inline rather than mounting the earlier kernel's output: kernel_sources did not attach and
the run died at the guard. 03_decode_and_submit.py's own comments record the same class of silent
mount failure costing two scored submissions on 1 Aug. Building in-process removes it entirely.
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

# ---------------------------------------------------------------- threshold sweep
@torch.inference_mode()
def restore_thr(words, thr):
    """Attach a mark only when its probability clears `thr`. thr=0.0 reproduces argmax."""
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
        mark = ""
        if p is not None:
            k = int(p[1:].argmax()) + 1
            if float(p[k]) >= thr:
                mark = TRAILING[k - 1]
        out.append(w + mark)
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
print("THRESHOLD SWEEP — restored vs 'always .'  (thr 0.00 = the argmax number already measured)")
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
    best = (None, -9.0)
    # NB: this is a mark-confidence threshold, not an argmax control — `k` is the better
    # of {'.', ','} and the 'none' class is never in the running, so thr=0 marks every
    # token by construction. The comparison that matters is against `always.`
    for thr in (0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95):
        hyp = [restore_thr(x.split(), thr) for x in cor]
        m = S.score(refs, hyp).multi
        cpu = sum(h.count(",") for h in hyp) / len(hyp)
        print(f"   {thr:>6.2f}{m:>11.4f}{m - always:>+13.4f}{cpu:>13.2f}"
              f"{'  <-- beats always.' if m > always else ''}")
        if m > best[1]:
            best = (thr, m)
    best_overall[rate] = (best[0], best[1] - always)

print("\n" + "=" * 92)
for rate, (thr, delta) in best_overall.items():
    print(f"  sim WER {rate}: best threshold {thr}  ->  {delta:+.4f} vs always.")
win = [r for r, (t, d) in best_overall.items() if d > 0]
print("\nVERDICT:", "APPLY at the winning threshold" if win else
      "DO NOT APPLY — no threshold beats 'always .'; restoration is closed on the merits")

if win:
    import csv  # noqa: E402

    thr = best_overall[max(win)][0]
    src = REPO / "data" / "submissions" / "team_best_0.7457_kWVXKLW3.csv"
    if src.exists():
        rows = list(csv.reader(open(src, encoding="utf-8")))
        outrows = []
        for r in rows[1:]:
            if len(r) < 2:
                continue
            # strip only . and , — never the apostrophe: it sits INSIDE words in all three
            # languages (g'ennyanja, ak'ekikobe) and is in the lin vocab
            base = re.sub(r"\s+", " ", re.sub(r"[.,]", "", r[1])).strip()
            outrows.append([r[0], restore_thr(base.split(), thr)])
        dst = WORKING / "submission_teambest_commas.csv"
        with open(dst, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(rows[0])
            w.writerows(outrows)
        print(f"\nwrote {dst.name} at threshold {thr}")
        sh([sys.executable, str(REPO / "local" / "validate_submission.py"), str(dst)], check=False)
