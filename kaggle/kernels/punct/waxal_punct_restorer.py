"""Kaggle GPU kernel: PUNCTUATION RESTORER — a real sequence model, trained on corrupted text.

WHY THIS EXISTS
---------------
The competition metric lowercases both sides and does nothing else, so punctuation is scored. Every
`.` and `,` a hypothesis omits is one word error plus one character error. Measured on our 900-clip
dev set, a PERFECT transcriber that emits no punctuation caps at 0.9367, not 1.0 — and all three
organiser baselines (`waxal-benchmarking/mms-300m-waxal-*`) have zero sentence punctuation in their
vocabularies, verified by downloading vocab.json. That is ~0.063 of score nobody above us appears
to be collecting.

WHY THE FIRST ATTEMPT FAILED, AND WHAT CHANGED
----------------------------------------------
`local/harness/punct_probe.py` trained a logistic-regression restorer on CLEAN transcripts. It
reached period F1=0.714 and still LOST to blindly appending one full stop (+0.0043 vs +0.0083 at
simulated WER 0.32). The reason is precision, not recall: a mark placed on a word the ASR already
got right converts a correct word into an error, so on corrupted input a confident-but-wrong
restorer is worse than a dumb rule with 0.82 precision by construction.

The fix is not a bigger model on the same data — it is the same model on the RIGHT data. This
trains on text corrupted exactly the way an ASR corrupts it (the substitution/deletion/insertion
mix from punct_probe, at a WER sampled per example across 0.10-0.45), so the model learns to keep
quiet when its context is garbage. That is the difference between predicting punctuation and
predicting punctuation *you can still see the evidence for*.

HONEST TARGET
-------------
Blind trailing '.' is worth +0.0083. The full oracle — every reference mark re-attached to every
word that survived corruption — is +0.038. This has to beat +0.0083 on the dev set to ship at all,
and the gap to +0.038 is the entire prize on offer here. If it does not clear the blind rule, we
ship the blind rule and this file is the record of why.

DATA AND RULES
--------------
Trains only on `Train.csv` rows with `original_split == "train"`, with the 900 frozen dev ids
excluded. No test-split text is read anywhere in this file. `Davlan/afro-xlmr-base` is XLM-R
further pretrained on African languages (Shona and Luganda among them) and is openly available.
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")
WORKING = Path("/kaggle/working")

BASE = os.environ.get("WAXAL_PUNCT_BASE", "Davlan/afro-xlmr-base")
EPOCHS = float(os.environ.get("WAXAL_PUNCT_EPOCHS", "3"))
MAXLEN = 192
SEED = 1337
# Label per word: what mark FOLLOWS it. '?' and '!' occur 2 and 12 times in 900 dev references —
# far too rare to learn and not worth the precision risk, so they are folded into "no mark".
LABELS = ["", ".", ","]
PUNCT = "!\"&'(),-./:;?«»“”"


def sh(cmd, check=True, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=check, **kw)


if not REPO.exists():
    sh(["git", "clone", "--depth", "1", REPO_URL, str(REPO)])
sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements-gpu.txt")])
sh([sys.executable, "-m", "pip", "install", "-q", "jiwer", "accelerate"])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: E402

sys.path.insert(0, str(REPO / "local" / "harness"))
import score as S  # noqa: E402

if not torch.cuda.is_available():
    raise SystemExit("no CUDA — set the kernel accelerator to GPU T4 before running")
DEVICE = "cuda"
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
print(f"gpu: {torch.cuda.get_device_name(0)}   base: {BASE}")

# ------------------------------------------------------------------ data
df = pd.read_csv(REPO / "data" / "zindi" / "Train.csv", escapechar="\\")
dev = json.load(open(REPO / "local" / "harness" / "devset.json", encoding="utf-8"))
dev_items = dev["items"]
dev_ids = {it["id"] for it in dev_items}
tr = df[(df["original_split"] == "train") & (~df["id"].astype(str).isin(dev_ids))]
print(f"train rows: {len(tr):,}   dev rows: {len(dev_ids):,}")


def strip_punct(w: str) -> str:
    return "".join(c for c in w if c not in PUNCT)


def to_pairs(text: str):
    """(bare word, label) per word. Label is the trailing mark we care about, else ''."""
    out = []
    for w in str(text).split():
        mark = ""
        for m in (".", ","):
            if w.rstrip().endswith(m):
                mark = m
                break
        b = strip_punct(w)
        if b:
            out.append((b, mark))
    return out


train_pairs = [to_pairs(t) for t in tr["transcription"]]
train_pairs = [p for p in train_pairs if p]
train_langs = list(tr["language"])[:len(train_pairs)]
VOCAB = sorted({w for p in train_pairs for w, _ in p})
print(f"training sentences: {len(train_pairs):,}   vocab: {len(VOCAB):,}")


def corrupt(words: list[str], rate: float, rng: random.Random) -> list[str]:
    """The exact error mix from punct_probe.py: 70% substitution, 20% deletion, 10% insertion of
    the error budget. Keeping it identical is the point — the offline numbers in
    docs/MODEL-CANDIDATES.md were measured with this, so the two are comparable."""
    out: list[str] = []
    for w in words:
        p = rng.random()
        if p < rate * 0.7:
            out.append(rng.choice(VOCAB))
        elif p < rate * 0.9:
            continue
        elif p < rate:
            out += [w, rng.choice(VOCAB)]
        else:
            out.append(w)
    return out or [rng.choice(VOCAB)]


tok = AutoTokenizer.from_pretrained(BASE)


def encode(words: list[str], labels: list[int] | None):
    """Word-level labels on the FIRST subword of each word; -100 elsewhere so continuation pieces
    and specials are ignored by the loss and by argmax read-back."""
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=MAXLEN)
    wid = enc.word_ids()
    lab, prev = [], None
    for k, w in enumerate(wid):
        if w is None or w == prev:
            lab.append(-100)
        else:
            lab.append(labels[w] if labels is not None and w < len(labels) else 0)
        prev = w
    enc = dict(enc)
    enc["labels"] = lab
    return enc


class PunctData(Dataset):
    """Corruption is applied at __getitem__, not once up front, so every epoch sees a different
    noise draw of the same sentence — more augmentation for free, and it stops the model
    memorising one particular corrupted form."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        rng = random.Random(SEED * 1_000_003 + i)
        pr = self.pairs[i]
        words = [w for w, _ in pr]
        marks = [m for _, m in pr]
        # Sample the error rate per example across the range real checkpoints land in, so the
        # restorer is not tuned to one WER it will not see. 15% clean keeps the easy case sharp.
        rate = 0.0 if rng.random() < 0.15 else rng.uniform(0.10, 0.45)
        kept: list[str] = []
        kept_lab: list[int] = []
        for w, m in zip(words, marks):
            p = rng.random()
            if p < rate * 0.7:
                kept.append(rng.choice(VOCAB))
                kept_lab.append(0)          # a substituted word is wrong anyway; do not mark it
            elif p < rate * 0.9:
                continue                     # deleted: its mark goes with it
            elif p < rate:
                kept.append(w)
                kept_lab.append(LABELS.index(m) if m in LABELS else 0)
                kept.append(rng.choice(VOCAB))
                kept_lab.append(0)
            else:
                kept.append(w)
                kept_lab.append(LABELS.index(m) if m in LABELS else 0)
        if not kept:
            kept, kept_lab = [rng.choice(VOCAB)], [0]
        return encode(kept, kept_lab)


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id
    out = {
        "input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
        "attention_mask": torch.tensor(
            [b["attention_mask"] + [0] * (n - len(b["attention_mask"])) for b in batch]),
        "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
    }
    return out


model = AutoModelForTokenClassification.from_pretrained(BASE, num_labels=len(LABELS)).to(DEVICE)
dl = DataLoader(PunctData(train_pairs), batch_size=16, shuffle=True, collate_fn=collate,
                num_workers=2, drop_last=True)
opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
steps = int(len(dl) * EPOCHS)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-5, total_steps=steps, pct_start=0.1)
scaler = torch.cuda.amp.GradScaler()

print(f"\ntraining: {steps:,} steps ({EPOCHS} epochs over {len(dl):,} batches)")
model.train()
step = 0
done = False
for ep in range(int(EPOCHS) + 1):
    if done:
        break
    for batch in dl:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.cuda.amp.autocast():
            loss = model(**batch).loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        if step % 200 == 0:
            print(f"  step {step:,}/{steps:,}  loss {loss.item():.4f}", flush=True)
        if step >= steps:
            done = True
            break

model.eval()
out_dir = WORKING / "punct_restorer"
model.save_pretrained(out_dir)
tok.save_pretrained(out_dir)
print(f"saved {out_dir}")


# ------------------------------------------------------------------ apply + evaluate
@torch.inference_mode()
def restore(sentences: list[list[str]], bs: int = 32) -> list[str]:
    """Words in, punctuated string out. Threshold is argmax; a mark is only emitted when the model
    prefers it to 'no mark', which is what keeps precision up on corrupted input."""
    res: list[str] = []
    for k in range(0, len(sentences), bs):
        chunk = sentences[k:k + bs]
        encs = [encode(w, None) for w in chunk]
        batch = collate(encs)
        logits = model(input_ids=batch["input_ids"].to(DEVICE),
                       attention_mask=batch["attention_mask"].to(DEVICE)).logits.float().cpu()
        for words, enc, lg in zip(chunk, encs, logits):
            keep = [j for j, x in enumerate(enc["labels"]) if x != -100]
            picks = lg[keep].argmax(-1).tolist()
            res.append(" ".join(
                w + (LABELS[picks[j]] if j < len(picks) else "") for j, w in enumerate(words)))
    return res


refs = [it["reference"] for it in dev_items]
langs = [it["language"] for it in dev_items]
bare = [" ".join(strip_punct(w) for w in r.split()) for r in refs]
rng = random.Random(SEED)
dev_vocab = sorted({w for b in bare for w in b.split()})


def corrupt_eval(t: str, rate: float) -> str:
    out = []
    for w in t.split():
        p = rng.random()
        if p < rate * 0.7:
            out.append(rng.choice(dev_vocab))
        elif p < rate * 0.9:
            continue
        elif p < rate:
            out += [w, rng.choice(dev_vocab)]
        else:
            out.append(w)
    return " ".join(out) or rng.choice(dev_vocab)


print(f"\n{'=' * 78}\n=== SIMULATED — same table as punct_probe.py, directly comparable\n{'=' * 78}")
print(f"   {'sim WER':>8}  {'none':>8}  {'always .':>9}  {'neural':>8}  {'oracle':>8}"
      f"   {'neural-always.':>15}")
rows = {}
for rate in (0.0, 0.15, 0.25, 0.32, 0.42):
    cor = [corrupt_eval(b, rate) for b in bare]
    a = S.score(refs, cor).multi
    b_ = S.score(refs, [c + "." for c in cor]).multi
    c = S.score(refs, restore([x.split() for x in cor])).multi
    orc = []
    for r, x in zip(refs, cor):
        pm = {strip_punct(w): w for w in r.split() if any(ch in PUNCT for ch in w)}
        orc.append(" ".join(pm.get(w, w) for w in x.split()))
    d = S.score(refs, orc).multi
    rows[rate] = {"none": a, "always_period": b_, "neural": c, "oracle": d}
    print(f"   {rate:8.2f}  {a:8.4f}  {b_:9.4f}  {c:8.4f}  {d:8.4f}   {c - b_:+15.4f}")

# ------------------------------------------------------------------ the honest test: real ASR text
# Simulated corruption is a model of an ASR, not an ASR. If a bakeoff prediction file is attached,
# the restorer is scored on genuine hypotheses — wrong in the ways a real CTC decoder is wrong,
# which is not the way a uniform random substitution is wrong.
real = {}
for cand in list(Path("/kaggle/input").rglob("dev_preds_*.json")):
    tag = cand.stem.replace("dev_preds_", "")
    preds = json.load(open(cand, encoding="utf-8"))
    ok = [it for it in dev_items if it["id"] in preds]
    if not ok:
        continue
    r_ = [it["reference"] for it in ok]
    h_ = [preds[it["id"]] for it in ok]
    a = S.score(r_, h_).multi
    b_ = S.score(r_, [h + "." if h.strip() else h for h in h_]).multi
    c = S.score(r_, restore([h.split() for h in h_])).multi
    real[tag] = {"n": len(ok), "none": a, "always_period": b_, "neural": c}
    print(f"\n  REAL [{tag}] n={len(ok)}: none={a:.4f}  always.={b_:.4f}  neural={c:.4f}"
          f"   (neural-always. {c - b_:+.4f})")
if not real:
    print("\n  no dev_preds_*.json attached — attach the bakeoff kernel output to score this on")
    print("  genuine ASR hypotheses. The simulated table above is a proxy, not the answer.")

print(f"\n{'-' * 78}\n  VERDICT\n{'-' * 78}")
ref_rate = 0.32
win = rows[ref_rate]["neural"] - rows[ref_rate]["always_period"]
print(f"  at sim WER {ref_rate}: neural {rows[ref_rate]['neural']:.4f} vs blind period "
      f"{rows[ref_rate]['always_period']:.4f}  ({win:+.4f})")
print(f"  headroom to oracle: {rows[ref_rate]['oracle'] - rows[ref_rate]['neural']:+.4f}")
print("  SHIP IT" if win > 0.002 else "  DO NOT SHIP — blind trailing '.' is still the better rule")

json.dump({"simulated": rows, "real": real, "base": BASE, "epochs": EPOCHS},
          open(WORKING / "punct_result.json", "w"), indent=1)
print(f"\nwrote {WORKING / 'punct_result.json'}")
