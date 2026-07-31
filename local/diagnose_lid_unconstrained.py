"""What language does LID think the phase-2 clips are, when we don't force its hand?

Stage 1 constrains mms-lid-256's argmax to {lin, sna, lug}, because those are the three languages
the challenge covers. Under that constraint phase 2 came back 1,403/1,500 Luganda while the same
model calibrated at 97.3% on phase-1 clips. Both numbers can be true at once if phase 2 contains
audio in languages that are NOT among the three: a constrained argmax has to put such a clip
somewhere, and for a Ugandan Bantu language the nearest of our three is always Luganda.

That is exactly what the phase-2 transcripts look like — `hukendera hu luguudo` uses hu- where
Luganda uses ku-, and `ni ndeeba` is not Luganda at all.

So: run the same model over the same clips with all 256 classes live and print what it actually
says. If the top-1 labels are dominated by non-target languages, our routing question is not
"which of three" but "is this one of the three at all", and that changes stage 3.

CPU only and fp32 — stage 2 owns the GPU and must not be disturbed. Budget a few minutes.

    python local/diagnose_lid_unconstrained.py [n_clips]
"""
from __future__ import annotations

import io
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

SEED = 1337                      # same seed as the pipeline; rerunning gives the same sample
LID_MODEL = "facebook/mms-lid-256"
TARGETS = ("lin", "sna", "lug")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40

HOME = Path("/teamspace/studios/this_studio")
WORK = (HOME if HOME.exists() else Path(__file__).resolve().parents[1]) / "waxal-work"
ZIP = WORK / "phase2_audio.zip"

if not ZIP.exists():
    raise SystemExit(f"no zip at {ZIP} — run stage 1 first")

print(f"loading {LID_MODEL} on CPU (fp32)...")
fe = AutoFeatureExtractor.from_pretrained(LID_MODEL)
lid = Wav2Vec2ForSequenceClassification.from_pretrained(LID_MODEL).eval()
id2label = lid.config.id2label
print(f"  {len(id2label)} languages in the head")

with zipfile.ZipFile(ZIP) as zf:
    names = sorted(n for n in zf.namelist()
                   if n.startswith("audio/") and n.lower().endswith(".wav"))
    print(f"  {len(names):,} phase-2 clips in the archive")
    sample = random.Random(SEED).sample(names, min(N, len(names)))

    top1: Counter[str] = Counter()
    in_target = 0
    rows = []
    for k, name in enumerate(sample):
        wav, sr = sf.read(io.BytesIO(zf.read(name)), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav[:16000 * 30]

        # attention_mask matters even for a batch of one on some HF versions, and costs nothing.
        inp = fe([wav], sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.inference_mode():
            logits = lid(inp.input_values,
                         attention_mask=inp.get("attention_mask")).logits[0]
        prob = torch.softmax(logits, dim=-1)
        best = torch.topk(prob, 5)

        labels = [id2label[int(i)] for i in best.indices]
        probs = [float(p) for p in best.values]
        top1[labels[0]] += 1
        in_target += labels[0] in TARGETS
        rows.append((Path(name).stem, labels, probs))
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(sample)}", flush=True)

print(f"\n--- top-1 over {len(sample)} phase-2 clips, ALL {len(id2label)} classes live ---")
for lang, n in top1.most_common(15):
    mark = "  <- target" if lang in TARGETS else ""
    print(f"  {lang:<8} {n:>4}  ({n / len(sample):5.1%}){mark}")

print(f"\n  top-1 is one of {TARGETS}: {in_target}/{len(sample)} ({in_target / len(sample):.1%})")
if in_target / len(sample) < 0.7:
    print("\n  !! Most phase-2 clips are NOT being identified as one of the three challenge\n"
          "     languages. Constraining the argmax to three classes then forces each of them\n"
          "     into the nearest neighbour, which for Ugandan Bantu is Luganda — and that is\n"
          "     the 94%-lug routing we saw. Read the per-clip table below before deciding what\n"
          "     stage 3 should do with these.")

print("\n--- per clip (top-5) ---")
for stem, labels, probs in rows[:25]:
    pretty = "  ".join(f"{lang}:{p:.2f}" for lang, p in zip(labels, probs))
    print(f"  {stem:<12} {pretty}")
