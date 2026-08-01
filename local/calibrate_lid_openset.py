"""How accurate is the OPEN-SET router on clips whose language we actually know?

Stage 1 prints this line and then scrolls it away behind 5,753 clips of decoding. Sbu asked for
the number and correctly refused to spend another GPU-hour re-running the whole stage to resurface
one print. This script produces the same number on its own: a few hundred phase-1 clips, LID only,
no MMS decoding, no submission written.

Why the number matters. Closed-set calibration (97.3%) asked LID a three-way question: is this
clip lin, sna or lug? Open-set routing asks a 2,396-way one, and a true-Luganda clip now has to
beat Runyankole, Lusoga, Kinyarwanda and every other neighbour rather than just Lingala and Shona.
If accuracy collapses here, open-set routing is scattering clips across near neighbours and phase 2
needs a confidence floor before we trust it. If it holds, the routing is sound and the luo/nyn-heavy
phase-2 mix is a fact about phase 2, not an artefact of loosening the label space.

The true language is the HF config the clip streams from — that is dataset metadata, not the
transcript. This script never reads a transcription column; it drops them on sight, same guard as
stage 1.

CPU is fine (LID is a one-shot classifier, no autoregression) and deliberate — stage 2 owns the
GPU. Budget ~10 min on CPU for the default 100 clips per language, most of it download.

    python local/calibrate_lid_openset.py [n_per_language]
"""
from __future__ import annotations

import io
import sys
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

SEED = 1337
LID_MODEL = "facebook/mms-lid-256"
ASR_MODEL = "facebook/mms-1b-all"
LANGS = ["lin", "sna", "lug"]
HF_CONFIGS = {"lin": "lin_asr", "sna": "sna_asr", "lug": "lug_asr"}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEVICE}  {N} clips per language\n")

# ---------------------------------------------------------------- the routing rule
# This has to be IDENTICAL to stage 1's or the number is about a different router. Stage 1 lets
# the argmax range over every LID label that mms-1b-all can actually decode, because a language
# we cannot decode is not a useful thing to route to.
from huggingface_hub import list_repo_files

HAS_ADAPTER = {f.split(".")[1] for f in list_repo_files(ASR_MODEL) if f.startswith("adapter.")}
print(f"{len(HAS_ADAPTER):,} adapters on {ASR_MODEL}")

fe = AutoFeatureExtractor.from_pretrained(LID_MODEL)
lid = Wav2Vec2ForSequenceClassification.from_pretrained(LID_MODEL).to(DEVICE).eval()
id2label = lid.config.id2label
route_idx = [i for i, l in id2label.items() if l in HAS_ADAPTER]
closed_idx = [i for i, l in id2label.items() if l in LANGS]
assert route_idx and closed_idx
print(f"open set: argmax over {len(route_idx):,} languages "
      f"(closed set was {len(closed_idx)})\n")


def decode(cell) -> np.ndarray:
    if isinstance(cell, dict) and cell.get("array") is not None:
        wav, sr = np.asarray(cell["array"], dtype="float32"), cell["sampling_rate"]
    else:
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav.astype(np.float32)


from datasets import Audio, load_dataset

open_hits = closed_hits = total = 0
confusion: dict[str, Counter[str]] = defaultdict(Counter)
low_conf: list[tuple[str, str, float]] = []

for true_lang, cfg in HF_CONFIGS.items():
    ds = load_dataset("google/WaxalNLP", cfg, split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    # === RULES GUARD: phase-1 ground truth is public and may never be read. ===
    ds = ds.remove_columns([c for c in ("transcription", "text") if c in ds.column_names])

    n = 0
    for row in ds:
        wav = decode(row["audio"])[:16000 * 30]
        inp = fe([wav], sampling_rate=16000, return_tensors="pt", padding=True)
        # attention_mask is not optional for this model — mms-lid-256 has feat_extract_norm
        # "layer", and omitting the mask is what made phase-2 routing collapse the first time.
        mask = inp.get("attention_mask")
        with torch.inference_mode():
            logits = lid(inp.input_values.to(DEVICE),
                         attention_mask=None if mask is None else mask.to(DEVICE)).logits[0]

        # Two argmaxes over one forward pass: what open-set routing picks, and what the old
        # three-class router would have picked on the same clip. Same logits, so the comparison
        # is exact rather than two runs that might have drifted.
        prob = torch.softmax(logits, dim=-1)
        open_pick = id2label[route_idx[int(logits[route_idx].argmax())]]
        closed_pick = id2label[closed_idx[int(logits[closed_idx].argmax())]]

        open_hits += open_pick == true_lang
        closed_hits += closed_pick == true_lang
        confusion[true_lang][open_pick] += 1
        if open_pick != true_lang:
            low_conf.append((true_lang, open_pick, float(prob[route_idx].max())))
        total += 1
        n += 1
        if n >= N:
            break
    print(f"  {cfg}: {n} clips")

print(f"\n--- open-set LID accuracy: {open_hits / total:.1%}  ({open_hits}/{total}) ---")
print(f"--- closed-set, same clips: {closed_hits / total:.1%}  ({closed_hits}/{total}) ---")

print("\nconfusion (rows = truth, open-set picks):")
for lang in LANGS:
    row = confusion[lang]
    got = "  ".join(f"{k}:{v}" for k, v in row.most_common(6))
    n = sum(row.values())
    print(f"  {lang}  ({n:>4})  acc {row[lang] / max(n, 1):5.1%}   {got}")

if low_conf:
    print(f"\n{len(low_conf)} misroutes, with the confidence LID had in the wrong answer:")
    for t, p, c in sorted(low_conf, key=lambda x: -x[2])[:12]:
        print(f"  {t} -> {p:<8} {c:.2f}")

drop = closed_hits - open_hits
print()
if open_hits / total >= 0.90:
    print(f"VERDICT: open-set routing holds ({open_hits / total:.1%}). Opening the label space cost"
          f" {drop} of {total} clips. Phase 2's luo/nyn-heavy mix is a property of phase 2, not of"
          f" the router. Safe to ship phase 2 on this routing.")
else:
    print(f"VERDICT: open-set routing is LEAKING — {drop} clips of {total} that the three-class"
          f" router got right now land on a neighbour. Do NOT ship phase 2 on this without a"
          f" confidence floor: route to the open-set pick only when it clears a threshold, and"
          f" fall back to the closed-set pick otherwise. Read the misroute table above to set it.")
