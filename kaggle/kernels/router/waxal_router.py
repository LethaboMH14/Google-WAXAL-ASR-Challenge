"""Kaggle GPU kernel: THE ROUTER — the single biggest lever we have, measured.

WHY THIS IS THE MOST IMPORTANT KERNEL IN THE REPO
-------------------------------------------------
Our one real leaderboard observation is 0.491944 (submission GNXR4Rkc, phase 2). The same
checkpoints, scored on dev with the language KNOWN, give 0.7453:

    mms-300m-waxal-lin  0.6893  x 0.459 word share
    mms-300m-waxal-sna  0.7815  x 0.366
    mms-300m-waxal-lug  0.8163  x 0.175   -> 0.7453 pooled, LM-free, no leaks

A 0.2533 gap. It is not the acoustic models, and after the two KenLM leaks were fixed it is not
the harness either. It is the router. Solving 0.4919 = p*0.7453 + (1-p)*m for the routing accuracy
p gives p = 0.66 if a misrouted clip scores 0, and p = 0.43 if it scores 0.30 (which is the
realistic figure: decoding Lingala with the Dholuo adapter still emits plausible Latin-script Bantu,
so CER stays well under 1).

That range is corroborated directly. local/diagnose_lid_unconstrained.py ran mms-lid-256 open-set
over phase-2 clips and got luo 42.5% / lug 27.5% / nyn 20% / guz,xog,kin,kam 2.5% each —
ZERO Lingala and ZERO Shona, on a corpus that is 43.9% Lingala and 41.1% Shona. We shipped that
routing. Most of the 1,500 clips that decide the prize were decoded in the wrong language.

So: the acoustic work is not the bottleneck and never was. Route correctly and the SAME models go
from 0.49 to 0.745; route correctly with the bakeoff lineup (0.7984 oracle) and we clear the
0.7257 at the top of the leaderboard. Nothing else available to us is worth a quarter of a point.

WHY A GENERIC LID KEEPS FAILING, AND WHAT TO USE INSTEAD
--------------------------------------------------------
mms-lid-256 has never seen this corpus. Asked to choose among 256 languages it picks the
neighbouring Bantu language it knows better; forced to choose among three it collapses onto
Luganda for anything Ugandan. Both failures are the same failure: it is scoring language identity
in the abstract, on audio whose channel, speakers and register it has no familiarity with.

The models that DO know this corpus are our own ASR checkpoints — fine-tuned on WaxalNLP itself.
A CTC model's per-frame confidence is a direct answer to "does this audio explain well under this
language's phone and character inventory". Run all three and take the most confident. That is the
`asr-conf` router below, and it costs three cheap forward passes per clip.

Second, independent candidate: Whisper's own language head. One decoder step over the encoder
output gives a posterior over language tokens from a completely different model family and
different training data than mms-lid, and different failure modes are what makes a vote worth
taking. Note the limit, verified before this ran: whisper-large-v3 carries <|ln|> and <|sn|> but
NOT <|lg|> — convert_tokens_to_ids returns unk_token_id for it. So Whisper is a Lingala/Shona
discriminator, worth keeping because those two are 82.5% of the metric's words, and it abstains
rather than votes whenever the question is about Luganda.

WHAT IS MEASURED, ON WHAT, AND WHY IT IS COMPARABLE
---------------------------------------------------
The labelled set is the first WAXAL_LID_N clips of each language's phase-1 TEST split, streamed in
file order — byte-for-byte the same clips waxal_lid_probe.py scores, so `asr-conf` and `whisper-lid`
here can be read directly against `mms-closed` / `mms-open` / `okwija` there. Ids give the
language for free.

RULES GUARD: the phase-1 ground-truth TRANSCRIPTS are public and using them is explicit grounds for
disqualification. Reading `lug_96114` -> "lug" is reading an id, not a label. The transcription
columns are dropped from the stream on arrival and nothing here can reach them.

SEED 1337, argmax everywhere, no sampling: rerunning reproduces the routing exactly.
"""

import io
import json
import os
import re
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

REPO_URL = "https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge"
REPO = Path("/kaggle/repo")
WORKING = Path("/kaggle/working")
PHASE2_URL = "https://storage.googleapis.com/waxalphase2/audio.zip"

LANGS = ["lin", "sna", "lug"]
HF_CONFIGS = {"lin": "lin_asr", "sna": "sna_asr", "lug": "lug_asr"}

# Same family, same recipe, same tokenizer style for all three — which is the whole point. Scores
# from three checkpoints are only comparable if the checkpoints are comparable, so the router does
# NOT use the bakeoff lineup (w2v-bert for lin, Whisper for sna, mms-300m for lug): a Whisper
# average log-prob and a CTC per-frame log-prob are not the same quantity and arg-maxing across
# them would measure architecture, not language. Route with the uniform trio, then DECODE with the
# per-language winners.
ROUTER_ASR = {
    "lin": "waxal-benchmarking/mms-300m-waxal-lin",
    "sna": "waxal-benchmarking/mms-300m-waxal-sna",
    "lug": "waxal-benchmarking/mms-300m-waxal-lug",
}
WHISPER_LID = "openai/whisper-large-v3"
WHISPER_CODE = {"ln": "lin", "sn": "sna", "lg": "lug"}   # Whisper's ISO-639-1 -> our ISO-639-3

N_PER_LANG = int(os.environ.get("WAXAL_LID_N", "400"))
MAX_SECONDS = 20
SEED = 1337

# Phase-1 truth, from id prefixes over all 4,253 rows. A phase-2 routing far from this is evidence
# of router bias, not of a differently-composed corpus.
PHASE1_MIX = {"lin": 0.439, "sna": 0.411, "lug": 0.150}
ORACLE = 0.7453          # mms-300m trio, dev, language known, LM-free
LB = 0.491944347         # what that same trio scored on phase 2, routed by open-set mms-lid


def sh(cmd, check=True, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=check, **kw)


if not REPO.exists():
    sh(["git", "clone", "--depth", "1", REPO_URL, str(REPO)])
sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements-gpu.txt")])

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402

if not torch.cuda.is_available():
    raise SystemExit("no CUDA — set the kernel accelerator to GPU T4 before running")
DEVICE = "cuda"
print(f"gpu: {torch.cuda.get_device_name(0)}")

# ------------------------------------------------------------------ 1. labelled audio
labelled: dict[str, np.ndarray] = {}
truth: dict[str, str] = {}
for lang, cfg in HF_CONFIGS.items():
    ds = load_dataset("google/WaxalNLP", cfg, split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    # RULES GUARD — see the module docstring. Drop them on arrival, not at point of use.
    ds = ds.remove_columns([c for c in ("transcription", "text") if c in ds.column_names])
    n = 0
    for row in ds:
        w = np.asarray(row["audio"]["array"], dtype=np.float32)
        if w.ndim > 1:
            w = w.mean(axis=1)
        labelled[str(row["id"])] = w[:16000 * MAX_SECONDS]
        truth[str(row["id"])] = lang
        n += 1
        if n >= N_PER_LANG:
            break
    print(f"  {lang}: {n} labelled clips", flush=True)
print(f"labelled set: {len(labelled):,} clips")

# ------------------------------------------------------------------ 2. phase-2 audio
phase2: dict[str, np.ndarray] = {}
zp = WORKING / "phase2_audio.zip"
if not zp.exists():
    os.system(f"wget -q -O {zp} {PHASE2_URL}")
if zp.exists() and zp.stat().st_size > 0:
    import librosa
    import soundfile as sf

    # The zip is not purely audio; a member that soundfile cannot open killed the LID probe's first
    # run. Filter by extension, then catch and COUNT what still fails — a router that silently
    # drops clips is worse than one that crashes, because the dropped rows go out blank.
    AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}
    skipped, failed = [], []
    with zipfile.ZipFile(zp) as zf:
        for nm in [x for x in zf.namelist() if not x.endswith("/")]:
            if nm.startswith("__MACOSX") or Path(nm).suffix.lower() not in AUDIO_EXT:
                skipped.append(nm)
                continue
            try:
                with zf.open(nm) as fh:
                    w, sr = sf.read(io.BytesIO(fh.read()), dtype="float32")
            except Exception as e:  # noqa: BLE001
                failed.append(f"{nm}: {type(e).__name__}")
                continue
            if w.ndim > 1:
                w = w.mean(axis=1)
            if sr != 16000:
                w = librosa.resample(w, orig_sr=sr, target_sr=16000)
            phase2[Path(nm).stem] = w.astype(np.float32)[:16000 * MAX_SECONDS]
    if skipped:
        print(f"  skipped {len(skipped):,} non-audio member(s), e.g. {skipped[:5]}")
    if failed:
        print(f"  FAILED to decode {len(failed):,} member(s), e.g. {failed[:5]}")
print(f"phase 2: {len(phase2):,} clips")
if phase2 and len(phase2) < 1500:
    print(f"  WARNING: expected 1,500 phase-2 clips, loaded {len(phase2):,}")

ALL = {**labelled, **phase2}
print(f"total to score: {len(ALL):,} clips\n")


# ------------------------------------------------------------------ 3. router A: ASR confidence
def ctc_confidence(model_id: str, clips: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Per-clip confidence features from one language's CTC model.

    Batch size 1, deliberately. Every feature here is a mean over TRUE frames, and in a padded
    batch the tail frames are padding — including them makes a short clip in a long batch look
    different from the same clip in a short batch, which would make the routing depend on batch
    composition. Computing valid lengths per architecture is doable but the checkpoints span
    Wav2Vec2ForCTC and Wav2Vec2BertForCTC with different downsampling; one clip at a time is
    ~100 ms on a T4 and removes the entire class of bug.
    """
    from transformers import AutoModelForCTC, AutoProcessor

    pr = AutoProcessor.from_pretrained(model_id)
    md = AutoModelForCTC.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE).eval()
    blank = md.config.pad_token_id if md.config.pad_token_id is not None else 0

    out: dict[str, dict[str, float]] = {}
    ids = list(clips)
    with torch.inference_mode():
        for k, i in enumerate(ids):
            inp = pr(clips[i], sampling_rate=16000, return_tensors="pt")
            feats = inp["input_values"] if "input_values" in inp else inp["input_features"]
            logits = md(feats.to(DEVICE).half()).logits.float()[0]        # (T, V)
            lp = torch.log_softmax(logits, dim=-1)
            top, arg = lp.max(dim=-1)
            nb = arg != blank
            # Four rules, because which one discriminates is an empirical question and we get all
            # four for free once the logits exist. mean_lp is the plain greedy-path score;
            # mean_lp_nb ignores blank frames, which matters because a model hearing an unfamiliar
            # language emits blank confidently and can otherwise win on silence; neg_entropy reads
            # the whole posterior rather than just its peak; blank_frac is reported for diagnosis.
            out[i] = {
                "mean_lp": float(top.mean()),
                "mean_lp_nb": float(top[nb].mean()) if bool(nb.any()) else -20.0,
                "neg_entropy": float(-(-(lp.exp() * lp).sum(-1)).mean()),
                "blank_frac": float((~nb).float().mean()),
            }
            if k % 500 == 0:
                print(f"  {model_id.split('/')[-1]} {k}/{len(ids)}", flush=True)
    del md
    torch.cuda.empty_cache()
    return out


print("=" * 78)
print("=== ROUTER A: asr-conf — three same-family CTC models, most confident wins")
print("=" * 78, flush=True)
conf: dict[str, dict[str, dict[str, float]]] = {}
for lang, mid in ROUTER_ASR.items():
    print(f"\nscoring every clip under {lang} ({mid})", flush=True)
    conf[lang] = ctc_confidence(mid, ALL)

BASE_RULES = ["mean_lp", "mean_lp_nb", "neg_entropy"]
asr_routes: dict[str, dict[str, str]] = {}
for rule in BASE_RULES:
    asr_routes[rule] = {i: max(LANGS, key=lambda lg: conf[lg][i][rule]) for i in ALL}

# Z-NORMALISED VARIANTS, and this is not a flourish. The three vocabularies are 74 / 53 / 40
# tokens (verified from their config.json), so the models are not competing on level terms: with
# 40 classes it is mechanically easier to concentrate probability mass on one of them than with
# 74, and both mean_lp and neg_entropy inherit that bias — max entropy alone is log V. A raw
# argmax across the three would partly be measuring vocabulary size and would tilt toward Luganda.
#
# Z-scoring each model's scores over the whole corpus removes any constant per-model offset,
# whatever its cause — vocab size, calibration temperature, fine-tuning duration — and changes the
# question from "which model likes this clip most" to "which model likes this clip most relative to
# how it treats audio in general". It uses no labels, so it is legitimate on the unlabelled phase-2
# clips too; it does assume the corpus is a mix rather than one language, which it is.
_ids = list(ALL)
zconf: dict[str, dict[str, dict[str, float]]] = {lg: {} for lg in LANGS}
for rule in BASE_RULES:
    for lg in LANGS:
        v = np.array([conf[lg][i][rule] for i in _ids], dtype=np.float64)
        mu, sd = float(v.mean()), float(v.std())
        if sd < 1e-9:                       # a constant score carries no information
            print(f"  WARNING: {lg}/{rule} has zero variance across the corpus — z is undefined, "
                  f"leaving it unnormalised")
            sd = 1.0
        for i, x in zip(_ids, (v - mu) / sd):
            zconf[lg].setdefault(i, {})[rule] = float(x)
    asr_routes[f"z:{rule}"] = {i: max(LANGS, key=lambda lg: zconf[lg][i][rule]) for i in ALL}
RULES = BASE_RULES + [f"z:{r}" for r in BASE_RULES]


# ------------------------------------------------------------------ 4. router B: Whisper's LID head
def whisper_lid(clips: dict[str, np.ndarray], bs: int = 8) -> dict[str, str]:
    """One decoder step from the SOT token; argmax over <|ln|>, <|sn|>, <|lg|>.

    Whisper's feature extractor pads every clip to a fixed 30 s window, so unlike the CTC path
    there is no variable-length pooling here and batching is safe.
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    pr = WhisperProcessor.from_pretrained(WHISPER_LID)
    md = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_LID, torch_dtype=torch.float16).to(DEVICE).eval()
    # The guard has to test against unk, not against None. Verified locally on whisper-large-v3:
    #   <|ln|> -> 50353   <|sn|> -> 50324   <|lg|> -> 50257 == unk_token_id
    # Whisper covers Lingala and Shona but NOT Luganda, and convert_tokens_to_ids answers with unk
    # rather than None, so a `if t is None` check passes happily and the argmax then runs over
    # <|ln|>, <|sn|> and <|unk|> — a router with a garbage third class. Test for unk.
    unk = pr.tokenizer.unk_token_id
    tok_ids, keep = [], []
    for code, lg in WHISPER_CODE.items():
        t = pr.tokenizer.convert_tokens_to_ids(f"<|{code}|>")
        if t is None or t < 0 or t == unk:
            print(f"  <|{code}|> is not in this tokenizer (got {t!r}, unk={unk}) — "
                  f"whisper-lid cannot answer {lg}")
            continue
        tok_ids.append(t)
        keep.append(lg)
    if len(keep) < 2:
        raise SystemExit("whisper-lid: fewer than two language tokens resolved")
    if set(keep) != set(LANGS):
        print(f"  whisper-lid is a {'/'.join(keep)} discriminator only. It is kept because those "
              f"two carry 82.5% of the metric's words, but it can never return the missing "
              f"language, so it must ABSTAIN rather than vote when the question is about one.")
    sot = md.config.decoder_start_token_id

    ids, out = list(clips), {}
    with torch.inference_mode():
        for k in range(0, len(ids), bs):
            ch = ids[k:k + bs]
            inp = pr([clips[i] for i in ch], sampling_rate=16000, return_tensors="pt")
            enc = md.model.encoder(inp.input_features.to(DEVICE).half())
            dec = torch.full((len(ch), 1), sot, dtype=torch.long, device=DEVICE)
            logits = md(decoder_input_ids=dec, encoder_outputs=enc).logits[:, 0].float()
            for i, p in zip(ch, logits[:, tok_ids].argmax(-1).cpu().numpy()):
                out[i] = keep[int(p)]
            if k % 400 == 0:
                print(f"  whisper-lid {k}/{len(ids)}", flush=True)
    del md
    torch.cuda.empty_cache()
    return out


print(f"\n{'=' * 78}\n=== ROUTER B: whisper-lid — {WHISPER_LID}\n{'=' * 78}", flush=True)
try:
    wroute = whisper_lid(ALL)
except Exception as e:  # noqa: BLE001 — one dead router must not cost us the other
    print(f"  whisper-lid FAILED: {type(e).__name__}: {e}")
    wroute = {}


# ------------------------------------------------------------------ 5. score every candidate
def report(name: str, route: dict[str, str]) -> float:
    """Accuracy + per-language recall on labelled clips, language mix on phase 2."""
    hit = sum(1 for i, t in truth.items() if route.get(i) == t)
    acc = hit / max(1, len(truth))
    print(f"\n  {name}")
    print(f"    accuracy: {acc:.4f}  ({hit}/{len(truth)})")
    for lg in LANGS:
        pool = [i for i, t in truth.items() if t == lg]
        got = sum(1 for i in pool if route.get(i) == lg)
        conf_ = Counter(route.get(i) for i in pool if route.get(i) != lg)
        print(f"      {lg}: recall {got / max(1, len(pool)):.4f} ({got}/{len(pool)})"
              f"   confused with {dict(conf_.most_common(3)) if conf_ else '-'}")
    ph2 = {i: route[i] for i in phase2 if i in route}
    if ph2:
        mix = Counter(ph2.values())
        tot = sum(mix.values())
        drift = sum(abs(mix.get(lg, 0) / tot - PHASE1_MIX[lg]) for lg in LANGS) / 2
        print("    phase-2 mix: "
              + "  ".join(f"{lg} {mix.get(lg, 0) / tot:.1%}" for lg in LANGS)
              + f"   (phase-1 truth lin 43.9% / sna 41.1% / lug 15.0%; "
              f"total variation {drift:.3f})")
    # What this routing is worth, if a correct clip scores ORACLE and a misroute scores ~0.30.
    print(f"    projected score at this accuracy: {acc * ORACLE + (1 - acc) * 0.30:.4f}"
          f"   (we are at {LB:.4f})")
    return acc


print(f"\n{'=' * 78}\n=== RESULTS\n{'=' * 78}")
cands: dict[str, dict[str, str]] = {f"asr-conf[{r}]": asr_routes[r] for r in RULES}
if wroute:
    cands["whisper-lid"] = wroute

# A vote across independent families. Two things it must get right:
#
# 1. The ASR base rule is chosen by MEASURED accuracy on the labelled clips, not picked in advance.
#    Which confidence statistic discriminates best is an empirical question and we have the labels
#    to answer it, so answering it beats asserting it.
# 2. Whisper ABSTAINS on any language it cannot name. whisper-large-v3 has no <|lg|>, so on a clip
#    the ASR router calls Luganda, Whisper's "lin" is not a dissenting opinion — it is the only
#    thing Whisper is able to say. Counting it as a vote would let a model systematically overrule
#    the one language it is blind to, which is worse than not consulting it at all.
if wroute:
    _acc_of = lambda r: sum(1 for i, t in truth.items() if r.get(i) == t) / max(1, len(truth))
    _ranked = sorted(RULES, key=lambda r: _acc_of(asr_routes[r]), reverse=True)
    first, second = asr_routes[_ranked[0]], asr_routes[_ranked[1]]
    covered = set(wroute.values())
    print(f"\n  vote: ASR rules {_ranked[0]} + {_ranked[1]} (best two by measured accuracy), "
          f"whisper voting only on {sorted(covered)}")
    vote, abstained = {}, 0
    for i in ALL:
        voters = [first[i], second[i]]
        if first[i] in covered and i in wroute:
            voters.append(wroute[i])
        else:
            abstained += 1
        best_, n = Counter(voters).most_common(1)[0]
        vote[i] = best_ if n >= 2 else first[i]   # ties go to the best-measured single rule
    print(f"  whisper abstained on {abstained:,} / {len(ALL):,} clips")
    cands[f"vote({_ranked[0]} + {_ranked[1]} + whisper)"] = vote

scores = {n: report(n, r) for n, r in cands.items()}

print(f"\n{'-' * 78}\n  VERDICT\n{'-' * 78}")
best = max(scores, key=scores.get)
ba = scores[best]
print(f"  best router: {best}   accuracy {ba:.4f}")
print(f"  the shipped open-set mms-lid routing implied ~0.43-0.66 accuracy (0.4919 observed)")
print(f"  at {ba:.4f}, the SAME mms-300m trio projects ~{ba * ORACLE + (1 - ba) * 0.30:.4f};")
print(f"  the bakeoff lineup (0.7984 oracle) projects ~{ba * 0.7984 + (1 - ba) * 0.30:.4f}")
print("\n  CAVEATS, plainly: accuracy is measured on phase-1 TEST audio, and phase 2 is described")
print("  as new recordings, so this is an upper bound. The 0.30 misroute figure is an")
print("  estimate, not a measurement — it comes from solving the 0.4919 observation rather than")
print("  from scoring misrouted clips, which is what WAXAL_MISROUTE=1 on stage 3 now fixes. The")
print("  RANKING between routers does not depend on that figure; only the projection does.")

# ------------------------------------------------------------------ 6. hand the routing to stage 3
for name, route in cands.items():
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    json.dump({i: route[i] for i in phase2 if i in route},
              open(WORKING / f"lang_map_{slug}.json", "w"), indent=1)
json.dump({i: cands[best][i] for i in phase2 if i in cands[best]},
          open(WORKING / "lang_map.json", "w"), indent=1)
json.dump({"accuracy": scores, "best": best, "n_labelled": len(truth), "n_phase2": len(phase2)},
          open(WORKING / "router_result.json", "w"), indent=1)
print(f"\n  wrote lang_map.json (the winner, {best}) plus one file per candidate.")
print("  Stage 3 reads lang_map.json for phase-2 routing — drop it in as a dataset and rerun.")

# The zip is 100% reconstructible from PHASE2_URL and /kaggle/working is the output volume.
if zp.exists():
    mb = zp.stat().st_size / 1e6
    zp.unlink()
    print(f"  cleaned {mb:,.0f} MB of regenerable cache from the output volume")
