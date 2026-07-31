"""Why is phase 2 routing 94% Luganda when LID calibrates at 97.3%?

Stage 1's LID calibration samples phase-1 ids only, because those are the ones whose language we
know from the id prefix. Phase-1 audio arrives via HuggingFace `load_dataset` + `cast_column`;
phase-2 audio arrives from a zip read with soundfile. Those are two different code paths, so a
calibration run on the first says nothing at all about the second. This script looks at the path
the calibration cannot see.

CPU only, on purpose — stage 2 is training and must not lose GPU memory to a diagnostic.

    python local/diagnose_phase2.py
"""
from __future__ import annotations

import io
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

HOME = Path("/teamspace/studios/this_studio")
WORK = (HOME if HOME.exists() else Path(__file__).resolve().parents[1]) / "waxal-work"
ZIP = WORK / "phase2_audio.zip"

if not ZIP.exists():
    raise SystemExit(f"no zip at {ZIP} — run stage 1 first")

with zipfile.ZipFile(ZIP) as zf:
    names = [n for n in zf.namelist() if not n.endswith("/")]

    # ---- 1. what is actually in the archive -------------------------------------------------
    print(f"entries (non-dir): {len(names):,}")
    print("\nby extension:")
    for ext, n in Counter(Path(x).suffix.lower() for x in names).most_common():
        print(f"  {ext or '(none)':<12} {n:,}")

    print("\nby top-level folder:")
    for top, n in Counter(x.split('/')[0] if '/' in x else '(root)' for x in names).most_common(10):
        print(f"  {top:<24} {n:,}")

    print("\nfirst 8 names:")
    for x in names[:8]:
        print(f"  {x}")

    # ---- 2. the thing stage 1 would not notice ----------------------------------------------
    # audio_store is keyed on Path(n).stem. Two entries sharing a stem means the second silently
    # overwrites the first, and nothing in stage 1's output would show it.
    stems = Counter(Path(x).stem for x in names)
    dupes = {s: c for s, c in stems.items() if c > 1}
    print(f"\nunique stems: {len(stems):,}   stems appearing more than once: {len(dupes):,}")
    if dupes:
        print("  !! stem collisions — audio_store[stem] keeps whichever comes LAST in the zip.")
        for s, c in list(dupes.items())[:5]:
            print(f"     {s} x{c}:")
            for x in [n for n in names if Path(n).stem == s]:
                print(f"       {x}")

    # ---- 3. anything that isn't audio is worth reading ---------------------------------------
    meta = [x for x in names if Path(x).suffix.lower() in
            {".csv", ".tsv", ".json", ".jsonl", ".txt", ".yaml", ".yml"}]
    if meta:
        print(f"\nNON-AUDIO FILES ({len(meta)}) — phase 2 was assumed to have no metadata:")
        for x in meta[:10]:
            print(f"  {x}   ({zf.getinfo(x).file_size:,} bytes)")
        head = zf.read(meta[0])[:400].decode("utf-8", "replace")
        print(f"\n  head of {meta[0]}:\n{head}")

    # ---- 4. do the waveforms look like the training data ------------------------------------
    audio = [x for x in names if Path(x).suffix.lower() in {".wav", ".flac", ".ogg", ".mp3", ".m4a"}]
    probe = audio[:: max(1, len(audio) // 12)][:12]
    print(f"\nprobing {len(probe)} clips (WAXAL train clips run 3-67 s, 16 kHz mono):")
    print(f"  {'name':<28}{'sr':>8}{'ch':>4}{'secs':>8}{'rms':>9}{'peak':>8}")
    srs, secs = [], []
    for x in probe:
        raw = zf.read(x)
        try:
            wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
        except Exception as e:                      # noqa: BLE001 - report, don't crash the probe
            print(f"  {Path(x).name:<28} UNREADABLE: {type(e).__name__}: {e}")
            continue
        ch = 1 if wav.ndim == 1 else wav.shape[1]
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        dur = len(wav) / sr
        srs.append(sr)
        secs.append(dur)
        print(f"  {Path(x).name:<28}{sr:>8}{ch:>4}{dur:>8.1f}"
              f"{float(np.sqrt(np.mean(wav ** 2))):>9.4f}{float(np.abs(wav).max()):>8.3f}")

    if srs:
        print(f"\n  sample rates seen: {dict(Counter(srs))}")
        print(f"  duration: min {min(secs):.1f}s  median {float(np.median(secs)):.1f}s  "
              f"max {max(secs):.1f}s")
        if float(np.median(secs)) < 2.0:
            print("  !! median under 2 s. WAXAL utterances average ~12 s; clips this short would\n"
                  "     starve both LID and CTC, and LID collapsing to one class is what that\n"
                  "     looks like from the outside.")
