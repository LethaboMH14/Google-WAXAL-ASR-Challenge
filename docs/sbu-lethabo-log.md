# Sbu <-> Lethabo log

Running back-and-forth between Sbu (executing on Lightning AI) and Lethabo (design/plan owner).
Newest entries at the bottom. Pull before you read, push after you write. Keep entries short —
this is a status channel, not the design doc (that's README.md/HANDOFF.md).

---

## 2026-07-31 — Sbu: stage 0 complete on Lightning (CPU studio)

All three languages passed, no THIN/TOO SMALL:

| Language | Corpus words | Target | % | Verdict |
|---|---|---|---|---|
| Lingala | 7,826,099 | 9,025,326 | 86.7% | OK |
| Shona | 19,367,839 | 5,366,168 | 360.9% | OK |
| Luganda | 7,482,535 | 9,168,243 | 81.6% | OK |

Shona overshot because `HPLT2.0_cleaned` hit its `max_words` cap at 12,000,087 words for that
language alone, before the other sources were even added — the script's own safeguard, not an
error.

Two minor notes, neither blocking:
- Saw an "unauthenticated requests to HF Hub, set HF_TOKEN" warning while resolving datasets.
  Everything still completed fine — flagging in case rate limits bite on stage 2/3's larger
  streaming pulls.
- `lm_sources.json` (the rules-required external-data disclosure) wrote successfully.

Also: `00_build_lm_corpus.py`'s closing line still prints `Now: Kaggle -> Output -> New Dataset
-> waxal-lm, mounted by stage 3` — leftover from before the Lightning move. Ignored it; output
already lives in persistent `waxal-work/` and stage 3 reads it directly. Same class of issue as
the stale "prefer L4" line flagged earlier in `setup_lightning.sh`. Two Kaggle-era printouts
found stale now — might be worth a quick sweep across all four scripts for others before they
cause confusion mid-run.

Moving to stage 1 (T4 GPU) next.
