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

---

## 2026-07-31 — Sbu: stage 1 complete, need the calibration number from this run

Ran clean on T4. `submission_01_mms_zeroshot_phase1.csv` and `..._phase2.csv` both wrote,
0 blanks on both (4,253 and 1,500 rows).

Language mix (from `lang_map.json`, derived after the run):

- Phase 1 (id-prefix, no LID involved): lin 1866, sna 1750, lug 638 — matches the documented
  split exactly.
- Phase 2 (LID-routed, all 1,500): lug 302, sna 1, lin 0, and ~1,197 spread across other
  languages — luo 718, nyn 285, kin 107, kam 41, xog 17, wol 7, nso 5, nya 4, ibo 3, umb 3,
  yao 2, swh 2, tha 1, zul 1, bem 1.

First reaction was that this was broken — HANDOFF said to check phase 2 against ~44/41/15
lin/sna/lug and this is nowhere near that. Pulled and read before flagging it though, and found
commit `e9b3885` ("open-set LID for phase 2 — it is not lin/sna/lug") already explains it: the
old 3-class-constrained LID couldn't say "this is Luo," so it forced everything toward Luganda
regardless of truth, and the commit has real evidence for that (wrong verb prefixes, Runyankole
grammar in the forced-lug transcripts). This run used the open-set version from that commit, so
the distribution above looks like the intended behavior, not a new bug.

What I don't have: the run's own printed `LID accuracy (open set): X%` calibration line — it
scrolled out of reach in the terminal and I didn't want to spend another GPU-hour re-running
just to resurface one line. Your commit message says to read that number before trusting the
routing. Can you paste it (or re-run the calibration cell) so we know phase 2's routing is
trustworthy before either of us uploads that file? Phase 1's file doesn't depend on LID at all
(language comes straight from the id prefix), so I'll validate and hold that one ready
regardless.

Also: HANDOFF §3's "check against ~44/41/15" guidance is now stale given the open-set fix —
might be worth updating so nobody else reads that number as the target and panics like I almost
did.

Not uploading anything yet. Waiting on the calibration confirmation.

---

## 2026-07-31 — Sbu: phase 1 validated, OK to upload

Ran `local/validate_submission.py` against `submission_01_mms_zeroshot_phase1.csv` on the
Studio. Result:

```
validating as phase 1 (4,253 rows expected)
rows: 4,253 | words per utt: mean 24.8 (ref 26.0) | uppercase ratio: 0.0000 (ref 0.0133)
WARN  commas present; confirm the CSV quoting survived a round-trip
OK — safe to upload.  (review warnings first)
```

No FAILs, one WARN (generic caution on any comma in the output — only 1 comma across all 4,253
rows, and the file it's reading from is the same round-tripped CSV, so the quoting clearly held).
Distribution vs Train.csv reference looks sane. This one's ready whenever you want to submit it
for the first leaderboard score — not blocked on the phase 2 calibration question above.

Not uploading it myself without your go-ahead, since it's the first submission and I'd rather
you make that call. Say the word and I'll have Sbu submit it, or tell me to hold.
