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

---

## 2026-07-31 — Lethabo: upload phase 1. Calibration script attached. Stale docs fixed.

Good run, and thank you for pulling before flagging the phase-2 distribution — that saved a
round-trip and you got to the right answer on your own.

**1. Upload phase 1. That's the go-ahead.** Reasons, so you can sanity-check the call rather than
just take it: phase 1's languages come from the id prefix and never touch LID, so none of the
open-set uncertainty applies to that file; it costs 1 of 5 daily submissions against 200 total;
and an unknown baseline is worth more to us right now than a marginally better one three days
from now. Push it and paste the score here.

**2. Your phase-2 distribution is the strongest evidence we have, and it's yours.** My call in
`e9b3885` was built on 40 sampled clips. You ran all 1,500 and got luo 718 / nyn 285 / lug 302 /
kin 107 / kam 41 / xog 17 — same shape, 37x the sample, and the tail (wol, nso, ibo, umb, yao,
tha, zul) is small enough to look like genuine LID noise rather than a second population. Add
that to the log as the number we cite from here on; drop my 40-clip one.

**3. The "phase 1 validated OK" vs "FAIL 11 NaN" thing is not a conflict — different files.** I
should say this plainly because on the surface it looks like one of us is wrong. I validated a
CSV built *before* `e9b3885`; that commit bundled three changes, not one: open-set routing, the
`BLANK_FILL` write-out, and apostrophe folding (`git log -S BLANK_FILL` and `-S APOSTROPHES` both
return only `e9b3885`). Your run was on `e9b3885`, which is exactly why you saw 0 blanks and no
`['ʼ', '’']` warning where I saw 11 NaN and the apostrophe warning. Your file is the good one.

**4. The calibration number: `python local/calibrate_lid_openset.py` (new, committed).** You were
right not to burn a GPU-hour re-running stage 1 for one print. The script does LID only over ~100
phase-1 clips per language — no MMS decoding, no submission — using the *identical* routing rule
stage 1 uses, and prints open-set accuracy, closed-set accuracy on the same logits, the confusion
matrix, and the confidence LID had in each wrong answer. CPU is fine, ~10 min, so it won't touch
stage 2's GPU. ≥90% and phase 2 routing is sound; below that it prints what a confidence floor
would need to look like. Paste the verdict here.

Note what that number does and doesn't tell us. It measures whether opening the label space makes
LID leak *known-Luganda* clips onto neighbours — a real risk and the one you're asking about. It
cannot tell us whether the phase-2 references are actually transcribed in Dholuo, because we have
no phase-2 labels and never will before the deadline. My position: even if they aren't, those
clips were producing forced-Luganda garbage before, so open-set can't lose us much and might win
a lot. Uncertainty stated rather than hidden.

**5. Stale docs — all fixed, and I swept the rest as you suggested.** Good catch, and the
`~44/41/15` one was the dangerous one: it would have made someone reject a *correct* phase-2 file.
Fixed in `HANDOFF.md` (both places, with the withdrawal called out explicitly so nobody trusts an
old copy), `README.md`, and the stage-3 printout. Also fixed: the two `Kaggle -> Output -> New
Dataset` lines, the `setup_lightning.sh` claim that an L4 "costs about the same" as a T4 (it does
not — 1.58 vs 0.19 credits/hour on a 15-credit monthly grant, and that line could have quietly
eaten your month), and the "Run on Kaggle GPU" headers on stages 1/2/3.

**6. One thing you should know before it bites you: stage 3 cannot decode phase 2 yet.** It still
routes closed-set, and unlike stage 1 it can't simply be switched — what it decodes *with* is a
w2v-bert fine-tuned on lin/sna/lug transcripts plus one KenLM per those three languages, and
neither can emit Dholuo. So **do not assume stage 3's phase-2 file beats stage 1's.** Stage 3 wins
on phase 1; stage 1's open-set output is currently the better phase-2 file. I've put that warning
in the script and in HANDOFF §3 so it can't be missed at 2am. The fix is a stage-3 split
(fine-tuned + KenLM for phase 1 and the lug slice of phase 2, MMS adapters for the rest) — I'm
designing it now, and if you want to take it instead, say so and it's yours.

Open question I don't have an answer to yet, if you want to dig: do HPLT or GlotCC carry enough
`luo` and `nyn` text to build KenLMs for them? If yes, the non-target clips get shallow fusion
too and the phase-2 file improves a lot. If no, they get greedy MMS and that's the ceiling.

---

## 1 Aug — Lethabo: everything is on Kaggle now, and a real bug in the LM corpus

**Lightning is out.** Not a preference — the paid GPUs are a rules problem ("no paid services or
free trials that require a credit card") and the free tier can't carry stage 2. Everything runs on
Kaggle from here. The cost of that call: the phase-1 and phase-2 CSVs you generated live on that
studio, so as of this morning **we have no submission file at all**. That's the thing I'm fixing
first, ahead of anything clever.

**1. Kaggle's real constraint is 2 concurrent batch sessions, not the GPU count.** I found this by
being refused: `Maximum batch GPU session count of 2 reached`. It counts CPU kernels too — our
CPU LM kernel was holding the slot. So the 30 GPU-hours/week is the budget, but *two jobs at a
time* is the schedule, and that's what actually decides the ordering. Worth knowing before you
queue anything.

**2. Stage 1 now has a kernel (`kaggle/kernels/baseline/`) and is queued behind the LM kernel.**
It's the zero-shot MMS pass — no training, ~1.5 h, writes both phase CSVs — so it's what ends the
no-submission state. Two things the wrapper adds around your script: it runs
`local/validate_submission.py` on both CSVs *inside* the kernel, so a malformed file is caught
before download rather than after it has burned one of five daily slots; and it deletes
`phase2_audio.zip` from `/kaggle/working` afterwards, since everything there becomes the kernel's
output and gets counted against the 20 GB cap.

**3. Found a genuine bug in stage 0, and it's on the lever we care most about.** Train.csv is
17,063 ASCII `'` and **zero** curly apostrophes — I counted — so `'` is the only apostrophe in the
stage 2 CTC vocab. Scraped web text is the opposite: U+2019 everywhere. `normalise()` kept both
characters and folded neither, while `CHARSET` comes from that Train-derived vocab. So we paid
twice, silently: `acceptable()` drops a line once >2% of its characters are outside CHARSET, and
one curly apostrophe in a short sentence is already ~1.5–3%, so whole sentences were being thrown
out of the corpus over punctuation; and the longer lines that passed kept a character the acoustic
model cannot emit, so pyctcdecode could never match those words. It bites hardest exactly where we
need the LM most — Luganda's `ng'` digraph puts an apostrophe inside very common words.

Fixed in `7b9cd95`. **Consequence: the LM kernel running right now was launched before the fix, so
its corpus is the degraded one and needs a rebuild.** I'm letting it finish rather than killing it
— it also produces the open-set LID number you asked for, which this bug doesn't touch — then
rebuilding once stage 1 has its slot. If you get to it first, it's just a re-push of
`kaggle/kernels/lm/`.

**4. Stage 3 was reading its inputs from a mount name that never exists.** On Kaggle a kernel's
output mounts at `/kaggle/input/<kernel-slug>`, so the checkpoint arrives as `waxal-stage2-train`
and stage 1's lang_map as `waxal-baseline` — stage 3 asked for both under `waxal-ckpt`. The
checkpoint load would have failed outright and the lang_map would have been silently skipped,
re-LID'ing every clip. `ART()` now resolves by *content* (find the mount that actually contains
`w2vbert-waxal` / `lm_corpus` / `lang_map.json`) with a `WAXAL_<NAME>` override, so renaming a
kernel doesn't require editing the script. Also added a preflight: the model doesn't load until
after ~5 GB of phase-2 audio has downloaded, so a missing checkpoint now fails in the first second
and prints the mounts it can see.

**5. Your open question is still open** (do HPLT/GlotCC carry enough `luo`/`nyn` text for KenLMs).
The apostrophe fix makes it slightly more likely to be worth it, since the same fold applies to
whatever those corpora give us. Still yours if you want it.
